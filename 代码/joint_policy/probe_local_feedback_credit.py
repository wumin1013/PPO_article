"""Same-state complete suffix returns outside the learned mixture interval."""
import time
import traceback
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from bounded_shape_return import terminal_shape_cost
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def objective(reference, rows, actions, env=None):
    cost, details=terminal_shape_cost(reference,rows,actions)
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    value=-len(rows)*.001+length/100-float(cost.sum())
    if env is not None and not env.success:
        value-=env.max_ticks*.001+1+max(0,length-env.progress()[0])/20
    return value,details


def main():
    torch.set_num_threads(1)
    source=BASE/'constrained_parameter_step_probe_20260916';model=source/'candidate_1.pt'
    original_ids=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    indices=[4,9,13,14,18,53]
    deltas=[np.eye(3)[i]*sign*np.array([.02,.3,.3]) for i in range(3) for sign in (-1,1)]
    deltas += [np.array([.02,-.3,.3]), np.array([-.02,.3,-.3])]
    stage=register('local_feedback_credit_probe_20260916',
        [__file__,model,source/'receipt.json',BASE/'probe_coordinated_suffix.py',BASE/'frame_feedback_policy.py',
         BASE/'bounded_shape_return.py',BASE/'contextual_support_credit_audit_20260916/receipt.json']+
        [source/'training_3'/f'{original_ids.index(i)}.npz' for i in indices],
        dict(training_updates=0,new_suffix_decisions=20000,new_physics_ticks=430000,wall_s=400,
             training_paths=indices,anchors_per_path=3,suffixes_per_anchor=8,
             latent_offsets=[d.tolist() for d in deltas],intervention_decisions=4),
        'Use exact saved states from six existing training paths, three anchors per path:80ms before '
        'strongest actual turning, at that turning, and a low-turn high-speed state. For straight paths use '
        'quarter/middle/three-quarter decisions. Six axial plus two explicitly joint latent interventions '
        'last80ms then frozen feedback resumes until complete stop or failure. Reuse all baseline physics. '
        'This is a local action-support/credit diagnostic, not learned performance or a deployable schedule. '
        'Original failure penalties and bounded terminal shape cost are retained. No development or sealed path used.')
    saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=FrameFeedbackPolicy(frame='heading',history_independent=True);policy.load_state_dict(saved['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    started=time.perf_counter();records=[];baselines=[]
    try:
        for path in indices:
            trace=source/'training_3'/f'{original_ids.index(path)}.npz'
            with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            base_return,base_cost=objective(ref,rows,actions)
            speed=np.linalg.norm(rows[:,2:4],axis=1)
            live=np.flatnonzero(rows[:,12]==0);n=min(len(actions)-1,(live[-1]+1)//20)
            if len(ref)==2:anchors=[n//4,n//2,3*n//4]
            else:
                peak=int(np.argmax(np.abs(rows[live,10]))//20)
                sample=np.arange(n)*20
                score=speed[sample]/100-abs(rows[sample,10])/10
                anchors=[max(0,peak-4),min(peak,n-1),int(np.argmax(score))]
            anchors=list(dict.fromkeys(anchors))
            for j in (n//4,n//2,3*n//4):
                if len(anchors)>=3:break
                if j not in anchors:anchors.append(j)
            baselines.append(dict(path=path,source_trace=str(trace.relative_to(BASE)),source_sha256=sha(trace),
                return_=base_return,time_s=len(rows)*.001,cost=base_cost,anchors=anchors))
            for anchor in anchors:
                frame=anchor*20;env=reconstruct(ref,rows,actions,frame,costs);initial=env.state()
                costs['actor_calls']+=1
                assert np.array_equal(env.decode_action(policy.act(env.observation(),True)[0]),actions[anchor])
                for k,delta in enumerate(deltas):
                    env.restore(initial);chunks=[];held=[];observations=[];base_latents=[];latents=[];count=0
                    while not env.done:
                        assert time.perf_counter()-started<400
                        assert costs['actor_calls']<20000 and costs['new_physics_ticks']<427880
                        obs=env.observation();latent=policy.act(obs,True)[0];costs['actor_calls']+=1
                        action=latent+delta if count<4 else latent
                        observations.append(obs);base_latents.append(latent);latents.append(action)
                        _,_,_,info=env.step(action);chunks.append(info['rows']);held.append(info['held_action']);count+=1
                        costs['new_physics_ticks']+=len(info['rows'])
                    full=np.vstack([rows[:frame],*chunks]);allactions=np.vstack([actions[:anchor],held])
                    audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    value,details=objective(ref,full,allactions,env)
                    shape=corner_shape(ref,full,allactions) if len(ref)==3 else None
                    if shape:shape['spatial']=geometric_turn_metrics(ref,full)
                    name=f'p{path}_a{anchor}_d{k}.npz'
                    np.savez_compressed(stage/name,reference=ref,rows=full,actions=allactions,
                        observation=np.asarray(observations),base_latent=np.asarray(base_latents),latent=np.asarray(latents),
                        anchor=anchor,delta=delta,intervention_decisions=min(4,count))
                    record=dict(path=path,anchor=anchor,delta_index=k,delta=delta.tolist(),trace=name,
                        success=success,reason=env.reason,return_=value,delta_return=value-base_return,
                        full_time_s=env.ticks*.001 if success else None,shape_cost=details,shape=shape,
                        audit=audit,geometry=env.tracker.snapshot(1.25),decisions=count)
                    records.append(record)
                subset=records[-8:]
                print('LOCAL',path,anchor,'successful',sum(r['success'] for r in subset),'improvements',
                      [(r['delta_index'],round(r['delta_return'],5)) for r in subset if r['success'] and r['delta_return']>.002],flush=True)
                write_json(stage/'results.json',dict(baselines=baselines,records=records,costs=costs,goal_complete=False))
        receipt(stage,dict(status='completed_local_complete_suffix_credit_probe',costs=costs,
                            suffixes=len(records),wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
