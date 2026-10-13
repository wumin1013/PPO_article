"""Training-only 2x2x2 entry-action interaction probe on the new learned actor."""
import itertools
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from complete_corner_metrics import complete_corner_metrics,coverage_costs
from bounded_shape_return import terminal_shape_cost
from probe_coordinated_suffix import reconstruct
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def score(ref,rows,actions,env=None):
    _,original=terminal_shape_cost(ref,rows,actions,discrete_first_label=False)
    shape=complete_corner_metrics(ref,rows)
    variants=coverage_costs(original,shape)
    length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
    base=-len(rows)*.001+length/100
    if env is not None and not env.success:
        base-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
    return dict(return_window=base-variants['window_two_edges']['cost'],
                return_full=base-variants['full_two_edges']['cost'],shape=shape,
                quality_window=variants['window_two_edges']['quality'],quality_full=variants['full_two_edges']['quality'])


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916'
    model=parent/'candidate_1.pt';source=parent/'training_1'
    indices=[0,1,6,7,12,13,14,15]
    stage=register('entry_action_interaction_probe_20260916',
        [__file__,model,parent/'receipt.json',BASE/'local_geometry_feedback_policy.py',
         BASE/'complete_corner_metrics.py',BASE/'bounded_shape_return.py',BASE/'probe_coordinated_suffix.py']+
         [source/f'{i}.npz' for i in indices],
        dict(new_suffixes=112,new_decisions=12000,new_physics_ticks=300000,wall_s=180,
             paths=indices,anchors_per_path=2,intervention_decisions=3,
             latent_offsets=dict(steering_signed_inward=.025,feed=-.3,preview=.3)),
        'Complete2x2x2 action-factorial probe at the new learned local-credit candidate. All paths are '
        'existing training geometries; both directions and20/80/140/155deg are represented. Anchors '
        'are80ms and20ms before maximum signed physical angular velocity, based on saved trajectories. '
        'Each nonzero combination lasts60ms, then original learned feedback resumes until full stop. '
        'Reuse000 baseline. Measure true XY relative to both edges, full heading/curvature and success, '
        'time and unchanged current return. Also rescore the existing complete-extent diagnostic to '
        'expose window-dependent rankings; this is analysis, not a simultaneous reward/controller change. '
        'Adopt an interaction hypothesis only if complete combined branches outperform relevant single '
        'factors on actual shape without introducing entry excursion or completion failure. These finite '
        'interventions never become runtime rules; do not claim learned behavior or generalized performance.')
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    started=time.monotonic();baselines=[];records=[];interactions=[]
    for path in indices:
        with np.load(source/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        base=score(ref,rows,actions);sign=np.sign(base['shape']['turn_deg'])
        speed=np.linalg.norm(rows[:,2:4],axis=1);omega=np.zeros(len(rows));valid=(speed>5)&(rows[:,12]==0)
        omega[valid]=sign*(rows[valid,2]*rows[valid,5]-rows[valid,3]*rows[valid,4])/speed[valid]**2
        peak=int(np.argmax(omega))//20;anchors=[max(0,peak-4),max(0,peak-1)]
        assert len(set(anchors))==2
        baselines.append(dict(path=path,**base,time_s=len(rows)*.001,anchors=anchors,
                              source_trace_sha256=sha(source/f'{path}.npz')))
        for anchor in anchors:
            frame=anchor*20;prefix=[]
            env=reconstruct(ref,rows,actions,frame,costs,observations=prefix);initial=env.state()
            costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(policy.act(env.observation(),True)[0]),actions[anchor])
            cells={(0,0,0):dict(**base,success=True)}
            for bits in itertools.product((0,1),repeat=3):
                if not any(bits):continue
                delta=np.asarray(bits)*np.array([sign*.025,-.3,.3])
                env.restore(initial);chunks=[];held=[];obs=[];latents=[];count=0
                while not env.done:
                    assert time.monotonic()-started<180
                    assert costs['actor_calls']<12000 and costs['new_physics_ticks']<297880
                    x=env.observation();latent=policy.act(x,True)[0];costs['actor_calls']+=1
                    action=latent+delta if count<3 else latent
                    _,_,_,info=env.step(action)
                    chunks.append(info['rows']);held.append(info['held_action']);obs.append(x);latents.append(action);count+=1
                    costs['new_physics_ticks']+=len(info['rows'])
                full=np.vstack([rows[:frame],*chunks]);allactions=np.vstack([actions[:anchor],held])
                audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                value=score(ref,full,allactions,env)
                name=f'p{path}_a{anchor}_'+''.join(map(str,bits))+'.npz'
                np.savez_compressed(stage/name,reference=ref,rows=full,actions=allactions,
                    observation=np.asarray(obs),prefix_observation=np.asarray(prefix),latent=np.asarray(latents),
                    anchor=anchor,delta=delta,intervention_decisions=min(count,3))
                item=dict(path=path,anchor=anchor,bits=list(bits),delta=delta.tolist(),trace=name,
                          success=success,reason=env.reason,full_time_s=len(full)*.001 if success else None,
                          audit=audit,geometry=env.tracker.snapshot(1.25),**value)
                records.append(item);cells[bits]=item
            cross=[]
            for axes in itertools.combinations(range(3),2):
                a=np.zeros(3,int);b=np.zeros(3,int);a[axes[0]]=1;b[axes[1]]=1
                cross.append(dict(axes=list(axes),window_return_interaction=float(cells[tuple(a+b)]['return_window']-
                    cells[tuple(a)]['return_window']-cells[tuple(b)]['return_window']+base['return_window']),
                    all_relevant_complete=all(cells[tuple(k)]['success'] for k in [a,b,a+b])))
            interactions.append(dict(path=path,anchor=anchor,pairwise=cross))
            candidates=[r for bits,r in cells.items() if r['success']]
            best=max(candidates,key=lambda r:r['return_window'])
            print('ENTRY_FACTORIAL',path,anchor,'success',sum(r['success'] for r in cells.values()),
                  'best',best.get('bits',[0,0,0]),'gain',round(best['return_window']-base['return_window'],4),
                  'E',round(base['shape']['full_normalized_energy'],3),'->',round(best['shape']['full_normalized_energy'],3),
                  'exit',round(base['shape']['out_edge_outward_mm'],3),'->',round(best['shape']['out_edge_outward_mm'],3),flush=True)
            write_json(stage/'results.json',dict(baselines=baselines,records=records,interactions=interactions,costs=costs))
    receipt(stage,dict(status='completed_entry_action_factorial_probe',costs=costs,
                        suffixes=len(records),successful=sum(r['success'] for r in records),wall_s=time.monotonic()-started))


if __name__=='__main__':main()
