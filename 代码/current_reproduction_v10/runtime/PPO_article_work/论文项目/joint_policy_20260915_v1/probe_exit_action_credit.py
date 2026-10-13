"""Same-state local exit-action credit under the audited two-edge objective."""
import time
import numpy as np
import torch
from bounded_shape_return import terminal_shape_cost
from complete_corner_metrics import complete_corner_metrics
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def complete_return(ref,rows,actions,env=None):
    cost,details=terminal_shape_cost(ref,rows,actions,discrete_first_label=False,two_edges=True)
    length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
    value=-len(rows)*.001+length/100-cost.sum()
    if env is not None and not env.success:
        value-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
    return float(value),details


def main():
    torch.set_num_threads(1)
    source=BASE/'local_geometry_full_training_audit_20260916/training_pool'
    model=BASE/'local_geometry_continuous_reward_ppo_20260916/policy_18645.pt'
    indices=[0,1,2,3,6,7,14,15]
    stage=register('exit_action_credit_probe_20260916',
        [__file__,model,BASE/'bounded_shape_return.py',BASE/'complete_corner_metrics.py',
         BASE/'local_geometry_feedback_policy.py',BASE/'probe_coordinated_suffix.py',
         BASE/'complete_two_edge_shape_audit_20260916/receipt.json',
         BASE/'two_edges_side_coverage_ppo_20260916/receipt.json']+[source/f'{i}.npz' for i in indices],
        dict(new_suffixes=96,new_decisions=16000,new_physics_ticks=350000,wall_s=240,
             paths=indices,anchors_per_path=2,intervention_decisions=3,latent_offset=[.025,.3,.3]),
        'Diagnose whether local action changes can improve true exit shape and complete two-edge return '
        'at the retained18645 policy, after the matched PPO side-cost change failed. Eight existing training '
        'corners span both directions and20/35/80/155deg. For each use the saved closest-vertex decision '
        'and the decision40ms before first exit-half-plane crossing by0.005mm. Six single-axis +/- '
        'interventions last60ms; resume the unchanged feedback to full stop. Reuse all baseline physics. '
        'Adopt only a credit/action-support hypothesis supported by successful complete improvements; '
        'these interventions are not learned behavior and are never deployed. No new objective weights, '
        'angle speed tables, development probes or sealed tests.')
    saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(saved['policy'])
    costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0,training_updates=0)
    started=time.monotonic();records=[];baselines=[]
    for path in indices:
        with np.load(source/f'{path}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        base_return,base_cost=complete_return(ref,rows,actions)
        base_shape=complete_corner_metrics(ref,rows)
        sign=np.sign(base_shape['turn_deg']);out=ref[2]-ref[1];out/=np.linalg.norm(out)
        side=(rows[:,:2]-ref[1])@(sign*np.array([-out[1],out[0]]))
        first=int(np.flatnonzero(side<-.005)[0])
        apex=int(np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1)))
        anchors=list(dict.fromkeys([apex//20,max(0,first//20-2)]))
        if len(anchors)==1:anchors.append(max(0,anchors[0]-3))
        baselines.append(dict(path=path,return_=base_return,time_s=len(rows)*.001,cost=base_cost,
                              shape=base_shape,anchors=anchors,trace_sha256=sha(source/f'{path}.npz')))
        for anchor in anchors:
            frame=anchor*20;prefix_obs=[]
            env=reconstruct(ref,rows,actions,frame,costs,observations=prefix_obs);initial=env.state()
            costs['actor_calls']+=1
            assert np.array_equal(env.decode_action(policy.act(env.observation(),True)[0]),actions[anchor])
            for axis in range(3):
                for direction in (-1,1):
                    delta=np.zeros(3);delta[axis]=direction*[.025,.3,.3][axis]*(sign if axis==0 else 1.)
                    env.restore(initial);chunks=[];held=[];observations=[];latents=[];count=0
                    while not env.done:
                        assert time.monotonic()-started<240
                        assert costs['actor_calls']<16000 and costs['new_physics_ticks']<347880
                        obs=env.observation();latent=policy.act(obs,True)[0];costs['actor_calls']+=1
                        action=latent+delta if count<3 else latent
                        _,_,_,info=env.step(action)
                        observations.append(obs);latents.append(action);held.append(info['held_action']);chunks.append(info['rows'])
                        costs['new_physics_ticks']+=len(info['rows']);count+=1
                    full=np.vstack([rows[:frame],*chunks]);allactions=np.vstack([actions[:anchor],held])
                    audit=difference_audit(ref[0],full)
                    success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                    value,details=complete_return(ref,full,allactions,env)
                    shape=complete_corner_metrics(ref,full)
                    name=f'p{path}_a{anchor}_axis{axis}_{direction:+d}.npz'
                    np.savez_compressed(stage/name,reference=ref,rows=full,actions=allactions,
                        prefix_observation=np.asarray(prefix_obs),observation=np.asarray(observations),
                        latent=np.asarray(latents),anchor=anchor,delta=delta,intervention_decisions=min(count,3))
                    records.append(dict(path=path,anchor=anchor,axis=axis,direction=direction,delta=delta.tolist(),
                        trace=name,success=success,reason=env.reason,return_=value,delta_return=value-base_return,
                        full_time_s=len(full)*.001 if success else None,cost=details,shape=shape,
                        audit=audit,geometry=env.tracker.snapshot(1.25),decisions=count))
            selected=records[-6:]
            best=max(selected,key=lambda r:r['return_'])
            print('EXIT_PROBE',path,anchor,'success',sum(r['success'] for r in selected),
                'best',best['axis'],best['direction'],'gain',round(best['delta_return'],4),
                'exit',round(base_shape['out_edge_outward_mm'],3),'->',round(best['shape']['out_edge_outward_mm'],3),flush=True)
            write_json(stage/'results.json',dict(baselines=baselines,records=records,costs=costs))
    good=[r for r in records if r['success'] and r['delta_return']>.005]
    strict=[r for r in good if r['shape']['out_edge_outward_mm']<.05 and r['shape']['in_edge_outward_mm']<.05 and r['shape']['heading_overshoot_deg']<2]
    summary=dict(suffixes=len(records),successful=sum(r['success'] for r in records),
                 improvements_gt5ms_equivalent=len(good),local_shape_screen_count=len(strict),
                 local_shape_screen_paths=sorted({r['path'] for r in strict}),goal_complete=False)
    write_json(stage/'summary.json',summary)
    receipt(stage,dict(status='completed_local_exit_action_credit_probe',costs=costs,summary=summary,wall_s=time.monotonic()-started))
    print(summary,flush=True)


if __name__=='__main__':main()
