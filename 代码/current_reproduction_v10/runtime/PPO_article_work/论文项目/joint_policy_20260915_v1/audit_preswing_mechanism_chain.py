"""Causal entry audit: command decomposition, objective coverage, finite branches."""
import math
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct, encode
from joint_runtime import FILTER_TAU, wrap
from bounded_shape_return import terminal_shape_cost
from complete_corner_metrics import complete_corner_metrics, coverage_costs
from validate_actions import difference_audit
from study_io import BASE, register, receipt, write_json


def objective(ref, rows, actions, env=None):
    _, old=terminal_shape_cost(ref,rows,actions,discrete_first_label=False)
    c=complete_corner_metrics(ref,rows)
    variants=coverage_costs(old,c)
    length=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
    base=-len(rows)*.001+length/100
    if env is not None and not env.success:
        base-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
    return dict(shape=c,quality_components=old,coverage_costs=variants,
        base_return=base,incoming_window_return=base-old['cost'],
        two_edge_window_return=base-variants['window_two_edges']['cost'],
        two_edge_full_return=base-variants['full_two_edges']['cost'])


def decomposition(ref, rows, actions):
    u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
    sign=float(np.sign(np.cross(u,v)));initial=math.atan2(u[1],u[0])
    ni=sign*np.array([-u[1],u[0]])
    apex=int(np.argmin(np.linalg.norm(rows[:,:2]-ref[1],axis=1)))
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    # Invert the recorded, exact first-order guidance filter; no physical replay.
    h=np.arctan2(rows[:,9],rows[:,8]);prev=np.r_[initial,h[:-1]]
    increment=rows[:,10]*.001
    target=np.array([wrap(x) for x in prev+increment/(1-math.exp(-.001/FILTER_TAU))])
    steer=actions[np.minimum(np.arange(len(rows))//20,len(actions)-1),0]
    beta=np.array([wrap(x) for x in target-steer])
    signed=lambda x:sign*np.array([wrap(v-initial) for v in x])
    total=signed(target);los=signed(beta);guide=signed(h)
    physical=sign*np.unwrap(np.arctan2(rows[:,3],rows[:,2])-initial)
    inside=(rows[:,:2]-ref[1])@ni
    target_negative=(np.arange(len(rows))<apex)&(total<np.deg2rad(-.05))
    indices=np.flatnonzero(target_negative)
    assert len(indices)
    first=int(indices[0]);anchor=first//20
    # Check the inverse filter against exact geometric reconstruction at anchor.
    return dict(sign=sign,anchor=anchor,first_target_negative_tick=first,
        first_target_negative_time_s=(first+1)*.001,
        first_physical_negative_time_s=next(((i+1)*.001 for i in range(apex)
            if speed[i]>5 and physical[i]<np.deg2rad(-.05)),None),
        first_outside005_time_s=next(((i+1)*.001 for i in range(apex) if inside[i]<-.005),None),
        negative_target_duration_s=float(target_negative.sum()*.001),
        negative_target_steer_outward_fraction=float(np.mean(sign*steer[indices]<0)),
        negative_target_los_inward_fraction=float(np.mean(los[indices]>=0)),
        first_target_deg=float(np.rad2deg(total[first])),
        first_los_deg=float(np.rad2deg(los[first])),
        first_signed_residual_deg=float(np.rad2deg(sign*steer[first])),
        minimum_preapex_target_deg=float(np.rad2deg(total[:apex].min())),
        minimum_preapex_actual_heading_deg=float(np.rad2deg(physical[:apex][speed[:apex]>5].min())),
        raw=dict(signed_target=total,signed_los=los,signed_guidance=guide,
                 signed_steering=sign*steer,signed_physical_heading=physical,signed_in_edge=inside))


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';pool=parent/'training_1'
    model=parent/'candidate_1.pt';indices=list(range(6,14))
    stage=register('preswing_mechanism_chain_20260916',
        [__file__,model,parent/'receipt.json',BASE/'bounded_shape_return.py',
         BASE/'complete_corner_metrics.py',BASE/'joint_runtime.py',BASE/'probe_coordinated_suffix.py']+
        [pool/f'{i}.npz' for i in indices],
        dict(new_suffixes=24,reused_baselines=8,actor_calls=5000,new_physics_ticks=120000,
             wall_s=180,intervention_decisions=6,signed_steering_latent_offset=.005,
             new_training_decisions=0),
        'Root-cause diagnosis on the retained learned model and8 existing training corners '
        'with observed preswing. Decompose saved signed LOS bearing, steering residual, '
        'filtered desired heading and physical heading. Anchor at the first negative '
        'combined target heading below-0.05deg before the apex; this is a diagnostic '
        'localization threshold, not a reward or acceptance rule. From the same exact '
        'state test signed steering latent offsets+/-0.005 and removal of outward '
        'steering residual for120ms; feed/preview feedback, physical executor and later '
        'original policy remain unchanged. Baseline physics is reused. Every branch '
        'finishes or retains its real failure. Compare entry XY and duration, complete '
        'time and the original incoming-window reward, later two-edge-window reward, '
        'and already defined full-extent diagnostic reward. Separately expose which '
        'objective terms are inactive because of allowances. No learning or runtime '
        'clamp adoption. Distinguish a physical generating cause, local reward '
        'preference and the still-unproven historical optimizer origin.')
    started=time.perf_counter();costs=dict(actor_calls=0,new_physics_ticks=0,saved_prefix_geometry_ticks=0)
    p=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    baseline=[];branches=[]
    for index in indices:
        with np.load(pool/f'{index}.npz') as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        d=decomposition(ref,rows,actions);raw=d.pop('raw');sign=d['sign'];anchor=d['anchor'];frame=anchor*20
        np.savez_compressed(stage/f'command_decomposition_{index}.npz',reference=ref,**raw)
        base=objective(ref,rows,actions)
        baseline.append(dict(path=index,decomposition=d,**base))
        env=reconstruct(ref,rows,actions,frame,costs);initial=env.state()
        original=p.act(env.observation(),True)[0];costs['actor_calls']+=1
        assert np.array_equal(env.decode_action(original),actions[anchor])
        s,_=env.progress();target,_=env.sample_reference(s+actions[anchor,2]);delta=target-env.position
        beta=math.atan2(delta[1],delta[0]);u=ref[1]-ref[0];incoming=math.atan2(u[1],u[0])
        error=abs(sign*wrap(beta-incoming)-raw['signed_los'][frame]);assert error<1e-11
        base['command_reconstruction_error_rad']=error
        for arm in ('inward_offset','outward_offset','remove_outward_residual'):
            env.restore(initial);chunks=[];held=[];obs=[];count=0
            while not env.done:
                assert time.perf_counter()-started<180
                assert costs['actor_calls']<5000 and costs['new_physics_ticks']<117880
                x=env.observation();latent=p.act(x,True)[0];costs['actor_calls']+=1
                if count<6:
                    if arm=='remove_outward_residual':
                        action=env.decode_action(latent)
                        if sign*action[0]<0:action[0]=0.;latent=encode(action)
                    else:latent[0]+=sign*(.005 if arm=='inward_offset' else -.005)
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);obs.append(x)
                costs['new_physics_ticks']+=len(info['rows']);count+=1
            full=np.vstack([rows[:frame],*chunks]);all_actions=np.vstack([actions[:anchor],held]) if anchor else np.asarray(held)
            audit=difference_audit(ref[0],full);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            name=f'p{index}_{arm}.npz'
            np.savez_compressed(stage/name,reference=ref,rows=full,actions=all_actions,observations=np.asarray(obs),anchor=anchor)
            value=objective(ref,full,all_actions,env)
            row=dict(path=index,arm=arm,success=success,reason=env.reason,trace=name,
                full_time_s=len(full)*.001 if success else None,audit=audit,**value)
            row['return_changes']={k:value[k]-base[k] for k in ('base_return','incoming_window_return','two_edge_window_return','two_edge_full_return')}
            branches.append(row)
            print('PRESWING_CAUSE',index,arm,success,'entry',round(base['shape']['in_edge_outward_mm'],5),
                round(value['shape']['in_edge_outward_mm'],5),'dR',row['return_changes'],flush=True)
        write_json(stage/'results.json',dict(baselines=baseline,branches=branches,costs=costs,
            historical_learning_cause_established=False,goal_complete=False))
    receipt(stage,dict(status='completed_preswing_command_and_local_objective_diagnosis',costs=costs,
        new_suffixes=len(branches),complete_suffixes=sum(r['success'] for r in branches),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
