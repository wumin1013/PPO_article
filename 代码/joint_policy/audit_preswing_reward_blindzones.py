"""Matched historical update: remove positional/turning deadzones independently."""
import copy
import math
import time
import numpy as np
import torch
from bounded_shape_return import terminal_shape_cost
from isolated_corner_shape import corner_shape
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from train_bounded_shape_ppo import update
from replay_preswing_learning_origin import query_targets
from probe_coordinated_suffix import reconstruct
from train_pilot import save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    replay=BASE/'preswing_learning_origin_replay_20260916'
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    stage=register('preswing_reward_deadzone_factorial_20260917',
        [__file__,BASE/'bounded_shape_return.py',BASE/'isolated_corner_shape.py',
         BASE/'train_bounded_shape_ppo.py',replay/'before_update_1.pt',replay/'before_update_2.pt',
         replay/'receipt.json',source/'episode_results.json',source/'paired_groups.json'],
        dict(new_physics_ticks=0,new_training_decisions=0,optimizer_replays=3,
             reused_original_update=1,saved_training_decisions=1194,wall_s=90),
        'Offline2x2 objective-mechanism experiment on the exact first historical batch. '
        'Factors only remove the original0.05mm entry-position deadzone and5degree '
        'reverse-turn allowance. Retain original local energy/reversal extent, weights, '
        'bounded terminal cost, every time/failure charge, samples, actor/Adam initialization, '
        'minibatch RNG and optimizer settings. Recompute same-path paired advantages '
        'exactly as original training. Reuse original00 optimizer result, and verify all '
        'saved00 shape costs. Other three arms are diagnostic optimizer replays, not '
        'new training collection or selected candidates. Measure target change at fixed '
        'preswing states and the actual advantage changes. Absence of a one-update '
        'reversal does not imply the revised objective could never help after training. '
        'This separates objective deadzones from the already-proven failure-credit effect.')
    started=time.perf_counter();costs=dict(new_physics_ticks=0,new_training_decisions=0,
        saved_prefix_geometry_ticks=0,optimizer_replays=0)
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    after=torch.load(replay/'before_update_2.pt',map_location='cpu',weights_only=False)
    metadata=read(source/'episode_results.json')['episodes']
    selected=[metadata[i] for i in before['episode_numbers']]
    loaded=[];shape_details=[]
    for m in selected:
        with np.load(source/'episodes'/m['trace']) as z:data={k:z[k].copy() for k in z.files}
        old,detail=terminal_shape_cost(data['reference'],data['rows'],data['actions'],discrete_first_label=False)
        assert np.allclose(old,data['shape_cost'],atol=1e-12,rtol=0)
        loaded.append(data)
        shape_details.append((detail,corner_shape(data['reference'],data['rows'],data['actions'])
                              if len(data['reference'])==3 else None))
    queries=[]
    for q in read(BASE/'preswing_mechanism_chain_20260916/results.json')['baselines']:
        i=q['path'];anchor=q['decomposition']['anchor']
        with np.load(BASE/'exit_local_credit_parameter_learning_20260916/training_1'/f'{i}.npz') as z:
            ref,rows,actions=[z[k] for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs);u=ref[1]-ref[0]
        queries.append(dict(path=i,anchor=anchor,env=env,observation=env.observation(),
            sign=q['decomposition']['sign'],incoming_heading=math.atan2(u[1],u[0])))
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    policy.load_state_dict(before['policy'])
    with torch.no_grad():initial_targets=query_targets(policy,queries)[0].numpy()
    records=[]
    for position,reverse in ((False,False),(True,False),(False,True),(True,True)):
        arm=f'position{int(position)}_reverse{int(reverse)}';batch=[];components=[]
        for m,data,(detail,shape) in zip(selected,loaded,shape_details):
            d={k:data[k].copy() for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal')}
            cost=detail['cost']
            if shape is not None:
                incoming=shape['outward_excursion_mm']/1.25 if position else detail['incoming']
                turning=shape['reverse_turning_total_deg']/abs(shape['turn_deg']) if reverse else detail['reverse']
                quality=detail['normalized_energy']+5*incoming+2*turning
                cost=float(-np.expm1(-quality))
                assert 0<=cost<=1
            d['reward']=data['base_reward'].copy();d['reward'][-1]-=cost
            if not position and not reverse:assert np.allclose(d['reward'],data['reward'],rtol=0,atol=1e-12)
            batch.append(d)
            components.append(dict(episode=m['episode'],path=m['path'],success=m['success'],
                shape_cost=cost,base_return=float(data['base_reward'].sum()),total_return=float(d['reward'].sum())))
        for k in range(0,len(batch),4):
            group=batch[k:k+4];total=np.array([d['reward'].sum() for d in group])
            advantages=(total-(total.sum()-total)/3)/max(float(total.std()),.02)
            for d,c,a in zip(group,components[k:k+4],advantages):
                d['paired_advantage']=np.full(len(d['reward']),a);c['paired_advantage']=float(a)
        policy.load_state_dict(before['policy'])
        if position or reverse:
            optim={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
                   'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
            for k,v in optim.items():v.load_state_dict(copy.deepcopy(before['optimizer'][k]))
            torch.set_rng_state(before['torch_rng'])
            stats=update(policy,optim,batch,paired_advantage=True)
            costs['optimizer_replays']+=1
            save_torch(stage/f'{arm}.pt',dict(policy=policy.state_dict(),diagnostic_only=True,arm=arm))
        else:policy.load_state_dict(after['policy']);stats=None
        with torch.no_grad():target=query_targets(policy,queries)[0].numpy()
        row=dict(arm=arm,position_deadzone_removed=position,reverse_allowance_removed=reverse,
            reused=not(position or reverse),targets_deg=np.rad2deg(target).tolist(),
            mean_target_change_deg=float(np.rad2deg(target-initial_targets).mean()),
            path12=[c for c in components if c['path']==12],optimizer_stats=stats)
        records.append(row)
        print('REWARD_CAUSE',arm,'mean_delta',row['mean_target_change_deg'],
              'targets',np.round(row['targets_deg'],4).tolist(),flush=True)
        write_json(stage/'results.json',dict(initial_targets_deg=np.rad2deg(initial_targets).tolist(),
            arms=records,costs=costs,original_shape_costs_verified=True,
            failure_ordering='Cost remains in[0,1]; unchanged Tmax+1+remaining/20 failure charge retains complete-success preference.',
            candidates_adopted=False))
    receipt(stage,dict(status='completed_matched_shape_deadzone_update_audit',costs=costs,
        original_shape_costs_verified=True,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
