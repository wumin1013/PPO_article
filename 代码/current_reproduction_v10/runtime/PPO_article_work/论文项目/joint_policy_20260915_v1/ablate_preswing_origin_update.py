"""Controlled actor-loss interventions on a proven historical PPO update."""
import math
import time
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from train_bounded_shape_ppo import update
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from replay_preswing_learning_origin import query_targets
from probe_coordinated_suffix import reconstruct
from complete_corner_metrics import complete_corner_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main(temporal=False):
    torch.set_num_threads(1)
    replay=BASE/'preswing_learning_origin_replay_20260916'
    training=BASE/'local_geometry_continuous_reward_ppo_20260916'
    stage=register('preswing_origin_temporal_loss_interventions_20260916' if temporal else 'preswing_origin_loss_interventions_20260916',
        [__file__,BASE/'train_bounded_shape_ppo.py',BASE/'replay_preswing_learning_origin.py',
         replay/'receipt.json',replay/'results.json',replay/'before_update_1.pt',
         replay/'before_update_2.pt',replay/'before_update_3.pt',
         training/'episode_results.json',training/'paired_groups.json',
         BASE/'learning_preparation_v4/paths.json'],
        dict(optimizer_replays=6 if temporal else 7,new_training_decisions=0,
             evaluation_actor_calls=0 if temporal else 5000,
             evaluation_physics_ticks=0 if temporal else 120000,wall_s=240,queries=list(range(6,14))),
        ('Further localize the already identified path12 contribution, without repeating '
        'original complete-mask or physical tests. Six masks: path12 successful episodes '
        'only, failed episodes only, and their respective before/after-first-outgoing-preview '
        'portions. The transition is determined from each saved episode first decision '
        'whose held preview reaches beyond the incoming segment. Keep earlier data and '
        'advantages fixed. These are optimizer-loss interventions only, not evidence that '
        'any single sampled action physically caused failure. ' if temporal else '')+
        'Causal optimizer-loss test of the first exactly reproduced historical update. '
        'Same pre-update weights, Adam states, samples, advantages, original full-batch '
        'normalization, minibatch RNG, clipping and critic updates in all arms. Only '
        'actor-loss sample masks differ: all, isolated-path12 only, other paths only, '
        'all-failed path groups only, all-failed groups excluded, successful episodes '
        'only, failed episodes only. All is a required implementation identity check '
        'against original post-update tensors. No new policy training collection. '
        'Measure combined targets at fixed current preswing states; then inspect actual '
        'full trajectories on8 original training corners for original first/second '
        'update policies and the isolated-only/other-only interventions. Reuse available '
        'initial-model geometry-matched trajectories, evaluating only missing paths. '
        'No candidate adoption or sealed testing. Failed prefixes never count as successful '
        'shape improvements; loss masks are diagnostic interventions, not a proposed '
        'exclusion of hard training paths or failed episodes.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,
        saved_prefix_geometry_ticks=0,optimizer_replays=0,new_training_decisions=0)
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    after1=torch.load(replay/'before_update_2.pt',map_location='cpu',weights_only=False)
    after2=torch.load(replay/'before_update_3.pt',map_location='cpu',weights_only=False)
    metadata=read(training/'episode_results.json')['episodes'];selected=[metadata[i] for i in before['episode_numbers']]
    batch=[];before_preview=[];preview_onsets=[]
    for m in selected:
        with np.load(training/'episodes'/m['trace']) as z:
            batch.append({k:z[k].copy() for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal')})
            n=len(z['actions']);early=np.ones(n,bool);onset=None
            if len(z['reference'])==3:
                ref=z['reference'];u=ref[1]-ref[0];u=u/np.linalg.norm(u)
                for j in range(n):
                    position=ref[0] if j==0 else z['rows'][j*20-1,:2]
                    if (ref[1]-position)@u < z['actions'][j,2]:onset=j;early[j:]=False;break
            before_preview.append(early);preview_onsets.append(dict(episode=m['episode'],path=m['path'],first_outgoing_preview_decision=onset))
    for start in range(0,len(batch),4):
        group=batch[start:start+4];total=np.array([d['reward'].sum() for d in group]);base=(total.sum()-total)/3
        for d,a in zip(group,total-base):d['paired_advantage']=np.full(len(d['reward']),a/max(float(total.std()),.02))
    ids=np.concatenate([np.full(len(d['reward']),m['path']) for m,d in zip(selected,batch)])
    successes=np.concatenate([np.full(len(d['reward']),m['success']) for m,d in zip(selected,batch)])
    failed_groups=[path for path in set(ids) if not any(m['success'] for m in selected if m['path']==path)]
    masks=dict(all=np.ones(len(ids),bool),isolated_only=ids<16,other_only=ids>=16,
        all_failed_groups_only=np.isin(ids,failed_groups),exclude_all_failed_groups=~np.isin(ids,failed_groups),
        successful_episodes_only=successes,failed_episodes_only=~successes)
    if temporal:
        early=np.concatenate(before_preview);local=ids==12
        masks=dict(path12_success_only=local&successes,path12_failed_only=local&~successes,
            path12_success_before_preview=local&successes&early,
            path12_success_after_preview=local&successes&~early,
            path12_failed_before_preview=local&~successes&early,
            path12_failed_after_preview=local&~successes&~early)
        write_json(stage/'phase_definition.json',dict(onsets=preview_onsets,
            definition='First saved20ms decision with held D exceeding remaining distance to the incoming vertex.'))
    queries=[]
    case_data=read(BASE/'preswing_mechanism_chain_20260916/results.json')['baselines']
    for b in case_data:
        i=b['path'];anchor=b['decomposition']['anchor']
        with np.load(BASE/'exit_local_credit_parameter_learning_20260916/training_1'/f'{i}.npz') as z:
            ref,rows,actions=[z[k] for k in ('reference','rows','actions')]
        env=reconstruct(ref,rows,actions,anchor*20,costs);u=ref[1]-ref[0]
        queries.append(dict(path=i,anchor=anchor,env=env,observation=env.observation(),
            sign=b['decomposition']['sign'],incoming_heading=math.atan2(u[1],u[0])))
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    records=[];model_states={}
    for arm,mask in masks.items():
        policy.load_state_dict(before['policy'])
        optim={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
               'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
        for k,opt in optim.items():opt.load_state_dict(before['optimizer'][k])
        torch.set_rng_state(before['torch_rng'])
        stats=update(policy,optim,batch,paired_advantage=True,actor_sample_mask=mask)
        costs['optimizer_replays']+=1
        if arm=='all':
            assert all(torch.equal(v,after1['policy'][k]) for k,v in policy.state_dict().items())
        with torch.no_grad():target,_=query_targets(policy,queries)
        name=f'{arm}.pt';save_torch(stage/name,dict(policy=policy.state_dict(),diagnostic_only=True,arm=arm))
        model_states[arm]=name
        row=dict(arm=arm,included_decisions=int(mask.sum()),target_deg=np.rad2deg(target.numpy()).tolist(),
            all_targets_inward=bool((target>=0).all()),actor_steps=stats['actor_minibatches'],
            original_identity_verified=arm=='all')
        records.append(row);write_json(stage/'loss_interventions.json',dict(records=records,
            query_paths=[q['path'] for q in queries],all_failed_paths=[int(i) for i in failed_groups]))
        print('LOSS_CAUSE',arm,np.round(row['target_deg'],4).tolist(),flush=True)
    if temporal:
        receipt(stage,dict(status='completed_historical_temporal_loss_interventions',costs=costs,
            original_update_identity_previously_verified=True,wall_s=time.perf_counter()-started))
        return
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];indices=list(range(6,14))
    initial_pool=BASE/'matched_endpoint_representation_learning_20260916/mask_endpoint_bearing_training_50'
    lookup={}
    for i,r in enumerate(read(initial_pool/'summary.json')['results']):
        path=initial_pool/f'{i}.npz'
        with np.load(path) as z:lookup[z['reference'].tobytes()]=(r,path)
    initial_results=[];missing=[]
    for i in indices:
        item=lookup.get(np.asarray(paths[i]).tobytes())
        if item:
            r,path=item
            with np.load(path) as z:shape=complete_corner_metrics(z['reference'],z['rows'])
            initial_results.append(dict(path=i,success=r['success'],trace=path.relative_to(BASE).as_posix(),
                reused=True,shape=shape,failed_prefix=not r['success']))
        else:missing.append(i)
    physical=[]
    arms=[('initial_missing',before['policy'],missing),('original_update1',after1['policy'],indices),
          ('original_update2',after2['policy'],indices)]
    for arm in ('isolated_only','other_only'):
        arms.append((arm,torch.load(stage/model_states[arm],weights_only=False)['policy'],indices))
    for arm,state,ids_to_run in arms:
        if not ids_to_run:continue
        policy.load_state_dict(state)
        results=evaluate(policy,[np.asarray(paths[i]) for i in ids_to_run],stage,arm,None,costs,started+240,environment_factory)
        detailed=[]
        for local,(i,r) in enumerate(zip(ids_to_run,results)):
            path=stage/arm/f'{local}.npz'
            with np.load(path) as z:shape=complete_corner_metrics(z['reference'],z['rows'])
            detailed.append(dict(path=i,success=r['success'],trace=path.relative_to(BASE).as_posix(),
                reused=False,failed_prefix=not r['success'],shape=shape,full_time_s=r['completion_time_s']))
        if arm=='initial_missing':initial_results.extend(detailed)
        else:physical.append(dict(arm=arm,results=detailed,complete_paths=sum(r['success'] for r in detailed)))
        write_json(stage/'physical_results.json',dict(initial=initial_results,arms=physical,costs=costs))
    assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=120000
    receipt(stage,dict(status='completed_matched_historical_actor_loss_interventions',costs=costs,
        original_update_identity_verified=True,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
