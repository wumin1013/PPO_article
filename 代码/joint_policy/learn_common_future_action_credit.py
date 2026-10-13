"""Same-state conditional-mean baselines using a common future RNG stream."""
import copy
import time
import numpy as np
import torch

from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from bounded_shape_return import terminal_shape_cost
from train_bounded_shape_ppo import update
from probe_coordinated_suffix import reconstruct
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit
from complete_corner_metrics import complete_corner_metrics
from train_pilot import evaluate, save_torch
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


KEYS=('observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal')


def main():
    torch.set_num_threads(1)
    replay=BASE/'preswing_learning_origin_replay_20260916'
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    metadata=read(source/'episode_results.json')['episodes']
    selected=[metadata[i] for i in before['episode_numbers']]
    stage=register('common_future_action_credit_learning_20260917',
        [__file__,replay/'before_update_1.pt',replay/'before_update_2.pt',replay/'receipt.json',
         source/'episode_results.json',source/'paired_groups.json',
         BASE/'conditional_feedback_policy.py',BASE/'train_bounded_shape_ppo.py',
         BASE/'probe_coordinated_suffix.py',BASE/'bounded_shape_return.py',
         BASE/'joint_runtime_v5.py',BASE/'validate_actions.py',BASE/'complete_corner_metrics.py',
         BASE/'causal_complete_shape_accounting_audit_20260917/receipt.json',
         BASE/'preswing_origin_loss_interventions_20260916/physical_results.json']+
         [source/'episodes'/m['trace'] for m in selected],
        dict(counterfactual_suffixes=1194,replayed_samples=1194,actor_calls=70000,
             new_physics_ticks=1650000,evaluation_actor_calls=3000,
             evaluation_physics_ticks=65000,wall_s=720,optimizer_updates=1,
             physical_review_training_paths=list(range(6,14))),
        'First direct corrective policy update for the proven early-failure credit mechanism. '
        'Reuse the exact first1194 on-policy samples and pre-update actor/Adam/RNG. Recover '
        'the original Gaussian random stream bitwise. For every sampled decision, branch '
        'from its exact physical/context state, replace only the CURRENT innovation by its '
        'conditional mean, and sample the unchanged AR policy with the original next RNG '
        'state until complete stop or true failure. The next stochastic innovation sequence '
        'is shared, including ordinary stochastic continuation beyond the original stop; '
        'it is never truncated at a stop time that depends on the original action. '
        'The counterfactual baseline depends on prior state and independent future random '
        'numbers, not the current innovation. Keep original reward, failure penalty, '
        'architecture, samples, same-path scale, PPO normalization/minibatches/Adam and '
        'critic targets. Only actor advantage changes from whole-episode leave-one-out '
        'return to observed return minus conditional-mean counterfactual return. This '
        'tests conditional action credit; usual sample-dependent normalization still applies. '
        'All1194 actions remain represented, including failures and all three action heads. '
        'One update only; evaluate8 existing training corners against saved original '
        'initial/first-update trajectories. A useful corrective mechanism requires8/8 '
        'complete, lower entry sum than original first update, no increased maximum exit '
        'or full reversal sum, and no entry increase versus original initial actor on '
        'any of8 paths. Otherwise stop and analyze; do not blindly retrain. No final '
        'adoption, development, sealed tests or claim that this historical initializer '
        'outperforms the retained strong model. No intervention schedule is deployed.')
    started=time.perf_counter()
    costs=dict(replayed_actor_calls=0,actor_calls=0,new_physics_ticks=0,
               saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    policy.load_state_dict(before['policy'])
    loaded=[];rng_after=[];rng_before=[]
    torch.manual_seed(9161043)
    for m in selected:
        with np.load(source/'episodes'/m['trace']) as z:d={k:z[k].copy() for k in z.files}
        policy.reset_context();before_states=[];after_states=[]
        for j,obs in enumerate(d['observation']):
            before_states.append(torch.get_rng_state())
            action,logp,value=policy.act(obs)
            after_states.append(torch.get_rng_state())
            assert np.array_equal(action,d['latent'][j]) and logp==d['old_log'][j] and value==d['value'][j]
            costs['replayed_actor_calls']+=1
        loaded.append(d);rng_before.append(before_states);rng_after.append(after_states)
    assert costs['replayed_actor_calls']==1194
    assert torch.equal(torch.get_rng_state(),before['torch_rng'])
    save_torch(stage/'common_future_rng.pt',dict(before=rng_before,after=rng_after,
        episodes=before['episode_numbers'],seed=9161043,original_sampling_bitwise_verified=True))
    records=[];counterfactual_advantages=[];reproduction=[]
    for episode_index,(m,d) in enumerate(zip(selected,loaded)):
        ref=d['reference'];rows=d['rows'];actions=d['actions']
        original_return=float(d['reward'].sum());local=[]
        # Validate the entire original stochastic dynamics once, without a replacement.
        if m['episode']==16:
            torch.set_rng_state(rng_before[episode_index][0]);policy.reset_context()
            env=environment_factory(ref);chunks=[];held=[]
            while not env.done:
                a,_,_=policy.act(env.observation());costs['actor_calls']+=1
                _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            assert np.array_equal(np.vstack(chunks),rows) and np.array_equal(held,actions)
            reproduction.append(dict(episode=16,rows_bitwise=True,actions_bitwise=True))
        for anchor in range(len(actions)):
            env=reconstruct(ref,rows,actions,anchor*20,costs)
            observation=env.observation();assert np.array_equal(observation,d['observation'][anchor])
            previous_obs=torch.tensor(d['previous_observation'][anchor])
            previous_z=torch.tensor(d['previous_latent'][anchor])
            live=torch.tensor(d['context_live'][anchor])
            with torch.no_grad():
                mean=policy.conditional(torch.tensor(observation),previous_obs,previous_z,live).mean.numpy()
            costs['actor_calls']+=1
            # The replacement innovation consumes no draw; restore the original NEXT draw.
            torch.set_rng_state(rng_after[episode_index][anchor])
            policy.previous_observation=observation.copy();policy.previous_latent=mean.copy()
            _,_,_,info=env.step(mean);chunks=[info['rows']];held=[info['held_action']]
            costs['new_physics_ticks']+=len(info['rows'])
            while not env.done:
                assert time.perf_counter()-started<720
                assert costs['actor_calls']<70000 and costs['new_physics_ticks']<1647880
                a,_,_=policy.act(env.observation());costs['actor_calls']+=1
                _,_,_,info=env.step(a);chunks.append(info['rows']);held.append(info['held_action'])
                costs['new_physics_ticks']+=len(info['rows'])
            suffix=np.vstack(chunks);suffix_actions=np.asarray(held)
            full=np.vstack([rows[:anchor*20],suffix]);all_actions=np.vstack([actions[:anchor],suffix_actions])
            audit=difference_audit(ref[0],full)
            complete=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            shape,details=terminal_shape_cost(ref,full,all_actions,discrete_first_label=False)
            length=float(np.linalg.norm(np.diff(ref,axis=0),axis=1).sum())
            total=-len(full)*.001+length/100-float(shape.sum())
            if not env.success:total-=env.max_ticks*.001+1+max(0.,length-env.progress()[0])/20
            name=f'ep{m["episode"]}_a{anchor}.npz'
            np.savez_compressed(stage/name,reference=ref,suffix_rows=suffix,suffix_actions=suffix_actions,
                original_episode=m['episode'],anchor=anchor,conditional_mean=mean)
            advantage=original_return-total;local.append(advantage)
            record=dict(episode=m['episode'],path=m['path'],anchor=anchor,
                original_success=m['success'],counterfactual_success=complete,
                counterfactual_reason=env.reason,observed_return=original_return,
                counterfactual_return=total,raw_action_advantage=advantage,
                full_time_s=len(full)*.001 if complete else None,trace=name,audit=audit,
                shape_cost=details,stochastic_extra_decisions=max(0,anchor+len(held)-len(actions)))
            records.append(record)
        counterfactual_advantages.append(np.asarray(local))
        write_json(stage/'counterfactual_results.json',dict(records=records,costs=costs,
            original_reproductions=reproduction,completed_episodes=episode_index+1))
        print('COMMON_FUTURE',m['episode'],m['path'],len(local),
            'complete',sum(r['counterfactual_success'] for r in records[-len(local):]),
            'nonzero',int((abs(np.asarray(local))>1e-6).sum()),'calls',costs['actor_calls'],flush=True)
    batch=[];scales=[]
    for k in range(0,len(loaded),4):
        totals=np.asarray([d['reward'].sum() for d in loaded[k:k+4]])
        scales.extend([max(float(totals.std()),.02)]*4)
    for d,a,scale in zip(loaded,counterfactual_advantages,scales):
        entry={k:d[k].copy() for k in KEYS};entry['paired_advantage']=a/scale;batch.append(entry)
    policy.load_state_dict(before['policy'])
    optim={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
           'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    for k,v in optim.items():v.load_state_dict(copy.deepcopy(before['optimizer'][k]))
    torch.set_rng_state(before['torch_rng'])
    stats=update(policy,optim,batch,paired_advantage=True)
    save_torch(stage/'policy.pt',dict(policy=policy.state_dict(),architecture='local_geometry_feedback',
        mask_endpoint=True,observation_dim=72,learning_method='conditional_mean_common_future_action_baseline',
        source_training_decisions=1194,optimizer_updates=1,mechanism_pilot=True))
    np.savez_compressed(stage/'advantages.npz',**{f'episode_{m["episode"]}':a for m,a in zip(selected,counterfactual_advantages)})
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];ids=list(range(6,14))
    results=evaluate(policy,[np.asarray(paths[i]) for i in ids],stage,'training_review',None,costs,started+720,environment_factory)
    physical=[]
    for index,(path,result) in enumerate(zip(ids,results)):
        with np.load(stage/'training_review'/f'{index}.npz') as z:shape=complete_corner_metrics(z['reference'],z['rows'])
        physical.append(dict(path=path,success=result['success'],failed_prefix=not result['success'],
            shape=shape,full_time_s=result['completion_time_s']))
    old=read(BASE/'preswing_origin_loss_interventions_20260916/physical_results.json')
    initial={r['path']:r for r in old['initial']}
    original=next(x['results'] for x in old['arms'] if x['arm']=='original_update1')
    all_complete=all(r['success'] for r in physical)
    checks=dict(all_complete=all_complete)
    if all_complete:
        checks.update(entry_sum_better_than_original_update=sum(r['shape']['in_edge_outward_mm'] for r in physical)<sum(r['shape']['in_edge_outward_mm'] for r in original),
            each_entry_nonworse_than_initializer=all(r['shape']['in_edge_outward_mm']<=initial[r['path']]['shape']['in_edge_outward_mm']+1e-6 for r in physical),
            max_exit_nonworse_than_original_update=max(r['shape']['out_edge_outward_mm'] for r in physical)<=max(r['shape']['out_edge_outward_mm'] for r in original)+1e-6,
            full_reverse_nonworse_than_original_update=sum(r['shape']['full_physical_reverse_deg'] for r in physical)<=sum(r['shape']['full_physical_reverse_deg'] for r in original)+1e-6)
    assert costs['actor_calls']<=70000 and costs['new_physics_ticks']<=1650000
    assert costs['evaluation_actor_calls']<=3000 and costs['evaluation_physics_ticks']<=65000
    write_json(stage/'results.json',dict(physical=physical,checks=checks,mechanism_eligible=all(checks.values()),
        optimizer=stats,costs=costs,counterfactual_suffixes=len(records),goal_complete=False))
    receipt(stage,dict(status='completed_common_future_action_credit_update',costs=costs,
        counterfactual_suffixes=len(records),mechanism_eligible=all(checks.values()),wall_s=time.perf_counter()-started))
    print('COMMON_FUTURE_LEARNING',checks,flush=True)


if __name__=='__main__':main()
