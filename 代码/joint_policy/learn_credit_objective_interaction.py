"""Reuse common-future rollouts for a credit-by-objective mechanism factorial."""
import copy
import time
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from probe_one_decision_credit import score_complete
from train_bounded_shape_ppo import update
from train_pilot import evaluate, save_torch
from complete_corner_metrics import complete_corner_metrics
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def checks(physical, original, initial):
    good=all(r['success'] for r in physical)
    result=dict(all_complete=good)
    if not good:return result
    result.update(entry_sum_better_than_original_update=sum(r['shape']['in_edge_outward_mm'] for r in physical)<sum(r['shape']['in_edge_outward_mm'] for r in original),
        each_entry_nonworse_than_initializer=all(r['shape']['in_edge_outward_mm']<=initial[r['path']]['shape']['in_edge_outward_mm']+1e-6 for r in physical),
        max_exit_nonworse_than_original_update=max(r['shape']['out_edge_outward_mm'] for r in physical)<=max(r['shape']['out_edge_outward_mm'] for r in original)+1e-6,
        full_reverse_nonworse_than_original_update=sum(r['shape']['full_physical_reverse_deg'] for r in physical)<=sum(r['shape']['full_physical_reverse_deg'] for r in original)+1e-6)
    return result


def main():
    torch.set_num_threads(1)
    branch=BASE/'common_future_action_credit_learning_20260917'
    assert (branch/'receipt.json').exists()
    replay=BASE/'preswing_learning_origin_replay_20260916'
    source=BASE/'local_geometry_continuous_reward_ppo_20260916'
    old=read(BASE/'preswing_origin_loss_interventions_20260916/physical_results.json')
    stage=register('common_future_credit_objective_interaction_20260917',
        [__file__,branch/'receipt.json',branch/'counterfactual_results.json',branch/'results.json',
         replay/'before_update_1.pt',replay/'before_update_2.pt',
         BASE/'causal_complete_shape_accounting_audit_20260917/terminal.pt',
         BASE/'causal_complete_shape_accounting_audit_20260917/receipt.json',
         BASE/'probe_one_decision_credit.py',BASE/'train_bounded_shape_ppo.py',
         BASE/'conditional_feedback_policy.py',BASE/'complete_corner_metrics.py',
         source/'episode_results.json',BASE/'learning_preparation_v4/paths.json'],
        dict(new_counterfactuals=0,optimizer_updates=1,evaluation_actor_calls=2500,
             evaluation_physics_ticks=55000,wall_s=120),
        'Explicit credit x objective2x2 interaction test on the identical1194 historical '
        'on-policy samples and same-state common-future baselines. Original vs independent '
        'full two-edge/no-allowance shape objective, and episode-paired vs conditional-mean '
        'action baseline. Reuse both old-objective actors/physics and the already-replayed '
        'new-objective paired actor. Rescore existing1194 complete counterfactuals; perform '
        'one new-objective action-baseline optimizer update with same initial actor,Adam,RNG, '
        'PPO settings and failure charges. Same-path scales are recomputed from that arm '
        'objective using the same formula. Compare new arms physically on8 original training '
        'corners using identical predeclared mechanism checks. This factorial separates '
        'credit/objective main effects and their interaction; it does not isolate the '
        'bundled individual components of the independent-zero shape revision. No '
        'additional physics for counterfactual collection, no development or sealed tests, '
        'no final adoption of a historical-initializer pilot. Stop if no candidate qualifies.')
    started=time.perf_counter();costs=dict(new_counterfactuals=0,optimizer_updates=0,
        evaluation_actor_calls=0,evaluation_physics_ticks=0)
    before=torch.load(replay/'before_update_1.pt',map_location='cpu',weights_only=False)
    metadata=read(source/'episode_results.json')['episodes'];selected=[metadata[i] for i in before['episode_numbers']]
    raw=read(branch/'counterfactual_results.json')['records'];batch=[];labels=[]
    for meta in selected:
        with np.load(source/'episodes'/meta['trace']) as z:d={k:z[k].copy() for k in z.files}
        ref=d['reference'];rows=d['rows'];actions=d['actions']
        cost=score_complete(ref,rows,actions)['independent_shape_cost']
        e={k:d[k].copy() for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','value','terminal')}
        e['reward']=d['base_reward'].copy();e['reward'][-1]-=cost;observed=float(e['reward'].sum())
        advantage=[]
        for record in [r for r in raw if r['episode']==meta['episode']]:
            anchor=record['anchor']
            if len(ref)==3:
                with np.load(branch/record['trace']) as z:suffix=z['suffix_rows'];held=z['suffix_actions']
                full=np.vstack([rows[:anchor*20],suffix]);full_actions=np.vstack([actions[:anchor],held])
                new_cost=score_complete(ref,full,full_actions)['independent_shape_cost']
            else:new_cost=0.
            new_return=record['counterfactual_return']+record['shape_cost']['cost']-new_cost
            advantage.append(observed-new_return)
            labels.append(dict(episode=meta['episode'],anchor=anchor,observed_return=observed,
                counterfactual_return=new_return,advantage=observed-new_return,
                success=record['counterfactual_success']))
        assert len(advantage)==len(actions)
        e['paired_advantage']=np.asarray(advantage);batch.append(e)
    for k in range(0,len(batch),4):
        totals=np.asarray([e['reward'].sum() for e in batch[k:k+4]])
        scale=max(float(totals.std()),.02)
        for e in batch[k:k+4]:e['paired_advantage']/=scale
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True);policy.load_state_dict(before['policy'])
    optim={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
           'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    for k,v in optim.items():v.load_state_dict(copy.deepcopy(before['optimizer'][k]))
    torch.set_rng_state(before['torch_rng']);stats=update(policy,optim,batch,paired_advantage=True);costs['optimizer_updates']+=1
    save_torch(stage/'revised_objective_action_credit.pt',dict(policy=policy.state_dict(),architecture='local_geometry_feedback',
        mask_endpoint=True,observation_dim=72,learning_method='conditional_mean_common_future_independent_shape',
        source_training_decisions=1194,optimizer_updates=1,mechanism_pilot=True))
    write_json(stage/'counterfactual_labels.json',dict(labels=labels,optimizer=stats))
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];ids=list(range(6,14));arms=[]
    for arm,model in [('revised_objective_paired',BASE/'causal_complete_shape_accounting_audit_20260917/terminal.pt'),
                      ('revised_objective_action_credit',stage/'revised_objective_action_credit.pt')]:
        policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
        results=evaluate(policy,[np.asarray(paths[i]) for i in ids],stage,arm,None,costs,started+120,environment_factory)
        physical=[]
        for index,(path,result) in enumerate(zip(ids,results)):
            with np.load(stage/arm/f'{index}.npz') as z:shape=complete_corner_metrics(z['reference'],z['rows'])
            physical.append(dict(path=path,success=result['success'],failed_prefix=not result['success'],
                shape=shape,full_time_s=result['completion_time_s']))
        arms.append(dict(arm=arm,physical=physical,model=str(model.relative_to(BASE)),reused_physics=False))
    arms.extend([dict(arm='original_objective_paired',physical=next(r['results'] for r in old['arms'] if r['arm']=='original_update1'),reused_physics=True),
                 dict(arm='original_objective_action_credit',physical=read(branch/'results.json')['physical'],reused_physics=True)])
    original=arms[-2]['physical'];initial={r['path']:r for r in old['initial']}
    for arm in arms:
        arm['checks']=checks(arm['physical'],original,initial)
        arm['mechanism_eligible']=all(arm['checks'].values())
        if all(r['success'] for r in arm['physical']):
            arm['entry_sum_mm']=sum(r['shape']['in_edge_outward_mm'] for r in arm['physical'])
            arm['exit_sum_mm']=sum(r['shape']['out_edge_outward_mm'] for r in arm['physical'])
            arm['reverse_sum_deg']=sum(r['shape']['full_physical_reverse_deg'] for r in arm['physical'])
        print('CREDIT_OBJECTIVE',arm['arm'],arm.get('entry_sum_mm'),arm.get('exit_sum_mm'),arm.get('reverse_sum_deg'),arm['checks'],flush=True)
    assert costs['evaluation_actor_calls']<=2500 and costs['evaluation_physics_ticks']<=55000
    write_json(stage/'results.json',dict(arms=arms,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_common_future_credit_objective_factorial',costs=costs,
        mechanism_eligible=[a['arm'] for a in arms if a['mechanism_eligible']],wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
