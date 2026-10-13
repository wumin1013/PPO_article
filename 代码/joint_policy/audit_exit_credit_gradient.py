"""Exactly replay two saved PPO batches and attribute their actor gradients."""
import time
import numpy as np
import torch
from conditional_feedback_policy import ConditionalLocalGeometryPolicy
from train_bounded_shape_ppo import update
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from complete_corner_metrics import complete_corner_metrics
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def flatten_gradient(value,parameters,retain_graph=True):
    values=torch.autograd.grad(value,parameters,retain_graph=retain_graph)
    return torch.cat([v.flatten() for v in values])


def main():
    source=BASE/'two_edges_side_coverage_ppo_20260916'
    probe=BASE/'exit_action_credit_probe_20260916'
    initial=BASE/'local_geometry_continuous_reward_ppo_20260916/policy_18645.pt'
    stage=register('exit_credit_gradient_audit_20260916',
        [__file__,initial,source/'results.json',source/'episode_results.json',source/'paired_groups.json',
         probe/'receipt.json',probe/'policy_direction_audit.json',BASE/'train_bounded_shape_ppo.py'],
        dict(replay_actor_calls=2144,replay_updates=2,new_training_decisions=0,
             evaluation_paths=8,evaluation_decisions=2000,evaluation_physics_ticks=45000,wall_s=120),
        'Exactly reproduce the first two completed two-edge PPO optimizer batches from saved samples '
        'and RNG. Measure separate normalized score gradients from isolated-corner, failure-containing '
        'non-corner and other groups. Compare with the existing finite60ms local-return sensitivity '
        'surrogate at16 training states, explicitly not an exact differential Q gradient. Save the '
        'recovered two-update policy and evaluate8 existing training corners to check actual early '
        'shape effects. No new training physics, no added learning iterations or sealed tests.')
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(9161087)
    policy=ConditionalLocalGeometryPolicy(72,mask_endpoint=True)
    policy.load_state_dict(torch.load(initial,map_location='cpu',weights_only=False)['policy'])
    torch.manual_seed(9161087);policy.reset_context()
    optimizer={'actor':torch.optim.Adam(policy.actor_parameters(),lr=3e-5),
               'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    query=read(probe/'policy_direction_audit.json')['records'];probe_rows=read(probe/'results.json')['records']
    observations=[];sensitivities=[]
    for row in query:
        trace=next(r for r in probe_rows if r['path']==row['path'] and r['anchor']==row['anchor'])
        with np.load(probe/trace['trace']) as z:observations.append(z['observation'][0])
        sensitivities.append(row['finite_offset_return_sensitivity'])
    qo=torch.tensor(np.asarray(observations),dtype=torch.float32)
    qg=torch.tensor(np.asarray(sensitivities),dtype=torch.float32)
    original_means=policy.distribution(qo).mean.detach()
    expected=read(source/'results.json')['updates'];episodes=read(source/'episode_results.json')['episodes']
    batch=[];group=[];batch_groups=[];decisions=0;update_index=0;reports=[]
    for meta in episodes:
        with np.load(source/'episodes'/meta['trace']) as z:
            data={k:z[k] for k in ('observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal')}
        policy.reset_context()
        for j,obs in enumerate(data['observation']):
            latent,logp,value=policy.act(obs)
            assert np.array_equal(latent,data['latent'][j]) and logp==data['old_log'][j] and value==data['value'][j]
        decisions+=len(data['reward']);batch.append(data);group.append((meta,data))
        if len(group)==4:
            total=np.array([d['reward'].sum() for _,d in group]);adv=total-(total.sum()-total)/3
            category='isolated' if 'normalized_energy' in group[0][0]['shape_cost'] else 'other_failure_group' if not all(m['success'] for m,_ in group) else 'other_success_group'
            for (_,d),a in zip(group,adv):
                d['paired_advantage']=np.full(len(d['reward']),a/max(float(total.std()),.02))
                d['category']=np.full(len(d['reward']),category)
            batch_groups.append(dict(path=meta['path'],category=category,episodes=[m['episode'] for m,_ in group]))
            group=[]
        if decisions!=expected[update_index]['decisions']:continue
        tensors={k:torch.as_tensor(np.concatenate([d[k] for d in batch]),dtype=torch.bool if k=='context_live' else torch.float32)
                 for k in ('observation','latent','previous_observation','previous_latent','context_live')}
        logp=policy.evaluate_conditioned(tensors['observation'],tensors['latent'],tensors['previous_observation'],tensors['previous_latent'],tensors['context_live'])[0]
        adv=torch.tensor(np.concatenate([d['paired_advantage'] for d in batch]),dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        parameters=policy.actor_parameters();category=np.concatenate([d['category'] for d in batch])
        local=flatten_gradient((policy.distribution(qo).mean*qg).sum(-1).mean(),parameters)
        gradients={}
        for key in ('isolated','other_failure_group','other_success_group','all'):
            mask=torch.tensor(category==key) if key!='all' else torch.ones(len(adv),dtype=torch.bool)
            gradient=flatten_gradient((logp*adv*mask).mean(),parameters)
            norm=float(gradient.norm())
            gradients[key]=dict(decisions=int(mask.sum()),norm=norm,
                cosine_to_local_surrogate=float(gradient@local/(gradient.norm()*local.norm())) if norm>1e-12 else None)
        before=policy.distribution(qo).mean.detach()
        stats=update(policy,optimizer,batch,paired_advantage=True)
        assert all(stats[k]==expected[update_index][k] for k in stats)
        after=policy.distribution(qo).mean.detach()
        report=dict(decisions=decisions,groups=batch_groups,gradients=gradients,exact_update_statistics=True,
                    local_linearized_delta=((after-before)*qg).sum(-1).tolist(),
                    cumulative_linearized_delta=((after-original_means)*qg).sum(-1).tolist())
        reports.append(report);update_index+=1;batch=[];batch_groups=[]
        print('GRADIENT',decisions,gradients,'surrogate_delta_mean',float(((after-before)*qg).sum(-1).mean()),flush=True)
        if update_index==2:break
    assert decisions==2144 and update_index==2
    save_torch(stage/'recovered_policy_2144.pt',dict(policy=policy.state_dict(),recovery='Exact saved-data replay, no new training physics'))
    indices=[0,1,2,3,6,7,14,15]
    paths=[np.asarray(read(BASE/'learning_preparation_v4/paths.json')['training'][i]) for i in indices]
    costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    results=evaluate(policy,paths,stage,'training_corners',None,costs,started+120,environment_factory)
    for i,row in enumerate(results):
        with np.load(stage/'training_corners'/f'{i}.npz') as z:row['shape']=complete_corner_metrics(z['reference'],z['rows'])
        row['original_path']=indices[i]
    assert costs['evaluation_actor_calls']<=2000 and costs['evaluation_physics_ticks']<=45000
    write_json(stage/'results.json',dict(replay=reports,training_evaluation=results,costs=costs,
        interpretation='Exact sampled PPO batch gradients and early-policy physical evaluation; local finite-offset alignment is approximate. This does not prove the historical origin of reverse steering.'))
    receipt(stage,dict(status='completed_exact_two_batch_credit_audit',replay_actor_calls=decisions,replay_updates=2,
                       new_training_decisions=0,costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
