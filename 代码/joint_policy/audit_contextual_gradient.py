"""Attribute the actual PPO actor gradient in the first failure-containing batch."""
import time
import numpy as np
import torch
from contextual_mixture_policy import ContextualMixturePolicy
from train_bounded_shape_ppo import update
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    source=BASE/'contextual_learned_function_ppo_20260916'
    stage=register('contextual_gradient_attribution_20260916',
        [__file__,source/'results.json',source/'episode_results.json',BASE/'train_bounded_shape_ppo.py'],
        dict(new_physics_ticks=0,replay_actor_calls=6000,wall_s=120),
        'Exactly reconstruct the first pre-update actor batch containing a failure; measure '
        'actual actor-gradient contributions from that group and the remaining successful groups. '
        'Compare the same samples with one proposed within-path return scaling, without updating or physics.')
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(9161009)
    a=torch.load(BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt',map_location='cpu',weights_only=False)
    b=torch.load(BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt',map_location='cpu',weights_only=False)
    policy=ContextualMixturePolicy(72,a['policy'],b['policy'])
    torch.manual_seed(9161009)
    optimizer={'actor':torch.optim.Adam(policy.actor_parameters(),lr=.0003),'critic':torch.optim.Adam(policy.critic.parameters(),lr=.001)}
    records=read(source/'episode_results.json')['episodes'];updates=read(source/'results.json')['updates']
    batch=[];group=[];all_groups=[];decisions=0;idx=0;found=False
    for e in records:
        with np.load(source/'episodes'/e['trace']) as z:
            data={k:z[k] for k in ['observation','latent','previous_observation','previous_latent','context_live','old_log','reward','value','terminal']}
        policy.reset_context()
        for i,obs in enumerate(data['observation']):
            z,logp,v=policy.act(obs)
            assert np.array_equal(z,data['latent'][i]) and logp==data['old_log'][i] and v==data['value'][i]
        group.append((e,data));batch.append(data);decisions+=len(data['reward'])
        if len(group)==4:
            totals=np.array([d['reward'].sum() for _,d in group]);advantages=totals-(totals.sum()-totals)/3
            for (_,d),adv in zip(group,advantages):
                d['paired_advantage']=np.full(len(d['reward']),adv)
                d['balanced_advantage']=np.full(len(d['reward']),adv/max(totals.std(),.02))
                d['failure_group']=np.full(len(d['reward']),not all(r['success'] for r,_ in group))
            all_groups.append(dict(episodes=[r['episode'] for r,_ in group],successes=sum(r['success'] for r,_ in group),returns=totals.tolist()))
            group=[]
        if decisions!=updates[idx]['decisions']:continue
        if any(np.any(d['failure_group']) for d in batch):found=True;break
        stats=update(policy,optimizer,batch,paired_advantage=True)
        assert all(stats[k]==updates[idx][k] for k in stats)
        idx+=1;batch=[];all_groups=[]
    assert found and decisions<=6000
    keys=['observation','latent','previous_observation','previous_latent','context_live']
    tensors={k:torch.as_tensor(np.concatenate([d[k] for d in batch]),dtype=torch.bool if k=='context_live' else torch.float32) for k in keys}
    logp=policy.evaluate_conditioned(tensors['observation'],tensors['latent'],tensors['previous_observation'],tensors['previous_latent'],tensors['context_live'])[0]
    mask=torch.tensor(np.concatenate([d['failure_group'] for d in batch]))
    parameters=policy.actor_parameters()
    report={}
    for name in ['paired_advantage','balanced_advantage']:
        adv=torch.tensor(np.concatenate([d[name] for d in batch]),dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        gradients=[]
        for select in [mask,~mask]:
            grads=torch.autograd.grad(-(logp*adv*select).mean(),parameters,retain_graph=True)
            gradients.append(torch.cat([g.flatten() for g in grads]))
        failed,success=gradients
        total=failed+success
        report[name]=dict(failure_group_gradient_norm=float(failed.norm()),other_success_group_gradient_norm=float(success.norm()),
            norm_ratio=float(failed.norm()/success.norm()),
            cosine_with_failure_group=float(torch.dot(total,failed)/(total.norm()*failed.norm())),
            cosine_with_success_groups=float(torch.dot(total,success)/(total.norm()*success.norm())),
            failure_group_decisions=int(mask.sum()),total_decisions=len(mask))
    write_json(stage/'results.json',dict(decisions=decisions,groups=all_groups,gradients=report,
        interpretation='Same pre-update actor and saved samples; gradient-scale imbalance is directly measured. This does not identify the historical cause of reverse commands or predict a successful new policy.'))
    receipt(stage,dict(status='exact_preupdate_gradient_attribution_complete',replay_actor_calls=decisions,physics_ticks=0,wall_s=time.perf_counter()-started))
    print(dict(decisions=decisions,gradients=report),flush=True)


if __name__=='__main__':main()
