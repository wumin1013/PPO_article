"""PPO with separate Monte Carlo critic fitting and refreshed GAE estimates.

All samples remain the fresh on-policy batch. Complete segments have exact
undiscounted return targets; only the last unfinished segment is bootstrapped.
The critic never changes actor likelihoods because the networks share no weights.
"""
import numpy as np
import torch
from joint_policy import advantages
from learning_update_v4 import optimizers

def segmented_advantages(rewards,values,next_values,terminals,segment_end,*,lam=.98):
    """Bootstrap unfinished segments without linking independent case trajectories."""
    result=np.zeros(len(rewards),dtype=np.float32);carry=0.
    for i in range(len(rewards)-1,-1,-1):
        live=1.-float(terminals[i])
        delta=rewards[i]+live*next_values[i]-values[i]
        carry=delta+live*(1.-float(segment_end[i]))*lam*carry
        result[i]=carry
    return result,result+np.asarray(values,dtype=np.float32)

def update(policy,optimizer,batch,*,epochs=6,minibatch=256,lam=.98):
    obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
    latent=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
    old_log=torch.as_tensor(batch['log_probability'],dtype=torch.float32)
    segmented='segment_end' in batch
    if segmented:
        _,returns=segmented_advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],batch['segment_end'],lam=1.)
    else:
        _,returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],lam=1.)
    targets=torch.as_tensor(returns,dtype=torch.float32)
    before=float((policy.value(obs)-targets).square().mean().detach())
    losses=[];critic_norms=[]
    for epoch in range(32):
        for ids in torch.randperm(len(obs)).split(minibatch):
            loss=.5*(policy.value(obs[ids])-targets[ids]).square().mean()
            optimizer['critic'].zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.)
            optimizer['critic'].step();losses.append(float(loss.detach()));critic_norms.append(float(norm))
    with torch.no_grad():
        current=policy.value(obs).numpy()
    if segmented:
        with torch.no_grad():following=policy.value(torch.as_tensor(np.asarray(batch['next_observation']),dtype=torch.float32)).numpy()
        adv,_=segmented_advantages(batch['reward'],current,following,batch['terminal'],batch['segment_end'],lam=lam)
    else:
        following=np.r_[current[1:],batch['next_value'][-1]]
        adv,_=advantages(batch['reward'],current,following,batch['terminal'],lam=lam)
    raw_std=float(adv.std());adv=torch.as_tensor(adv)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    parameters=list(policy.actor.parameters())+[policy.log_std]
    statistics=[];stopped=False;actor_steps=0
    for epoch in range(epochs):
        for ids in torch.randperm(len(obs)).split(minibatch):
            logp,entropy,_=policy.evaluate(obs[ids],latent[ids])
            logratio=logp-old_log[ids];ratio=logratio.exp()
            kl=float(((ratio-1.)-logratio).mean().detach())
            loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()-.001*entropy.mean()
            norm=0.
            if not stopped and kl<=.03:
                optimizer['actor'].zero_grad(set_to_none=True);loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,.5));optimizer['actor'].step();actor_steps+=1
            else:stopped=True
            statistics.append([float(loss.detach()),kl,float(entropy.mean().detach()),norm])
    result=dict(zip(['actor_loss','approximate_kl','latent_entropy','actor_gradient_norm'],np.mean(statistics,axis=0).tolist()))
    result.update(value_loss=float(np.mean(losses)),critic_gradient_norm=float(np.mean(critic_norms)),
        critic_target='Undiscounted Monte Carlo with final unfinished-segment bootstrap',
        critic_mse_before=before,critic_mse_after=float(np.mean((current-returns)**2)),
        refreshed_advantage_std=raw_std,critic_minibatches=len(losses),actor_minibatches=actor_steps,actor_early_stop=stopped)
    if segmented:result.update(critic_target='Undiscounted return with separate unfinished-segment bootstraps',independent_segments=int(np.count_nonzero(batch['segment_end'])))
    return result
