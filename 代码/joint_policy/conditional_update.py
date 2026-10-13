# Derived from learning_update_v4.py; only history-conditioned likelihood evaluation and context tensors differ.
"""PPO with separate actor/critic optimization and measured KL stopping."""
import numpy as np
import torch
from joint_policy import advantages


def update(policy, optimizer, batch, *, epochs=6, minibatch=256, lam=.98):
    obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
    latent=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
    old_log=torch.as_tensor(batch['log_probability'],dtype=torch.float32)
    previous_obs=torch.as_tensor(np.asarray(batch['previous_observation']),dtype=torch.float32)
    previous_latent=torch.as_tensor(np.asarray(batch['previous_latent']),dtype=torch.float32)
    live=torch.as_tensor(batch['context_live'],dtype=torch.bool)
    adv,returns=advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],lam=lam)
    adv=torch.as_tensor(adv)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns=torch.as_tensor(returns)
    actor_parameters=list(policy.actor.parameters())+[policy.log_std]
    statistics=[]
    actor_steps=0
    stopped=False
    for epoch in range(epochs):
        for indices in torch.randperm(len(obs)).split(minibatch):
            logp,entropy,value=policy.evaluate_conditioned(obs[indices],latent[indices],previous_obs[indices],previous_latent[indices],live[indices])
            logratio=logp-old_log[indices]
            ratio=torch.exp(logratio)
            kl=float(((ratio-1)-logratio).mean().detach())
            surrogate=torch.minimum(ratio*adv[indices],ratio.clamp(.8,1.2)*adv[indices])
            actor_loss=-surrogate.mean()-.001*entropy.mean()
            actor_norm=0.
            if not stopped and kl<=.03:
                optimizer['actor'].zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_norm=float(torch.nn.utils.clip_grad_norm_(actor_parameters,.5))
                optimizer['actor'].step()
                actor_steps+=1
            else:
                stopped=True
            value_loss=.5*(value-returns[indices]).square().mean()
            optimizer['critic'].zero_grad(set_to_none=True)
            value_loss.backward()
            critic_norm=float(torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.))
            optimizer['critic'].step()
            statistics.append([float(actor_loss.detach()),float(value_loss.detach()),
                kl,float(entropy.mean().detach()),actor_norm,critic_norm])
    result=dict(zip(['actor_loss','value_loss','approximate_kl','latent_entropy',
        'actor_gradient_norm','critic_gradient_norm'],np.mean(statistics,axis=0).tolist()))
    result.update(actor_minibatches=actor_steps,critic_minibatches=len(statistics),
                  actor_early_stop=stopped)
    return result
