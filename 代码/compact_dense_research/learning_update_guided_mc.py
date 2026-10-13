"""On-policy PPO with explicitly recorded auxiliary action supervision."""
import numpy as np
import torch
from learning_update_guarded_mc import PostStepGuard
from learning_update_compact_mc import optimizers,segmented_advantages

AUXILIARY_WEIGHT=.5
ACTION_SCALE=(.015,.15,.3)


def update(policy,optimizer,batch,*,epochs=6,minibatch=256,lam=.98):
    obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
    following_obs=torch.as_tensor(np.asarray(batch['next_observation']),dtype=torch.float32)
    latent=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
    old_log=torch.as_tensor(batch['log_probability'],dtype=torch.float32)
    teacher=torch.as_tensor(np.asarray(batch['teacher_target']),dtype=torch.float32)
    scale=torch.tensor(ACTION_SCALE)
    with torch.no_grad():
        collected=policy.distribution(obs).log_prob(latent).sum(-1)
        initial_error=float((collected-old_log).abs().max())
    assert np.isfinite(initial_error) and initial_error<=1e-3
    guard=PostStepGuard(policy,optimizer['actor'],obs,latent,old_log)
    _,returns=segmented_advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],batch['segment_end'],lam=1.)
    targets=torch.as_tensor(returns,dtype=torch.float32)
    before=float((policy.value(obs)-targets).square().mean().detach())
    critic_losses=[]
    for epoch in range(32):
        for ids in torch.randperm(len(obs)).split(minibatch):
            loss=.5*(policy.value(obs[ids])-targets[ids]).square().mean()
            optimizer['critic'].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.)
            optimizer['critic'].step();critic_losses.append(float(loss.detach()))
    with torch.no_grad():
        current=policy.value(obs).numpy();following=policy.value(following_obs).numpy()
    advantages,_=segmented_advantages(batch['reward'],current,following,batch['terminal'],batch['segment_end'],lam=lam)
    raw_std=float(advantages.std())
    adv=torch.as_tensor(advantages)
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    parameters=list(policy.actor.parameters())+[policy.log_std]
    stats=[];stopped=False
    with torch.no_grad():
        aux_before=float(((policy.actor(obs)-teacher)/scale).square().mean())
    for epoch in range(epochs):
        for ids in torch.randperm(len(obs)).split(minibatch):
            logp,entropy,_=policy.evaluate(obs[ids],latent[ids])
            logratio=logp-old_log[ids];ratio=logratio.exp()
            kl=float(((ratio-1.)-logratio).mean().detach())
            ppo_loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
            aux=((policy.actor(obs[ids])-teacher[ids])/scale).square().mean()
            loss=ppo_loss-.001*entropy.mean()+AUXILIARY_WEIGHT*aux
            if not stopped and kl<=.03:
                guard.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,.5);guard.step()
            else:
                stopped=True
            stats.append([float(ppo_loss.detach()),float(aux.detach()),kl])
    exact,sampled=guard.measure()
    assert max(exact,sampled)<=guard.limit+1e-10
    with torch.no_grad():
        aux_after=float(((policy.actor(obs)-teacher)/scale).square().mean())
    result=dict(zip(['ppo_actor_loss','auxiliary_action_loss','approximate_kl'],np.mean(stats,axis=0).tolist()))
    result.update(value_loss=float(np.mean(critic_losses)),critic_mse_before=before,
        critic_mse_after=float(np.mean((current-returns)**2)),refreshed_advantage_std=raw_std,
        actor_minibatches=guard.accepted,rejected_actor_minibatches=guard.rejected,
        actor_early_stop=stopped,post_update_exact_kl=exact,post_update_sampled_kl=sampled,
        post_step_kl_limit=guard.limit,post_step_attempts=guard.attempts,
        auxiliary_weight=AUXILIARY_WEIGHT,auxiliary_scales=ACTION_SCALE,
        auxiliary_mse_before=aux_before,auxiliary_mse_after=aux_after,
        initial_log_probability_max_error=initial_error,independent_segments=int(np.count_nonzero(batch['segment_end'])),
        training_kind='Fresh on-policy PPO plus explicit auxiliary action supervision on current visited states')
    return result
