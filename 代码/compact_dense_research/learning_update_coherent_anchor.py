"""Fresh PPO plus a single incumbent's action-equivalent feedback anchors."""
import numpy as np
import torch
from learning_update_compact_mc import optimizers,segmented_advantages
from learning_update_guarded_mc import PostStepGuard


def equivalent_loss(prediction,target,scale):
    error=prediction-target
    error=torch.where(target>=1.,-torch.relu(1.02-prediction),error)
    error=torch.where(target<=-1.,torch.relu(prediction+1.02),error)
    return (error/scale).square().mean()


def update(policy,optimizer,batch,anchors,*,kl_limit=.01,epochs=6,minibatch=256):
    obs=torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
    next_obs=torch.as_tensor(np.asarray(batch['next_observation']),dtype=torch.float32)
    action=torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
    old_log=torch.as_tensor(batch['log_probability'],dtype=torch.float32)
    target=torch.as_tensor(np.asarray(batch['teacher_target']),dtype=torch.float32)
    scale=torch.tensor([.015,.15,.3][:policy.action_dim])
    with torch.no_grad():initial_error=float(abs(policy.distribution(obs).log_prob(action).sum(-1)-old_log).max())
    roundoff_audit=None
    if initial_error>1e-3:
        # Narrow fixed exploration magnifies float32 GEMV/GEMM rounding.
        # Replay the actual single-observation sampling computation before
        # accepting a batch: never excuse a changed collection policy.
        with torch.no_grad():
            individual=[policy.distribution(x) for x in obs]
            replay=torch.stack([d.log_prob(z).sum() for d,z in zip(individual,action)])
            pointwise_error=float(abs(replay-old_log).max())
            pointwise_mean=torch.stack([d.mean for d in individual])
            batched=policy.distribution(obs)
            raw_error=float(abs(batched.mean-pointwise_mean).max())
            normalized_error=float(abs((batched.mean-pointwise_mean)/batched.scale).max())
        assert pointwise_error<=1e-6,('Actual sampling likelihood changed',pointwise_error)
        assert raw_error<=2e-6 and normalized_error<=.002,('Unexpected matrix roundoff',raw_error,normalized_error)
        roundoff_audit=dict(pointwise_log_probability_error=pointwise_error,
            maximum_raw_mean_discrepancy=raw_error,maximum_normalized_mean_discrepancy=normalized_error,
            recorded_sampling_log_probabilities_preserved=True)
    for group in optimizer['actor'].param_groups:group['lr']=3e-4
    guard=PostStepGuard(policy,optimizer['actor'],obs,action,old_log,limit=kl_limit)
    _,returns=segmented_advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],batch['segment_end'],lam=1.)
    returns=torch.tensor(returns,dtype=torch.float32)
    critic_losses=[]
    for _ in range(32):
        for ids in torch.randperm(len(obs)).split(minibatch):
            loss=.5*(policy.value(obs[ids])-returns[ids]).square().mean()
            optimizer['critic'].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1.);optimizer['critic'].step()
            critic_losses.append(float(loss.detach()))
    with torch.no_grad():
        values=policy.value(obs).numpy();following=policy.value(next_obs).numpy()
    adv,_=segmented_advantages(batch['reward'],values,following,batch['terminal'],batch['segment_end'],lam=.98)
    advantage=torch.tensor(adv);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    stats=[]
    for _ in range(epochs):
        for ids in torch.randperm(len(obs)).split(minibatch):
            logp,entropy,_=policy.evaluate(obs[ids],action[ids])
            ratio=(logp-old_log[ids]).exp()
            surrogate=-torch.minimum(ratio*advantage[ids],ratio.clamp(.8,1.2)*advantage[ids]).mean()
            visited=equivalent_loss(policy.actor(obs[ids]),target[ids],scale)
            anchor_ids=torch.randint(len(anchors[0]),(minibatch,))
            retained=equivalent_loss(policy.actor(anchors[0][anchor_ids]),anchors[1][anchor_ids],scale)
            loss=surrogate+.25*(visited+retained)-.001*entropy.mean()
            guard.zero_grad(set_to_none=True);loss.backward()
            actor_parameters=list(policy.actor.parameters())+[policy.log_std]
            torch.nn.utils.clip_grad_norm_([p for p in actor_parameters if p.requires_grad],.5)
            guard.step()
            stats.append([float(surrogate.detach()),float(visited.detach()),float(retained.detach())])
    exact,sampled=guard.measure();assert max(exact,sampled)<=kl_limit+1e-10
    return dict(ppo_actor_loss=float(np.mean(stats,axis=0)[0]),visited_anchor_loss=float(np.mean(stats,axis=0)[1]),
        retained_anchor_loss=float(np.mean(stats,axis=0)[2]),value_loss=float(np.mean(critic_losses)),
        post_update_exact_kl=exact,post_update_sampled_kl=sampled,post_step_kl_limit=kl_limit,
        actor_minibatches=guard.accepted,rejected_actor_minibatches=guard.rejected,
        post_step_attempts=guard.attempts,initial_log_probability_max_error=initial_error,
        verified_float32_reduction_roundoff=roundoff_audit,
        auxiliary_weight_each=.25,training_kind='Fresh on-policy PPO with explicit incumbent action-equivalence regularization;retained anchors are auxiliary offline samples,not PPO rollouts')
