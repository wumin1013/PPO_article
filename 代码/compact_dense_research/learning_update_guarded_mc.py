"""Reuse the existing PPO objective with transactional post-step KL protection.

The guard measures the full fresh batch against the collection policy. Rejected
Adam steps restore both parameters and optimizer moments, then halve the step.
The same routine supports both two- and three-output residual policies.
"""
import copy
import numpy as np
import torch
from learning_update_compact_mc import optimizers, update as original_update


class PostStepGuard:
    def __init__(self, policy, optimizer, observation, latent, old_log,
                 limit=.03, max_backtracks=8):
        self.policy = policy
        self.optimizer = optimizer
        self.observation = observation
        self.latent = latent
        self.old_log = old_log
        self.limit = limit
        self.max_backtracks = max_backtracks
        self.parameters = list(policy.actor.parameters()) + [policy.log_std]
        with torch.no_grad():
            distribution = policy.distribution(observation)
            self.old_mean = distribution.mean.detach().clone().double()
            self.old_scale = distribution.scale.detach().clone().double()
        self.attempts = []
        self.accepted = 0
        self.rejected = 0

    def zero_grad(self, **kwargs):
        return self.optimizer.zero_grad(**kwargs)

    @torch.no_grad()
    def measure(self):
        distribution = self.policy.distribution(self.observation)
        mean, scale = distribution.mean.double(), distribution.scale.double()
        kl = (torch.log(scale / self.old_scale) +
              (self.old_scale.square() + (self.old_mean - mean).square()) /
              (2. * scale.square()) - .5).sum(-1).mean()
        logratio = distribution.log_prob(self.latent).sum(-1).double() - self.old_log.double()
        sampled_kl = (logratio.expm1() - logratio).mean()
        return float(kl), float(sampled_kl)

    def step(self):
        saved_parameters = [p.detach().clone() for p in self.parameters]
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        rates = [group['lr'] for group in self.optimizer.param_groups]
        accepted = False
        for backtrack in range(self.max_backtracks + 1):
            if backtrack:
                self.restore(saved_parameters, saved_optimizer)
            for group, rate in zip(self.optimizer.param_groups, rates):
                group['lr'] = rate * 2. ** (-backtrack)
            self.optimizer.step()
            exact, sampled = self.measure()
            accepted = (np.isfinite(exact) and np.isfinite(sampled) and
                        max(exact, sampled) <= self.limit)
            self.attempts.append(dict(backtrack=backtrack, exact_kl=exact,
                sampled_kl=sampled, accepted=bool(accepted),
                learning_rates=[group['lr'] for group in self.optimizer.param_groups]))
            if accepted:
                self.accepted += 1
                break
        if not accepted:
            self.restore(saved_parameters, saved_optimizer)
            self.rejected += 1

    @torch.no_grad()
    def restore(self, parameters, optimizer_state):
        for current, saved in zip(self.parameters, parameters):
            current.copy_(saved)
        self.optimizer.load_state_dict(optimizer_state)


def update(policy, optimizer, batch, *, epochs=6, minibatch=256, lam=.98):
    observation = torch.as_tensor(np.asarray(batch['observation']), dtype=torch.float32)
    latent = torch.as_tensor(np.asarray(batch['latent']), dtype=torch.float32)
    old_log = torch.as_tensor(batch['log_probability'], dtype=torch.float32)
    with torch.no_grad():
        collected_log = policy.distribution(observation).log_prob(latent).sum(-1)
        initial_error = float((collected_log - old_log).abs().max())
    if not np.isfinite(initial_error) or initial_error > 1e-3:
        raise ValueError(f'Collection-policy log probability mismatch: {initial_error}')
    guard = PostStepGuard(policy, optimizer['actor'], observation, latent, old_log)
    result = original_update(policy, dict(actor=guard, critic=optimizer['critic']),
                             batch, epochs=epochs, minibatch=minibatch, lam=lam)
    exact, sampled = guard.measure()
    assert max(exact, sampled) <= guard.limit + 1e-10
    result.update(actor_minibatch_attempts=result['actor_minibatches'],
        actor_minibatches=guard.accepted, rejected_actor_minibatches=guard.rejected,
        initial_log_probability_max_error=initial_error,
        post_update_exact_kl=exact, post_update_sampled_kl=sampled,
        post_step_kl_limit=guard.limit, post_step_attempts=guard.attempts,
        actor_learning_rates=[group['lr'] for group in optimizer['actor'].param_groups],
        update_guard='Full-batch exact Gaussian and sampled KL; parameter/moment rollback; bounded step halving')
    return result
