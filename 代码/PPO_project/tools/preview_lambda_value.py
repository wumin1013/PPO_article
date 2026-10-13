"""Single critic-target candidate; frozen PPO source remains unchanged.

The update below is a static copy of PreviewEventPPO.update. Its only learning
change is the unnormalized, detached GAE lambda-return critic target. Actor
loss, rollout boundaries, density masks, optimizers and all guards are retained.
"""
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

import preview_coherent_distance as coherent
from src.algorithms.preview_event_ppo import _gradient_diagnostics, _moments, rl_utils


def update_lambda_return(self, transition_dict):
    self.critic.eval()  # Disable dropout for rollout targets and every epoch.
    count = len(transition_dict["states"])
    if count == 0:
        raise ValueError("Cannot update from an empty rollout")
    tensors = {}
    for key, shape in (("states", (count, self.state_dim)), ("next_states", (count, self.state_dim)),
                       ("actions", (count, 3))):
        array = np.asarray(transition_dict[key], dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{key} must be finite with shape {shape}")
        tensors[key] = torch.as_tensor(array, device=self.device)
    reward_array = np.asarray(transition_dict["rewards"], dtype=np.float32)
    if reward_array.shape not in ((count,), (count, 1)) or not np.isfinite(reward_array).all():
        raise ValueError("rewards must be a finite vector with N entries")
    rewards = torch.as_tensor(reward_array, device=self.device).reshape(count, 1)
    states, next_states, actions = (tensors[key] for key in ("states", "next_states", "actions"))
    masks = self._batch_masks(transition_dict, count)
    ends, terminated, truncated = self._episode_flags(transition_dict, count)
    with torch.no_grad():
        targets = rewards + self.gamma * self.critic(next_states) * (1 - terminated)
        delta = targets - self.critic(states)
        raw_advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, delta.cpu(), ends.cpu()).to(self.device)
        value_targets = self.critic(states) + raw_advantage
        advantage = raw_advantage
        if advantage.std(unbiased=False) > 1e-6:
            advantage = torch.clamp((advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8), -10, 10)
        old_mu, old_std = self.actor(states)
        old_log_probs = self.latent_log_prob(old_mu, old_std, actions, masks)
        old_log_probability_error = 0.0
        if "old_log_probs" in transition_dict:
            supplied = torch.as_tensor(np.asarray(transition_dict["old_log_probs"]), dtype=torch.float32, device=self.device)
            if supplied.shape not in ((count,), (count, 1)) or not bool(torch.isfinite(supplied).all()):
                raise ValueError("old_log_probs must be a finite vector with N entries")
            supplied = supplied.reshape(count, 1)
            old_log_probability_error = float(torch.max(torch.abs(supplied - old_log_probs)))
            if old_log_probability_error > 1e-4:
                raise ValueError("Stored old_log_probs do not match current rollout policy, raw latent actions and event masks")
            old_log_probs = supplied
    event = masks[:, 2].bool()
    epoch_rows = []
    for epoch in range(self.epochs):
        mu, std = self.actor(states)
        log_probs = self.latent_log_prob(mu, std, actions, masks)
        raw_log_ratio = log_probs - old_log_probs
        # Retain the preceding study's common numerical guard. Report its
        # activation separately from PPO clipping of the surrogate.
        log_ratio = torch.clamp(raw_log_ratio, -5., 5.)
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
        entropy = self.masked_entropy(mu, std, masks).mean() * self.ent_coef
        anchor_kl = self.motion_anchor_kl(mu, std)
        actor_loss = policy_loss - entropy + self.anchor_beta * anchor_kl
        critic_loss = F.smooth_l1_loss(self.critic(states), value_targets)
        if not bool(torch.isfinite(actor_loss) & torch.isfinite(critic_loss)):
            raise FloatingPointError("Non-finite event PPO loss; no optimizer step taken")
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        gradients = _gradient_diagnostics(self.actor)
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
        with torch.no_grad():
            outside = (ratio < 1 - self.eps) | (ratio > 1 + self.eps)
            clipped_branch = ((advantage > 0) & (ratio > 1 + self.eps)) | ((advantage < 0) & (ratio < 1 - self.eps))
            kl_by_dim = torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mu, old_std), torch.distributions.Normal(mu, std))
            row = dict(epoch=epoch, actor_loss=float(actor_loss), critic_loss=float(critic_loss),
                policy_loss=float(policy_loss), entropy_bonus=float(entropy),
                motion_anchor_kl=float(anchor_kl), motion_anchor_beta=self.anchor_beta,
                anchor_loss=float(self.anchor_beta * anchor_kl),
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                ratio_outside_clip_fraction=float(outside.float().mean()),
                surrogate_clipped_fraction=float(clipped_branch.float().mean()),
                event_ratio_outside_clip_fraction=float(outside[event].float().mean()) if bool(event.any()) else 0.,
                event_surrogate_clipped_fraction=float(clipped_branch[event].float().mean()) if bool(event.any()) else 0.,
                log_ratio_guard_fraction=float((raw_log_ratio.abs() > 5).float().mean()),
                old_new_kl_masked_mean=float((kl_by_dim * masks).sum(1).mean()),
                old_new_kl_physics_mean=float(kl_by_dim[:, :2].sum(1).mean()),
                old_new_kl_preview_event_mean=float(kl_by_dim[event, 2].mean()) if bool(event.any()) else 0.,
                preview_mu_min=float(mu[:, 2].min()), preview_mu_max=float(mu[:, 2].max()),
                preview_std_mean=float(std[:, 2].mean()), physics_mu_mean=mu[:, :2].mean(0).tolist(),
                physics_std_mean=std[:, :2].mean(0).tolist(),
                std_by_action_mean=std.mean(0).tolist(), std_by_action_min=std.min(0).values.tolist(),
                std_by_action_max=std.max(0).values.tolist(),
                actor_gradient_norm_before_clip=float(actor_norm), critic_gradient_norm_before_clip=float(critic_norm),
                actor_gradient_clip_scale=min(1., self.max_grad_norm / (float(actor_norm) + 1e-6)), **gradients)
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        epoch_rows.append(row)
    self.logged_steps += count
    self.last_update_diagnostics = dict(step=self.logged_steps, transition_count=count,
        preview_decision_count=int(event.sum()), preview_decision_fraction=float(event.float().mean()),
        physical_decision_count=count, active_action_dims=list(self.active_action_dims),
        terminated_count=int(terminated.sum()), truncated_count=int(truncated.sum()), episode_end_count=int(ends.sum()),
        std_mode=self.std_mode, reward=_moments(rewards), td_delta=_moments(delta),
        raw_advantage=_moments(raw_advantage), normalized_advantage=_moments(advantage),
        lambda_value_target=_moments(value_targets),
        td_target=_moments(targets), old_log_probability_max_absolute_error=old_log_probability_error,
        preview_latent_clipped_fraction=float(((actions[event, 2] < 0) | (actions[event, 2] > 1)).float().mean()) if bool(event.any()) else 0.,
        normalization="Mean over20ms macro decisions; no preview-event or duration multiplier",
        macro_period_s=.020,
        critic_regression_target="Frozen unnormalized GAE plus old state value; td_target logs the one-step comparison only",
        likelihood="Joint latent Gaussian on newly sampled dimensions; held preview excluded by event mask",
        epochs=epoch_rows)
    return float(actor_loss.detach()), float(critic_loss.detach())


def make_agent(cfg, env):
    """Reuse the registered common initialization and event-aware action masks."""
    agent = coherent.make_agent(cfg, env)
    agent.update = MethodType(update_lambda_return, agent)
    return agent
