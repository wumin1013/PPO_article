"""PPO on a 1 ms time axis with preview decisions marked as events.

Physical controls are sampled each tick. The third latent action contributes to
the joint likelihood and entropy only when a new preview decision is made.
Held preview values are state, not repeated stochastic actions. Losses remain
means over all physical ticks; there is no event-count or duration multiplier.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.ppo import PPOContinuous, PolicyNetContinuous
from src.utils import rl_utils


class PreviewEventPolicy(PolicyNetContinuous):
    """Existing shared 3-head network, optionally without the std=0.01 floor.

    std_fix uses 0.001 + softplus(raw_std), capped at 2 like the legacy policy.
    The positive 0.001 floor is additive, so initialization at 0.01 has a
    nonzero derivative. The upper cap retains a zero derivative above 2.
    Parameter names/shapes are identical to the existing three-head policy.
    """

    def __init__(self, state_dim, hidden_dim, action_dim=3, std_mode="legacy"):
        if action_dim != 3:
            raise ValueError("PreviewEventPolicy requires three network actions")
        if std_mode not in ("legacy", "std_fix"):
            raise ValueError("std_mode must be legacy or std_fix")
        super().__init__(state_dim, hidden_dim, action_dim)
        self.std_mode = std_mode

    def forward(self, x):
        if self.std_mode == "legacy":
            return super().forward(x)
        x = self.ln1(F.elu(self.shared_fc1(x)))
        x = self.ln2(F.elu(self.shared_fc2(x)))
        x = self.ln3(F.elu(self.shared_fc3(x)))
        raw_mu = self.mu_head(x)
        mu = torch.cat((torch.tanh(raw_mu[:, :1]), torch.sigmoid(raw_mu[:, 1:])), dim=1)
        std = F.softplus(self.std_head(x)) + 0.001
        return mu, torch.clamp(std, max=2.0)


def _moments(tensor):
    value = tensor.detach()
    return dict(mean=float(value.mean()), std=float(value.std(unbiased=False)),
                min=float(value.min()), max=float(value.max()))


def _gradient_diagnostics(actor):
    result = {}
    squared = 0.0
    for name, parameter in actor.named_parameters():
        if parameter.grad is not None and not name.startswith(("mu_head", "std_head")):
            squared += float(parameter.grad.detach().square().sum())
    result["trunk_gradient_norm"] = math.sqrt(squared)
    for head in ("mu", "std"):
        layer = getattr(actor, head + "_head")
        for index, label in enumerate(("angular", "feed", "preview")):
            weight = layer.weight.grad[index]
            bias = layer.bias.grad[index]
            result[f"{label}_{head}_weight_gradient_norm"] = float(weight.norm())
            result[f"{label}_{head}_bias_gradient"] = float(bias)
            result[f"{label}_{head}_head_gradient_norm"] = float(torch.sqrt(weight.square().sum() + bias.square()))
        result[f"physics_{head}_weight_gradient_norm"] = float(layer.weight.grad[:2].norm())
    return result


class PreviewEventPPO(PPOContinuous):
    """Event-masked joint PPO with the existing shared policy/value networks.

    The caller owns preview scheduling, held-D/phase observations and execution
    clipping. update() consumes original latent actions plus preview_decisions
    or action_masks. dones marks resets; terminated suppresses bootstrap;
    truncated ends GAE while retaining bootstrap from the actual final state.
    """

    def __init__(self, *args, std_mode="legacy", anchor_beta=1.0,
                 anchor_mean=(0.0, 0.05), anchor_std=(0.01, 0.01),
                 initial_mean=(0.0, 0.05, (3.0 - 0.8) / 3.8),
                 initial_std=(0.01, 0.01, 0.05), **kwargs):
        super().__init__(*args, **kwargs)
        if self.action_dim != 3 or self.active_action_dims not in ((0, 1), (0, 1, 2)):
            raise ValueError("Use action_dim=3 and active_action_dims=(0,1) or (0,1,2)")
        if not math.isfinite(anchor_beta) or anchor_beta < 0:
            raise ValueError("anchor_beta must be finite and nonnegative")
        means, scales = np.asarray(initial_mean, float), np.asarray(initial_std, float)
        if means.shape != (3,) or not np.isfinite(means).all() or not (-1 < means[0] < 1) or np.any((means[1:] <= 0) | (means[1:] >= 1)):
            raise ValueError("initial_mean must be interior to the three action bounds")
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any((scales <= 0.001) | (scales >= 2.0)):
            raise ValueError("initial_std must have three values strictly between .001 and 2")
        anchor_mean_array, anchor_std_array = np.asarray(anchor_mean, float), np.asarray(anchor_std, float)
        if anchor_mean_array.shape != (2,) or anchor_std_array.shape != (2,) or not np.isfinite(anchor_mean_array).all() or not np.isfinite(anchor_std_array).all() or np.any(anchor_std_array <= 0):
            raise ValueError("anchor_mean/std must be finite two-element vectors with positive std")
        # Replace only the forward implementation, retaining the base policy
        # weights and the caller's RNG stream. This enables legacy replay.
        original_state = self.actor.state_dict()
        with torch.random.fork_rng(devices=[]):
            actor = PreviewEventPolicy(self.state_dim, self.actor.shared_fc1.out_features,
                                       self.action_dim, std_mode=std_mode).to(self.device)
        actor.load_state_dict(original_state)
        self.actor = actor
        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.std_mode = std_mode
        self.anchor_beta = float(anchor_beta)
        self.anchor_mean = torch.tensor(anchor_mean_array, dtype=torch.float32, device=self.device)
        self.anchor_std = torch.tensor(anchor_std_array, dtype=torch.float32, device=self.device)
        self.initial_preview = float(means[2])
        with torch.no_grad():
            self.actor.mu_head.weight.zero_()
            self.actor.mu_head.bias.copy_(torch.tensor(
                [math.atanh(means[0]), *(math.log(x / (1 - x)) for x in means[1:])],
                dtype=torch.float32, device=self.device))
            self.actor.std_head.weight.zero_()
            self.actor.std_head.bias.copy_(torch.tensor(
                [math.log(math.expm1(x - 0.001)) for x in scales],
                dtype=torch.float32, device=self.device))
        self.logged_steps = 0
        self.last_update_diagnostics = {}

    @torch.no_grad()
    def take_action(self, state, preview_decision=True, held_preview=None, return_info=False):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,) or not np.isfinite(state_array).all():
            raise ValueError("state must be a finite vector of state_dim values")
        event = bool(preview_decision) and 2 in self.active_action_dims
        if not event and held_preview is None and 2 in self.active_action_dims:
            raise ValueError("A nondecision direct tick requires held_preview")
        held = self.initial_preview if held_preview is None else float(held_preview)
        if not math.isfinite(held):
            raise ValueError("held_preview must be finite")
        if not hasattr(self, "state_tensor"):
            self.state_tensor = torch.empty((1, self.state_dim), dtype=torch.float32, device=self.device)
        self.state_tensor[0] = torch.as_tensor(state_array, device=self.device)
        mu, std = self.actor(self.state_tensor)
        # Consume a common three-variate draw to retain legacy physical RNG
        # alignment. Off-event, the unused third draw is discarded, not scored
        # or executed as a new decision.
        latent = torch.distributions.Normal(mu, std).sample()
        if not event:
            latent[0, 2] = held
        mask = torch.tensor([[1., 1., float(event)]], device=self.device)
        action = latent[0].cpu().numpy().copy()
        if return_info:
            return action, dict(action_mask=mask[0].cpu().numpy().copy(),
                                old_log_prob=float(self.latent_log_prob(mu, std, latent, mask)[0, 0]))
        return action

    @staticmethod
    def latent_log_prob(mu, std, actions, action_masks):
        # Mask before log_prob too: a huge but irrelevant held value must not
        # overflow to -inf and then produce 0 * -inf = nan.
        scored_actions = torch.where(action_masks.bool(), actions, mu.detach())
        return (torch.distributions.Normal(mu, std).log_prob(scored_actions) * action_masks).sum(dim=1, keepdim=True)

    @staticmethod
    def masked_entropy(mu, std, action_masks):
        return (torch.distributions.Normal(mu, std).entropy() * action_masks).sum(dim=1, keepdim=True)

    def motion_anchor_kl(self, mu, std):
        current = torch.distributions.Normal(mu[:, :2], std[:, :2])
        anchor = torch.distributions.Normal(self.anchor_mean, self.anchor_std)
        return torch.distributions.kl_divergence(current, anchor).sum(dim=1).mean()

    def _binary_vector(self, value, count, name):
        array = np.asarray(value)
        if array.shape not in ((count,), (count, 1)) or not np.isin(array, (0, 1)).all():
            raise ValueError(f"{name} must be a binary vector with {count} entries")
        return torch.as_tensor(array, dtype=torch.float32, device=self.device).reshape(count, 1)

    def _batch_masks(self, transitions, count):
        if "preview_decisions" not in transitions and "action_masks" not in transitions:
            if 2 in self.active_action_dims:
                raise ValueError("Direct PPO requires explicit preview_decisions or action_masks")
            return torch.tensor([[1., 1., 0.]], device=self.device).repeat(count, 1)
        decisions = None
        if "preview_decisions" in transitions:
            decisions = self._binary_vector(transitions["preview_decisions"], count, "preview_decisions")
            if 2 not in self.active_action_dims:
                decisions = torch.zeros_like(decisions)
        if "action_masks" in transitions:
            array = np.asarray(transitions["action_masks"])
            if array.shape != (count, 3) or not np.isin(array, (0, 1)).all() or not np.all(array[:, :2] == 1):
                raise ValueError("action_masks must have shape (N,3), binary entries and two active physical actions")
            masks = torch.as_tensor(array, dtype=torch.float32, device=self.device)
            if 2 not in self.active_action_dims and bool(masks[:, 2].any()):
                raise ValueError("Fixed PPO cannot score preview actions")
            if decisions is not None and not torch.equal(masks[:, 2:3], decisions):
                raise ValueError("action_masks and preview_decisions disagree")
            return masks
        return torch.cat((torch.ones((count, 2), device=self.device), decisions), dim=1)

    def _episode_flags(self, transitions, count):
        zeros = torch.zeros((count, 1), device=self.device)
        ends = self._binary_vector(transitions["dones"], count, "dones") if "dones" in transitions else None
        truncated = self._binary_vector(transitions["truncated"], count, "truncated") if "truncated" in transitions else None
        if "terminated" in transitions:
            terminated = self._binary_vector(transitions["terminated"], count, "terminated")
        elif ends is not None:
            terminated = ends * (1 - (zeros if truncated is None else truncated))
        else:
            terminated = zeros
        if truncated is None:
            truncated = zeros if ends is None else ends * (1 - terminated)
        combined = torch.maximum(terminated, truncated)
        if ends is not None and not torch.equal(ends, combined):
            raise ValueError("dones must equal terminated or truncated")
        return combined, terminated, truncated

    def update(self, transition_dict):
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
            critic_loss = F.smooth_l1_loss(self.critic(states), targets)
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
            td_target=_moments(targets), old_log_probability_max_absolute_error=old_log_probability_error,
            preview_latent_clipped_fraction=float(((actions[event, 2] < 0) | (actions[event, 2] > 1)).float().mean()) if bool(event.any()) else 0.,
            normalization="Mean over all 1 ms physical ticks; no preview-event or duration multiplier",
            likelihood="Joint latent Gaussian on newly sampled dimensions; held preview excluded by event mask",
            epochs=epoch_rows)
        return float(actor_loss.detach()), float(critic_loss.detach())
