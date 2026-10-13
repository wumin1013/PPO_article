"""Joint PPO decisions at 20 ms, with the frozen event-PPO update reused.

The environment holds each bounded action while integrating KCM at 1 ms. This
agent scores one original latent draw per macro transition, never its repeated
physical executions. Removing a control dimension masks its density, entropy,
and policy gradients; inactive executed angular action is zero and preview is
the configured fixed value.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.algorithms.preview_event_ppo import PreviewEventPPO


class PreviewMacroPPO(PreviewEventPPO):
    """Existing three-head network, joint macro sampling and constant masks.

    One update row is one macro decision. The caller supplies the sum of its
    physical-tick rewards, the actual final observation, and ordinary episode
    termination/truncation flags. Full macro transitions last macro_period_s;
    a shorter terminal transition has no bootstrap. A shorter nonterminal time
    limit needs duration-aware discounting outside this fixed-period protocol.
    """

    def __init__(self, *args, active_action_dims=(0, 1, 2),
                 fixed_preview=(3.0 - 0.8) / 3.8, macro_period_s=0.020, **kwargs):
        active = tuple(active_action_dims)
        if active not in ((0, 1, 2), (0, 1), (1, 2), (1,)):
            raise ValueError("Macro active_action_dims must be (0,1,2), (0,1), (1,2), or (1,)")
        if not math.isfinite(macro_period_s) or macro_period_s <= 0:
            raise ValueError("macro_period_s must be finite and positive")
        if not math.isfinite(fixed_preview) or not 0 <= fixed_preview <= 1:
            raise ValueError("fixed_preview must be finite and in [0,1]")
        if kwargs.get('anchor_beta', 0.0) != 0:
            raise ValueError("Macro protocol requires anchor_beta=0")
        kwargs['anchor_beta'] = 0.0
        kwargs.setdefault('std_mode', 'std_fix')
        kwargs.setdefault('action_dim', 3)
        kwargs.setdefault('actor_lr', 3e-4)
        kwargs.setdefault('critic_lr', 3e-4)
        kwargs.setdefault('gamma', math.exp(-macro_period_s / 10.0))
        kwargs.setdefault('lmbda', 1.0)
        kwargs.setdefault('initial_mean', (0.0, 0.05, (3.0 - 0.8) / 3.8))
        kwargs.setdefault('initial_std', (0.03, 0.05, 0.15))
        # The frozen parent constructor only accepts both physical dimensions.
        # Build its unchanged networks using the full mask, then specialize the
        # decision semantics here; no parent source or architecture is changed.
        super().__init__(*args, active_action_dims=(0, 1, 2), **kwargs)
        self.active_action_dims = active
        self.fixed_preview = float(fixed_preview)
        self.macro_period_s = float(macro_period_s)
        self.action_mask = np.asarray([float(index in active) for index in range(3)], np.float32)

    @torch.no_grad()
    def take_action(self, state, return_info=False, deterministic=False):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,) or not np.isfinite(state_array).all():
            raise ValueError("state must be a finite vector of state_dim values")
        state_tensor = torch.as_tensor(state_array[None], device=self.device)
        mu, std = self.actor(state_tensor)
        # Draw all three dimensions to maintain common RNG alignment between
        # matched controls, then discard inactive dimensions before scoring.
        latent = mu.clone() if deterministic else torch.distributions.Normal(mu, std).sample()
        if 0 not in self.active_action_dims:
            latent[0, 0] = 0.0
        if 2 not in self.active_action_dims:
            latent[0, 2] = self.fixed_preview
        mask = torch.as_tensor(self.action_mask[None], device=self.device)
        action = latent[0].cpu().numpy().copy()
        if return_info:
            return action, dict(action_mask=self.action_mask.copy(),
                old_log_prob=float(self.latent_log_prob(mu, std, latent, mask)[0, 0]),
                mu=mu[0].cpu().numpy().copy(), std=std[0].cpu().numpy().copy(),
                deterministic=bool(deterministic), macro_period_s=self.macro_period_s)
        return action

    def _batch_masks(self, transitions, count):
        expected = np.broadcast_to(self.action_mask, (count, 3))
        if 'action_masks' in transitions:
            supplied = np.asarray(transitions['action_masks'])
            if supplied.shape != (count, 3) or not np.array_equal(supplied, expected):
                raise ValueError("Macro action_masks must match the constant active-action mask on every row")
        if 'preview_decisions' in transitions:
            decisions = self._binary_vector(transitions['preview_decisions'], count, 'preview_decisions')
            if not bool(torch.all(decisions == float(2 in self.active_action_dims))):
                raise ValueError("Macro preview_decisions must match whether preview is active on every row")
        return torch.as_tensor(expected.copy(), dtype=torch.float32, device=self.device)

    def motion_anchor_kl(self, mu, std):
        # No anchor term or inactive physical-head gradient route in this study.
        return mu.new_zeros(())

    def update(self, transition_dict):
        losses = super().update(transition_dict)
        diagnostics = self.last_update_diagnostics
        count = diagnostics['transition_count']
        diagnostics.update(
            macro_decision_count=count,
            macro_period_s=self.macro_period_s,
            nominal_physical_duration_s=count * self.macro_period_s,
            decision_count_by_action=[count if index in self.active_action_dims else 0 for index in range(3)],
            active_action_mask=self.action_mask.tolist(),
            normalization="Mean over macro decisions; each reward is the sum of its physical-tick rewards",
            likelihood="One joint raw latent Gaussian density per macro decision, restricted to active dimensions",
            physical_decision_count=count,
            motion_anchor_disabled=True,
        )
        return losses
