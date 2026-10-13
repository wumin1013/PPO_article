"""Small PPO correction around the preserved historical controller.

The inherited72-dimensional controller and474 causal memory scalars remain.
Only the new17-input residual Actor/Critic is trained; this is not dimensional
reduction of the complete method or independent from-scratch PPO training.
"""
import copy
import json
import math
import numpy as np
import torch
from compact_control import CompactPolicy, CompactRuntime, feature_names, local_progress
from candidate_policy_factory import candidate_policy
from turn_aware_projector import environment_factory
from cache_corner_contexts import memory_state, restore_memory
from local_arc_support import install_local_arc_support, ClusteredPreviewFloorPolicy
from evaluate_historical_dense import HIST

_selection = None
_metadata = None


def inherited_inputs():
    global _selection, _metadata
    if _selection is None:
        _selection = json.loads((HIST / 'selected_policy.json').read_text(encoding='utf-8'))
        _metadata = torch.load(_selection['model'], map_location='cpu', weights_only=False)
    return _selection, _metadata


class ResidualPolicy(CompactPolicy):
    def __init__(self, dimension=17, action_dim=3, seed=922301):
        if (dimension, action_dim) != (17, 3):
            raise ValueError('This pilot trains three corrections with17 local/nominal inputs')
        super().__init__(dimension, action_dim, seed)
        with torch.no_grad():
            self.actor[-1].bias.zero_()
            self.log_std.copy_(torch.log(torch.tensor([.01, .04, .04])))


class HistoricalResidualRuntime:
    def __init__(self, reference, *, dimension=17, action_dim=3, preview_rule_mm=3.,
                 allow_swing=True, potential_speed=30., remaining_encoding='log'):
        if (dimension, action_dim, allow_swing, remaining_encoding) != (17, 3, True, 'log'):
            raise ValueError('Unsupported residual training interface')
        selection, metadata = inherited_inputs()
        self.env = environment_factory(reference)
        # The feature encoder reads this static field; inherited observation is unchanged.
        self.env.dimension = 14
        self.base, self.arc_layer = install_local_arc_support(candidate_policy(metadata, selection))
        self.base = ClusteredPreviewFloorPolicy(self.base, preview_rule_mm)
        self.base.bind(self.env)
        self.potential_speed = float(potential_speed)
        self.cached_decision = None
        self.nominal_latent = None
        self.cached_observation = None
        self.correction_active = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _prepare_nominal(self):
        if self.cached_decision != self.env.decisions and not self.env.done:
            original = self.env.observation()
            self.nominal_latent = self.base.act(original, True)[0].copy()
            self.cached_decision = self.env.decisions
            self.correction_active = np.count_nonzero(abs(self.base.sharp_s-self.env.progress()[0]) <= 30.) >= 2

    def observation(self):
        self._prepare_nominal()
        local = CompactRuntime.observation(self.env)
        remaining = max(0., self.env.corridor.arcs[-1]-local_progress(self.env))
        local[-1] = math.log1p(remaining/30.)/8.
        nominal = np.tanh(self.nominal_latent) if self.nominal_latent is not None else np.zeros(3)
        self.cached_observation = np.r_[local, nominal].astype(np.float32)
        return self.cached_observation.copy()

    def feature_names(self):
        names = feature_names(14)
        names[-1] = 'log1p_remaining_distance_over30_div8'
        return names + ['nominal_steering_normalized', 'nominal_feed_normalized', 'nominal_preview_normalized']

    def potential(self):
        return (self.env.progress()[0]-self.env.corridor.arcs[-1])/self.potential_speed

    def step(self, residual):
        self._prepare_nominal()
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (3,) or not np.isfinite(residual).all():
            raise ValueError('Three finite raw residual actions required')
        if not self.correction_active or not np.any(residual):
            combined = self.nominal_latent.copy()
        else:
            bounded = np.clip(np.tanh(self.nominal_latent)+residual, -1.+1e-10, 1.-1e-10)
            combined = np.arctanh(bounded)
        phi = self.potential()
        _, _, done, info = self.env.step(combined)
        reward = -info['elapsed_s']+(0. if done else self.potential())-phi
        if done and not self.env.success:
            remaining = max(0., self.env.corridor.arcs[-1]-self.env.progress()[0])
            reward -= self.env.max_ticks*.001+1.+remaining/20.
        info['historical_nominal_latent'] = self.nominal_latent.copy()
        info['residual_latent'] = residual.copy()
        info['combined_latent'] = combined.copy()
        return self.observation(), float(reward), done, info

    def state(self):
        return dict(physical=self.env.state(), inherited_memory=memory_state(self.base),
                    cached_decision=self.cached_decision, nominal_latent=copy.deepcopy(self.nominal_latent),
                    cached_observation=copy.deepcopy(self.cached_observation), correction_active=self.correction_active,
                    projector_counts=dict(self.env.experimental_projector.counts))

    def restore(self, state):
        self.env.restore(state['physical'])
        restore_memory(self.base, state['inherited_memory'])
        self.cached_decision = state['cached_decision']
        self.nominal_latent = copy.deepcopy(state['nominal_latent'])
        self.cached_observation = copy.deepcopy(state['cached_observation'])
        self.correction_active = state['correction_active']
        self.env.experimental_projector.counts = dict(state['projector_counts'])
