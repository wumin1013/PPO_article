"""Exclude clustered discrete turns from the historical distributed-arc residual.

This rule only gates the supplementary arc feed residual, not the learned feed
or the physical velocity limit. It is shared geometry logic, without case IDs.
"""
import numpy as np
from arc_cruise_policy import ArcCruiseResidualPolicy, arc_feature


class LocalArcSupportPolicy:
    def __init__(self, original):
        self.base = original.base
        self.arc_amplitude = original.arc_amplitude
        self.arc_scale = original.arc_scale
        self.suppressed_calls = 0
        self.total_calls = 0

    def bind(self, env):
        self.env = env
        t = env.corridor.tangents
        change = np.arctan2(t[:-1, 0]*t[1:, 1]-t[:-1, 1]*t[1:, 0], (t[:-1]*t[1:]).sum(1))
        self.sharp_s = env.corridor.arcs[1:-1][abs(change) > np.deg2rad(10.)]
        self.suppressed_calls = self.total_calls = 0
        return self.base.bind(env)

    def act(self, observation, deterministic=True):
        latent = self.base.act(observation, deterministic)[0]
        s = self.env.progress()[0]
        clustered = np.count_nonzero(abs(self.sharp_s-s) <= 30.) >= 2
        self.total_calls += 1
        if clustered:
            self.suppressed_calls += 1
        elif self.arc_amplitude:
            latent = latent.copy()
            latent[1] += self.arc_amplitude*arc_feature(observation, self.arc_scale)
        return latent, 0., 0.


def install_local_arc_support(policy):
    parent = None
    node = policy
    while node is not None:
        if isinstance(node, ArcCruiseResidualPolicy):
            replacement = LocalArcSupportPolicy(node)
            if parent is None:
                return replacement, replacement
            parent.base = replacement
            return policy, replacement
        parent, node = node, getattr(node, 'base', None)
    raise ValueError('Expected one inherited arc residual layer')


class ClusteredPreviewFloorPolicy:
    """Diagnostic lower bound on preview only in clustered-turn neighborhoods."""
    def __init__(self, base, floor_mm):
        self.base = base
        self.floor_mm = float(floor_mm)
        self.changed_calls = 0

    def bind(self, env):
        self.env = env
        t = env.corridor.tangents
        turn = np.arctan2(t[:-1, 0]*t[1:, 1]-t[:-1, 1]*t[1:, 0], (t[:-1]*t[1:]).sum(1))
        self.sharp_s = env.corridor.arcs[1:-1][abs(turn) > np.deg2rad(10.)]
        self.changed_calls = 0
        return self.base.bind(env)

    def act(self, observation, deterministic=True):
        latent = self.base.act(observation, deterministic)[0]
        clustered = np.count_nonzero(abs(self.sharp_s-self.env.progress()[0]) <= 30.) >= 2
        if clustered and self.env.fixed_d is None:
            preview = self.env.decode_action(latent, None)[2]
            if preview < self.floor_mm:
                latent = latent.copy()
                latent[2] = np.arctanh((self.floor_mm-6.4)/5.6)
                self.changed_calls += 1
        return latent, 0., 0.


class ClusteredCornerFeedPolicy:
    """Geometric local-feed prior; physical feasibility is audited separately."""
    def __init__(self, base, reserve):
        self.base, self.reserve = base, float(reserve)
        self.changed_calls = 0

    def bind(self, env):
        self.env = env
        t = env.corridor.tangents
        angle = abs(np.arctan2(t[:-1, 0]*t[1:, 1]-t[:-1, 1]*t[1:, 0], (t[:-1]*t[1:]).sum(1)))
        mask = angle > np.deg2rad(10.)
        self.sharp_s = env.corridor.arcs[1:-1][mask]
        half = angle[mask]/2.
        # Circular fillet estimates use the same one-sided Euclidean tolerance;
        # spacing limits avoid overlapping nominal tangency intervals.
        spacing = np.minimum(env.corridor.lengths[:-1], env.corridor.lengths[1:])[mask]
        radius = np.minimum(1.25/(1.-np.cos(half)), .45*spacing/np.tan(half))
        self.tangency = radius*np.tan(half)
        self.corner_speed = self.reserve*np.minimum(np.sqrt(2000.*radius), np.cbrt(62500.*radius**2))
        self.changed_calls = 0
        return self.base.bind(env)

    def act(self, observation, deterministic=True):
        latent = self.base.act(observation, deterministic)[0]
        distance = abs(self.sharp_s-self.env.progress()[0])
        if np.count_nonzero(distance <= 30.) >= 2:
            room = np.maximum(distance-self.tangency-2., 0.)
            limit = min(100., float(np.sqrt(self.corner_speed**2+2.*1000.*room).min()))
            feed = self.env.decode_action(latent, self.env.fixed_d)[1]
            if feed > limit:
                latent = latent.copy()
                latent[1] = np.arctanh(np.clip(limit/50.-1., -.999999999, .999999999))
                self.changed_calls += 1
        return latent, 0., 0.


class ClusteredJointFeedbackPolicy:
    """Shared three-action error correction, selected by explicit parameter search."""
    def __init__(self, base, coefficients):
        self.base = base
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (3,):
            raise ValueError('Three joint feedback coefficients required')
        self.changed_calls = 0

    def bind(self, env):
        self.env = env
        t = env.corridor.tangents
        turn = np.arctan2(t[:-1, 0]*t[1:, 1]-t[:-1, 1]*t[1:, 0], (t[:-1]*t[1:]).sum(1))
        self.sharp_s = env.corridor.arcs[1:-1][abs(turn) > np.deg2rad(10.)]
        self.changed_calls = 0
        return self.base.bind(env)

    def act(self, observation, deterministic=True):
        from turn_feedback_policy import turn_feedback_features
        latent = self.base.act(observation, deterministic)[0]
        if np.count_nonzero(abs(self.sharp_s-self.env.progress()[0]) <= 30.) >= 2:
            correction = self.coefficients*turn_feedback_features(observation)
            latent = latent.astype(float)+correction
            self.changed_calls += int(np.any(correction))
        return latent, 0., 0.
