"""Learned joint residual on observable decision innovations, with unchanged ZOH."""
import math
import numpy as np
from deep_joint_credit_policy import load_credit_policy
from probe_joint_target_coordinates import guide_angle


SCALES = np.array([.1, .2, .2])


def innovation_features(env, latent):
    requested = env.decode_action(latent, env.fixed_d)
    old_beta = guide_angle(env, env.held)
    new_beta = guide_angle(env, requested)
    target = new_beta + requested[0]
    jump = math.sin(target - old_beta - env.held[0])
    gap = math.sin(target - env.heading)
    speed = np.linalg.norm(env.velocity)
    lag = (math.cos(env.heading)*env.velocity[1]
           - math.sin(env.heading)*env.velocity[0]) / max(speed, 5.)
    # Odd steering features and even feed/preview features preserve reflection.
    x = np.tanh(np.array([jump, gap, lag])/.5)
    features = np.stack([x**3, x**2, x**2], axis=1)*SCALES
    return features, np.array([jump, gap, lag])


class DecisionInnovationPolicy:
    def __init__(self, parent, coefficients):
        self.base = load_credit_policy(parent)
        self.coefficients = np.asarray(coefficients, dtype=float).reshape(3, 3)
        self.env = None
        self.last_features = None
        self.last_inputs = None

    def bind(self, env):
        self.env = env
        return env

    def act(self, observation, deterministic=True):
        original = self.base.act(observation, deterministic)[0]
        features, inputs = innovation_features(self.env, original)
        self.last_features, self.last_inputs = features, inputs
        # Exactly zero preserves the parent's dtype and arithmetic bitwise.
        if not np.any(self.coefficients):
            return original, 0., 0.
        residual = (features*self.coefficients).sum(axis=0)
        return (original+residual).astype(original.dtype), 0., 0.


def metadata(parent, coefficients):
    return dict(architecture='decision_innovation_joint_residual', parent=parent,
                coefficients=np.asarray(coefficients).reshape(3, 3),
                feature_scales=SCALES.copy(),
                learning_method='complete_closed_loop_joint_parameter_optimization',
                action_interval_ms=20, physical_tick_ms=1,
                temporal_interpolation=False, accepted=False)
