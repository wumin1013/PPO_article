"""Exact reflection-equivariant actor using the existing local-frame state.

Group averaging is an architecture constraint, not an angle-specific control
rule. Both three-action outputs still depend on learned network weights.
"""
import torch
from joint_policy import JointPolicy


NORMAL_FEATURES = [1, 3, 5, 7, 8, 12, 14]+list(range(23, 42, 2))+list(range(43, 62, 2))


def reflect_observation(observation):
    if observation.shape[-1] != 72:
        raise ValueError('Reflection map must match the registered full observation schema')
    mirrored = observation.clone()
    mirrored[..., NORMAL_FEATURES] *= -1
    return mirrored


class ReflectionPolicy(JointPolicy):
    def distribution(self, observation):
        if self.reference_actor is not None:
            raise ValueError('This bounded candidate supports joint/fixed only')
        sign = observation.new_tensor([-1., 1., 1.])
        mean = .5*(self.actor(observation)+sign*self.actor(reflect_observation(observation)))
        return torch.distributions.Normal(mean, torch.exp(self.log_std.clamp(-4., -.3)))
