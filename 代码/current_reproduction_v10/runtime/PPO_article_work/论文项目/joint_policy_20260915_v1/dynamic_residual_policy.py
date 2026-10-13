"""Learned state-linear action residual with exact reflection symmetry.

All residual weights start at zero and are learned from complete episodes.
Inputs are existing normalized lateral/longitudinal error, velocity,
acceleration, and filtered heading components (observation 0:8). There are
no gate-triggered overrides, angle tables, speed caps or trajectory templates.
"""
import torch
from reflection_policy import ReflectionPolicy,reflect_observation


class DynamicResidualPolicy(ReflectionPolicy):
    def __init__(self,observation_dim,**kwargs):
        super().__init__(observation_dim,**kwargs)
        self.dynamic_head=torch.nn.Linear(8,3)
        torch.nn.init.zeros_(self.dynamic_head.weight)
        torch.nn.init.zeros_(self.dynamic_head.bias)

    def distribution(self,observation):
        base=super().distribution(observation)
        sign=observation.new_tensor([-1.,1.,1.])
        delta=.5*(self.dynamic_head(observation[...,:8])+
                   sign*self.dynamic_head(reflect_observation(observation)[...,:8]))
        return torch.distributions.Normal(base.mean+delta,base.stddev)


class PreviewResidualPolicy(DynamicResidualPolicy):
    """Learned affine correction from all ten existing preview point vectors."""
    def __init__(self,observation_dim,**kwargs):
        super().__init__(observation_dim,**kwargs)
        self.preview_head=torch.nn.Linear(20,3)
        torch.nn.init.zeros_(self.preview_head.weight)
        torch.nn.init.zeros_(self.preview_head.bias)

    def distribution(self,observation):
        base=super().distribution(observation)
        sign=observation.new_tensor([-1.,1.,1.])
        delta=.5*(self.preview_head(observation[...,22:42])+
                   sign*self.preview_head(reflect_observation(observation)[...,22:42]))
        return torch.distributions.Normal(base.mean+delta,base.stddev)
