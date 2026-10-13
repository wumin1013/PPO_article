"""Direct learned access to existing canonical state, with the same reflection symmetry."""
import torch
from canonical_gate_state import CanonicalGatePolicy
from reflection_policy import reflect_observation,NORMAL_FEATURES

RAW_STEERING_INDICES=tuple(i for i in NORMAL_FEATURES if i not in (8,14))


def raw_state_features(observation):
    x=observation.clone()
    x[...,8:11]=x.new_tensor([0.,.3,.25])
    x[...,13:15]=0.
    mirrored=reflect_observation(x)
    return torch.stack([.5*(x-mirrored),.5*(x+mirrored),.5*(x+mirrored)],-2)


class RawStateSkipPolicy(CanonicalGatePolicy):
    def __init__(self,centers=256):
        super().__init__(centers)
        self.raw_skip_coefficients=torch.nn.Parameter(torch.zeros(3,72))

    def distribution(self,observation):
        base=super().distribution(observation)
        residual=(raw_state_features(observation)*self.raw_skip_coefficients).sum(-1)
        return torch.distributions.Normal(base.mean+residual*self.residual_scale,base.scale)
