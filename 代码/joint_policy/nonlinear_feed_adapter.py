"""Learned nonlinear feed residual; all inputs are existing observable state."""
import torch
from torch import nn
from canonical_gate_state import CanonicalGatePolicy
from reflection_policy import reflect_observation


def masked_observation(observation):
    x=observation.clone()
    x[...,8:11]=x.new_tensor([0.,.3,.25]);x[...,13:15]=0.
    return x


class NonlinearFeedAdapter(CanonicalGatePolicy):
    def __init__(self, centers=256):
        super().__init__(centers)
        self.register_buffer('adapter_mean',torch.zeros(72))
        self.register_buffer('adapter_std',torch.ones(72))
        self.feed_adapter=nn.Sequential(nn.Linear(72,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.feed_adapter[-1].weight);nn.init.zeros_(self.feed_adapter[-1].bias)

    def feed_correction(self,observation):
        x=masked_observation(observation)
        a=(x-self.adapter_mean)/self.adapter_std
        b=(reflect_observation(x)-self.adapter_mean)/self.adapter_std
        return .5*(self.feed_adapter(a)+self.feed_adapter(b)).squeeze(-1)

    def distribution(self,observation):
        base=super().distribution(observation)
        mean=base.mean.clone();mean[...,1]+=self.feed_correction(observation)
        return torch.distributions.Normal(mean,base.scale)
