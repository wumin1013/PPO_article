"""Small learned residual on frozen feedback features; no rollout schedule."""
import numpy as np
import torch
from torch import nn
from frame_feedback_policy import FrameFeedbackPolicy, heading_frame
from reflection_policy import reflect_observation


class LocalResidualPolicy(nn.Module):
    def __init__(self, base_state):
        super().__init__()
        self.base=FrameFeedbackPolicy(frame='heading',history_independent=True)
        self.base.load_state_dict(base_state)
        for p in self.base.parameters():p.requires_grad_(False)
        self.register_buffer('weights',torch.zeros(3,97))
        self.register_buffer('scale',torch.tensor([.02,.3,.3]))

    def features(self, observation):
        x=observation.clone();x[...,8:11]=x.new_tensor([0.,.3,.25])
        a=self.base.actor[:-1](heading_frame(x))
        b=self.base.actor[:-1](heading_frame(reflect_observation(x)))
        odd=.5*(a-b);even=.5*(a+b)
        odd=torch.cat([odd,torch.zeros_like(odd[...,:1])],dim=-1)
        even=torch.cat([even,torch.ones_like(even[...,:1])],dim=-1)
        return torch.stack([odd,even,even],dim=-2)

    def mean(self,observation):
        residual=(self.features(observation)*self.weights).sum(-1)*self.scale
        return self.base.distribution(observation).mean+residual

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        if not deterministic:raise ValueError('Local residual is a deterministic fitted actor')
        x=torch.as_tensor(observation,dtype=torch.float32)
        return self.mean(x).numpy(),0.,float(self.base.value(x))


def load_local_policy(saved):
    base={k[5:]:v for k,v in saved['policy'].items() if k.startswith('base.')}
    model=LocalResidualPolicy(base);model.load_state_dict(saved['policy']);return model
