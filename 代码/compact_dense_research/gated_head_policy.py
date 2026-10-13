"""One14input policy with frozen steering and a learned feed/preview correction."""
import torch
from torch import nn
from compact_fine_policy import FineCompactPolicy


def observable_gate(x):
    bend=torch.hypot(1.-x[...,5],x[...,13])
    unit=((torch.maximum(bend,x[...,6].abs())-.005)/.145).clamp(0.,1.)
    return unit.square()*(3.-2.*unit)


class GatedActor(nn.Module):
    def __init__(self,backbone,preview_only=False):
        super().__init__();self.backbone=backbone
        self.backbone.requires_grad_(False)
        self.preview_only=bool(preview_only)
        self.correction=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1 if preview_only else 2))
        nn.init.zeros_(self.correction[2].weight);nn.init.zeros_(self.correction[2].bias)
        self.register_buffer('output_scale',torch.tensor([.4] if preview_only else [.08,.4]))

    def corrected(self,features,base,x):
        delta=torch.tanh(self.correction(features))*self.output_scale*observable_gate(x).unsqueeze(-1)
        return torch.cat([base[...,1:2],base[...,2:3]+delta],dim=-1) if self.preview_only else base[...,1:]+delta

    def forward(self,x):
        features=self.backbone[:4](x);base=self.backbone[4:](features)
        return torch.cat([base[...,0:1],self.corrected(features,base,x)],dim=-1)


class GatedHeadPolicy(FineCompactPolicy):
    def __init__(self,seed=922302,preview_only=False):
        super().__init__(seed=seed,feed_gain=1.)
        self.actor=GatedActor(self.actor,preview_only)

    def initialize(self,base):
        self.actor.backbone.load_state_dict(base.actor.state_dict())
        self.critic.load_state_dict(base.critic.state_dict())
        with torch.no_grad():self.log_std.copy_(base.log_std)
