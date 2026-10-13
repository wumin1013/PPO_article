"""Two-output ablation with the corresponding static local feature capacity."""
import torch
from torch import nn
from direct_two_action_control import DirectTwoPolicy
from localized_preview_head import LocalizedActor


class LocalizedTwoActor(LocalizedActor):
    def __init__(self,backbone,count=192):
        super().__init__(backbone,count)
        self.correction=nn.Linear(count,2,bias=False)
        nn.init.zeros_(self.correction.weight)
        self.register_buffer('output_scale',torch.tensor([.01,.08]))
        # Give the ablation the opportunity to improve its entire own actor.
        self.backbone.requires_grad_(True)

    def forward(self,x):
        return self.backbone(x)+torch.tanh(self.correction(self.features(x)))*self.output_scale


class LocalizedTwoPolicy(DirectTwoPolicy):
    def __init__(self,seed=922401,count=192):
        super().__init__(seed)
        self.actor=LocalizedTwoActor(self.actor,count)

    def initialize(self,base):
        self.actor.backbone.load_state_dict(base.actor.state_dict())
        self.critic.load_state_dict(base.critic.state_dict())
        with torch.no_grad():self.log_std.copy_(base.log_std)


def policy_from(saved,seed):
    if saved.get('policy_class')=='LocalizedTwoPolicy':
        policy=LocalizedTwoPolicy(seed,count=saved['local_basis_count'])
        policy.load_state_dict(saved['policy'])
        return policy
    from research_policy_loader import from_saved
    return from_saved(saved,seed)
