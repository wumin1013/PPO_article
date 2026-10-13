"""Static compact-support features over14current inputs; no temporal memory."""
import torch
from torch import nn
from compact_fine_policy import FineCompactPolicy


class LocalizedActor(nn.Module):
    def __init__(self,backbone,count=96,left_support=False,positive_only=False):
        super().__init__();self.backbone=backbone;self.backbone.requires_grad_(False)
        self.left_support=bool(left_support);self.positive_only=bool(positive_only)
        self.register_buffer('centers',torch.zeros(count,14))
        self.register_buffer('feature_scale',torch.ones(14))
        self.register_buffer('radius',torch.tensor(1.))
        self.correction=nn.Linear(count,1,bias=False);nn.init.zeros_(self.correction.weight)

    def features(self,x):
        distance=torch.sqrt((((x.unsqueeze(-2)-self.centers)/self.feature_scale).square()).sum(-1).clamp_min(1e-20))/self.radius
        q=(1.-distance).clamp_min(0.)
        result=q.pow(4)*(4.*distance+1.)
        if self.left_support:
            def smooth(v):
                u=v.clamp(0.,1.);return u.square()*(3.-2.*u)
            ahead=smooth(x[...,13].abs()/.001)
            left=smooth(x[...,13]/.001)+(1.-ahead)*smooth(x[...,6]/.1)
            result=result*left.unsqueeze(-1)
        return result

    def forward(self,x):
        base=self.backbone(x)
        delta=.4*torch.tanh(self.correction(self.features(x)))
        if self.positive_only:delta=delta.clamp_min(0.)
        return torch.cat([base[...,:2],base[...,2:3]+delta],dim=-1)


class LocalizedPreviewPolicy(FineCompactPolicy):
    def __init__(self,seed=922302,count=96,left_support=False,positive_only=False):
        super().__init__(seed=seed);self.actor=LocalizedActor(self.actor,count,left_support,positive_only)

    def initialize(self,base):
        self.actor.backbone.load_state_dict(base.actor.state_dict());self.critic.load_state_dict(base.critic.state_dict())
        with torch.no_grad():self.log_std.copy_(base.log_std)


def physics_worker(connection,dt,args,stage,deadline):
    import train_mixed_resolution_ppo as runtime
    previous=runtime.policy_from
    def factory(saved,seed):
        if saved.get('policy_class')!='LocalizedPreviewPolicy':return previous(saved,seed)
        policy=LocalizedPreviewPolicy(seed,count=saved['local_basis_count'],left_support=saved.get('left_turn_support',False),positive_only=saved.get('nonnegative_preview_correction',False))
        policy.load_state_dict(saved['policy']);return policy
    runtime.policy_from=factory
    runtime.physics_worker(connection,dt,args,stage,deadline)
