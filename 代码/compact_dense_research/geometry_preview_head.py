"""Small experimental observable-geometry head; coefficients are explicit."""
import torch
from torch import nn
from compact_fine_policy import FineCompactPolicy
from gated_head_policy import observable_gate


def smooth_unit(x):
    u=x.clamp(0.,1.)
    return u.square()*(3.-2.*u)


class GeometryPreviewActor(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.backbone=backbone;self.backbone.requires_grad_(False)
        self.gains=nn.Parameter(torch.zeros(3))
        self.register_buffer('acute_mid_degrees',torch.tensor(132.))

    def features(self,x):
        # For an approaching isolated corner, the reference chord gives
        # tan(theta/2)=(1-far_tangent)/far_normal. This is a current-observation
        # feature, not a lookup of a case, corner ID, route coordinate, or time.
        left=smooth_unit(x[...,13]/.02)
        theta=2.*torch.atan2((1.-x[...,5]).clamp_min(0.),x[...,13].clamp_min(1e-8))*180./torch.pi
        acute=smooth_unit((theta-self.acute_mid_degrees+2.)/4.)
        after=smooth_unit(x[...,6]/.2)*(1.-left)
        return torch.stack([left,left*acute,after],dim=-1)*observable_gate(x).unsqueeze(-1)

    def forward(self,x):
        base=self.backbone(x);factor=1.+(self.features(x)*self.gains).sum(-1)
        preview=((6.4+5.6*base[...,2])*factor-6.4)/5.6
        # Keep unmodified coordinates exact; final shared action decoder clips.
        return torch.cat([base[...,:2],preview.unsqueeze(-1)],dim=-1)


class GeometryPreviewPolicy(FineCompactPolicy):
    def __init__(self,seed=922302):
        super().__init__(seed=seed);self.actor=GeometryPreviewActor(self.actor)

    def initialize(self,base):
        self.actor.backbone.load_state_dict(base.actor.state_dict())
        self.critic.load_state_dict(base.critic.state_dict())
        with torch.no_grad():self.log_std.copy_(base.log_std)


def physics_worker(connection,dt,args,stage,deadline):
    # Factory extension is process-local; the shared runner and executor files
    # remain unchanged while other registered training rounds are active.
    import train_mixed_resolution_ppo as runtime
    previous=runtime.policy_from
    def factory(saved,seed):
        if saved.get('policy_class')!='GeometryPreviewPolicy':return previous(saved,seed)
        policy=GeometryPreviewPolicy(seed);policy.load_state_dict(saved['policy']);return policy
    runtime.policy_from=factory
    runtime.physics_worker(connection,dt,args,stage,deadline)
