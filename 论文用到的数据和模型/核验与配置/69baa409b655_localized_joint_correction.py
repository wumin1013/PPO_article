"""Three-coordinate local refinement of one14input memoryless policy."""
import torch
from torch import nn
from compact_fine_policy import FineCompactPolicy
from localized_preview_head import LocalizedActor,LocalizedPreviewPolicy


class LocalizedJointActor(LocalizedActor):
    def __init__(self,backbone,count=128):
        super().__init__(backbone,count)
        self.correction=nn.Linear(count,3,bias=False);nn.init.zeros_(self.correction.weight)
        self.register_buffer('output_scale',torch.tensor([.01,.08,.4]))

    def forward(self,x):
        return self.backbone(x)+torch.tanh(self.correction(self.features(x)))*self.output_scale


class LocalizedJointPolicy(FineCompactPolicy):
    def __init__(self,seed=922302,count=128,base_count=96):
        super().__init__(seed=seed)
        previous=LocalizedPreviewPolicy(seed,count=base_count)
        self.actor=LocalizedJointActor(previous.actor,count)

    def initialize(self,base):
        self.actor.backbone.load_state_dict(base.actor.state_dict());self.critic.load_state_dict(base.critic.state_dict())
        with torch.no_grad():self.log_std.copy_(base.log_std)


def physics_worker(connection,dt,args,stage,deadline):
    import train_mixed_resolution_ppo as runtime
    previous=runtime.policy_from
    def factory(saved,seed):
        if saved.get('policy_class')=='LocalizedJointPolicy':
            policy=LocalizedJointPolicy(seed,count=saved['joint_basis_count'],base_count=saved['base_local_basis_count'])
        elif saved.get('policy_class')=='LocalizedPreviewPolicy':
            policy=LocalizedPreviewPolicy(seed,count=saved['local_basis_count'],left_support=saved.get('left_turn_support',False),positive_only=saved.get('nonnegative_preview_correction',False))
        else:return previous(saved,seed)
        policy.load_state_dict(saved['policy']);return policy
    runtime.policy_from=factory
    runtime.physics_worker(connection,dt,args,stage,deadline)
