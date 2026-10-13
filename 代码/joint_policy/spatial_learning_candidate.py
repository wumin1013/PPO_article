"""Learning-only spatial compression and input-matched fresh initialization."""
import math
import numpy as np
import torch
from compact_observation_candidate import compact_observation, feature_names
from event_observation_candidate import LearningDesignRuntime
from joint_policy import JointPolicy

DISTANCES=(3.,6.,12.,24.)
REPRESENTATIONS=('original72','spatial22')


class SpatialLearningRuntime(LearningDesignRuntime):
    def __init__(self,*args,representation='original72',**kwargs):
        if representation not in REPRESENTATIONS:
            raise ValueError(representation)
        self.spatial=representation=='spatial22'
        super().__init__(*args,representation='original72',**kwargs)

    def observation(self):
        return compact_observation(self,DISTANCES) if self.spatial else super().observation()


def environment_factory(reference,*,fixed_d=None,max_time=None,representation='original72',reward_mode='original'):
    length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
    return SpatialLearningRuntime(reference,fixed_d=fixed_d,max_time=max(5.,length/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),
        representation=representation,reward_mode=reward_mode)


def fresh_policy(observation_dim,*,mode,seed):
    """No pretrained weight truncation; matching hidden-layer initialization streams."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        policy=JointPolicy(observation_dim,mode=mode)
        for head_index,head in enumerate((policy.actor,policy.critic)):
            for layer_index,layer in enumerate(head):
                if isinstance(layer,torch.nn.Linear):
                    torch.manual_seed(seed+1000*head_index+10*layer_index)
                    layer.reset_parameters()
        with torch.no_grad():
            policy.actor[-1].weight.zero_()
            policy.actor[-1].bias.copy_(torch.tensor([0.,math.atanh(-.4),math.atanh((3.-6.4)/5.6)]))
            policy.critic[-1].weight.zero_()
            policy.critic[-1].bias.zero_()
            policy.log_std.copy_(torch.log(torch.tensor([.06,.12,.20])))
    return policy


def spatial_feature_names():
    return feature_names(DISTANCES)
