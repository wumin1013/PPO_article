"""Information-preserving frame encoding for a learned three-action actor."""
import numpy as np
import torch
from reflection_policy import ReflectionPolicy,reflect_observation


PAIRS=[0,2,4,11,13]+list(range(22,62,2))
NORMAL_PAIRS=[index+1 for index in PAIRS]


def heading_frame(observation):
    """Rotate vector pairs only; retain all scalar information including angle."""
    c=observation[...,6:7];s=observation[...,7:8];result=observation.clone()
    x=observation[...,PAIRS];y=observation[...,NORMAL_PAIRS]
    result[...,PAIRS]=c*x+s*y
    result[...,NORMAL_PAIRS]=-s*x+c*y
    return result


class FrameFeedbackPolicy(ReflectionPolicy):
    def __init__(self,observation_dim=72,*,frame='reference',history_independent=False):
        if frame not in ('reference','heading'):raise ValueError(frame)
        super().__init__(observation_dim)
        self.frame=frame
        self.history_independent=history_independent
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor([0.,np.arctanh(.8),np.arctanh((4.-6.4)/5.6)],dtype=torch.float32))

    def distribution(self,observation):
        if self.history_independent:
            observation=observation.clone()
            observation[...,8:11]=observation.new_tensor([0.,.3,.25])
        encode=heading_frame if self.frame=='heading' else lambda x:x
        sign=observation.new_tensor([-1.,1.,1.])
        mean=.5*(self.actor(encode(observation))+sign*self.actor(encode(reflect_observation(observation))))
        return torch.distributions.Normal(mean,torch.exp(self.log_std.clamp(-4.,-.3)))
