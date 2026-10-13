"""Three learned residual outputs sharing observable nonlinear state features."""
import torch
from torch import nn
from nonlinear_feed_adapter import NonlinearFeedAdapter, masked_observation
from reflection_policy import reflect_observation


class NonlinearJointAdapter(NonlinearFeedAdapter):
    def __init__(self, centers=256):
        super().__init__(centers)
        self.joint_adapter = nn.Sequential(nn.Linear(72, 64), nn.Tanh(),
                                           nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
        self.register_buffer('joint_scale', torch.tensor([.015, .06, .06]))
        nn.init.zeros_(self.joint_adapter[-1].weight)
        nn.init.zeros_(self.joint_adapter[-1].bias)

    def correction(self, observation):
        x = masked_observation(observation)
        a = (x-self.adapter_mean)/self.adapter_std
        b = (reflect_observation(x)-self.adapter_mean)/self.adapter_std
        sign = x.new_tensor([-1., 1., 1.])
        return .5*(self.joint_adapter(a)+sign*self.joint_adapter(b))*self.joint_scale

    def distribution(self, observation):
        base = super().distribution(observation)
        return torch.distributions.Normal(base.mean+self.correction(observation), base.scale)

    def initialize_from_feed(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith('joint_') for k in missing)
        self.joint_adapter[0].load_state_dict(self.feed_adapter[0].state_dict())
        self.joint_adapter[2].load_state_dict(self.feed_adapter[2].state_dict())
