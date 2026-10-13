"""Fourteen-input actor with explicit optional feed activation and finer exploration."""
import torch
from torch import nn
from compact_control import CompactPolicy


class FeedResponse(nn.Module):
    """Fixed monotone output activation; no new input, parameter, or hidden state."""
    def __init__(self,gain=1.):
        super().__init__()
        self.gain=float(gain)

    def forward(self,action):
        feed=action[...,1].double()
        feed=(feed+(self.gain-1.)*torch.relu(feed-.6)).to(action.dtype)
        return torch.stack([action[...,0],feed,action[...,2]],dim=-1)


class FineCompactPolicy(CompactPolicy):
    def __init__(self,dimension=14,action_dim=3,seed=922302,*,feed_gain=1.):
        if (dimension,action_dim)!=(14,3):
            raise ValueError('This experiment uses the same14inputs and3outputs')
        super().__init__(dimension,action_dim,seed)
        self.feed_gain=float(feed_gain)
        self.actor.add_module('feed_response',FeedResponse(feed_gain))

    def distribution(self,x):
        return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-8.,-.3).exp())
