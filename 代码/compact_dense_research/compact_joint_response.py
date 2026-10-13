"""Explicit, memoryless output responses for a finite diagnostic search."""
import torch
from torch import nn
from compact_fine_policy import FineCompactPolicy


class PreviewResponse(nn.Module):
    def __init__(self, gain=1.):
        super().__init__()
        self.gain = float(gain)

    def forward(self, action):
        if self.gain == 1.:
            return action
        preview = ((6.4 + 5.6 * action[..., 2].double()) * self.gain - 6.4) / 5.6
        return torch.stack([action[..., 0], action[..., 1], preview.to(action.dtype)], dim=-1)


class JointResponsePolicy(FineCompactPolicy):
    def __init__(self, *, feed_gain=1., preview_gain=1., seed=922302):
        super().__init__(14, 3, seed, feed_gain=feed_gain)
        self.preview_gain = float(preview_gain)
        self.actor.add_module('preview_response', PreviewResponse(preview_gain))
