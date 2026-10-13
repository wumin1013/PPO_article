"""Small learned coefficient space over nonlinear three-action functions."""
import torch
from torch import nn
from canonical_gate_state import CanonicalGatePolicy
from nonlinear_joint_adapter import NonlinearJointAdapter


class CompleteJointFunctionPolicy(nn.Module):
    def __init__(self, joint_state, feed_state):
        super().__init__()
        self.joint = NonlinearJointAdapter()
        self.feed = NonlinearJointAdapter()
        self.joint.load_state_dict(joint_state)
        self.feed.load_state_dict(feed_state)
        # learned coefficients: three joint heads, extra feed feature, original feed residual
        self.coefficients = nn.Parameter(torch.tensor([0., 0., 0., 2., 1.]))

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('Complete-function search uses deterministic rollouts')
        x = torch.as_tensor(observation, dtype=torch.float32)
        canonical = CanonicalGatePolicy.distribution(self.joint, x).mean
        c = self.coefficients
        mean = canonical.clone()
        mean[..., 1] += c[4]*self.joint.feed_correction(x)
        mean += self.joint.correction(x)*c[:3]
        mean[..., 1] += c[3]*self.feed.correction(x)[..., 1]
        return mean.numpy(), 0., 0.
