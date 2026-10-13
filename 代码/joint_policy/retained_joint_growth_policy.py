"""Jointly learn state-dependent growth from the verified retained policy."""
import numpy as np
import torch
from torch import nn
from canonical_gate_state import CanonicalGatePolicy
from nonlinear_feed_adapter import NonlinearFeedAdapter
from raw_state_joint_policy import LOCAL_INDICES
from reflection_policy import reflect_observation


class RetainedJointGrowthPolicy(nn.Module):
    def __init__(self, retained, feed, basis, transform):
        super().__init__()
        self.base = CanonicalGatePolicy(256)
        self.base.load_state_dict(retained)
        self.feed_source = NonlinearFeedAdapter()
        self.feed_source.load_state_dict(feed)
        for name, value in basis.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.register_buffer('transform', torch.as_tensor(transform, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(self.transform.shape[1]))

    def raw_features(self, observation):
        def kernel(x):
            x = (x[..., LOCAL_INDICES]-self.mean)/self.std
            return torch.exp(-(x.unsqueeze(-2)-self.centers).square().mean(-1)/(2*self.width.square()))
        a, b = kernel(observation), kernel(reflect_observation(observation))
        odd, even = .5*(a-b), .5*(a+b)
        n = len(self.centers)
        result = observation.new_zeros(observation.shape[:-1]+(3, 3*n+1))
        result[..., 0, :n] = odd
        # The previously fitted nonlinear function is only a feature. Its
        # state-dependent amplitude, steering and preview weights start at zero
        # and learn solely from new complete closed-loop consequences.
        feed = self.feed_source.feed_correction(observation)
        result[..., 1, n:2*n] = even*feed.unsqueeze(-1)
        result[..., 1, -1] = feed
        result[..., 2, 2*n:3*n] = even
        return result

    def function_features(self, observation):
        return self.raw_features(observation)@self.transform

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(observation, dtype=torch.float32)
        delta = (self.function_features(x)*self.coefficients).sum(-1)
        return self.base.act(observation, deterministic)[0]+delta.numpy(), 0., 0.


def growth_basis_transform(retained, feed, basis, observations):
    count = len(basis['centers'])
    provisional = RetainedJointGrowthPolicy(retained, feed, basis, np.zeros((3*count+1, 1)))
    with torch.no_grad():
        f = provisional.raw_features(torch.as_tensor(observations, dtype=torch.float32)).numpy().astype(float)
    transform, audit = [], []
    for head, limit in enumerate((.02, .75, .10)):
        indices = np.flatnonzero(abs(f[:, head]).max(0) > 1e-14)
        x = f[:, head, indices]
        _, s, v = np.linalg.svd(x, full_matrices=False)
        rank = min(6, int((s > s[0]*1e-3).sum()))
        rotation = v[:rank].T/s[:rank]
        rotation *= limit/np.maximum(abs(x@rotation).max(0), 1e-12)
        column = np.zeros((f.shape[-1], rank))
        column[indices] = rotation
        transform.append(column)
        audit.append(dict(head=head, retained_rank=rank, singular_values=s.tolist(),
                          maximum_latent_feature=limit))
    return np.column_stack(transform), dict(heads=audit, observation_states=len(observations),
        source='Retained-policy observable states only; no desired action/speed labels.',
        use='Nonlinear feed function is a frozen feature, not an adopted policy.')
