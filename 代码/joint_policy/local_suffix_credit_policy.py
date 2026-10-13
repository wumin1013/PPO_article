"""Local smooth state functions learned from complete-consequence action credit."""
import numpy as np
import torch
from torch import nn
from raw_state_joint_policy import LOCAL_INDICES
from reflection_policy import reflect_observation


class LocalSuffixCreditPolicy(nn.Module):
    def __init__(self, parent, mean, std, centers, radius, precision='float32'):
        super().__init__()
        from deep_joint_credit_policy import load_credit_policy
        self.base = load_credit_policy(parent)
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        dtype = torch.float64 if precision == 'float64' else torch.float32
        for name, value in dict(mean=mean, std=std, centers=centers, radius=radius).items():
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        self.register_buffer('output_scale', torch.tensor([.03, .2, .2], dtype=dtype))
        self.register_buffer('step_scale', torch.tensor(1., dtype=dtype))
        self.coefficients = nn.Parameter(torch.zeros(len(centers), 3, dtype=dtype))

    def features(self, observation):
        def kernel(x):
            x = (x[..., LOCAL_INDICES]-self.mean)/self.std
            r = torch.sqrt((x.unsqueeze(-2)-self.centers).square().mean(-1))/self.radius
            return torch.clamp(1-r, min=0.).pow(4)*(1+4*r)
        a, b = kernel(observation), kernel(reflect_observation(observation))
        return torch.stack([.5*(a-b), .5*(a+b), .5*(a+b)], -1)

    def correction(self, observation):
        return (self.features(observation)*self.coefficients).sum(-2)*self.output_scale*self.step_scale

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(observation, dtype=torch.float32)
        original = self.base.act(observation, deterministic)[0]
        return (original+self.correction(x).numpy()).astype(original.dtype), 0., 0.


def local_basis(observations, mean, std):
    x = (observations[:, LOCAL_INDICES]-mean)/std
    y = (reflect_observation(torch.as_tensor(observations)).numpy()[:, LOCAL_INDICES]-mean)/std
    distance = np.sqrt(np.minimum(((x[:, None]-x[None, :])**2).mean(-1),
                                 ((x[:, None]-y[None, :])**2).mean(-1)))
    # Radius comes from state-space neighbors only; no path, angle or time index.
    distance[distance < .001] = np.inf
    nearest = np.sort(distance, axis=1)[:, 3]
    radius = np.maximum(1.5*nearest, .08)
    assert np.isfinite(radius).all()
    return x, radius
