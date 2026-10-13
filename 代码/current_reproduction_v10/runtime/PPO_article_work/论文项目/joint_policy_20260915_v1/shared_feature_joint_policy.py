"""Learn joint output weights in a data-derived shared nonlinear feature subspace."""
import numpy as np
import torch
from torch import nn
from complete_joint_function_policy import CompleteJointFunctionPolicy
from nonlinear_feed_adapter import masked_observation
from reflection_policy import reflect_observation


class SharedFeatureJointPolicy(nn.Module):
    def __init__(self, parent, basis):
        super().__init__()
        self.base = CompleteJointFunctionPolicy(parent['joint_state'], parent['feed_state'])
        with torch.no_grad():
            self.base.coefficients.copy_(torch.as_tensor(parent['coefficients']))
        for name, value in basis.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(14))

    def hidden(self, observation):
        x = masked_observation(observation)
        p = self.base.joint
        a = p.joint_adapter[:4]((x-p.adapter_mean)/p.adapter_std)
        b = p.joint_adapter[:4]((reflect_observation(x)-p.adapter_mean)/p.adapter_std)
        return .5*(a-b), .5*(a+b)

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(observation, dtype=torch.float32)
        odd, even = self.hidden(x)
        f0 = (odd@self.odd_projection)/self.odd_scale
        f1 = ((even-self.even_mean)@self.even_projection)/self.even_scale
        f1 = torch.cat([f1, torch.ones_like(f1[..., :1])], dim=-1)
        c = self.coefficients
        delta = torch.stack([.01*(f0*c[:4]).sum(-1),
                             .05*(f1*c[4:9]).sum(-1), .05*(f1*c[9:14]).sum(-1)], -1)
        mean = self.base.act(observation, deterministic)[0]
        return mean+delta.numpy(), 0., 0.


def construct_basis(parent, observations):
    # Basis uses only state covariance, never angle/path identities or desired actions.
    empty = dict(odd_projection=np.zeros((64, 4)), even_projection=np.zeros((64, 4)),
                 even_mean=np.zeros(64), odd_scale=np.ones(4), even_scale=np.ones(4))
    policy = SharedFeatureJointPolicy(parent, empty)
    with torch.no_grad():
        odd, even = policy.hidden(torch.as_tensor(observations, dtype=torch.float32))
        odd, even = odd.numpy(), even.numpy()
    mean = even.mean(0)
    _, so, vo = np.linalg.svd(odd, full_matrices=False)
    _, se, ve = np.linalg.svd(even-mean, full_matrices=False)
    po, pe = vo[:4].T, ve[:4].T
    oscale = np.maximum(abs(odd@po).max(0), 1e-5)
    escale = np.maximum(abs((even-mean)@pe).max(0), 1e-5)
    return dict(odd_projection=po, even_projection=pe, even_mean=mean,
                odd_scale=oscale, even_scale=escale), dict(
                    odd_variance_fraction=float((so[:4]**2).sum()/(so**2).sum()),
                    even_variance_fraction=float((se[:4]**2).sum()/(se**2).sum()),
                    observable_states=len(observations), parameter_dimensions=14)
