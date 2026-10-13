"""Joint radial residual on continuous local physical/geometric observations."""
import numpy as np
import torch
from torch import nn
from shared_feature_joint_policy import SharedFeatureJointPolicy
from reflection_policy import reflect_observation


LOCAL_INDICES = list(range(8))+[11, 12, 15]+list(range(22, 72))


class RawStateJointPolicy(nn.Module):
    def __init__(self, parent, basis):
        super().__init__()
        spec = parent['policy_spec']
        self.base = SharedFeatureJointPolicy(spec['parent'], spec['basis'])
        with torch.no_grad():
            self.base.coefficients.copy_(torch.as_tensor(parent['coefficients']))
        for name, value in basis.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(3*len(self.centers)))

    def function_features(self, observation):
        a = (observation[..., LOCAL_INDICES]-self.mean)/self.std
        b = (reflect_observation(observation)[..., LOCAL_INDICES]-self.mean)/self.std
        def kernel(x):
            distance = (x.unsqueeze(-2)-self.centers).square().mean(-1)
            return torch.exp(-distance/(2*self.width.square()))
        ka, kb = kernel(a), kernel(b)
        odd, even = .5*(ka-kb), .5*(ka+kb)
        size = len(self.centers)
        shape = observation.shape[:-1]+(3, 3*size)
        result = observation.new_zeros(shape)
        result[..., 0, :size] = .01*odd
        result[..., 1, size:2*size] = .05*even
        result[..., 2, 2*size:] = .05*even
        return result

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(observation, dtype=torch.float32)
        delta = (self.function_features(x)*self.coefficients).sum(-1)
        return self.base.act(observation, deterministic)[0]+delta.numpy(), 0., 0.


def construct_raw_basis(observations, count=16):
    obs = torch.as_tensor(observations, dtype=torch.float32)
    mirror = reflect_observation(obs).numpy()
    x, y = np.asarray(observations)[:, LOCAL_INDICES], mirror[:, LOCAL_INDICES]
    both = np.vstack([x, y])
    mean = both.mean(0)
    std = np.maximum(both.std(0), .05)
    x, y = (x-mean)/std, (y-mean)/std
    # Deterministic coverage of state space modulo reflection, not of path/angle IDs.
    selected = [int(np.argmax(np.mean(x*x, axis=1)))]
    distance = np.full(len(x), np.inf)
    for index in range(count):
        center = selected[-1]
        distance = np.minimum(distance, np.minimum(np.mean((x-x[center])**2, axis=1),
                                                  np.mean((y-x[center])**2, axis=1)))
        if index+1 < count:
            selected.append(int(np.argmax(distance)))
    centers = x[selected]
    pair = ((centers[:, None]-centers[None, :])**2).mean(-1)
    pair[np.eye(count, dtype=bool)] = np.inf
    width = max(float(np.sqrt(np.median(pair.min(1)))*.6), .1)
    return dict(centers=centers, mean=mean, std=std, width=np.array(width)), dict(
        states=len(x), centers=count, dimensions=len(LOCAL_INDICES), parameters=3*count,
        width=width, maximum_nearest_center_distance=float(np.sqrt(distance.max())),
        excluded_inputs='Previous actions, endpoint global vector, global progress, gate ordinal, decision time, brake/rest counters.',
        learned_outputs='Steering, feed, preview; no path ID, angle table, online switching or trajectory template.')
