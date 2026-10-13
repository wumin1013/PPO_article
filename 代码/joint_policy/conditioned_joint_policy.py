"""Reparameterize the existing joint state-function span by observed sensitivity."""
import numpy as np
import torch
from torch import nn
from raw_state_joint_policy import RawStateJointPolicy


class ConditionedJointPolicy(nn.Module):
    def __init__(self, parent, transform):
        super().__init__()
        spec = parent['policy_spec']
        self.base = RawStateJointPolicy(spec['parent'], spec['basis'])
        with torch.no_grad():
            self.base.coefficients.copy_(torch.as_tensor(parent['coefficients']))
        self.register_buffer('transform', torch.as_tensor(transform, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(self.transform.shape[1]))

    def function_features(self, observation):
        return self.base.function_features(observation)@self.transform

    @torch.no_grad()
    def act(self, observation, deterministic=True):
        x = torch.as_tensor(observation, dtype=torch.float32)
        delta = (self.function_features(x)*self.coefficients).sum(-1)
        return self.base.act(observation, deterministic)[0]+delta.numpy(), 0., 0.


def condition_basis(parent, observations, relative_cutoff=1e-3):
    spec = parent['policy_spec']
    raw = RawStateJointPolicy(spec['parent'], spec['basis'])
    with torch.no_grad():
        features = raw.function_features(torch.as_tensor(observations, dtype=torch.float32)).numpy().astype(float)
    columns, audit = [], []
    count = features.shape[-1]//3
    for head, scale in enumerate((.01, .05, .05)):
        indices = np.arange(head*count, (head+1)*count)
        x = features[:, head, indices]
        _, singular, vectors = np.linalg.svd(x, full_matrices=False)
        keep = singular > relative_cutoff*singular[0]
        rotation = vectors[keep].T/singular[keep]
        normalized = x@rotation
        rotation *= scale/np.maximum(abs(normalized).max(0), 1e-15)
        full = np.zeros((features.shape[-1], rotation.shape[1]))
        full[indices] = rotation
        columns.append(full)
        audit.append(dict(head=head, original_singular_values=singular.tolist(),
            retained_rank=int(keep.sum()), original_rank=count,
            maximum_feature_per_direction=scale,
            original_condition=float(singular[0]/max(singular[-1], 1e-30))))
    transform = np.column_stack(columns)
    return transform, dict(heads=audit, observations=len(observations),
        raw_dimensions=features.shape[-1], conditioned_dimensions=transform.shape[1],
        relative_singular_cutoff=relative_cutoff,
        scope='Same raw state-function span, with near-null directions omitted; no new action labels or deployed rules.')
