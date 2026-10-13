"""Fresh trainable actor on explicitly shared historical Wendland feature maps.

Only static buffers can be extracted from the historical checkpoint. Both arms
allocate the same network; fixed preview masks the third output at the interface.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn


STATIC_KEYS = {
    'preview_centers': 'actor.backbone.centers',
    'preview_scale': 'actor.backbone.feature_scale',
    'preview_radius': 'actor.backbone.radius',
    'joint_centers': 'actor.centers',
    'joint_scale': 'actor.feature_scale',
    'joint_radius': 'actor.radius',
    'output_scale': 'actor.output_scale',
}


def tensor_hash(tensors):
    """Stable content digest independent of torch serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def extract_static_assets(checkpoint):
    """Whitelist buffers: this function never returns trained network weights."""
    path = Path(checkpoint)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    weights = saved['policy']
    arrays = {key: weights[source].detach().cpu().clone()
              for key, source in STATIC_KEYS.items()}
    assert arrays['preview_centers'].shape == (96, 14)
    assert arrays['joint_centers'].shape == (192, 14)
    assert arrays['output_scale'].shape == (3,)
    assert all(torch.isfinite(v).all() for v in arrays.values())
    assert torch.all(arrays['preview_scale'] > 0)
    assert torch.all(arrays['joint_scale'] > 0)
    assert torch.all(arrays['preview_radius'] > 0)
    assert torch.all(arrays['joint_radius'] > 0)
    return dict(format='strong_v4_static_feature_assets', tensors=arrays,
                source_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_buffer_keys=STATIC_KEYS, tensor_sha256=tensor_hash(arrays),
                contains_historical_trainable_tensors=False)


def wendland_features(x, centers, scale, radius):
    distance = torch.sqrt((((x.unsqueeze(-2)-centers)/scale).square())
                          .sum(-1).clamp_min(1e-20))/radius
    return (1.-distance).clamp_min(0.).pow(4)*(4.*distance+1.)


class StrongActorV4(nn.Module):
    def __init__(self, seed, assets):
        super().__init__()
        arrays = assets['tensors']
        if set(arrays) != set(STATIC_KEYS):
            raise ValueError('Static asset keys differ from the explicit whitelist')
        for key, value in arrays.items():
            self.register_buffer(key, value.detach().clone().to(torch.float32))
        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            self.base = nn.Sequential(nn.Linear(14, 64), nn.Tanh(),
                                      nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
            self.preview_correction = nn.Linear(96, 1, bias=False)
            self.joint_correction = nn.Linear(192, 3, bias=False)
            nn.init.zeros_(self.preview_correction.weight)
            nn.init.zeros_(self.joint_correction.weight)

    def features(self, x):
        return (wendland_features(x, self.preview_centers, self.preview_scale,
                                  self.preview_radius),
                wendland_features(x, self.joint_centers, self.joint_scale,
                                  self.joint_radius))

    def from_features(self, x, preview, joint):
        raw = self.base(x)
        delta = .4*torch.tanh(self.preview_correction(preview))
        raw = torch.cat([raw[..., :2], raw[..., 2:3]+delta], dim=-1)
        return raw+torch.tanh(self.joint_correction(joint))*self.output_scale

    def forward(self, x):
        return self.from_features(x, *self.features(x))


class StrongPolicyV4(nn.Module):
    def __init__(self, seed, assets, arm='adaptive', fixed_preview_mm=3.):
        super().__init__()
        if arm not in ('adaptive', 'fixed'):
            raise ValueError('Expected adaptive or fixed arm')
        if float(fixed_preview_mm) != 3.:
            raise ValueError('Existing targets_two are registered for 3 mm only')
        self.seed = int(seed)
        self.arm = arm
        self.fixed_preview_mm = float(fixed_preview_mm)
        self.action_dim = 3 if arm == 'adaptive' else 2
        self.actor = StrongActorV4(seed, assets)

    def forward(self, observations):
        x = torch.as_tensor(observations, dtype=torch.float32)
        if x.shape[-1] != 14:
            raise ValueError('Expected fourteen physical/geometric observations')
        return self.actor(x)[..., :self.action_dim]

    @torch.no_grad()
    def mean(self, observations):
        return self(observations)

    @torch.no_grad()
    def act(self, observations, deterministic=True):
        if not deterministic:
            raise ValueError('This offline-only pilot has no registered exploration policy')
        return self.mean(observations).numpy(), None, None

    def decoded(self, observations):
        bounded = np.clip(self.mean(observations).numpy().astype(float),
                          -1.+1e-10, 1.-1e-10)
        feed = 50.*(1.+bounded[..., 1])
        preview = (6.4+5.6*bounded[..., 2] if self.arm == 'adaptive'
                   else np.full_like(feed, self.fixed_preview_mm))
        return np.stack([np.pi/2*bounded[..., 0], feed, preview], axis=-1)


def load_strong_policy(path):
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'strong_policy_v4':
        raise ValueError('Unrecognized strong-policy checkpoint format')
    assets = dict(tensors={key: saved['policy']['actor.'+key] for key in STATIC_KEYS})
    policy = StrongPolicyV4(saved['seed'], assets, saved['arm'], saved['fixed_preview_mm'])
    policy.load_state_dict(saved['policy'], strict=True)
    if tensor_hash(dict(policy.named_parameters())) != saved['trainable_tensor_sha256']:
        raise ValueError('Trainable tensor content hash mismatch')
    policy.eval()
    return policy, saved
