"""PPO adapter around an own-seed v4 fit, without changing its actor formula."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from strong_policy_v4 import StrongPolicyV4, STATIC_KEYS, load_strong_policy, tensor_hash


class StrongPPOV4(nn.Module):
    def __init__(self, fitted):
        super().__init__()
        self.seed, self.arm = fitted.seed, fitted.arm
        self.action_dim, self.fixed_preview_mm = fitted.action_dim, fitted.fixed_preview_mm
        self.actor = fitted.actor
        self.actor.requires_grad_(True)
        with torch.random.fork_rng():
            torch.manual_seed(self.seed+100000)
            self.critic = nn.Sequential(nn.Linear(14, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.log(torch.tensor([.002, .01, .005])))
        self.action_generator = torch.Generator(device='cpu').manual_seed(self.seed+200000)

    def dist(self, x):
        mean = self.actor(x)[..., :self.action_dim]
        return torch.distributions.Normal(mean, self.log_std[:self.action_dim].clamp(-8., -2.).exp())

    def value(self, x):
        return self.critic(x).squeeze(-1)

    @torch.no_grad()
    def mean(self, observations):
        return self.actor(torch.as_tensor(observations, dtype=torch.float32))[..., :self.action_dim]

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        x = torch.as_tensor(observations, dtype=torch.float32)
        dist = self.dist(x)
        if deterministic:
            action = dist.mean
        else:
            # Always draw three coordinates so paired arms share their first two.
            noise = torch.randn((*dist.mean.shape[:-1], 3), generator=self.action_generator)
            action = dist.mean+dist.scale*noise[..., :self.action_dim]
        return action.numpy(), float(dist.log_prob(action).sum(-1)), float(self.value(x))


def from_fit(path):
    fitted, metadata = load_strong_policy(path)
    return StrongPPOV4(fitted), metadata


def load_strong_ppo(path):
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    if saved.get('format') != 'strong_ppo_v4':
        raise ValueError('Unrecognized strong PPO checkpoint')
    assets = dict(tensors={key:saved['policy']['actor.'+key] for key in STATIC_KEYS})
    base = StrongPolicyV4(saved['seed'], assets, saved['arm'], saved['fixed_preview_mm'])
    policy = StrongPPOV4(base)
    policy.load_state_dict(saved['policy'], strict=True)
    policy.action_generator.set_state(saved['action_rng_state'])
    if tensor_hash(dict(policy.named_parameters())) != saved['trainable_tensor_sha256']:
        raise ValueError('Strong PPO tensor content hash mismatch')
    policy.eval()
    return policy, saved


def interleaved_gae(records, gamma=.99, lam=.95):
    """Trace only within each program/episode; bootstrap every truncated tail."""
    advantages = np.empty(len(records), dtype=np.float32)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record['path'], record['episode']), []).append(index)
    for indices in groups.values():
        carry = 0.
        for index in reversed(indices):
            record = records[index]
            live = 1.-float(record['done'])
            delta = record['reward']+gamma*live*record['next_value']-record['value']
            carry = delta+gamma*lam*live*carry
            advantages[index] = carry
    returns = advantages+np.asarray([r['value'] for r in records], dtype=np.float32)
    return advantages, returns
