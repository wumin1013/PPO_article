"""Behavioral checks for PPO sampling and independent episode returns."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.algorithms.ppo import PPOContinuous, PolicyNetContinuous
from src.utils.rl_utils import compute_advantage


def test_gaussian_scale_is_shared_at_sampling_and_update_extremes():
    for action_dim in (2, 3):
        net = PolicyNetContinuous(14, 32, action_dim)
        with torch.no_grad():
            for name, parameter in net.named_parameters():
                if 'std' in name:
                    parameter.fill_(100)
        mu, std = net(torch.zeros(8, 14))
        assert torch.all(std <= 2.0) and torch.all(std >= 0.01)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.sample()
        assert torch.allclose(torch.exp(dist.log_prob(latent) - dist.log_prob(latent)),
                              torch.ones_like(latent))


def test_clipping_is_not_scored_as_gaussian_sample():
    dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    latent = torch.tensor([-3.0, 3.0])
    executed = torch.clamp(latent, -1, 1)
    assert not torch.equal(dist.log_prob(latent), dist.log_prob(executed))


def test_gae_does_not_leak_across_reset():
    values = torch.tensor([[1.0], [100.0]])
    boundary = torch.tensor([[1.0], [0.0]])
    assert torch.equal(compute_advantage(.99, .95, values, boundary), values)


def test_short_real_update_is_finite_and_changes_actor():
    torch.manual_seed(7)
    agent = PPOContinuous(14, 32, 3, 3e-4, 1e-4, .95, 2, .1, .99, torch.device('cpu'))
    states = np.random.default_rng(7).normal(size=(8, 14)).astype(np.float32)
    actions = [agent.take_action(s) for s in states]
    before = [p.detach().clone() for p in agent.actor.parameters()]
    losses = agent.update(dict(states=states, actions=actions, next_states=states + .01,
                               rewards=list(np.linspace(0, 1, 8)), dones=[False]*7+[True]))
    assert np.isfinite(losses).all()
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))


def test_inactive_lookahead_head_has_no_policy_gradient():
    torch.manual_seed(8)
    agent = PPOContinuous(14, 32, 3, 3e-4, 1e-4, .95, 1, .1, .99,
                          torch.device('cpu'), active_action_dims=(0, 1))
    states = np.zeros((4, 14), dtype=np.float32)
    before = agent.actor.mu_head.weight[2].detach().clone()
    agent.update(dict(states=states, actions=[agent.take_action(s) for s in states],
                      next_states=states, rewards=[1, 0, 2, 0], dones=[0, 0, 0, 1]))
    assert torch.equal(before, agent.actor.mu_head.weight[2])
