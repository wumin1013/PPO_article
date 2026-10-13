"""Event probabilities, reset/bootstrap boundaries and std-fix behavior."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
sys.path.insert(0, str(ROOT / "PPO_project/tools"))

from src.algorithms.preview_event_ppo import PreviewEventPPO
from preview_direct_research import AnchoredPPO


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(**overrides):
    config = dict(state_dim=26, hidden_dim=32, action_dim=3,
                  actor_lr=1e-5, critic_lr=1e-4, gamma=.99, lmbda=1.,
                  epochs=2, eps=.1, ent_coef=.0003, device=torch.device("cpu"),
                  active_action_dims=(0, 1, 2))
    config.update(overrides)
    return config


def rollout(agent, count=16, every=4):
    rng = np.random.default_rng(5)
    states = rng.normal(0, .1, (count, agent.state_dim)).astype(np.float32)
    decisions = np.arange(count) % every == 0
    actions, old_log_probs = [], []
    held = agent.initial_preview
    for state, event in zip(states, decisions):
        latent, info = agent.take_action(state, preview_decision=event,
                                         held_preview=held, return_info=True)
        if event:
            held = float(latent[2])
        actions.append(latent)
        old_log_probs.append(info["old_log_prob"])
    return dict(states=states, actions=np.asarray(actions), next_states=states + .01,
                rewards=np.linspace(-.1, .2, count), dones=[False] * (count - 1) + [True],
                terminated=[False] * (count - 1) + [True],
                preview_decisions=decisions, old_log_probs=old_log_probs)


def initialize_legacy_heads(agent):
    preview = (3. - .8) / 3.8
    with torch.no_grad():
        agent.actor.mu_head.weight.zero_()
        agent.actor.mu_head.bias.copy_(torch.tensor([0., math.log(.05/.95), math.log(preview/(1-preview))]))
        agent.actor.std_head.weight.zero_()
        agent.actor.std_head.bias.fill_(math.log(math.expm1(.01-.001)))
        agent.actor.std_head.bias[2] = math.log(math.expm1(.05-.001))


@pytest.mark.parametrize("state_dim", [20, 26, 28])
def test_state_dimensions_and_legacy_initialization_preserve_rng(state_dim):
    config = options(state_dim=state_dim)
    torch.manual_seed(13)
    original = AnchoredPPO(**config)
    initialize_legacy_heads(original)
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(13)
    candidate = PreviewEventPPO(**config, std_mode="legacy")
    assert torch.equal(torch.get_rng_state(), original_rng)
    for before, after in zip(original.actor.parameters(), candidate.actor.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for before, after in zip(original.critic.parameters(), candidate.critic.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    states = np.random.default_rng(7).normal(size=(8, state_dim)).astype(np.float32)
    torch.manual_seed(67)
    old_actions = np.array([original.take_action(state) for state in states])
    torch.manual_seed(67)
    new_actions = np.array([candidate.take_action(state) for state in states])
    np.testing.assert_array_equal(old_actions, new_actions)


@pytest.mark.parametrize("active", [(0, 1), (0, 1, 2)])
def test_every_tick_legacy_update_matches_archived_algorithm(active):
    config = options(active_action_dims=active, state_dim=20)
    torch.manual_seed(19)
    original = AnchoredPPO(**config)
    initialize_legacy_heads(original)
    torch.manual_seed(19)
    candidate = PreviewEventPPO(**config, std_mode="legacy")
    data = rollout(candidate, every=1)
    # The archived algorithm scores its own pre-update density. This test uses
    # the same route, isolating event-mask arithmetic from serialization error.
    data.pop("old_log_probs")
    old_loss = original.update(data)
    new_loss = candidate.update(data)
    np.testing.assert_allclose(old_loss, new_loss, rtol=0, atol=1e-7)
    for name, before in original.actor.state_dict().items():
        torch.testing.assert_close(before, candidate.actor.state_dict()[name], rtol=0, atol=1e-7)
    for name, before in original.critic.state_dict().items():
        torch.testing.assert_close(before, candidate.critic.state_dict()[name], rtol=0, atol=1e-7)


def test_std_fix_has_positive_physical_derivative_at_same_initial_scale():
    torch.manual_seed(23)
    legacy = PreviewEventPPO(**options(), std_mode="legacy")
    torch.manual_seed(23)
    fixed = PreviewEventPPO(**options(), std_mode="std_fix")
    states = torch.zeros(8, 26)
    mu_legacy, std_legacy = legacy.actor(states)
    mu_fixed, std_fixed = fixed.actor(states)
    torch.testing.assert_close(mu_legacy, mu_fixed, rtol=0, atol=0)
    torch.testing.assert_close(std_legacy, std_fixed, rtol=1e-6, atol=1e-8)
    std_legacy[:, :2].sum().backward()
    std_fixed[:, :2].sum().backward()
    assert torch.count_nonzero(legacy.actor.std_head.bias.grad[:2]) == 0
    assert torch.all(fixed.actor.std_head.bias.grad[:2] > 0)
    # The maximum cap is intentionally retained, not advertised as an
    # everywhere-positive derivative parameterization.
    with torch.no_grad():
        fixed.actor.std_head.bias.fill_(100.)
    assert torch.all(fixed.actor(states)[1] == 2.)


def test_sampling_uses_softplus_std_below_legacy_floor_and_preserves_raw_latent():
    agent = PreviewEventPPO(**options(), std_mode="std_fix")
    with torch.no_grad():
        agent.actor.std_head.bias.fill_(math.log(math.expm1(.005-.001)))
        agent.actor.mu_head.bias[1] = -20.
    state = np.zeros(26, np.float32)
    with torch.no_grad():
        mu, std = agent.actor(torch.tensor(state[None]))
    assert torch.all(std < .01)
    torch.manual_seed(21)
    expected = torch.distributions.Normal(mu, std).sample()[0].numpy()
    torch.manual_seed(21)
    action, info = agent.take_action(state, return_info=True)
    np.testing.assert_array_equal(action, expected)
    # Force a reproducible negative feed latent: no physical clipping belongs
    # in this sampler or its scored density.
    negatives = [agent.take_action(state)[1] for _ in range(40)]
    assert any(value < 0 for value in negatives)
    dist = torch.distributions.Normal(mu, std)
    expected_logp = float(dist.log_prob(torch.tensor(action[None])).sum())
    assert info["old_log_prob"] == pytest.approx(expected_logp)


def test_held_preview_is_returned_but_only_physical_density_is_scored():
    agent = PreviewEventPPO(**options())
    state = np.zeros(26, np.float32)
    torch.manual_seed(47)
    sampled = agent.take_action(state, preview_decision=True)
    torch.manual_seed(47)
    held, info = agent.take_action(state, preview_decision=False, held_preview=.81, return_info=True)
    np.testing.assert_array_equal(sampled[:2], held[:2])
    assert held[2] == pytest.approx(.81)
    np.testing.assert_array_equal(info["action_mask"], [1, 1, 0])
    with torch.no_grad():
        mu, std = agent.actor(torch.tensor(state[None]))
        expected = torch.distributions.Normal(mu, std).log_prob(torch.tensor(held[None]))[:, :2].sum()
    assert info["old_log_prob"] == pytest.approx(float(expected))
    with pytest.raises(ValueError, match="held_preview"):
        agent.take_action(state, preview_decision=False)


def test_masked_log_probability_and_entropy_have_no_held_preview_gradient():
    mu = torch.full((4, 3), .5, requires_grad=True)
    std = torch.full((4, 3), .2, requires_grad=True)
    actions = torch.tensor([[.3, .4, .8], [.2, .6, 1e30], [.6, .3, -1e30], [.5, .7, .99]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.float32)
    loss = PreviewEventPPO.latent_log_prob(mu, std, actions, mask).mean()
    assert torch.isfinite(loss)
    loss.backward(retain_graph=True)
    assert mu.grad[0, 2] != 0 and std.grad[0, 2] != 0
    assert torch.count_nonzero(mu.grad[1:, 2]) == 0
    assert torch.count_nonzero(std.grad[1:, 2]) == 0
    std.grad.zero_()
    entropy = PreviewEventPPO.masked_entropy(mu, std, mask).mean()
    entropy.backward()
    assert std.grad[0, 2] == pytest.approx(1 / (.2 * 4))
    assert torch.count_nonzero(std.grad[1:, 2]) == 0


@pytest.mark.parametrize("active", [(0, 1), (0, 1, 2)])
def test_inactive_held_values_cannot_change_update(active):
    torch.manual_seed(77)
    original = PreviewEventPPO(**options(active_action_dims=active), std_mode="std_fix")
    candidate = copy.deepcopy(original)
    data = rollout(original)
    data["preview_decisions"][:] = False
    data.pop("old_log_probs")
    changed = copy.deepcopy(data)
    changed["actions"][:, 2] = 1e30
    first = original.update(data)
    second = candidate.update(changed)
    np.testing.assert_array_equal(first, second)
    for name, value in original.actor.state_dict().items():
        torch.testing.assert_close(value, candidate.actor.state_dict()[name], rtol=0, atol=0)
    for epoch in original.last_update_diagnostics["epochs"]:
        assert epoch["preview_mu_head_gradient_norm"] == 0
        assert epoch["preview_std_head_gradient_norm"] == 0
    assert original.last_update_diagnostics["preview_decision_count"] == 0


def test_timeout_bootstraps_but_gae_stops_before_next_episode():
    config = options(gamma=.9, lmbda=1., epochs=1, critic_lr=0.)
    agent = PreviewEventPPO(**config)
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
        agent.critic.fc4.bias.fill_(1.)
    data = rollout(agent, count=2, every=1)
    data.update(rewards=[1., 100.], dones=[True, False], terminated=[False, False], truncated=[True, False])
    agent.update(data)
    diag = agent.last_update_diagnostics
    assert diag["td_target"]["min"] == pytest.approx(1.9)
    assert diag["raw_advantage"]["min"] == pytest.approx(.9, abs=1e-6)
    assert diag["raw_advantage"]["max"] == pytest.approx(99.9)
    assert diag["truncated_count"] == 1 and diag["terminated_count"] == 0
    assert agent.critic.training is False
    terminal = PreviewEventPPO(**config)
    terminal.actor.load_state_dict(agent.actor.state_dict())
    terminal.critic.load_state_dict(agent.critic.state_dict())
    data.pop("old_log_probs")
    data["terminated"] = [True, False]
    # If both flags are true at a time limit coinciding with completion,
    # actual termination still suppresses bootstrap.
    terminal.update(data)
    assert terminal.last_update_diagnostics["td_target"]["min"] == pytest.approx(1.)
    assert terminal.last_update_diagnostics["raw_advantage"]["min"] == pytest.approx(0.)


def test_actual_ratio_clip_fractions_and_analytic_kl_match_samples(monkeypatch):
    agent = PreviewEventPPO(**options(actor_lr=.003, critic_lr=0., lmbda=0., epochs=3), anchor_beta=0.)
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
    data = rollout(agent, count=24, every=3)
    states = torch.tensor(data["states"])
    actions = torch.tensor(data["actions"])
    masks = torch.ones((24, 3))
    masks[:, 2] = torch.tensor(data["preview_decisions"], dtype=torch.float32)
    rewards = torch.tensor(data["rewards"], dtype=torch.float32).view(-1, 1)
    advantage = (rewards-rewards.mean())/(rewards.std(unbiased=False)+1e-8)
    with torch.no_grad():
        old_mu, old_std = agent.actor(states)
    old_logp = torch.tensor(data["old_log_probs"]).view(-1, 1)
    captured = []
    original_step = agent.actor_optimizer.step

    def capture_then_step(*args, **kwargs):
        with torch.no_grad():
            mu, std = agent.actor(states)
            logp = agent.latent_log_prob(mu, std, actions, masks)
            ratio = torch.exp(torch.clamp(logp-old_logp, -5, 5))
            outside = ((ratio < .9) | (ratio > 1.1)).float().mean()
            clipped = ((torch.clamp(ratio, .9, 1.1)*advantage) < ratio*advantage).float().mean()
            kl = torch.distributions.kl_divergence(torch.distributions.Normal(old_mu, old_std),
                                                torch.distributions.Normal(mu, std))
            captured.append((float(outside), float(clipped), float((kl*masks).sum(1).mean())))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(agent.actor_optimizer, "step", capture_then_step)
    agent.update(data)
    for row, expected in zip(agent.last_update_diagnostics["epochs"], captured):
        assert row["ratio_outside_clip_fraction"] == pytest.approx(expected[0])
        assert row["surrogate_clipped_fraction"] == pytest.approx(expected[1])
        assert row["old_new_kl_masked_mean"] == pytest.approx(expected[2])
    assert captured[0] == pytest.approx((0., 0., 0.))
    assert any(row[0] > 0 for row in captured[1:])


def test_stored_density_or_conflicting_masks_rejected_before_optimizer_step():
    agent = PreviewEventPPO(**options())
    data = rollout(agent)
    before = copy.deepcopy(agent.actor.state_dict())
    data["old_log_probs"][0] += 1.
    with pytest.raises(ValueError, match="old_log_probs"):
        agent.update(data)
    for name, parameter in agent.actor.state_dict().items():
        assert torch.equal(before[name], parameter)
    data.pop("old_log_probs")
    data["action_masks"] = np.ones((16, 3))
    with pytest.raises(ValueError, match="disagree"):
        agent.update(data)
    data.pop("action_masks")
    data.pop("preview_decisions")
    with pytest.raises(ValueError, match="explicit preview_decisions"):
        agent.update(data)


@pytest.mark.parametrize("std_mode", ["legacy", "std_fix"])
def test_short_event_update_is_finite_and_diagnostics_are_complete(std_mode):
    torch.manual_seed(89)
    agent = PreviewEventPPO(**options(), std_mode=std_mode, anchor_beta=1.)
    data = rollout(agent, every=4)
    before = agent.actor.mu_head.weight.detach().clone()
    losses = agent.update(data)
    assert np.isfinite(losses).all()
    assert not torch.equal(before, agent.actor.mu_head.weight)
    diag = agent.last_update_diagnostics
    assert diag["preview_decision_count"] == 4 and diag["physical_decision_count"] == 16
    assert diag["normalized_advantage"]["std"] == pytest.approx(1., abs=1e-6)
    assert diag["old_log_probability_max_absolute_error"] <= 1e-4
    assert len(diag["epochs"]) == 2
    for epoch in diag["epochs"]:
        assert 0 < epoch["actor_gradient_clip_scale"] <= 1
        assert epoch["preview_mu_head_gradient_norm"] > 0
        if std_mode == "std_fix":
            assert epoch["angular_std_head_gradient_norm"] > 0
        else:
            assert epoch["angular_std_head_gradient_norm"] == 0
