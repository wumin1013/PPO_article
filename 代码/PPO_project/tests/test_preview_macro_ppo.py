"""Macro-policy density, inactive-head isolation and episode-boundary contracts.

Updates below use tiny synthetic rollouts only; no environment or training run
is started. The existing three-head policy and event PPO remain untouched.
"""
import copy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
sys.path.insert(0, str(ROOT / "PPO_project/tools"))
LEGACY_SOURCES = (
    ROOT / "PPO_project/src/algorithms/ppo.py",
    ROOT / "PPO_project/src/algorithms/preview_event_ppo.py",
)
LEGACY_HASHES = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in LEGACY_SOURCES}

from src.algorithms.ppo import PolicyNetContinuous, ValueNet
from src.algorithms.preview_event_ppo import PreviewEventPPO, PreviewEventPolicy
from preview_macro_ppo import PreviewMacroPPO


ACTIVE_GROUPS = ((0, 1, 2), (0, 1), (1, 2), (1,))
FIXED_PREVIEW = (3.0 - 0.8) / 3.8


@pytest.fixture(autouse=True)
def single_torch_thread_and_preserved_sources():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
    for path, digest in LEGACY_HASHES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def options(**overrides):
    config = dict(state_dim=26, hidden_dim=32, action_dim=3,
                  epochs=2, eps=.1, ent_coef=.0003,
                  device=torch.device("cpu"), active_action_dims=(0, 1, 2))
    config.update(overrides)
    return config


def macro_rollout(agent, count=12):
    states = np.random.default_rng(17).normal(0, .1, (count, agent.state_dim)).astype(np.float32)
    actions, masks, old_log_probs = [], [], []
    for state in states:
        action, info = agent.take_action(state, return_info=True)
        actions.append(action)
        masks.append(info["action_mask"])
        old_log_probs.append(info["old_log_prob"])
    return dict(states=states, actions=np.asarray(actions, dtype=np.float32),
                next_states=states + .01, rewards=np.linspace(-.2, .3, count),
                dones=[False] * (count - 1) + [True],
                terminated=[False] * (count - 1) + [True],
                action_masks=np.asarray(masks, dtype=np.float32), old_log_probs=old_log_probs)


def assert_parameters_equal(first, second):
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("active", ACTIVE_GROUPS)
def test_each_macro_sample_scores_exactly_its_active_gaussian_coordinates(active):
    torch.manual_seed(41)
    agent = PreviewMacroPPO(**options(active_action_dims=active))
    state = np.zeros(26, dtype=np.float32)
    samples = []
    for _ in range(8):
        action, info = agent.take_action(state, return_info=True)
        action = np.asarray(action)
        with torch.no_grad():
            mu, std = agent.actor(torch.as_tensor(state[None]))
        expected_mask = np.array([index in active for index in range(3)], dtype=float)
        np.testing.assert_array_equal(info["action_mask"], expected_mask)
        np.testing.assert_allclose(np.asarray(info["mu"]).reshape(-1), mu.numpy()[0], rtol=0, atol=0)
        np.testing.assert_allclose(np.asarray(info["std"]).reshape(-1), std.numpy()[0], rtol=0, atol=0)
        # Independent scalar Gaussian formula, without calling the implementation.
        mean, scale = mu.numpy()[0], std.numpy()[0]
        expected = sum(-.5 * ((float(action[i]) - float(mean[i])) / float(scale[i])) ** 2
                       - math.log(float(scale[i])) - .5 * math.log(2 * math.pi) for i in active)
        assert info["old_log_prob"] == pytest.approx(expected, abs=2e-6)
        if 0 not in active:
            assert action[0] == 0.0
        if 2 not in active:
            assert action[2] == pytest.approx(FIXED_PREVIEW)
        samples.append(action)
    assert np.all(np.std(np.asarray(samples)[:, active], axis=0) > 0)


@pytest.mark.parametrize("active", ACTIVE_GROUPS)
def test_deterministic_action_returns_active_means_and_fixed_inactive_values(active):
    agent = PreviewMacroPPO(**options(active_action_dims=active), fixed_preview=.37)
    state = np.linspace(-.2, .2, 26, dtype=np.float32)
    with torch.no_grad():
        expected = agent.actor(torch.as_tensor(state[None]))[0].numpy()[0]
    if 0 not in active:
        expected[0] = 0.
    if 2 not in active:
        expected[2] = .37
    before_rng = torch.get_rng_state().clone()
    action, info = agent.take_action(state, return_info=True, deterministic=True)
    np.testing.assert_allclose(action, expected, rtol=0, atol=1e-7)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert np.isfinite(info["old_log_prob"])


@pytest.mark.parametrize("active", ACTIVE_GROUPS[1:])
def test_inactive_coordinates_have_zero_density_and_entropy_gradient(active):
    mu = torch.full((4, 3), .4, requires_grad=True)
    std = torch.full((4, 3), .2, requires_grad=True)
    actions = torch.full((4, 3), .7)
    inactive = [index for index in range(3) if index not in active]
    actions[:, inactive] = 1e30
    mask = torch.tensor([[index in active for index in range(3)]] * 4, dtype=torch.float32)
    logp = PreviewMacroPPO.latent_log_prob(mu, std, actions, mask).mean()
    assert torch.isfinite(logp)
    logp.backward(retain_graph=True)
    assert torch.count_nonzero(mu.grad[:, inactive]) == 0
    assert torch.count_nonzero(std.grad[:, inactive]) == 0
    assert torch.all(mu.grad[:, active] != 0)
    std.grad.zero_()
    PreviewMacroPPO.masked_entropy(mu, std, mask).mean().backward()
    assert torch.count_nonzero(std.grad[:, inactive]) == 0
    assert torch.all(std.grad[:, active] > 0)


@pytest.mark.parametrize("active", ACTIVE_GROUPS[1:])
def test_extreme_inactive_actions_do_not_change_losses_parameters_or_gradients(active):
    torch.manual_seed(67)
    original = PreviewMacroPPO(**options(active_action_dims=active))
    candidate = copy.deepcopy(original)
    data = macro_rollout(original)
    changed = copy.deepcopy(data)
    inactive = [index for index in range(3) if index not in active]
    for index in inactive:
        changed["actions"][:, index] = np.where(np.arange(len(data["states"])) % 2, -1e30, 1e30)
    before_mu = original.actor.mu_head.weight.detach().clone()
    first, second = original.update(data), candidate.update(changed)
    np.testing.assert_array_equal(first, second)
    assert_parameters_equal(original.actor, candidate.actor)
    assert_parameters_equal(original.critic, candidate.critic)
    for row in original.last_update_diagnostics["epochs"]:
        for index in inactive:
            name = ("angular", "feed", "preview")[index]
            assert row[f"{name}_mu_head_gradient_norm"] == 0
            assert row[f"{name}_std_head_gradient_norm"] == 0
    torch.testing.assert_close(original.actor.mu_head.weight[inactive], before_mu[inactive], rtol=0, atol=0)
    assert not torch.equal(original.actor.mu_head.weight[list(active)], before_mu[list(active)])


@pytest.mark.parametrize("active", ACTIVE_GROUPS)
def test_conflicting_stored_rollout_probability_is_rejected_without_update(active):
    agent = PreviewMacroPPO(**options(active_action_dims=active))
    data = macro_rollout(agent)
    before_actor, before_critic = copy.deepcopy(agent.actor), copy.deepcopy(agent.critic)
    data["old_log_probs"][0] += 1.
    with pytest.raises(ValueError, match="old_log_probs"):
        agent.update(data)
    assert_parameters_equal(before_actor, agent.actor)
    assert_parameters_equal(before_critic, agent.critic)
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state


@pytest.mark.parametrize("active", ACTIVE_GROUPS)
@pytest.mark.parametrize("defect", ("changed_row", "nonbinary", "wrong_shape"))
def test_invalid_or_inconsistent_macro_masks_are_rejected(active, defect):
    agent = PreviewMacroPPO(**options(active_action_dims=active))
    data = macro_rollout(agent)
    if defect == "changed_row":
        # Every macro transition owns exactly the registered action dimensions.
        data["action_masks"][0, active[-1]] = 0
    elif defect == "nonbinary":
        data["action_masks"][0, 1] = .5
    else:
        data["action_masks"] = data["action_masks"][:, :2]
    with pytest.raises(ValueError):
        agent.update(data)
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state


@pytest.mark.parametrize("active", ((0, 1, 2), (1, 2)))
def test_adaptive_macro_cannot_silently_reuse_event_hold_flags(active):
    agent = PreviewMacroPPO(**options(active_action_dims=active))
    data = macro_rollout(agent)
    data["preview_decisions"] = np.ones(len(data["states"]), dtype=bool)
    data["preview_decisions"][0] = False
    with pytest.raises(ValueError):
        agent.update(data)
    assert not agent.actor_optimizer.state


def test_macro_defaults_use_physical_period_without_motion_anchor():
    agent = PreviewMacroPPO(**options())
    assert agent.gamma == pytest.approx(math.exp(-.02 / 10.))
    assert agent.lmbda == 1.
    assert agent.anchor_beta == 0.
    assert agent.std_mode == "std_fix"
    assert agent.actor_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert agent.critic_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    with torch.no_grad():
        _, std = agent.actor(torch.zeros(2, 26))
    torch.testing.assert_close(std, torch.tensor([[.03, .05, .15]] * 2), rtol=1e-6, atol=1e-8)
    slower = PreviewMacroPPO(**options(), macro_period_s=.04)
    assert slower.gamma == pytest.approx(math.exp(-.04 / 10.))


@pytest.mark.parametrize("terminated,truncated", ((False, True), (True, False), (True, True)))
def test_macro_bootstrap_and_gae_respect_actual_episode_boundaries(terminated, truncated):
    agent = PreviewMacroPPO(**options(epochs=1, critic_lr=0.))
    with torch.no_grad():
        for parameter in agent.critic.parameters():
            parameter.zero_()
        agent.critic.fc4.bias.fill_(1.)
    data = macro_rollout(agent, count=2)
    data.update(rewards=[1., 100.], dones=[True, False],
                terminated=[terminated, False], truncated=[truncated, False])
    agent.update(data)
    diagnostics = agent.last_update_diagnostics
    expected_target = 1. if terminated else 1. + agent.gamma
    assert diagnostics["td_target"]["min"] == pytest.approx(expected_target, abs=1e-6)
    assert diagnostics["raw_advantage"]["min"] == pytest.approx(expected_target - 1., abs=1e-6)
    assert diagnostics["raw_advantage"]["max"] == pytest.approx(99. + agent.gamma, abs=1e-5)
    assert diagnostics["terminated_count"] == int(terminated)
    assert diagnostics["truncated_count"] == int(truncated)
    assert agent.critic.training is False


def test_macro_reuses_original_network_shapes_without_patching_legacy_classes():
    original_forward = PolicyNetContinuous.forward
    original_event_update = PreviewEventPPO.update
    agent = PreviewMacroPPO(**options())
    original_actor = PolicyNetContinuous(26, 32, 3)
    original_critic = ValueNet(26, 32)
    assert isinstance(agent.actor, PreviewEventPolicy)
    for current, original in ((agent.actor, original_actor), (agent.critic, original_critic)):
        assert {name: value.shape for name, value in current.state_dict().items()} == {
            name: value.shape for name, value in original.state_dict().items()}
    agent.update(macro_rollout(agent))
    assert PolicyNetContinuous.forward is original_forward
    assert PreviewEventPPO.update is original_event_update


def test_full_macro_update_matches_original_event_ppo_at_identical_decisions():
    matched = options(actor_lr=3e-4, critic_lr=3e-4,
                      gamma=math.exp(-.02 / 10.), lmbda=1.,
                      std_mode="std_fix", anchor_beta=0.,
                      initial_mean=(0., .05, FIXED_PREVIEW), initial_std=(.03, .05, .15))
    torch.manual_seed(83)
    original = PreviewEventPPO(**matched)
    torch.manual_seed(83)
    macro = PreviewMacroPPO(**matched)
    assert_parameters_equal(original.actor, macro.actor)
    assert_parameters_equal(original.critic, macro.critic)
    data = macro_rollout(macro, count=16)
    data["preview_decisions"] = np.ones(16, dtype=bool)
    first, second = original.update(data), macro.update(data)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    for prior, current in ((original.actor, macro.actor), (original.critic, macro.critic)):
        for name, value in prior.state_dict().items():
            torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=1e-7)
