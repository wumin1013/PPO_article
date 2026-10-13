"""Real event holding, masked likelihood, neutral migration and exact resume."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_coherent_distance as coherent
import preview_full_batch_study as full
import preview_restart_closed_loop as execution
from test_preview_full_batch import assert_tree_equal

REFERENCE = np.asarray([[0., 0.], [8., 0.], [8., 5.]])


@pytest.fixture(autouse=True)
def one_thread_and_shared_execution():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with execution.corrected_execution():
        yield
    torch.set_num_threads(previous)


def config(method='adaptive', period=200):
    cfg = full.configuration('preview_time', method, 901)
    common = full.configuration('preview_time', 'fixed3', 901)
    for key in ('init_checkpoint', 'init_checkpoint_sha256'):
        cfg['revision'][key] = common['revision'][key]
    return coherent.configure(cfg, period_ms=period)


def environment(cfg=None, path='unit'):
    return coherent.CoherentDistanceEnvironment(cfg or config(), path, reference=REFERENCE)


def agent(cfg=None):
    cfg = cfg or config()
    return coherent.make_agent(cfg, type('Env', (), {'observation_dim': 26})())


@pytest.mark.parametrize('period', [20, 200])
def test_shared_phase_and_physical_hold_change_only_at_registered_events(period):
    adaptive = environment(config(period=period))
    fixed = environment(config('fixed3', period))
    held = (3.-.8)/3.8
    for macro in range(22):
        tick = macro*20
        expected_phase = ((period-tick % period) % period)/period
        assert adaptive.state[21] == fixed.state[21] == np.float32(expected_phase)
        proposed = .15 if macro % 2 else .85
        is_event = tick % period == 0
        if is_event:
            held = proposed
        adaptive_start = len(adaptive.trace)
        fixed_start = len(fixed.trace)
        adaptive.step(np.asarray([0., -4., proposed]))
        fixed.step(np.asarray([0., -4., proposed]))
        assert not adaptive.done and not fixed.done
        assert adaptive.inner.tick == fixed.inner.tick == tick+20
        assert adaptive.inner.period == fixed.inner.period == period
        assert adaptive.inner.held_preview == held
        assert fixed.inner.held_preview == pytest.approx((3.-.8)/3.8)
        for index, row in enumerate(adaptive.trace[adaptive_start:]):
            assert row['lookahead_mm'] == pytest.approx(.8+3.8*held)
            assert row['preview_event'] == row['control_status__preview_event'] == float(is_event and index == 0)
        assert all(row['preview_event'] == row['control_status__preview_event'] == 0.
                   for row in fixed.trace[fixed_start:])


@pytest.mark.parametrize('period', [True, 0, 19, 100, 201])
def test_invalid_period_is_rejected_without_mutating_parent(period):
    cfg = full.configuration('preview_time', 'adaptive', 901)
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        coherent.configure(cfg, period_ms=period)
    assert cfg == before


def test_configuration_and_phase_observation_are_read_only():
    cfg = full.configuration('preview_time', 'adaptive', 901)
    before = copy.deepcopy(cfg)
    configured = coherent.configure(cfg, period_ms=200)
    assert cfg == before
    for key in ('ppo', 'environment', 'kinematic_constraints', 'reward_weights'):
        assert configured[key] == cfg[key]
    env = environment(config())
    env.step(np.asarray([0., -4., .4]))
    observation = env.state.copy()
    tracker = env.inner.env.env.coverage.snapshot()
    raw = env.inner.raw
    reference_state = (raw._theta_ref_filtered, raw._theta_ref_filter_update_count,
                       raw._theta_ref_filter_update_step)
    for _ in range(3):
        env._sense()
    np.testing.assert_array_equal(env.state, observation)
    assert_tree_equal(tracker, env.inner.env.env.coverage.snapshot())
    assert reference_state == (raw._theta_ref_filtered, raw._theta_ref_filter_update_count,
                               raw._theta_ref_filter_update_step)
    assert env.inner.tick == 20


@pytest.mark.parametrize('method', ['adaptive', 'fixed3'])
def test_event_mask_and_three_draw_rng_alignment(method):
    policy = agent(config(method))
    state = environment(config(method)).state.copy()
    state[20] = .37
    for phase in (0., .9, .5, .1):
        state[21] = phase
        torch.manual_seed(752)
        before = torch.get_rng_state()
        action, info = policy.take_action(state, return_info=True)
        observed_rng = torch.get_rng_state()
        torch.set_rng_state(before)
        with torch.no_grad():
            mu, std = policy.actor(torch.as_tensor(state[None]))
            torch.distributions.Normal(mu, std).sample()
        assert torch.equal(torch.get_rng_state(), observed_rng)
        expected_event = method == 'adaptive' and phase == 0.
        np.testing.assert_array_equal(info['action_mask'], [0., 1., float(expected_event)])
        assert info['preview_decision'] == expected_event
        if method == 'adaptive' and phase:
            assert action[2] == state[20]
        if method == 'fixed3':
            assert action[2] == pytest.approx((3.-.8)/3.8)


def test_held_D_has_no_density_entropy_or_direct_head_gradient():
    policy = agent()
    state = environment().state.copy()
    state[21], state[20] = .9, .42
    action, info = policy.take_action(state, return_info=True)
    mu, std = policy.actor(torch.as_tensor(state[None]))
    mask = torch.as_tensor(info['action_mask'][None])
    original = torch.as_tensor(action[None])
    altered = original.clone()
    altered[:, 2] = 100.
    first = policy.latent_log_prob(mu, std, original, mask)
    second = policy.latent_log_prob(mu, std, altered, mask)
    torch.testing.assert_close(first, second, rtol=0., atol=0.)
    assert first.item() == pytest.approx(info['old_log_prob'], abs=1e-6)
    policy.actor.zero_grad()
    (first.sum()+policy.masked_entropy(mu, std, mask).sum()).backward()
    assert torch.count_nonzero(policy.actor.mu_head.weight.grad[2]) == 0
    assert torch.count_nonzero(policy.actor.std_head.weight.grad[2]) == 0
    assert float(policy.actor.mu_head.weight.grad[1].norm()) > 0


@pytest.mark.parametrize('method', ['adaptive', 'fixed3'])
def test_batch_masks_match_observed_phase_and_reject_rescoring(method):
    policy = agent(config(method))
    states = np.tile(environment(config(method)).state, (4, 1))
    states[:, 21] = [0., .9, .1, 0.]
    masks = np.zeros((4, 3), dtype=np.float32)
    masks[:, 1] = 1.
    if method == 'adaptive':
        masks[[0, 3], 2] = 1.
    batch = dict(states=states, action_masks=masks, preview_decisions=masks[:, 2])
    np.testing.assert_array_equal(policy._batch_masks(batch, 4).numpy(), masks)
    invalid = copy.deepcopy(batch)
    invalid['action_masks'][1, 2] = 1.
    with pytest.raises(ValueError):
        policy._batch_masks(invalid, 4)
    invalid = copy.deepcopy(batch)
    invalid['preview_decisions'] = 1.-masks[:, 2]
    with pytest.raises(ValueError):
        policy._batch_masks(invalid, 4)
    with pytest.raises(ValueError):
        policy._batch_masks(dict(states=states), 4)


@pytest.mark.parametrize('method', ['adaptive', 'fixed3'])
def test_common_initializer_phase_migration_preserves_outputs_and_fresh_adam(method):
    cfg = config(method)
    old = full.local.make_agent(cfg, type('Env', (), {'observation_dim': 26})())
    migrated = agent(cfg)
    off = agent(config(method, 20))
    fixed = agent(config('fixed3'))
    states = torch.as_tensor(np.random.default_rng(71).normal(0, .1, (12, 26)).astype('float32'))
    states[:, 21] = 0.
    new_states = states.clone()
    new_states[:, 21] = torch.linspace(0., .9, 12)
    old.critic.eval()
    migrated.critic.eval()
    with torch.no_grad():
        for first, second in zip(old.actor(states), migrated.actor(new_states)):
            torch.testing.assert_close(first, second, rtol=0., atol=0.)
        torch.testing.assert_close(old.critic(states), migrated.critic(new_states), rtol=0., atol=0.)
    for name, layer in (('actor', 'shared_fc1'), ('critic', 'fc1')):
        assert torch.count_nonzero(getattr(getattr(migrated, name), layer).weight[:, 21]) == 0
        assert_tree_equal(getattr(migrated, name).state_dict(), getattr(off, name).state_dict())
        assert_tree_equal(getattr(migrated, name).state_dict(), getattr(fixed, name).state_dict())
    assert not migrated.actor_optimizer.state and not migrated.critic_optimizer.state


def test_period20_matches_original_macro_controller_on_same_initializer():
    cfg = config(period=20)
    changed = environment(cfg)
    original = full.local.LocalCornerEnvironment(cfg, 'unit', reference=REFERENCE)
    changed_agent = coherent.make_agent(cfg, changed)
    old_agent = full.local.make_agent(cfg, original)
    for _ in range(24):
        new_action = changed_agent.take_action(changed.state, deterministic=True)
        old_action = old_agent.take_action(original.state, deterministic=True)
        np.testing.assert_array_equal(new_action, old_action)
        new_state, new_reward, new_done, _ = changed.step(new_action)
        old_state, old_reward, old_done, _ = original.step(old_action)
        np.testing.assert_array_equal(new_state, old_state)
        assert (new_reward, new_done) == (old_reward, old_done)
        if new_done:
            break
    for key in ('x_mm', 'y_mm', 'v_mm_s', 'a_mm_s2', 'heading_rad', 'lookahead_mm'):
        np.testing.assert_array_equal([row[key] for row in changed.trace], [row[key] for row in original.trace])

