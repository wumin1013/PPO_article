"""Physical parity and full-duration accounting for the isolated objective."""
import copy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_coherent_distance as coherent
import preview_full_batch_study as full
import preview_macro_study as macro
import preview_restart_closed_loop as execution
import preview_undiscounted_objective as objective


FROZEN = [ROOT/'PPO_project/tools'/name for name in (
    'preview_macro_study.py', 'preview_coherent_distance.py',
    'preview_research_v2.py', 'terminal_metrics.py', 'terminal_braking.py')]
FROZEN_HASHES = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FROZEN}
COUNTS = {'environments': 0, 'macro_steps': 0, 'physical_ticks': 0, 'tail_ticks': 0,
          'synthetic_ppo_updates': 0, 'synthetic_ppo_rows': 0}
# These two existing perf_counter diagnostics measure wall time rather than
# physical state, action or reward; replay cannot promise identical timings.
TIMING_KEYS = {'control_status__direct_mapping_us',
               'control_status__direct_observation_features_us',
               'direct_mapping_us', 'direct_observation_features_us'}


def assert_tree_equal(first, second, path='root'):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second), path
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), path
        for key in first:
            if key not in TIMING_KEYS:
                assert_tree_equal(first[key], second[key], f'{path}.{key}')
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second), path
        for index, (a, b) in enumerate(zip(first, second)):
            assert_tree_equal(a, b, f'{path}[{index}]')
    else:
        np.testing.assert_equal(first, second, err_msg=path)


@pytest.fixture(autouse=True)
def unchanged_execution_and_sources():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with execution.corrected_execution():
        yield
    torch.set_num_threads(previous)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FROZEN} == FROZEN_HASHES


@pytest.fixture(scope='module', autouse=True)
def print_actual_test_costs():
    yield
    print('\nUNDISCOUNTED_OBJECTIVE_TEST_COSTS=' + repr(COUNTS))


def configuration(method='adaptive', period=20):
    cfg = full.configuration('preview_time', method, 901)
    return coherent.configure(cfg, period_ms=period)


def environment(cls=objective.UndiscountedEnvironment, *, cfg=None, reference=None):
    cfg = configuration() if cfg is None else cfg
    if cls is objective.UndiscountedEnvironment:
        cfg = objective.configure(cfg)
    reference = np.asarray([[0., 0.], [4., 0.]]) if reference is None else reference
    env = cls(cfg, 'synthetic_objective_unit', reference=reference)
    COUNTS['environments'] += 1
    return env


def step(env, action):
    physics_before, commands_before = env.physics_steps, env.command_steps
    result = env.step(action)
    physical = env.physics_steps - physics_before
    COUNTS['macro_steps'] += 1
    COUNTS['physical_ticks'] += physical
    COUNTS['tail_ticks'] += env.command_steps - commands_before - physical
    return result


def assert_trace_equal(first, second):
    assert len(first.trace) == len(second.trace)
    for a, b in zip(first.trace, second.trace):
        assert a.keys() == b.keys()
        for key in a:
            if key not in TIMING_KEYS:
                np.testing.assert_equal(a[key], b[key], err_msg=key)
    assert first.result == second.result


def finish(env, *, base=None):
    total = 0.
    action = np.asarray([0., math.log(.05/.95), (3.-.8)/3.8])
    for _ in range(1000):
        start = len(env.trace)
        physics = env.physics_steps
        state, reward, done, info = step(env, action)
        if base is not None:
            old_state, _, old_done, old_info = step(base, action)
            np.testing.assert_array_equal(state, old_state)
            assert done == old_done
            assert_tree_equal(info, old_info)
        total += reward
        if done:
            return total, reward, env.trace[start:], env.physics_steps - physics, info
    pytest.fail('Synthetic line did not terminate')


def assert_complete_return(env, total):
    physical = [row for row in env.trace[1:] if 'reward' in row]
    initial = physical[0]['reward_components__ordered_initial_s_mm']
    final = physical[-1]['reward_components__ordered_covered_s_mm']
    bonus = 10. if env.result['success'] else -20.
    assert initial > 0.
    expected = final - initial - env.result['elapsed_time_s'] + bonus
    assert total == pytest.approx(expected, abs=1e-11)
    assert total != pytest.approx(final - env.result['elapsed_time_s'] + bonus, abs=1e-4)


def test_config_changes_only_gamma_and_objective_metadata():
    cfg = configuration(period=200)
    before = copy.deepcopy(cfg)
    changed = objective.configure(cfg)
    assert cfg == before
    assert changed['ppo']['gamma'] == 1. and changed['ppo']['lmbda'] == 1.
    assert changed['environment']['max_steps'] == 60000
    changed['ppo']['gamma'] = cfg['ppo']['gamma']
    for key in ('undiscounted_objective', 'discount_change'):
        changed['revision'].pop(key)
    changed['revision']['objective'] = cfg['revision']['objective']
    assert changed == cfg


@pytest.mark.parametrize('invalid', [0, -1, float('inf'), True, 1.5])
def test_nonfinite_or_invalid_horizon_rejected_without_mutation(invalid):
    cfg = configuration()
    cfg['environment']['max_steps'] = invalid
    cfg['revision']['episode_max_steps'] = invalid
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError, match='finite positive integer'):
        objective.configure(cfg)
    assert cfg == before


def test_effective_horizon_mismatch_and_changed_physical_period_are_rejected():
    cfg = configuration()
    cfg['revision']['episode_max_steps'] = 27
    with pytest.raises(ValueError, match='must agree'):
        objective.configure(cfg)
    cfg = configuration()
    cfg['environment']['interpolation_period'] = .002
    with pytest.raises(ValueError, match='1 ms'):
        objective.configure(cfg)


def test_unconfigured_or_inconsistent_gamma_cannot_silently_use_new_reward():
    cfg = configuration()
    with pytest.raises(ValueError, match='configure'):
        objective.UndiscountedEnvironment(cfg, 'unit', reference=np.asarray([[0., 0.], [4., 0.]]))
    cfg = objective.configure(cfg)
    cfg['ppo']['gamma'] = .99
    with pytest.raises(ValueError, match='gamma=1'):
        objective.UndiscountedEnvironment(cfg, 'unit', reference=np.asarray([[0., 0.], [4., 0.]]))


@pytest.mark.parametrize('method', ['fixed3', 'adaptive'])
@pytest.mark.parametrize('period', [20, 200])
def test_same_actions_preserve_every_trace_value_and_incremental_reward(method, period):
    cfg = configuration(method, period)
    reference = np.asarray([[0., 0.], [8., 0.], [8., 5.]])
    base = environment(coherent.CoherentDistanceEnvironment, cfg=cfg, reference=reference)
    changed = environment(cfg=cfg, reference=reference)
    for index in range(13):
        action = np.asarray([0., -4., .15 if index % 2 else .85])
        old_state, old_reward, old_done, old_info = step(base, action)
        start = len(changed.trace)
        state, reward, done, info = step(changed, action)
        np.testing.assert_array_equal(state, old_state)
        assert done == old_done and not done
        assert_tree_equal(info, old_info)
        rows = changed.trace[start:]
        assert reward == pytest.approx(sum(row['reward_components__progress_mm'] for row in rows) - .020, abs=1e-14)
        assert old_reward != pytest.approx(reward, abs=1e-9)
    assert_trace_equal(base, changed)


def test_complete_success_counts_initial_coverage_and_brake_rest_exactly_once():
    base = environment(coherent.CoherentDistanceEnvironment)
    changed = environment()
    total, reward, last_rows, physical_count, info = finish(changed, base=base)
    assert_trace_equal(base, changed)
    assert changed.result['success'] and changed.result['full_rest_satisfied']
    assert info['done_reason'] == 'success'
    assert 0 < physical_count <= 20
    tail = len(last_rows) - physical_count
    assert tail > 0 and all(row['terminal_braking'] == 1. for row in last_rows[physical_count:])
    assert reward == pytest.approx(sum(row['reward'] for row in last_rows[:physical_count]) - tail*.001 + 10., abs=1e-12)
    assert_complete_return(changed, total)
    with pytest.raises(RuntimeError, match='Completed'):
        changed.step([0., -4., .5])


def test_mid_macro_horizon_failure_has_no_tail_or_success_bonus():
    cfg = configuration()
    cfg['environment']['max_steps'] = 27
    cfg['revision']['episode_max_steps'] = 27
    base = environment(coherent.CoherentDistanceEnvironment, cfg=cfg)
    changed = environment(cfg=cfg)
    total, reward, last_rows, physical_count, info = finish(changed, base=base)
    assert_trace_equal(base, changed)
    assert not changed.result['success']
    assert changed.physics_steps == changed.command_steps == 27
    assert physical_count == len(last_rows) == 7
    assert reward == pytest.approx(sum(row['reward'] for row in last_rows) - 20., abs=1e-12)
    assert_complete_return(changed, total)


def test_full_audit_failure_after_braking_keeps_failure_value(monkeypatch):
    env = environment()
    original = env._audit

    def failed_audit(reason, claimed):
        result = original(reason, claimed)
        result['success'] = False
        result['failure_reasons'].append('synthetic_complete_audit_failure')
        return result

    monkeypatch.setattr(env, '_audit', failed_audit)
    total, reward, rows, physical, info = finish(env)
    assert info['done_reason'] == 'terminal_audit_failure'
    assert env.result['full_rest_satisfied'] and not env.result['success']
    assert len(rows) > physical
    assert reward == pytest.approx(sum(row['reward'] for row in rows[:physical]) - (len(rows)-physical)*.001 - 20., abs=1e-12)
    assert_complete_return(env, total)


def test_brake_planning_failure_has_no_invented_tail(monkeypatch):
    env = environment()

    def unavailable(*args, **kwargs):
        raise ValueError('synthetic unavailable brake')

    monkeypatch.setattr(macro, 'plan_terminal_brake', unavailable)
    total, reward, rows, physical, info = finish(env)
    assert info['done_reason'] == 'terminal_brake_failure'
    assert len(rows) == physical
    assert env.command_steps == env.physics_steps
    assert reward == pytest.approx(sum(row['reward'] for row in rows) - 20., abs=1e-12)
    assert_complete_return(env, total)


def test_matching_existing_ppo_uses_gamma_one_without_optimizer_steps():
    cfg = objective.configure(configuration())
    env = environment(cfg=cfg)
    policy = coherent.make_agent(cfg, env)
    assert policy.gamma == 1. and policy.lmbda == 1.
    assert not policy.actor_optimizer.state and not policy.critic_optimizer.state
    assert policy.logged_steps == 0


def import_runner():
    directory = ROOT/'论文项目/revision_20260910_continued/root_cause_iteration_v1'
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import undiscounted_study
    return undiscounted_study


def test_runner_install_is_explicit_and_restores_without_registration():
    runner = import_runner()
    names = ('OUT', 'configuration', 'source_state', 'prepare', 'coherent', 'PERIODS')
    before = {name: getattr(runner.study, name) for name in names}
    with pytest.raises(RuntimeError, match='synthetic exit'):
        with runner.installed():
            cfg = runner.study.configuration('preview_time', 'adaptive', 901, period_ms=20)
            assert cfg['ppo']['gamma'] == 1.
            assert runner.study.coherent.CoherentDistanceEnvironment is objective.UndiscountedEnvironment
            assert runner.study.coherent.make_agent is runner.bounded.bounded.make_agent
            raise RuntimeError('synthetic exit')
    for name, value in before.items():
        assert getattr(runner.study, name) is value


def test_runner_real_128_decision_updates_and_resume_are_exact(tmp_path):
    import random

    runner = import_runner()
    before_counts = dict(COUNTS)
    with runner.installed():
        cfg = runner.study.configuration('preview_time', 'adaptive', 901, period_ms=20)
        cfg['revision']['training_path_ids'] = ['synthetic_objective_unit']
        cfg['environment']['max_steps'] = cfg['revision']['episode_max_steps'] = 87
        factory_class = runner.study.coherent.CoherentDistanceEnvironment
        agent_factory = runner.study.coherent.make_agent

        def env_factory(config, path_id):
            env = environment(factory_class, cfg=config)
            original_step = env.step

            def counted(action):
                physics, commands = env.physics_steps, env.command_steps
                result = original_step(action)
                count = env.physics_steps - physics
                COUNTS['macro_steps'] += 1
                COUNTS['physical_ticks'] += count
                COUNTS['tail_ticks'] += env.command_steps - commands - count
                return result

            env.step = counted
            return env

        def make(path):
            return full.FullBatchTrainer(cfg, path, env_factory, agent_factory,
                                         source_hash='synthetic-undiscounted-resume-unit')

        original = make(tmp_path/'continuous')
        original.advance(128)
        COUNTS['synthetic_ppo_updates'] += len(original.updates)
        COUNTS['synthetic_ppo_rows'] += original.agent.logged_steps
        expected = {name: copy.deepcopy(getattr(original.agent, name).state_dict()) for name in
                    ('actor', 'critic', 'actor_optimizer', 'critic_optimizer')}
        expected_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        continued = make(tmp_path/'resumed')
        continued.advance(17)
        continued = make(tmp_path/'resumed')
        continued.advance(67)
        continued = make(tmp_path/'resumed')
        continued.advance(128)
        COUNTS['synthetic_ppo_updates'] += len(continued.updates)
        COUNTS['synthetic_ppo_rows'] += continued.agent.logged_steps
        for name in expected:
            assert_tree_equal(expected[name], getattr(continued.agent, name).state_dict())
        assert_tree_equal(expected_rng, (torch.get_rng_state(), np.random.get_state(), random.getstate()))
        assert_tree_equal(original.updates, continued.updates)
        assert_tree_equal(original.episodes, continued.episodes)
        assert_tree_equal(original.env.trace, continued.env.trace)
        assert_tree_equal(original.buffer, continued.buffer)
        assert all(update['transition_count'] == 64 for update in original.updates)
        assert original.agent.gamma == continued.agent.gamma == 1.
    delta = {key: COUNTS[key] - before_counts[key] for key in COUNTS}
    print('\nSYNTHETIC_128_REFERENCE_AND_RESUME_COSTS=' + repr(delta))
