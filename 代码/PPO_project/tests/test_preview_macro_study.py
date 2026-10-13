"""Macro execution, full-command rewards and checkpoint recovery contracts."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_macro_study as study

FROZEN = [ROOT/'PPO_project/tools'/name for name in
          ('terminal_metrics.py', 'terminal_braking.py', 'preview_macro_ppo.py')]
FROZEN_HASHES = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FROZEN}


@pytest.fixture(autouse=True)
def one_thread_and_frozen_sources():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FROZEN} == FROZEN_HASHES


def line_env(candidate='preview_time', method='adaptive', length=4.):
    cfg = study.configuration(candidate, method, 101)
    return study.MacroEnvironment(cfg, 'unit_line', reference=np.array([[0., 0.], [length, 0.]]))


@pytest.mark.parametrize('candidate', study.CANDIDATES)
@pytest.mark.parametrize('method', study.METHODS)
def test_fixed_and_adaptive_share_registered_motion_and_training_changes(candidate, method):
    cfg = study.configuration(candidate, method, 101)
    control = study.configuration(candidate, 'adaptive', 101)
    assert cfg['ppo'] == control['ppo']
    assert cfg['kinematic_constraints'] == control['kinematic_constraints']
    assert cfg['environment'] == control['environment']
    assert cfg['revision']['macro_period_steps'] == 20
    assert cfg['revision']['motion_anchor_beta'] == 0.
    assert cfg['ppo']['gamma'] == pytest.approx(math.exp(-.020/10))
    expected = ([0] if candidate == 'joint_tracking' else []) + [1] + ([2] if method == 'adaptive' else [])
    assert cfg['revision']['active_action_dims'] == expected
    assert cfg['revision']['fixed_distance_mm'] == study.DISTANCES[method]
    assert cfg['revision']['training_path_ids'] == control['revision']['training_path_ids']


@pytest.mark.parametrize('candidate,method', [('joint_tracking','adaptive'), ('preview_tracking','adaptive'),
                                             ('joint_tracking','fixed1p2'), ('preview_time','fixed4')])
def test_one_latent_draw_is_held_for_twenty_actual_kcm_ticks(candidate, method, monkeypatch):
    env = line_env(candidate, method, length=20.)
    agent = study.make_agent(env.cfg, env)
    torch.manual_seed(71)
    action, choice = agent.take_action(env.state, return_info=True)
    distribution = torch.distributions.Normal(torch.tensor(choice['mu']), torch.tensor(choice['std']))
    expected_logp = float((distribution.log_prob(torch.tensor(action))*torch.tensor(choice['action_mask'])).sum())
    assert choice['old_log_prob'] == pytest.approx(expected_logp, abs=2e-6)
    calls = []
    original_step = env.inner.step

    def capture(values):
        calls.append(np.asarray(values).copy())
        return original_step(values)

    monkeypatch.setattr(env.inner, 'step', capture)
    _, reward, done, _ = env.step(action)
    assert not done
    assert env.physics_steps == env.command_steps == 20
    assert len(calls) == 20
    # Scored coordinates preserve the original latent exactly. Inactive fixed
    # preview is a physical configuration constant, represented in float64 by
    # the environment rather than the policy sampler's float32 return array.
    executed = np.asarray(action, float)
    if method != 'adaptive':
        executed[2] = (study.DISTANCES[method]-.8)/3.8
    np.testing.assert_array_equal(calls, np.tile(executed, (20, 1)))
    np.testing.assert_allclose(np.diff([row['timestamp_s'] for row in env.trace]), .001, atol=1e-15)
    assert sum(row.get('macro_decision', 0.) for row in env.trace) == 1
    assert sum(row.get('preview_event', 0.) for row in env.trace) == int(method == 'adaptive')
    expected_reward = sum(math.exp(-.001*k/10)*row['reward'] for k,row in enumerate(env.trace[1:]))
    assert reward == pytest.approx(expected_reward, abs=1e-12)
    if method != 'adaptive':
        assert all(row['lookahead_mm'] == pytest.approx(study.DISTANCES[method]) for row in env.trace)


def finish_line(env, *, corrupt_prefix=False):
    if corrupt_prefix:
        env.step([0., .05, .5])
        env.trace[5]['y_mm'] += 100.
    for _ in range(100):
        before = len(env.trace)
        physics_before = env.physics_steps
        _, reward, done, info = env.step([0., .05, .5])
        if done:
            return reward, info, env.trace[before:], env.physics_steps-physics_before
    pytest.fail('Small straight reference did not terminate in 100 macro steps')


@pytest.mark.parametrize('candidate,method', [('preview_time','adaptive'), ('preview_tracking','fixed3')])
def test_success_requires_complete_brake_and_reward_counts_every_tail_tick(candidate, method):
    env = line_env(candidate, method)
    reward, info, last_rows, physical_count = finish_line(env)
    assert env.result['success'] and env.result['full_rest_satisfied']
    assert env.result['internal_terminal_rest']
    assert info['done_reason'] == 'success'
    assert env.command_steps > env.physics_steps
    assert env.result['elapsed_time_s'] == pytest.approx(env.command_steps*.001)
    assert env.last_macro_duration_s == pytest.approx(len(last_rows)*.001)
    assert all(check['satisfied'] for check in env.result['constraints'].values())
    expected = sum(math.exp(-.001*k/10)*row['reward'] for k,row in enumerate(last_rows[:physical_count]))
    expected -= sum(.001*math.exp(-.001*k/10) for k in range(physical_count, len(last_rows)))
    expected += 10.*math.exp(-.001*len(last_rows)/10)
    assert reward == pytest.approx(expected, abs=1e-11)
    assert not any(row.get('terminal_braking') for row in last_rows[:physical_count])
    assert all(row['terminal_braking'] == 1. for row in last_rows[physical_count:])
    with pytest.raises(RuntimeError, match='Completed'):
        env.step([0., .05, .5])


def test_invalid_prefix_cannot_receive_success_bonus_after_good_braking():
    env = line_env()
    reward, info, last_rows, physical_count = finish_line(env, corrupt_prefix=True)
    assert env.result['full_rest_satisfied']
    assert not env.result['success']
    assert any(reason.startswith('constraint_') for reason in env.result['failure_reasons'])
    assert 'contour_tolerance_exceeded' in env.result['failure_reasons']
    assert info['done_reason'] == 'terminal_audit_failure'
    expected = sum(math.exp(-.001*k/10)*row['reward'] for k,row in enumerate(last_rows[:physical_count]))
    expected -= sum(.001*math.exp(-.001*k/10) for k in range(physical_count,len(last_rows)))
    expected -= 20.*math.exp(-.001*len(last_rows)/10)
    assert reward == pytest.approx(expected, abs=1e-11)


def test_failed_brake_is_retained_as_failure(monkeypatch):
    env = line_env()

    def unavailable(*args, **kwargs):
        raise ValueError('fixture infeasible braking state')

    monkeypatch.setattr(study, 'plan_terminal_brake', unavailable)
    _, info, last_rows, physical_count = finish_line(env)
    assert not env.result['success']
    assert info['done_reason'] == 'terminal_brake_failure'
    assert info['brake_failure'] == 'fixture infeasible braking state'
    assert len(last_rows) == physical_count
    assert env.command_steps == env.physics_steps


class TinyEpisode:
    """One-macro terminal fixture isolates checkpoint transactions, not motion."""
    def __init__(self, cfg, path_id):
        self.state = np.zeros(4, np.float32)
        self.observation_dim = 4
        self.done = False
        self.physics_steps = self.command_steps = 0
        self.result = None
        self.trace = [{'timestamp_s': 0., 'x_mm': 0., 'y_mm': 0.}]

    def step(self, action):
        if self.done:
            raise RuntimeError('Completed macro episode')
        self.state = np.asarray(action.tolist()+[1.], np.float32)
        self.physics_steps, self.command_steps = 1, 24
        self.done = True
        self.result = {'success': True}
        self.trace.append({'timestamp_s': .024, 'x_mm': .001, 'y_mm': 0.})
        return self.state.copy(), .1, True, {'done_reason': 'success'}


def install_tiny_training(monkeypatch, destination):
    original_configuration = study.configuration

    def small_config(candidate, method, seed):
        cfg = original_configuration(candidate, method, seed)
        cfg['ppo'].update(hidden_dim=8, epochs=1)
        cfg['revision']['training_path_ids'] = ['tiny_a', 'tiny_b']
        return cfg

    def protocol():
        value = {'source': {'source_sha256': 'test-fixture', 'source_files': {}}}
        study.old.write(destination/'protocol.json', value)
        return value

    monkeypatch.setattr(study, 'OUT', destination)
    monkeypatch.setattr(study, 'configuration', small_config)
    monkeypatch.setattr(study, 'prepare', protocol)
    monkeypatch.setattr(study, 'MacroEnvironment', TinyEpisode)


@pytest.mark.parametrize('crash_decision', [1, 3])
def test_resume_after_checkpoint_commit_finishes_episode_and_budget_outputs(monkeypatch, tmp_path, crash_decision):
    destination = tmp_path/'study'
    install_tiny_training(monkeypatch, destination)
    original_replace = Path.replace
    crashed = False

    def crash_after_committed_checkpoint(path, target):
        nonlocal crashed
        result = original_replace(path, target)
        if path.name == 'checkpoint.tmp' and not crashed:
            checkpoint = torch.load(target, map_location='cpu', weights_only=False)
            if checkpoint['decisions'] == crash_decision:
                crashed = True
                raise RuntimeError('simulated process loss after checkpoint commit')
        return result

    monkeypatch.setattr(Path, 'replace', crash_after_committed_checkpoint)
    with pytest.raises(RuntimeError, match='simulated process loss'):
        study.train('preview_time', 'adaptive', 913, 3, 'smoke')
    assert crashed
    monkeypatch.setattr(Path, 'replace', original_replace)
    run = study.train('preview_time', 'adaptive', 913, 3, 'smoke')
    checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['episode'] == 3
    assert checkpoint['episode_actions'] == []
    assert checkpoint['completed_episode_physics_steps'] == 3
    assert checkpoint['completed_episode_command_steps'] == 72
    assert (run/'training_final_partial.npz').exists()
    summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    assert summary['decisions'] == summary['episodes'] == 3
    assert summary['physics_steps'] == 3 and summary['command_steps'] == 72
    assert summary['checkpoint_sha256'] == study.old.sha(run/'checkpoint.pt')
    episodes = [json.loads(row) for row in (run/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['episode'] for row in episodes] == [0, 1, 2]
    updates = [json.loads(row) for row in (run/'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['decisions'] for row in updates] == [1, 2, 3]
    assert [row['step'] for row in updates] == [1, 2, 3]


def validation_fixture(monkeypatch, tmp_path):
    """Use tiny deterministic records to test storage binding, not performance."""
    run = tmp_path/'run'
    run.mkdir()
    cfg = study.configuration('preview_time', 'adaptive', 917)
    (run/'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    source = {'source_sha256': 'validation-fixture', 'source_files': {}}
    study.old.write(run/'provenance.json', dict(source_sha256=source['source_sha256'],
                                               config_sha256=study.old.sha(run/'config.yaml')))
    actor = torch.nn.Linear(4, 3)
    torch.save(dict(actor=actor.state_dict(), decisions=1), run/'checkpoint.pt')
    data = tmp_path/'data'
    data.mkdir()
    paths = {'paths': [{'id': name, 'split': 'validation'} for name in ('cache_a', 'cache_b')]}
    study.old.write(data/'manifest.json', paths)
    monkeypatch.setattr(study.old, 'DATA', data)
    monkeypatch.setattr(study.old, 'manifest', lambda: paths)
    monkeypatch.setattr(study, 'prepare', lambda: {'source': source})
    calls = []

    class CacheEpisode:
        def __init__(self, config, path_id):
            calls.append(path_id)
            self.state = np.zeros(4, np.float32)
            self.done = False
            self.trace = [dict(timestamp_s=0., x_mm=0., y_mm=0., v_mm_s=0., lookahead_mm=3.)]
            self.result = None

        def step(self, action):
            self.done = True
            self.trace.append(dict(timestamp_s=.024, x_mm=.001, y_mm=0., v_mm_s=0., lookahead_mm=3.))
            self.result = dict(success=True, path_completed=True, elapsed_time_s=.024, failure_reasons=[])

    class CacheAgent:
        def __init__(self):
            self.actor = torch.nn.Linear(4, 3)

        def take_action(self, state, deterministic=False):
            assert deterministic
            return np.array([0., .05, .5])

    monkeypatch.setattr(study, 'MacroEnvironment', CacheEpisode)
    monkeypatch.setattr(study, 'make_agent', lambda cfg, env: CacheAgent())
    return run, CacheEpisode, calls


def create_partial_validation(monkeypatch, run, environment):
    def interrupt_second_path(cfg, path_id):
        if path_id == 'cache_b':
            raise RuntimeError('fixture validation interruption')
        return environment(cfg, path_id)

    monkeypatch.setattr(study, 'MacroEnvironment', interrupt_second_path)
    with pytest.raises(RuntimeError, match='fixture validation interruption'):
        study.evaluate(run)
    assert (run/'validation/cache_a.json').exists()
    assert not (run/'validation/summary.json').exists()
    monkeypatch.setattr(study, 'MacroEnvironment', environment)


def test_partial_validation_reuses_bound_complete_path_without_reexecution(monkeypatch, tmp_path):
    run, environment, calls = validation_fixture(monkeypatch, tmp_path)
    create_partial_validation(monkeypatch, run, environment)
    prior = {name: (run/'validation'/name).read_bytes() for name in ('binding.json','cache_a.json','cache_a.npz')}
    calls.clear()
    rows = study.evaluate(run)
    assert calls == ['cache_b']
    assert [row['path_id'] for row in rows] == ['cache_a', 'cache_b']
    assert {name: (run/'validation'/name).read_bytes() for name in prior} == prior
    assert (run/'validation/summary.json').exists()
    with pytest.raises(FileExistsError, match='completed validation'):
        study.evaluate(run)


def test_partial_validation_rejects_changed_checkpoint_without_overwrite(monkeypatch, tmp_path):
    run, environment, calls = validation_fixture(monkeypatch, tmp_path)
    create_partial_validation(monkeypatch, run, environment)
    prior = (run/'validation/cache_a.npz').read_bytes()
    checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    checkpoint['actor']['weight'] += 1.
    torch.save(checkpoint, run/'checkpoint.pt')
    calls.clear()
    with pytest.raises(ValueError, match='different model/config/source'):
        study.evaluate(run)
    assert calls == []
    assert (run/'validation/cache_a.npz').read_bytes() == prior


@pytest.mark.parametrize('alteration', ['trace', 'row_hash'])
def test_partial_validation_rejects_changed_complete_cache(monkeypatch, tmp_path, alteration):
    run, environment, calls = validation_fixture(monkeypatch, tmp_path)
    create_partial_validation(monkeypatch, run, environment)
    if alteration == 'trace':
        path = run/'validation/cache_a.npz'
        path.write_bytes(path.read_bytes()+b'fixture corruption')
    else:
        path = run/'validation/cache_a.json'
        row = study.old.read(path)
        row['source_sha256'] = 'changed fixture source'
        study.old.write(path, row)
    damaged = path.read_bytes()
    calls.clear()
    with pytest.raises(ValueError, match='Cached validation row or trace changed'):
        study.evaluate(run)
    assert calls == []
    assert path.read_bytes() == damaged


def test_orphan_trace_is_kept_and_new_output_is_independent(monkeypatch, tmp_path):
    run, _, calls = validation_fixture(monkeypatch, tmp_path)
    output = run/'validation'
    output.mkdir()
    orphan = output/'cache_a.npz'
    orphan.write_bytes(b'uncommitted fixture trace')
    with pytest.raises(FileExistsError, match='orphaned partial trace'):
        study.evaluate(run)
    assert calls == []
    alternate = tmp_path/'new_validation_attempt'
    rows = study.evaluate(run, output=alternate)
    assert len(rows) == 2
    assert (alternate/'binding.json').exists()
    assert (alternate/'summary.json').exists()
    assert orphan.read_bytes() == b'uncommitted fixture trace'
