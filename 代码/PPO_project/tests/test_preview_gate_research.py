"""Gate-study semantics and an isolated train/evaluate/provenance smoke flow."""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_gate_research as study


def path_from_headings(headings):
    angles = np.deg2rad(headings)
    return np.vstack([[0., 0.], np.cumsum(.5*np.c_[np.cos(angles), np.sin(angles)], axis=0)])


def test_gate_geometry_preserves_oscillations_and_vetoes_fold():
    oscillatory = study.GateGeometry(path_from_headings([0., 60., -60., 60., -60., 0.]))
    geometry = oscillatory.at(0.)
    assert geometry['q'] == pytest.approx(1.)
    assert geometry['tangent_range_rad'] == pytest.approx(np.deg2rad(120.))
    assert geometry['gate'] == 1. and geometry['prior_mm'] == pytest.approx(4.)
    folded = study.GateGeometry(path_from_headings([0., 90., 180., 90.])).at(0.)
    assert folded['q'] > 0. and folded['gate'] == 0. and folded['prior_mm'] == 2.3
    assert oscillatory.at(100.)['prior_mm'] == 2.3


def test_features_are_orientation_and_collinear_subdivision_invariant():
    points = path_from_headings([0., 60., -60., 60., -60., 0.])
    original = study.GateGeometry(points).at(0.)
    rotation = np.array([[.6, -.8], [.8, .6]])
    transformed = study.GateGeometry(points@rotation.T+[10., -2.]).at(0.)
    finer = np.vstack([p for a, b in zip(points[:-1], points[1:]) for p in (a, (a+b)/2)]+[points[-1]])
    subdivided = study.GateGeometry(finer).at(0.)
    for key in original:
        assert transformed[key] == pytest.approx(original[key], abs=1e-12)
        assert subdivided[key] == pytest.approx(original[key], abs=1e-12)


def test_range_feature_is_normalized_but_not_clipped_at_one():
    feature = study.GateGeometry(path_from_headings([0., 90., 180., 270., 360.])).at(0.)
    assert feature['range_over_pi'] == pytest.approx(2.)
    assert feature['gate'] == 0.


@pytest.mark.parametrize('variant', study.PILOT_VARIANTS)
def test_every_variant_shares_twenty_state_inputs_and_training_changes(variant):
    cfg = study.config_for(variant, 101)
    env, observation = study.make_env(cfg, 'synthetic_test_only', reference=np.array([[0., 0.], [100., 0.]]))
    assert observation.shape == env.state.shape == env.observation_space.shape == (20,)
    assert env.observation_dim == 20 and env.raw.observation_dim == 18
    assert env.action_space_dim == (3 if variant == 'gate_residual' else 2)
    assert cfg['ppo']['actor_lr'] == 1e-5 and cfg['ppo']['lmbda'] == .999
    assert cfg['revision']['extra_observation_features'][1].endswith('no clipping')
    assert not env.raw.lookahead_control_enabled
    agent = study.make_agent(cfg, env)
    assert agent.active_action_dims == tuple(cfg['revision']['active_action_dims'])
    assert agent.actor.shared_fc1.in_features == agent.critic.fc1.in_features == 20
    with torch.no_grad():
        mean, std = agent.actor(torch.as_tensor(observation).unsqueeze(0))
    np.testing.assert_allclose(mean[0].numpy(), [0., .05, .5], atol=1e-7)
    np.testing.assert_allclose(std[0].numpy(), [.01, .01, .05], atol=1e-7)


class FakeRaw:
    observation_dim = 18
    action_space = gym.spaces.Box(np.array([-1., 0.], dtype=np.float32), np.ones(2, dtype=np.float32))
    observation_space = gym.spaces.Box(-10., 10., shape=(18,), dtype=np.float32)

    def __init__(self):
        self.current_position = np.zeros(2)
        self.lookahead_dist = 2.3
        self.state = np.zeros(18, dtype=np.float32)

    def _project_onto_progress_path(self, position):
        return None, None, position[0], None, None

    def _resolve_active_lookahead_dist(self):
        self._lookahead_dist_active = self.lookahead_dist

    def _build_state(self):
        return np.full(18, self._lookahead_dist_active, dtype=np.float32)


class FakeObjective:
    def __init__(self):
        self.env = FakeRaw()

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self):
        self.env.current_position[:] = 0.

    def step(self, action):
        self.env.received_action = np.asarray(action)
        self.env._resolve_active_lookahead_dist()
        self.env.current_position[0] += 1.
        self.env.state = self.env._build_state()
        return self.env.state.copy(), 1., False, dict(control_status={})


def timing_environment(variant):
    env = study.GatePreviewEnvironment(FakeObjective(), study.config_for(variant, 101), [[0., 0.], [100., 0.]])
    env.geometry = SimpleNamespace(at=lambda s: dict(q=s/10., range_over_pi=s/5.,
        prior_mm=2.3+s/10., gate=1., total_turn_rad=s, signed_turn_rad=s/2.))
    env.initialize_observation()
    return env


@pytest.mark.parametrize('preview_u,expected_distance', [(0., 2.), (.5, 2.3), (1., 2.6)])
def test_residual_mapping_and_control_next_observation_timing(preview_u, expected_distance):
    env = timing_environment('gate_residual')
    observation, _, _, info = env.step([0., .05, preview_u])
    np.testing.assert_array_equal(env.raw.received_action, [0., .05])
    np.testing.assert_allclose(observation[:18], expected_distance)
    np.testing.assert_allclose(observation[-2:], [.1, .2])
    status = info['control_status']
    assert status['gate_action_arc_mm'] == 0.
    assert status['gate_observation_arc_mm'] == 1.
    assert status['gate_prior_mm'] == 2.3
    assert status['gate_selected_mm'] == pytest.approx(expected_distance)
    assert status['gate_decision_us'] >= 0. and status['gate_observation_features_us'] >= 0.
    # The next action uses new geometry, without changing the last nextobs D.
    second = env.step([0., .05, .5])[0]
    np.testing.assert_allclose(second[:18], 2.4)
    env.reset()
    np.testing.assert_allclose(env.state[-2:], [0., 0.])
    np.testing.assert_allclose(env.state[:18], 2.3)


@pytest.mark.parametrize('variant,distance', [('fixed_short', 1.2), ('fixed_mid', 2.3),
                                              ('fixed_center', 3.), ('fixed_long', 4.)])
def test_fixed_distance_keeps_identical_extra_geometry_features(variant, distance):
    env = timing_environment(variant)
    observation = env.step([0., .05])[0]
    np.testing.assert_allclose(observation[:18], distance)
    np.testing.assert_allclose(observation[-2:], [.1, .2])
    assert env.action_space_dim == 2


def test_gate_only_disables_learned_residual_but_keeps_geometry_prior():
    env = timing_environment('gate_only')
    env.step([0., .05])
    observation, _, _, info = env.step([0., .05])
    np.testing.assert_allclose(observation[:18], 2.4)
    assert info['control_status']['gate_residual_mm'] == 0.
    with pytest.raises(ValueError, match='dimension'):
        env.step([0., .05, 1.])


def test_import_does_not_change_backend_adapters():
    before = {key: getattr(study.backend, key) for key in ('OUT', 'make_env', 'config_for', 'sample_row')}
    importlib.reload(study)
    assert before == {key: getattr(study.backend, key) for key in before}


def test_source_provenance_includes_study_v2_and_selector_implementation():
    source = study.source_state()['source_files']
    assert 'PPO_project/tools/preview_gate_research.py' in source
    assert 'PPO_project/tools/preview_research_v2.py' in source
    assert 'PPO_project/src/utils/ordered_coverage.py' in source


@pytest.fixture
def isolated_study(tmp_path, monkeypatch):
    data = tmp_path/'data'
    data.mkdir()
    fake_manifest = dict(paths=[dict(id=f'train_{i}', split='train') for i in range(3)]
        + [dict(id=f'validation_{i}', split='validation') for i in range(3)]
        + [dict(id='never_load_test_geometry', split='test')])
    study.backend.write_json(data/'manifest.json', fake_manifest)
    monkeypatch.setattr(study, 'DATA', data)
    monkeypatch.setattr(study, 'OUT', tmp_path/'study')
    monkeypatch.setattr(study, 'manifest', lambda: copy.deepcopy(fake_manifest))
    def fake_path(name):
        assert name != 'never_load_test_geometry', 'Test geometry must not be read'
        return np.array([[0., 0.], [100., 0.]])
    monkeypatch.setattr(study, 'path_points', fake_path)
    config = study.config_for
    def small_config(variant, seed):
        cfg = config(variant, seed)
        cfg['revision']['episode_max_steps'] = 8
        cfg['ppo'].update(hidden_dim=32, epochs=1)
        return cfg
    monkeypatch.setattr(study, 'config_for', small_config)
    monkeypatch.setattr(study, 'source_state', lambda: dict(source_sha256='isolated-gate-study', source_files={}))
    for key in ('OUT', 'DEFAULT_DATA_DIR', 'load_manifest', 'load_path', 'config_for',
                'make_env', 'make_agent', 'source_state', 'sample_row'):
        monkeypatch.setattr(study.backend, key, getattr(study.backend, key))
    study.activate_backend()
    study.prepare()
    return study


def test_short_training_resume_and_complete_validation_pipeline(isolated_study):
    runner = isolated_study
    run = runner.train('gate_residual', 7, 'smoke', 17)
    checkpoint = torch.load(run/'checkpoint.pt', weights_only=False)
    assert checkpoint['steps'] == 17
    assert checkpoint['actor']['shared_fc1.weight'].shape[1] == 20
    before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    runner.train('gate_residual', 7, 'smoke', 17)
    after = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
    assert before == after
    runner.train('gate_residual', 7, 'smoke', 25)
    checkpoint = torch.load(run/'checkpoint.pt', weights_only=False)
    assert checkpoint['steps'] == 25
    rows = runner.evaluate(run, 'validation')
    assert len(rows) == 3 and all(row['sample_count'] == 9 for row in rows)
    checked_run, checked_rows = runner._validated_run('smoke', 'gate_residual', 7, 25)
    assert checked_run == run and checked_rows == rows
    trace = np.load(run/'evaluation_validation_nominal/validation_0.npz')
    assert 'control_status__gate_decision_us' in trace['columns']
    assert 'control_status__gate_observation_q' in trace['columns']
    assert 'control_status__gate_observation_features_us' in trace['columns']
    # No freeze: rejection occurs before any held-out geometry is read.
    with pytest.raises(FileNotFoundError):
        runner.evaluate(run, 'test')
    with pytest.raises(ValueError, match='exact registered budget'):
        runner._validated_run('smoke', 'gate_residual', 7, 24)
    payload = json.loads((run/'evaluation_validation_nominal/summary.json').read_text(encoding='utf-8'))
    payload['rows'][0]['checkpoint_sha256'] = 'stale-checkpoint'
    runner.backend.write_json(run/'evaluation_validation_nominal/summary.json', payload)
    with pytest.raises(ValueError, match='stale'):
        runner._validated_run('smoke', 'gate_residual', 7, 25)


def test_protocol_change_blocks_training(isolated_study, monkeypatch):
    monkeypatch.setattr(isolated_study, 'source_state', lambda: dict(source_sha256='changed', source_files={}))
    with pytest.raises(ValueError, match='Protocol/source'):
        isolated_study.train('gate_only', 101, 'pilot', 16)


def test_registered_training_budgets_and_selection_requirements(isolated_study):
    with pytest.raises(ValueError, match='violates'):
        isolated_study.train('fixed_mid', 7, 'pilot', 32)
    with pytest.raises(ValueError, match='violates'):
        isolated_study.train('gate_only', 101, 'pilot', study.PILOT_STEPS+1)
    with pytest.raises(ValueError, match='Select'):
        isolated_study.train('gate_only', 11, 'formal', 32)


def test_select_uses_all_registered_pilots_and_preserves_selection(isolated_study, monkeypatch):
    called = []
    def validated(stage, variant, seed, budget):
        called.append((stage, variant, seed, budget))
        # Center is the strongest fixed pilot; gate successes do not select D.
        success = variant in ('fixed_center', 'gate_only', 'gate_residual')
        rows = [dict(success=success, path_completed=success, geometry_feasible=True,
                     ordered_progress=1. if success else .3, elapsed_time_s=3.) for _ in range(3)]
        return Path('unused'), rows
    monkeypatch.setattr(isolated_study, '_validated_run', validated)
    selection = isolated_study.select()
    value = json.loads(selection.read_text(encoding='utf-8'))
    assert value['selected_fixed'] == 'fixed_center' and value['fixed_distance_mm'] == 3.
    assert len(called) == len(study.PILOT_VARIANTS)
    assert all(c[-1] == study.PILOT_STEPS for c in called)
    assert isolated_study.config_for('selected_fixed', 11)['revision']['fixed_distance_mm'] == 3.
    with pytest.raises(FileExistsError):
        isolated_study.select()


def test_freeze_refuses_incomplete_formal_run(isolated_study, monkeypatch):
    isolated_study.backend.write_json(isolated_study.OUT/'selection.json', dict(
        source_sha256='isolated-gate-study', protocol_sha256=study.backend.sha(study.OUT/'protocol.json')))
    monkeypatch.setattr(isolated_study, '_validated_run', lambda *args: (_ for _ in ()).throw(ValueError('Incomplete budget')))
    with pytest.raises(ValueError, match='Incomplete'):
        isolated_study.freeze()
    assert not (isolated_study.OUT/'freeze.json').exists()


def test_freeze_collects_exactly_all_fifteen_registered_models(isolated_study, monkeypatch):
    runner = isolated_study
    runner.backend.write_json(runner.OUT/'selection.json', dict(selected_fixed='fixed_center', fixed_distance_mm=3.,
        source_sha256='isolated-gate-study', protocol_sha256=runner.backend.sha(runner.OUT/'protocol.json')))
    calls = []
    def validated(stage, variant, seed, budget):
        calls.append((stage, variant, seed, budget))
        run = runner.OUT/stage/f'{variant}_seed{seed}'
        run.mkdir(parents=True)
        (run/'checkpoint.pt').write_bytes(f'model {variant} {seed}'.encode())
        (run/'config.yaml').write_text(f'variant: {variant}\nseed: {seed}\n', encoding='utf-8')
        return run, [dict(success=True, path_completed=True, geometry_feasible=True,
                          ordered_progress=1., elapsed_time_s=4.) for _ in range(3)]
    monkeypatch.setattr(runner, '_validated_run', validated)
    frozen_path = runner.freeze()
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    assert len(frozen['models']) == len(frozen['configs']) == 15
    assert len(calls) == 15 and all(c[0] == 'formal' and c[3] == 131072 for c in calls)
    assert {c[1] for c in calls} == set(study.FORMAL_VARIANTS)
    assert {c[2] for c in calls} == set(study.SEEDS)
    assert frozen['primary_adaptive_candidate'] == 'gate_residual'
    assert frozen['rule_only_control'] == 'gate_only'
    assert frozen['test_results_used_for_selection'] is False
    with pytest.raises(FileExistsError):
        runner.freeze()
