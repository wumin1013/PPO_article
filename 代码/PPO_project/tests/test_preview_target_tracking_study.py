"""Single-change integration, namespace isolation and real smoke recovery."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_target_tracking_study as target


@pytest.fixture(autouse=True)
def isolated_threads_and_bindings():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    target.uninstall(); torch.set_num_threads(threads)


def arguments(v=0., a=0., target_feed=5., omega=.2, alpha=0., target_omega=-.3):
    return (v, a, omega, alpha, target_feed, target_omega,
            .001, 100., 2000., 62500., 2*np.pi, 10*np.pi, 1000.)


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor): assert torch.equal(first, second)
    elif isinstance(first, np.ndarray): np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second): assert_tree_equal(a, b)
    else: assert first == second


def test_same_input_angular_projection_exact_and_feed_anticipates_release():
    rng = np.random.default_rng(51)
    for _ in range(100):
        args = arguments(v=rng.uniform(1, 99), target_feed=rng.uniform(0, 100),
                         omega=rng.uniform(-2, 2), target_omega=rng.uniform(-5, 5))
        assert target.apply_target_tracking_constraints(*args)[3:] == target.ORIGINAL_APPLY(*args)[3:]
    old_v = new_v = old_a = new_a = 0.
    old, new = [], []
    for _ in range(100):
        old_v, old_a, *_ = target.ORIGINAL_APPLY(*arguments(old_v, old_a))
        new_v, new_a, *_ = target.apply_target_tracking_constraints(*arguments(new_v, new_a))
        old.append(old_v); new.append(new_v)
    assert max(old) > 5.1
    assert max(new) <= 5.+1e-8
    assert new[-1] == pytest.approx(5., abs=1e-8)


def test_scoped_install_is_idempotent_and_restores_every_binding_on_error():
    original_core = target.full.core.MacroEnvironment
    original_brake = target.full.core.plan_terminal_brake
    with pytest.raises(RuntimeError, match='probe'):
        with target.installed():
            assert target.install() is target.full
            with target.installed():
                assert target.cnc_env.apply_kinematic_constraints is target.apply_target_tracking_constraints
            assert target.cnc_env.apply_kinematic_constraints is target.apply_target_tracking_constraints
            assert target.full.core.plan_terminal_brake is original_brake
            assert target.kinematics.apply_kinematic_constraints is target.ORIGINAL_APPLY
            assert target.full.MILESTONES == (0, 256, 512, 1024, 2048, 4096)
            raise RuntimeError('probe')
    assert target.full.OUT == target.BASE_OUT
    assert target.full.core.MacroEnvironment is original_core
    assert target.cnc_env.apply_kinematic_constraints is target.ORIGINAL_APPLY


def test_refuses_to_chain_on_another_candidate(monkeypatch):
    monkeypatch.setattr(target.cnc_env, 'apply_kinematic_constraints', lambda *args: None)
    with pytest.raises(RuntimeError, match='fresh process'):
        target.install()


@pytest.mark.parametrize('method', target.METHODS)
def test_feed_wrapper_is_only_config_change_and_initializers_match(method):
    original = target.BASE_CONFIG('preview_time', method, 101)
    changed = target.configuration('preview_time', method, 101)
    for key in original:
        if key != 'revision': assert original[key] == changed[key]
    for key in ('init_checkpoint', 'init_checkpoint_sha256', 'corner_observation',
                'feed_logit_std', 'training_batch_macro_decisions', 'fixed_distance_mm',
                'active_action_dims', 'terminal_definition', 'validation_milestones'):
        assert original['revision'][key] == changed['revision'][key]


def test_new_source_extends_parent_without_changing_frozen_source_members():
    original = target.BASE_SOURCE()['source_files']
    changed = target.source_state()['source_files']
    assert all(changed[name] == digest for name, digest in original.items())
    assert set(changed)-set(original) == {
        'PPO_project/tools/preview_target_tracking.py',
        'PPO_project/tools/preview_target_tracking_study.py'}


def test_protocol_binds_parent_and_rejects_source_change(tmp_path, monkeypatch):
    monkeypatch.setattr(target, 'OUT', tmp_path/'candidate')
    first = target.prepare()
    assert first['parent_protocol_sha256'] == target.full.core.old.sha(target.BASE_OUT/'protocol.json')
    assert first['terminal_brake_version'] == 'Original v2; v3 is not installed'
    altered = copy.deepcopy(first['source']); altered['source_sha256'] = 'different'
    monkeypatch.setattr(target, 'source_state', lambda: altered)
    with pytest.raises(ValueError, match='registered target-tracking'):
        target.prepare()


def test_real_seed901_smoke64_and_interrupted_resume_match(tmp_path):
    with target.installed():
        cfg = target.configuration('preview_time', 'adaptive', 901)
        source_hash = target.source_state()['source_sha256']
        first = target.full.FullBatchTrainer(cfg, tmp_path/'whole', source_hash=source_hash)
        first.advance(64)
        expected = {name: copy.deepcopy(getattr(first.agent, name).state_dict()) for name in
                    ('actor', 'critic', 'actor_optimizer', 'critic_optimizer')}
        expected_rng = torch.get_rng_state()
        second = target.full.FullBatchTrainer(cfg, tmp_path/'resumed', source_hash=source_hash)
        second.advance(17)
        assert second.agent.logged_steps == 0 and len(second.buffer['states']) == 17
        second = target.full.FullBatchTrainer(cfg, tmp_path/'resumed', source_hash=source_hash)
        second.advance(64)
        for name in expected: assert_tree_equal(expected[name], getattr(second.agent, name).state_dict())
        assert_tree_equal(first.env.state, second.env.state)
        assert_tree_equal(first.updates, second.updates)
        assert torch.equal(expected_rng, torch.get_rng_state())
        assert first.updates[0]['transition_count'] == 64
        assert first.updates[0]['old_log_probability_max_absolute_error'] < 1e-4


def test_original_cli_dispatch_writes_new_protocol_bound_snapshot(tmp_path, monkeypatch):
    output = tmp_path/'candidate'
    monkeypatch.setattr(target, 'OUT', output)
    monkeypatch.setattr(sys, 'argv', ['preview_target_tracking_study.py', 'train',
        '--method', 'adaptive', '--seed', '901', '--stage', 'smoke', '--decisions', '0'])
    target.main()
    snapshot = output/'smoke/preview_time_adaptive_seed901/checkpoints/step0'
    summary = target.full.core.old.read(snapshot/'training_summary.json')
    provenance = target.full.core.old.read(snapshot/'provenance.json')
    assert summary['decisions'] == 0 and summary['optimizer_updates'] == 0
    assert provenance['protocol_sha256'] == target.full.core.old.sha(output/'protocol.json')
    assert provenance['source_sha256'] == target.source_state()['source_sha256']
    manifest = target.full.core.old.read(snapshot/'snapshot_manifest.json')
    assert all(target.full.core.old.sha(snapshot/name) == digest for name, digest in manifest['files'].items())
    assert target.full.OUT == target.BASE_OUT
    assert target.cnc_env.apply_kinematic_constraints is target.ORIGINAL_APPLY
