"""Synthetic execution/provenance gates; never train or read held-out paths."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
import preview_restart_closed_loop as study


def row(path='mixed_lengths_validation', success=True, elapsed=3., model='m'):
    return dict(model_label=model, path_id=path, success=success, elapsed_time_s=elapsed)


def test_original_success_cannot_be_traded_for_success_elsewhere():
    before = [row('a'), row('b', False)]
    after = [row('a', False, .1), row('b')]
    gate = study.screening_gate(before, after)
    assert not gate['eligible_for_separate_training_screen']
    assert gate['original_success_regressions'] == [['m', 'a']]


def test_new_success_opens_training_screen_but_never_adoption():
    gate = study.screening_gate([row(success=False)], [row()])
    assert gate['eligible_for_separate_training_screen'] and not gate['adopted']
    assert gate['common_success_before_time_s'] is None


def test_time_compares_only_identical_fully_successful_pairs():
    gate = study.screening_gate([row('a'), row('b', False, 100.)],
        [row('a', elapsed=3.1), row('b', False, .1)])
    assert not gate['eligible_for_separate_training_screen']
    assert gate['common_success_after_time_s'] == 3.1
    assert study.screening_gate([row()], [row(elapsed=2.9)])['eligible_for_separate_training_screen']


@pytest.mark.parametrize('rows', [[row()], [row('a'), row('b'), row('c')],
    [row(study.PATH_IDS[0]), row(study.PATH_IDS[0]), row(study.PATH_IDS[2])]])
def test_partial_duplicate_or_wrong_path_sets_rejected(rows):
    with pytest.raises(ValueError):
        study.require_paths(rows)


def test_test_path_is_rejected_before_any_loader(monkeypatch):
    monkeypatch.setattr(study.old, 'points', lambda _: pytest.fail('Must not load test geometry'))
    with pytest.raises(ValueError):
        study.reference('dense_corners_holdout')


def test_correction_binding_is_restored_after_failure():
    original = study.cnc_env.apply_kinematic_constraints
    with pytest.raises(RuntimeError, match='test failure'):
        with study.corrected_execution():
            assert study.cnc_env.apply_kinematic_constraints is study.target.apply_target_tracking_constraints
            raise RuntimeError('test failure')
    assert study.cnc_env.apply_kinematic_constraints is original


def test_configuration_changes_only_execution_metadata(monkeypatch):
    monkeypatch.setattr(study.target, 'configuration', lambda *args: {'revision': {'feed_tracking': {'algorithm': 'target'}}})
    cfg = {'seed': 11, 'revision': {'adaptive_preview': True}, 'ppo': {'hidden_dim': 256}}
    result = study.evaluation_config(cfg, {'source': 'training'})
    assert result['revision'].pop('feed_tracking') == {'algorithm': 'target'}
    assert result['revision'].pop('evaluation_transfer') == {'source': 'training'}
    assert result == cfg


def test_changed_input_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    p = tmp_path/'model'
    p.write_text('original', encoding='utf-8')
    protocol = {'input_sha256': {'model': study.sha(p)}}
    study.verify_inputs(protocol)
    p.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Bound input changed'):
        study.verify_inputs(protocol)


def test_actual_trace_archive_contract(tmp_path):
    path = tmp_path/'trace.npz'
    study.old.save_trace(path, [dict(timestamp_s=0., x_mm=1., y_mm=2.),
                                dict(timestamp_s=.001, x_mm=1.001, y_mm=2.)])
    columns = study.load_trace(path)
    np.testing.assert_array_equal(columns['timestamp_s'], [0., .001])
    np.testing.assert_array_equal(columns['x_mm'], [1., 1.001])


def test_diagnostics_preserve_zero_heading_event_and_coverage_failure():
    t = np.arange(8)*.001
    p = np.array([[0., 0.], [.001, 0.], [.002, 0.], [.002, 0.],
        [.003, .0001], [.004, .0002], [.005, .0003], [.006, .0004]])
    cols = dict(timestamp_s=t, x_mm=p[:, 0], y_mm=p[:, 1], heading_rad=np.arange(8)*.01,
        **{'p4_status__v_intent': np.full(8, 1.)})
    report = study.diagnostics(cols, np.array([[0., 0.], [5., 0.]]))
    assert report['interior_zero_speed_samples'] == 1
    assert report['interior_zero_speed_times_s'] == [.003]
    assert report['first_coverage_omission'] is not None
    assert report['zero_speed_heading_windows'][0]['rows'][3]['output_angular_velocity_rad_s'] == 0.
