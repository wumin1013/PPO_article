"""Independent rest-window regressions, including hidden stopping spikes."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from terminal_metrics import evaluate_terminal_trajectory


LIMITS = dict(MAX_VEL=100.0, MAX_ACC=2000.0, MAX_JERK=62500.0,
              MAX_ANG_VEL=2*np.pi, MAX_ANG_ACC=10*np.pi, MAX_ANG_JERK=1000.0)


def audit(feed, *, dt=.001, times=None, **options):
    feed = np.asarray(feed, float)
    timestamps = np.arange(len(feed)) * dt if times is None else np.asarray(times, float)
    positions = np.c_[np.r_[0., np.cumsum(feed[1:] * np.diff(timestamps))], np.zeros(len(feed))]
    reference = [[0., 0.], [max(float(positions[-1, 0]), .001), 0.]]
    return evaluate_terminal_trajectory(reference, positions, timestamps,
        tolerance_mm=.01, limits=LIMITS, initial_heading=0.,
        initial_velocity=[feed[0], 0.], initial_acceleration=[0., 0.], **options)


def smooth_stop():
    # Four symmetric jerk phases: bounded acceleration, speed and terminal a=0.
    jerk = np.r_[np.full(10, 1000.), np.full(20, -1000.), np.full(10, 1000.)]
    acceleration = np.cumsum(jerk) * .001
    feed = np.cumsum(acceleration) * .001
    feed[np.abs(feed) < 1e-14] = 0.
    return np.r_[0., feed, np.zeros(21)]


def test_complete_bounded_stop_and_hold_passes():
    result = audit(smooth_stop())
    assert result['success']
    assert result['full_rest_satisfied']
    assert result['terminal_rest_window']['complete']
    assert result['terminal_rest_window']['sample_count'] == 21
    assert result['constraint_relative_tolerance'] == 1e-5
    assert result['terminal_axis_rest_diagnostic_only']
    assert not result['axis_constraints_implemented_by_evaluator']


def test_last_speed_zero_does_not_hide_nonzero_acceleration():
    result = audit([.0005] * 30 + [0.])
    assert result['terminal_feed_mm_s'] == 0.
    assert not result['full_rest_satisfied']
    assert 'rest_feed_acceleration_mm_s2' in result['rest_failure_reasons']


def test_twenty_millisecond_window_required():
    result = audit(np.zeros(20))
    assert not result['success']
    assert 'rest_window_incomplete' in result['rest_failure_reasons']
    assert audit(np.zeros(21))['full_rest_satisfied']


def test_full_window_not_only_last_sample():
    result = audit(np.r_[np.full(10, .002), np.zeros(15)])
    assert result['terminal_feed_mm_s'] == 0.
    assert not result['full_rest_satisfied']
    assert 'rest_feed_mm_s' in result['rest_failure_reasons']


def test_appending_rest_does_not_erase_full_path_jerk_violation():
    result = audit(np.r_[0., np.full(50, 5.), np.zeros(30)])
    assert result['full_rest_satisfied']
    assert not result['success']
    assert 'constraint_feed_jerk_mm_s3' in result['failure_reasons']


def test_angular_motion_inside_window_is_checked():
    t = np.arange(31) * .001
    heading = .01 * t
    increments = .0005 * .001 * np.c_[np.cos(heading[1:]), np.sin(heading[1:])]
    p = np.vstack(([0., 0.], np.cumsum(increments, axis=0)))
    result = evaluate_terminal_trajectory([[0., 0.], [.001, 0.]], p, t,
        tolerance_mm=.01, limits=LIMITS, initial_heading=0.,
        initial_velocity=[.0005, 0.], initial_acceleration=[0., 0.])
    assert not result['full_rest_satisfied']
    assert 'rest_angular_velocity_rad_s' in result['rest_failure_reasons']


def test_small_angular_velocity_does_not_hide_angular_acceleration():
    t = np.arange(31) * .001
    omega = np.where(np.arange(31) % 2, 5e-5, -5e-5)
    heading = np.cumsum(omega) * .001
    increments = .0005 * .001 * np.c_[np.cos(heading[1:]), np.sin(heading[1:])]
    p = np.vstack(([0., 0.], np.cumsum(increments, axis=0)))
    result = evaluate_terminal_trajectory([[0., 0.], [.001, 0.]], p, t,
        tolerance_mm=.01, limits=LIMITS, initial_heading=float(heading[0]),
        initial_velocity=[.0005*np.cos(heading[0]), .0005*np.sin(heading[0])],
        initial_acceleration=[0., 0.])
    assert result['terminal_rest_checks']['angular_velocity_rad_s']['satisfied']
    assert not result['full_rest_satisfied']
    assert 'rest_angular_acceleration_rad_s2' in result['rest_failure_reasons']


@pytest.mark.parametrize('timestamps', [[0., .001, .001], [0., .002, .001], [0., np.nan, .002]])
def test_bad_time_is_retained_as_failure(timestamps):
    result = audit([0., 0., 0.], times=timestamps)
    assert not result['valid_trace']
    assert not result['success']
    assert not result['full_rest_satisfied']
    assert 'invalid_trace' in result['failure_reasons']


def test_nan_position_is_retained_as_failure():
    result = audit([0., np.nan, 0.])
    assert not result['valid_trace']
    assert not result['full_rest_satisfied']


def test_nonuniform_window_includes_earlier_boundary_sample():
    result = audit(np.zeros(7), times=[0., .004, .009, .014, .019, .024, .027])
    assert result['full_rest_satisfied']
    assert result['terminal_rest_window']['start_timestamp_s'] == .004


def test_original_geometry_failure_not_overridden_by_rest():
    p = np.zeros((30, 2))
    result = evaluate_terminal_trajectory([[0., 0.], [10., 0.]], p, np.arange(30)*.001,
        tolerance_mm=.01, limits=LIMITS, initial_heading=0.,
        initial_velocity=[0., 0.], initial_acceleration=[0., 0.])
    assert result['full_rest_satisfied']
    assert not result['success']
    assert 'ordered_path_incomplete' in result['failure_reasons']


def test_rest_policy_rejects_unknown_or_nonpositive_fields():
    with pytest.raises(ValueError, match='Unknown'):
        audit(np.zeros(30), rest_policy={'window_ms': 20.})
    with pytest.raises(ValueError, match='positive'):
        audit(np.zeros(30), rest_policy={'window_s': 0.})
