"""Correctness checks for the versioned baseline retimer, not performance tests."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from two_step_corrected_retiming import (LIMITS, peaks_and_ratios, retime_fixed_path,
                                         sampled_quintic_commands)
from terminal_metrics import evaluate_terminal_trajectory


def test_straight_retimer_meets_full_output_and_stop_audit():
    reference = np.array([[0., 0.], [16., 0.]])
    plan, attempts = retime_fixed_path(reference)
    assert plan is not None
    positions, times, arc, initial, kin = plan
    assert np.all(np.diff(arc) >= 0)
    assert np.array_equal(positions[0], reference[0])
    assert np.array_equal(positions[-1], reference[-1])
    assert np.all(kin["feed_mm_s"][-21:] == 0)
    result = evaluate_terminal_trajectory(reference, positions, times, tolerance_mm=1.25,
        limits=LIMITS, witness_spacing_mm=.01, **initial)
    assert result["success"], result["failure_reasons"]
    assert all(attempt["scalar_satisfied"] for attempt in attempts)


def test_discrete_motion_uses_actual_positions_and_no_endpoint_truncation():
    reference = np.array([[0., 0.], [16., 0.]])
    positions, times, arc, initial = sampled_quintic_commands(reference, 400)
    kin, _, _ = peaks_and_ratios(positions, times, initial, LIMITS)
    expected_v = np.diff(arc) / np.diff(times)
    np.testing.assert_allclose(kin["feed_mm_s"][1:], expected_v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(kin["feed_acceleration_mm_s2"][1:], np.diff(kin["feed_mm_s"]) / np.diff(times))
    assert arc[-26] == 16. and arc[-27] < 16.
    assert kin["feed_mm_s"][-26] > 0.  # Real final moving sample, not a zero-speed snap.
    assert np.all(kin["feed_mm_s"][-25:] == 0.)


def test_polyline_corner_is_not_declared_angular_feasible_by_time_scaling():
    path = np.array([[0., 0.], [8., 0.], [8., 8.]])
    p, t, _, initial = sampled_quintic_commands(path, 2000)
    _, _, ratios = peaks_and_ratios(p, t, initial, LIMITS)
    assert ratios["angular_velocity_rad_s"] > 1.
    assert ratios["angular_acceleration_rad_s2"] > 1.
    assert ratios["angular_jerk_rad_s3"] > 1.


@pytest.mark.parametrize("points", [np.array([[0., 0.]]), np.array([[0., 0.], [0., 0.]]), np.array([[0., 0.], [np.nan, 0.]])])
def test_invalid_fixed_geometry_is_rejected(points):
    with pytest.raises(ValueError):
        sampled_quintic_commands(points, 400)
