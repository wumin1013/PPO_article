"""Numerical attribution must preserve endpoint alignment and original counts."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from preview_roundoff_diagnostic import angular_from_heading, angular_from_positions, stats


def test_three_differences_align_with_current_sample():
    dt = .001
    jerk = np.array([0., 1000., -500., 250., -100.])
    alpha = np.cumsum(jerk) * dt
    omega = np.cumsum(alpha) * dt
    heading = np.cumsum(omega) * dt
    result = angular_from_heading(heading, dt)
    np.testing.assert_allclose(result['omega'], omega, atol=1e-14)
    np.testing.assert_allclose(result['alpha'], alpha, atol=1e-12)
    np.testing.assert_allclose(result['jerk'][1:], jerk[1:], atol=1e-8)
    assert np.isnan(result['jerk'][0])


def test_coordinate_heading_perturbation_gets_differenced_three_times():
    dt = .001
    heading = np.arange(20, dtype=float) * .001
    perturbation = 1e-12 * (-1.) ** np.arange(20)
    original = angular_from_heading(heading, dt)
    perturbed = angular_from_heading(heading + perturbation, dt)
    error = angular_from_heading(perturbation, dt)
    np.testing.assert_allclose(perturbed['jerk'][1:] - original['jerk'][1:], error['jerk'][1:], atol=1e-8)
    np.testing.assert_allclose(np.abs(error['jerk'][3:]), .008, rtol=1e-12)


def test_position_heading_matches_saved_endpoint_convention():
    dt = .001
    theta = np.linspace(0., .04, 30)
    displacement = .005 * np.column_stack((np.cos(theta[1:]), np.sin(theta[1:])))
    positions = np.vstack((np.zeros(2), np.cumsum(displacement, axis=0)))
    result = angular_from_positions(positions, dt, theta[0])
    np.testing.assert_allclose(result['heading'], theta, atol=1e-14)
    assert stats(np.array([np.nan, 1000., 1000.005, 1000.02]), 1000., 1e-5) == {
        'peak_abs': 1000.02, 'violating_samples': 1, 'checked_samples': 3}
