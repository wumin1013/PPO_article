from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT)]
from preview_target_tracking import project_feed_target


PARAMETERS = dict(dt=.001, max_vel=100., max_acc=2000., max_jerk=62500.)


def rollout(targets, velocity=0., acceleration=0.):
    values = [(velocity, acceleration, 0.)]
    for target in targets:
        velocity, acceleration, jerk = project_feed_target(
            velocity, acceleration, float(target), **PARAMETERS)
        values.append((velocity, acceleration, jerk))
    return np.asarray(values)


def audit_actual_output(values):
    v, a, j = values.T
    measured_a = np.r_[a[0], np.diff(v)/PARAMETERS['dt']]
    measured_j = np.diff(measured_a)/PARAMETERS['dt']
    np.testing.assert_allclose(measured_a, a, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(measured_j, j[1:], atol=1e-8, rtol=1e-11)
    assert np.min(v) >= 0
    assert np.max(v) <= PARAMETERS['max_vel']*(1+1e-9)
    assert np.max(np.abs(measured_a)) <= PARAMETERS['max_acc']*(1+1e-9)
    assert np.max(np.abs(measured_j)) <= PARAMETERS['max_jerk']*(1+1e-9)


@pytest.mark.parametrize('initial,target', [(0., 5.), (0., 100.), (100., 5.),
    (10., 3.7464523), (5., 0.), (100., 0.), (5., 100.), (5., 5.)])
def test_constant_target_settles_without_crossing_when_initially_viable(initial, target):
    values = rollout(np.full(500, target), initial)
    audit_actual_output(values)
    lo, hi = sorted((initial, target))
    assert np.min(values[:, 0]) >= lo-1e-8
    assert np.max(values[:, 0]) <= hi+1e-8
    assert np.all(np.abs(values[-100:, 0]-target) < 1e-8)
    assert np.all(np.abs(values[-100:, 1]) < 1e-8)


@pytest.mark.parametrize('initial,acc,target', [(10., 500., 10.), (10., -500., 10.),
                                              (20., -1000., 19.)])
def test_unreachable_new_target_releases_acceleration_without_fake_no_overshoot(initial, acc, target):
    values = rollout(np.full(500, target), initial, acc)
    audit_actual_output(values)
    first = values[1]
    assert first[1] == pytest.approx(acc-np.sign(acc)*PARAMETERS['max_jerk']*PARAMETERS['dt'])
    assert np.sign(first[0]-initial) == np.sign(acc)
    assert values[-1, 0] == pytest.approx(target, abs=1e-8)
    assert values[-1, 1] == pytest.approx(0, abs=1e-8)


def test_changing_targets_preserve_actual_motion_bounds_and_eventual_rest():
    targets = np.repeat([5., 50., 3., 100., 0., 30., 0.], 200)
    values = rollout(targets)
    audit_actual_output(values)
    assert np.all(np.abs(values[-20:, :2]) < 1e-8)


def test_fast_target_changes_do_not_replace_the_hard_viability_projection():
    rng = np.random.default_rng(731)
    targets = np.r_[rng.uniform(0, 100, 1500), np.zeros(500)]
    values = rollout(targets)
    audit_actual_output(values)
    assert np.all(np.abs(values[-20:, :2]) < 1e-8)


@pytest.mark.parametrize('velocity,acceleration', [(-1., 0.), (101., 0.), (0., -400.), (100., 400.)])
def test_invalid_or_nonviable_state_is_rejected(velocity, acceleration):
    with pytest.raises(ValueError):
        project_feed_target(velocity, acceleration, 5., **PARAMETERS)


@pytest.mark.parametrize('target', [np.nan, np.inf, -np.inf])
def test_nonfinite_target_is_rejected(target):
    with pytest.raises(ValueError):
        project_feed_target(5., 0., target, **PARAMETERS)


def test_outside_target_is_an_intent_clamp_not_an_output_velocity_clip():
    values = rollout(np.full(500, -10.), 50.)
    audit_actual_output(values)
    assert values[1, 0] > 0
    assert values[-1, 0] == pytest.approx(0, abs=1e-8)
