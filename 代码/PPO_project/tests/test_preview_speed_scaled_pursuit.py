"""Geometric invariants and real-environment control boundary checks."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_speed_scaled_pursuit as pursuit
import preview_local_corner_study as local
from src.environment.cnc_env import Env


@pytest.mark.parametrize('angle', [-1., -.2, 0., .4, 1.])
def test_same_target_geometry_has_same_curvature_at_different_feed(angle):
    assert pursuit.pursuit_rate(4., angle, 3.)/4. == pytest.approx(pursuit.pursuit_rate(17., angle, 3.)/17.)


def test_circle_target_requests_exact_circle_curvature():
    radius, angle, speed = 5., .3, 7.
    chord = 2.*radius*math.sin(angle)
    assert pursuit.pursuit_rate(speed, angle, chord) == pytest.approx(speed/radius)
    assert pursuit.pursuit_rate(speed, -angle, chord) == pytest.approx(-speed/radius)


def test_zero_feed_and_degenerate_target_do_not_request_hidden_rotation():
    assert pursuit.pursuit_rate(0., 1., 3.) == 0.
    assert pursuit.pursuit_rate(5., 1., 0.) == 0.


@pytest.mark.parametrize('args', [(np.nan, 0., 1.), (1., np.inf, 2.), (-1., 0., 2.), (1., 0., -1.)])
def test_invalid_physics_is_rejected(args):
    with pytest.raises(ValueError):
        pursuit.pursuit_rate(*args)


def test_context_restores_original_on_exception():
    original = Env._resolve_heading_intent
    with pytest.raises(RuntimeError):
        with pursuit.speed_scaled_execution():
            assert Env._resolve_heading_intent is pursuit.speed_scaled_heading_intent
            raise RuntimeError('intentional fixture')
    assert Env._resolve_heading_intent is original


def test_real_macro_keeps_twenty_physics_steps_and_one_reference_update_per_tick():
    cfg = local.configuration('preview_time', 'adaptive', 901)
    reference = np.asarray([[0., 0.], [2., 0.], [2., 5.], [8., 5.]])
    with pursuit.speed_scaled_execution():
        env = local.LocalCornerEnvironment(cfg, 'unit', reference=reference)
        raw = env.inner.raw
        # Stationary geometry contains a nonzero lookahead angle.
        assert abs(raw._raw_path_direction(raw.current_position, lookahead_distance=3.)) > .1
        assert raw._resolve_heading_intent(0., 5.) == 0.
        first = raw._theta_ref_filter_update_count
        env.step(np.asarray([0., math.log(.05/.95), (3.-.8)/3.8]))
        assert env.physics_steps == 20
        assert raw._theta_ref_filter_update_count-first == 20
        assert env.state.shape == (26,)
        assert max(abs(row.get('omega_rad_s', 0.)) for row in env.trace) <= raw.MAX_ANG_VEL
        assert max(abs(row.get('alpha_rad_s2', 0.)) for row in env.trace) <= raw.MAX_ANG_ACC+1e-7
        assert max(abs(row.get('angular_jerk_rad_s3', 0.)) for row in env.trace) <= raw.MAX_ANG_JERK+1e-7


def test_fixed_and_adaptive_use_the_identical_public_heading_law():
    point = np.asarray([[0., 0.], [2., 0.], [2., 5.], [8., 5.]])
    with pursuit.speed_scaled_execution():
        fixed = local.LocalCornerEnvironment(local.configuration('preview_time', 'fixed3', 901), 'unit', reference=point)
        adaptive = local.LocalCornerEnvironment(local.configuration('preview_time', 'adaptive', 901), 'unit', reference=point)
        latent = np.asarray([0., math.log(.05/.95), (3.-.8)/3.8])
        for _ in range(4):
            fixed.step(latent); adaptive.step(latent)
            np.testing.assert_array_equal(fixed.inner.raw.current_position, adaptive.inner.raw.current_position)
            assert fixed.inner.raw.heading == adaptive.inner.raw.heading
