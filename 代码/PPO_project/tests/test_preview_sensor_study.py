"""Separate policy control of D from its geometry sensing, without state leaks."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'PPO_project'), str(ROOT/'PPO_project/tools')]
import preview_sensor_study as sensor


def environment(method='adaptive'):
    cfg = sensor.configuration('preview_time', method, 101)
    return sensor.SensorEnvironment(cfg, 'sensor_unit', reference=np.array([[0.,0.],[3.,0.],[3.,8.]]))


def test_geometry_is_invariant_to_control_distance_and_queries_do_not_mutate_state():
    env = environment()
    raw = env.inner.raw
    before = (raw._theta_ref_filter_update_count, raw._theta_ref_filtered, raw._progress_s_max)
    observations, controls = [], []
    for distance in (1.2, 3., 4.):
        raw._lookahead_dist_active = distance
        observations.append(sensor.fixed_geometry(raw))
        controls.append(raw._raw_path_direction(raw.current_position))
        assert raw._lookahead_dist_active == distance
    np.testing.assert_array_equal(observations[0], observations[1])
    np.testing.assert_array_equal(observations[1], observations[2])
    assert controls[0] != controls[2]
    assert before == (raw._theta_ref_filter_update_count, raw._theta_ref_filtered, raw._progress_s_max)


def test_only_six_policy_state_coordinates_are_replaced_and_next_state_is_current():
    env = environment()
    initial_inner = env.inner.state
    unchanged = [i for i in range(26) if not 12 <= i < 18]
    np.testing.assert_array_equal(env.state[unchanged], initial_inner[unchanged])
    observation, _, done, _ = env.step([0., .05, .1])
    assert not done
    assert env.physics_steps == 20
    np.testing.assert_array_equal(observation, env.state)
    np.testing.assert_array_equal(observation[12:18], sensor.fixed_geometry(env.inner.raw))
    np.testing.assert_array_equal(observation[unchanged], env.inner.state[unchanged])
    assert observation[20] == pytest.approx(.1)
    assert env.trace[-1]['lookahead_mm'] == pytest.approx(1.18)


@pytest.mark.parametrize('method', sensor.core.METHODS)
def test_sensor_change_is_common_and_actions_reward_limits_are_unchanged(method):
    cfg = sensor.configuration('preview_time', method, 101)
    previous = sensor.BASE_CONFIG('preview_time', method, 101)
    for key in ('ppo', 'environment', 'reward_weights', 'kinematic_constraints'):
        assert cfg[key] == previous[key]
    assert cfg['revision']['active_action_dims'] == previous['revision']['active_action_dims']
    assert cfg['revision']['observation_dim'] == 26
    env = environment(method)
    np.testing.assert_array_equal(env.state[12:18], environment().state[12:18])


def test_control_preview_still_changes_executed_motion():
    short, long = environment(), environment()
    for _ in range(25):
        short.step([0., .05, (1.2-.8)/3.8])
        long.step([0., .05, (4.-.8)/3.8])
    assert abs(short.inner.raw.heading-long.inner.raw.heading) > .01
    assert np.linalg.norm(short.inner.raw.current_position-long.inner.raw.current_position) > .01
