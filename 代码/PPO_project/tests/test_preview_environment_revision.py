"""Regression tests for opt-in reference/query and ordered-turn corrections."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.cnc_env import Env


def make_env(path=None, *, filter_mode="once_per_step", turn_mode="ordered", finite_scan=True):
    weights = {
        "minimal_mode": True,
        "boundary": {"enabled": False},
        "control_authority": {"mode": "blend", "tangent_blend": .2,
            "theta_ref_smooth_alpha": .25,
            "reference_filter_update_mode": filter_mode,
            "turn_projection_mode": turn_mode},
        "lookahead": {"distance": 1.2, "ref_in_control": True,
            "finite_episode_preview": True, "finite_episode_turn_scan": finite_scan},
        "lookahead_control": {"enabled": False, "min_dist": .1, "max_dist": 3.},
        "p4": {"stall_steps": 1000, "speed_cap_enabled": False},
    }
    return Env(device="cpu", epsilon=.5, interpolation_period=.02,
        MAX_VEL=2., MAX_ACC=4., MAX_JERK=20.,
        MAX_ANG_VEL=3., MAX_ANG_ACC=8., MAX_ANG_JERK=40.,
        Pm=path if path is not None else [[0., 0.], [.6, 0.], [.6, 2.], [3., 2.]],
        max_steps=2000, reward_weights=weights, lookahead_obs_scales=[.5, 1., 2.])


def square_points():
    vertices = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.], [0., 0.]])
    return np.vstack([a + (b-a)*np.linspace(0., 1., 10, endpoint=False)[:, None]
        for a, b in zip(vertices[:-1], vertices[1:])] + [vertices[-1:]])


class PreviewEnvironmentRevisionTests(unittest.TestCase):
    def test_default_modes_retain_legacy_behavior(self):
        env = Env(device="cpu", epsilon=.5, interpolation_period=.02,
            MAX_VEL=2., MAX_ACC=4., MAX_JERK=20., MAX_ANG_VEL=3.,
            MAX_ANG_ACC=8., MAX_ANG_JERK=40., max_steps=2000,
            Pm=[[0., 0.], [2., 0.]], reward_weights={})
        self.assertEqual(env.reference_filter_update_mode, "legacy_query")
        self.assertEqual(env.turn_projection_mode, "legacy_rtree")
        self.assertFalse(env.finite_episode_turn_scan)

    def test_invalid_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference_filter_update_mode"):
            make_env(filter_mode="typo")
        with self.assertRaisesRegex(ValueError, "turn_projection_mode"):
            make_env(turn_mode="typo")

    def test_raw_queries_preserve_filter_action_and_projection_anchor(self):
        env = make_env()
        env.reset()
        names = ["_theta_ref_filtered", "_theta_ref_last", "_theta_ref_filter_update_step",
            "_theta_ref_filter_update_count", "_lookahead_action_u", "_lookahead_dist_active",
            "_lookahead_control_last", "_ordered_projection_position", "_ordered_projection_s",
            "_ordered_projection_cache", "_progress_prev_seg_idx", "_progress_s_max"]
        before = {name: copy.deepcopy(getattr(env, name)) for name in names}
        short = env._raw_path_direction(env.current_position, lookahead_distance=.1)
        long = env._raw_path_direction(env.current_position, lookahead_distance=2.)
        self.assertNotEqual(short, long)
        for point in ([100., -20.], [.6, 1.5], [0., 0.]):
            env._raw_path_direction(point, lookahead_distance=.5)
            env._get_path_direction(point, record=True)
        for name in names:
            if isinstance(before[name], np.ndarray):
                np.testing.assert_array_equal(getattr(env, name), before[name])
            else:
                self.assertEqual(getattr(env, name), before[name], name)

    def test_normal_step_advances_reference_once_and_queries_do_not_change_commands(self):
        clean, observed = make_env(), make_env()
        clean.reset()
        observed.reset()
        for step in range(60):
            previous = clean._theta_ref_filtered
            target = clean._raw_path_direction(clean.current_position)
            expected = clean._wrap_angle(previous + .25 * clean._wrap_angle(target-previous))
            for _ in range(4):
                observed._get_path_direction(observed.current_position, record=True)
                observed.calculate_direction_deviation(observed.current_position)
                observed._raw_path_direction([3., -1.], lookahead_distance=.3)
            action = [(-1.) ** (step // 7) * .15, .4]
            left = clean.step(action)
            right = observed.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            np.testing.assert_array_equal(clean.current_position, observed.current_position)
            self.assertEqual(left[1:3], right[1:3])
            self.assertAlmostEqual(clean._theta_ref_filtered, expected, places=13)
            self.assertEqual(clean._theta_ref_filter_update_count, step+1)
            self.assertEqual(observed._theta_ref_filter_update_count, step+1)
            self.assertEqual(clean._theta_ref_filter_update_step, clean.current_step)

    def test_same_step_reference_reuse_and_reset(self):
        env = make_env()
        env.reset()
        self.assertEqual(env._theta_ref_filter_update_count, 0)
        env.step([0., .2])
        once = env._theta_ref_filtered
        env._advance_path_reference(env.current_position)
        env._advance_path_reference(env.current_position)
        self.assertEqual(env._theta_ref_filtered, once)
        self.assertEqual(env._theta_ref_filter_update_count, 1)
        env.reset()
        self.assertEqual(env._theta_ref_filter_update_count, 0)
        self.assertEqual(env._theta_ref_filter_update_step, -1)

    def test_legacy_query_mode_still_exposes_historical_ema_side_effect(self):
        env = make_env(filter_mode="legacy_query")
        env.reset()
        a = env._get_path_direction(env.current_position)
        b = env._get_path_direction(env.current_position)
        self.assertNotEqual(a, b)

    def test_ordered_turn_projection_rejects_nearby_remote_branch(self):
        path = [[0., 0.], [10., 0.], [10., .1], [0., .1]]
        legacy = make_env(path, turn_mode="legacy_rtree")
        ordered = make_env(path)
        arcs = []
        for env in (legacy, ordered):
            env.reset()
            # Exercise the genuine global projection fallback used without R-tree.
            env.rtree_idx = None
            env.current_position = np.array([2., .09])
            ordered_arc = env._project_onto_progress_path(env.current_position)[2]
            arc = env._get_current_arc_length()
            arcs.append(arc)
            seen = []
            def scan(s):
                seen.append(s)
                return dict(dist_to_turn=1., turn_angle=math.pi/2., turn_sign=1,
                    turn_s=s+1., is_isolated_corner=True)
            env._scan_for_next_turn = scan
            env._compute_turn_info()
            self.assertEqual(seen, [arc])
            if env.turn_projection_mode == "ordered":
                self.assertEqual(arc, ordered_arc)
                self.assertEqual(env._update_segment_info()[0], 0)
        self.assertAlmostEqual(arcs[0], 18.1)
        self.assertAlmostEqual(arcs[1], 2.)

    def test_finite_turn_scan_ignores_next_lap_but_preserves_upcoming_corners(self):
        periodic = make_env(square_points(), finite_scan=False)
        finite = make_env(square_points(), finite_scan=True)
        periodic.reset()
        finite.reset()
        early_a, early_b = periodic._scan_for_next_turn(1.), finite._scan_for_next_turn(1.)
        self.assertEqual(early_a["turn_s"], early_b["turn_s"])
        self.assertAlmostEqual(early_a["dist_to_turn"], early_b["dist_to_turn"])
        late_a, late_b = periodic._scan_for_next_turn(7.7), finite._scan_for_next_turn(7.7)
        self.assertAlmostEqual(late_a["dist_to_turn"], .3, places=12)
        self.assertAlmostEqual(late_a["turn_angle"], math.pi/2.)
        self.assertTrue(math.isinf(late_b["dist_to_turn"]))
        self.assertEqual(late_b["turn_angle"], 0.)
        self.assertFalse(finite.reached_target)

    def test_finite_turn_scan_does_not_change_open_path(self):
        a, b = make_env(finite_scan=False), make_env(finite_scan=True)
        a.reset()
        b.reset()
        for arc in (.1, .5, 1., 2.8, 4.):
            left, right = a._scan_for_next_turn(arc), b._scan_for_next_turn(arc)
            for key in ("dist_to_turn", "turn_angle", "turn_sign", "is_isolated_corner"):
                self.assertEqual(left[key], right[key])

    def test_corrected_modes_do_not_relax_nearby_finish_or_closed_start(self):
        env = make_env([[0., 0.], [10., 0.], [10., .1], [0., .1]])
        env.reset()
        env.current_position = np.array([0., .1])
        env.turn_info = env._compute_turn_info()
        env.state = env._build_state()
        self.assertLess(env.state[6], .01)
        self.assertNotEqual(env._check_termination({}, False)[1], "success")
        closed = make_env(square_points())
        closed.reset()
        for _ in range(5):
            _, _, _, info = closed.step([0., 0.])
            self.assertNotEqual(info["done_reason"], "success")
            self.assertFalse(closed.lap_completed)


if __name__ == "__main__":
    unittest.main()
