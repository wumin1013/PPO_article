"""Regression tests for the 2026-09-08 environment audit."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.environment.cnc_env import Env
from src.environment.kinematics import apply_kinematic_constraints
from src.environment.reward import RewardCalculator, RewardContext
from src.algorithms.kcm import KinematicConstraintModule


def environment(path=None, *, reference=None, adaptive=False):
    weights = {
        "minimal_mode": True,
        "w_s": 12.0,
        "boundary": {"enabled": True},
        "control_authority": {"mode": "legacy"},
        "lookahead": {"distance": 1.0, "ref_in_control": True},
        "lookahead_control": {"enabled": adaptive, "policy_action": adaptive,
                              "min_dist": 0.5, "max_dist": 3.0, "mix_gain": 1.0,
                              "straight_dist": 1.0, "corner_dist": 1.0},
        "cornerness": {"enabled": True, "reference_distance": reference},
        "p4": {"stall_steps": 1000, "speed_cap_enabled": False},
    }
    return Env(device="cpu", epsilon=0.2, interpolation_period=0.02,
               MAX_VEL=1.0, MAX_ACC=4.0, MAX_JERK=20.0,
               MAX_ANG_VEL=3.0, MAX_ANG_ACC=8.0, MAX_ANG_JERK=40.0,
               Pm=path if path is not None else [[0., 0.], [2., 0.]],
               max_steps=1000, reward_weights=weights, lookahead_obs_scales=[1.0, 2.0])


class RevisionEnvironmentTests(unittest.TestCase):
    def test_scalar_output_differences_obey_constraints_through_reversals(self):
        previous = (0., 0., 0., 0.)
        dt = 0.01
        for step in range(2500):
            velocity_intent = 2.0 if (step // 137) % 2 else 0.0
            omega_intent = 3.0 if (step // 101) % 2 else -3.0
            result = apply_kinematic_constraints(*previous, velocity_intent, omega_intent,
                                                dt, 2., 5., 25., 3., 7., 35.)
            v, a, j, w, alpha, jw = result
            self.assertGreaterEqual(v, -1e-10)
            for value, bound in [(v, 2.), (a, 5.), (j, 25.), (w, 3.), (alpha, 7.), (jw, 35.)]:
                self.assertLessEqual(abs(value), bound + 1e-7)
            self.assertAlmostEqual(a, (v - previous[0]) / dt, places=9)
            self.assertAlmostEqual(j, (a - previous[1]) / dt, places=8)
            self.assertAlmostEqual(alpha, (w - previous[2]) / dt, places=9)
            self.assertAlmostEqual(jw, (alpha - previous[3]) / dt, places=8)
            previous = (v, a, w, alpha)

    def test_kcm_wrapper_uses_same_integrator(self):
        module = KinematicConstraintModule(2., 5., 25., 3., 7., 35., 0.01)
        state = dict(velocity=0., acceleration=0., angular_vel=0., angular_acc=0.)
        action, _, result = module.apply_constraints((2., 1.), state)
        expected = apply_kinematic_constraints(0., 0., 0., 0., 1., 2., .01, 2., 5., 25., 3., 7., 35.)
        np.testing.assert_allclose([result[k] for k in ["velocity", "acceleration", "jerk", "angular_vel", "angular_acc", "angular_jerk"]], expected)
        np.testing.assert_allclose(action, [expected[3], expected[0]])

    def test_guidance_heading_is_constrained_before_position_integration(self):
        for mode in ["legacy", "blend", "integrated"]:
            env = environment([[0., 0.], [.8, 0.], [.8, 1.], [2., 1.]], adaptive=True)
            env.reward_weights["boundary"]["enabled"] = False
            env.control_authority_mode = mode
            env.control_tangent_blend_active = .2
            env.reset()
            old_heading, old_omega, old_alpha = env.heading, 0., 0.
            for step in range(160):
                old_position = env.current_position.copy()
                _, _, _, info = env.step([(-1.) ** (step // 17), .8, float((step // 11) % 2)])
                dt = env.interpolation_period
                omega = env._wrap_angle(env.heading - old_heading) / dt
                alpha = (omega - old_omega) / dt
                jerk = (alpha - old_alpha) / dt
                self.assertAlmostEqual(omega, env.angular_vel, places=10)
                self.assertAlmostEqual(alpha, env.angular_acc, places=8)
                self.assertAlmostEqual(jerk, env.angular_jerk, places=6)
                self.assertLessEqual(abs(omega), env.MAX_ANG_VEL + 1e-7)
                self.assertLessEqual(abs(alpha), env.MAX_ANG_ACC + 1e-7)
                self.assertLessEqual(abs(jerk), env.MAX_ANG_JERK + 1e-6)
                self.assertAlmostEqual(np.linalg.norm(env.current_position - old_position) / dt, env.velocity, places=10)
                old_heading, old_omega, old_alpha = env.heading, omega, alpha

    def test_open_path_step_reports_success(self):
        env = environment()
        env.reset()
        for _ in range(200):
            _, _, done, info = env.step([0., 1.])
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(info["done_reason"], "success")
        self.assertGreaterEqual(info["progress"], .99)
        self.assertLessEqual(info["ordered_contour_error"], env.half_epsilon)

    def test_nearby_finish_cannot_skip_route(self):
        env = environment([[0., 0.], [10., 0.], [10., .1], [0., .1]])
        env.reset()
        env.current_position = np.array([0., .1])
        env.state = env._build_state()
        done, reason = env._check_termination({}, False)
        self.assertFalse(done)
        self.assertNotEqual(reason, "success")
        self.assertLess(env.state[6], .01)
        indices = [env._project_onto_progress_path(env.current_position)[1] for _ in range(20)]
        self.assertEqual(len(set(indices)), 1)

    def test_closed_start_does_not_complete_lap(self):
        env = environment([[0., 0.], [10., 0.], [10., 10.], [0., 10.], [0., 0.]])
        env.reset()
        for _ in range(5):
            _, _, done, info = env.step([0., 0.])
        self.assertFalse(done)
        self.assertFalse(env.lap_completed)
        self.assertLess(info["progress"], .01)

    def test_retrace_progress_is_not_rewarded_twice(self):
        calculator = RewardCalculator({"minimal_mode": True, "w_s": 12.}, 1., 3., .1, 20., 8., 40.)
        rewards = []
        for progress in [.6, .4, .6]:
            context = RewardContext(contour_error=0., progress=progress, velocity=0., heading_error=0.,
                                    jerk=0., angular_jerk=0., angular_acc=0., kcm_intervention=0.)
            _, parts = calculator.calculate_reward(context)
            rewards.append(parts["r_progress"])
        self.assertGreater(rewards[0], 0.)
        self.assertEqual(rewards[1:], [0., 0.])

    def test_reward_horizon_decouples_only_evaluation_geometry(self):
        path = [[0., 0.], [10., 0.], [10., 10.]]
        observations, directions, coupled, decoupled = [], [], [], []
        for reference in [None, 4.6]:
            env = environment(path, reference=reference, adaptive=True)
            env.reset()
            env.current_position = np.array([8., 0.])
            deltas = []
            for action in [0., 1.]:
                env._lookahead_action_u = action
                directions.append(env._get_path_direction(env.current_position))
                observations.append(env._compute_lookahead_observation())
                deltas.append(env._get_reward_geometry_delta())
            if reference is None:
                coupled = deltas
            else:
                decoupled = deltas
        self.assertNotEqual(coupled[0], coupled[1])
        np.testing.assert_allclose(decoupled, [math.pi / 2., math.pi / 2.])
        self.assertNotEqual(directions[0], directions[1])
        np.testing.assert_allclose(directions[:2], directions[2:])
        self.assertFalse(np.array_equal(observations[0], observations[1]))
        np.testing.assert_allclose(observations[:2], observations[2:])

    def test_action_drives_next_observation_at_selected_horizon(self):
        env = environment([[0., 0.], [1., 0.], [1., 2.]], adaptive=True)
        env.reset()
        observation, _, _, info = env.step([0., .1, 1.])
        self.assertAlmostEqual(info["lookahead_status"]["lookahead_dist"], 3.)
        np.testing.assert_allclose(observation[-4:], env._compute_lookahead_observation())

    def test_open_endpoint_clips_target_without_nan(self):
        env = environment()
        env.reset()
        env.current_position = np.array([2., 0.])
        observation = env._compute_lookahead_observation()
        self.assertTrue(np.isfinite(observation).all())
        np.testing.assert_allclose(observation, np.zeros(4))

    def test_contour_error_includes_distance_beyond_endpoint(self):
        env = environment()
        env.reset()
        self.assertAlmostEqual(env.get_contour_error(np.array([3., 0.])), 1.)


def write_diagnostics(output_directory):
    """Write measured before/after kernel traces from the preserved Git source."""
    import csv
    import hashlib
    import json
    import subprocess
    from datetime import datetime, timezone

    project = Path(__file__).resolve().parents[2]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(
        ["git", "show", "d8ff2b5:PPO_project/src/environment/kinematics.py"], cwd=project).decode("utf-8")
    namespace = {"__name__": "revision_legacy_kinematics"}
    exec(compile(source, "git:d8ff2b5:kinematics.py", "exec"), namespace)
    rows, summaries = [], {}
    for label, kernel in [("before_d8ff2b5", namespace["apply_kinematic_constraints"]),
                          ("corrected", apply_kinematic_constraints)]:
        previous = (0., 0., 0., 0.)
        previous_output_acc = previous_output_alpha = 0.
        peaks = dict(acceleration=0., jerk=0., angular_acceleration=0., angular_jerk=0.,
                     logged_acceleration_discrepancy=0., logged_jerk_discrepancy=0.)
        violations = 0
        for step in range(1000):
            v_intent = 2. if (step // 137) % 2 else 0.
            w_intent = 3. if (step // 101) % 2 else -3.
            v, a, j, w, alpha, jw = kernel(*previous, v_intent, w_intent, .01, 2., 5., 25., 3., 7., 35.)
            output_a, output_alpha = (v - previous[0]) / .01, (w - previous[2]) / .01
            output_j, output_jw = (output_a - previous_output_acc) / .01, (output_alpha - previous_output_alpha) / .01
            values = dict(acceleration=abs(output_a), jerk=abs(output_j), angular_acceleration=abs(output_alpha),
                          angular_jerk=abs(output_jw), logged_acceleration_discrepancy=abs(a-output_a),
                          logged_jerk_discrepancy=abs(j-output_j))
            for key, value in values.items():
                peaks[key] = max(peaks[key], float(value))
            violations += int(abs(output_a)>5.+1e-7 or abs(output_j)>25.+1e-7 or abs(output_alpha)>7.+1e-7 or abs(output_jw)>35.+1e-7)
            rows.append(dict(version=label, sample=step+1, time_s=(step+1)*.01,
                             velocity_intent=v_intent, angular_intent=w_intent,
                             velocity=v, logged_acceleration=a, logged_jerk=j,
                             angular_velocity=w, logged_angular_acceleration=alpha, logged_angular_jerk=jw,
                             output_acceleration=output_a, output_jerk=output_j,
                             output_angular_acceleration=output_alpha, output_angular_jerk=output_jw))
            previous = (v, a, w, alpha)
            previous_output_acc, previous_output_alpha = output_a, output_alpha
        summaries[label] = {**peaks, "violation_samples": violations, "samples": 1000}
    with (output / "kcm_before_after_trace.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    geometry = []
    for reference in [None, 4.6]:
        env = environment([[0., 0.], [10., 0.], [10., 10.]], reference=reference, adaptive=True)
        env.reward_weights["minimal_mode"] = False
        env.reward_weights["cornerness"].update(theta0_deg=30., ema_tau_steps=1, w_track0=5., w_track_min=1., w_smooth0=3.)
        env.reset()
        env.current_position = np.array([8., 0.])
        for action in [0., 1.]:
            env._lookahead_action_u = action
            direction = env._get_path_direction(env.current_position)
            observation = env._compute_lookahead_observation()
            env.reward_calculator.reset()
            context = RewardContext(contour_error=.05, progress=0., velocity=.5, heading_error=0.,
                                    jerk=0., angular_jerk=0., angular_acc=2., kcm_intervention=0.,
                                    lookahead_delta_theta=env._get_reward_geometry_delta())
            reward, components = env.reward_calculator.calculate_reward(context)
            geometry.append(dict(reference_distance=reference, action=action,
                                 active_distance=env._lookahead_dist_active, reference_direction=direction,
                                 observation=observation.tolist(),
                                 observation_delta=env._lookahead_obs_last["delta_theta_far"],
                                 reward_delta=env._get_reward_geometry_delta(), reward=reward,
                                 reward_components=components))
    record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), baseline_commit="d8ff2b5",
                  command="python PPO_project/tests/test_revision_environment.py --diagnostics 论文项目/revision_20260908/environment_diagnostics",
                  constraints=dict(dt=.01, max_vel=2., max_acc=5., max_jerk=25., max_omega=3., max_alpha=7., max_angular_jerk=35.),
                  code_sha256={str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [project/"PPO_project/src/environment/kinematics.py",
                                            project/"PPO_project/src/environment/cnc_env.py",
                                            project/"PPO_project/src/environment/reward.py",
                                            Path(__file__).resolve()]},
                  kcm=summaries, geometry=geometry,
                  interpretation="Deterministic mechanism diagnostics; not independently trained performance results.")
    (output/"environment_diagnostics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        write_diagnostics(sys.argv[2])
    else:
        unittest.main()
