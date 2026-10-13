"""Meaningful geometry, discretization, kinematics and provenance regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PPO_ROOT))
sys.path.insert(0, str(PPO_ROOT / "tools"))
from src.utils.revision_paths import generate_suite, load_path, import_cam, _cubic
from src.utils.revision_metrics import ordered_projection, independent_kinematics, evaluate_trajectory, summarize_runs
from src.utils.comparison_metrics import project_points_to_polyline
from src.algorithms.two_step_baseline import (run_two_step_baseline, TwoStepConstraints, smooth_reference_path,
    _compression_preserves_ordered_polyline, _macro_vertices_for_corner_smoothing, SmoothingParams)
from machine_data import export_commands, analyze_measurement, create_templates


class RevisionPathTests(unittest.TestCase):
    def test_split_hash_and_geometry_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = generate_suite(temp)
            self.assertEqual(len(manifest["paths"]), 9)
            self.assertEqual(len({row["reference_sha256"] for row in manifest["paths"]}), 9)
            for split in ["train", "validation", "test"]:
                records = [row for row in manifest["paths"] if row["split"] == split]
                self.assertEqual(len(records), 3)
                for row in records:
                    self.assertEqual(row["source"]["kind"], "synthetic")
                    self.assertGreater(row["statistics"]["path_length_mm"], 10)
                    self.assertLess(row["statistics"]["path_length_mm"], 90)
                    self.assertTrue(np.isfinite(load_path(row["id"], temp)).all())
            initial = (Path(temp) / "manifest.json").read_bytes()
            generate_suite(temp)
            self.assertEqual(initial, (Path(temp) / "manifest.json").read_bytes())
            path = Path(temp) / manifest["paths"][0]["reference_file"]
            path.write_text(path.read_text() + "100,100\n")
            with self.assertRaises(ValueError):
                load_path(manifest["paths"][0]["id"], temp)

    def test_cubic_convex_hull_bound(self):
        control = np.array([[0, 0], [4, 0], [-2, 9], [7, 3]], float)
        sampled = _cubic(control, .005, .6)
        t = np.linspace(0, 1, 10001)[:, None]
        analytic = (1-t)**3 * control[0] + 3*(1-t)**2*t*control[1] + 3*(1-t)*t*t*control[2] + t**3*control[3]
        error = project_points_to_polyline(analytic, sampled, closed=False).contour_error
        self.assertLessEqual(error.max(), .005 + 1e-10)
        self.assertLessEqual(np.linalg.norm(np.diff(sampled, axis=0), axis=1).max(), .6 + 1e-10)

    def test_cam_import_preserves_order_and_requires_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv = root / "source.csv"
            csv.write_text("x_mm,y_mm\n0,0\n1,0\n1,0\n1,1\n", encoding="utf-8")
            metadata = root / "meta.json"
            metadata.write_text("{}")
            with self.assertRaises(ValueError):
                import_cam(csv, metadata, root)
            metadata.write_text(json.dumps(dict(source_description="unit-test declared export", cam_software="fixture",
                export_settings="XY polyline", units="mm", coordinate_system="workpiece", discretization_error_mm="unknown")))
            result = import_cam(csv, metadata, root)
            self.assertEqual(result["split"], "pending_development")
            self.assertEqual(result["source"]["preprocessing"]["removed_adjacent_duplicates"], 1)
            self.assertEqual(result["source"]["source_sha256"], hashlib.sha256(csv.read_bytes()).hexdigest())


class RevisionMetricTests(unittest.TestCase):
    def test_closed_start_is_not_completion_and_shortcut_fails(self):
        square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
        stuck = ordered_projection(square, np.zeros((20, 2)), tolerance_mm=.2)
        self.assertFalse(stuck["ordered_coverage_complete"])
        self.assertLess(stuck["ordered_progress"][-1], .01)
        shortcut = ordered_projection(square, np.array([[0, 0], [10, 10], [0, 0]]), tolerance_mm=.2)
        self.assertFalse(shortcut["ordered_coverage_complete"])
        valid = ordered_projection(square, square, tolerance_mm=.2)
        self.assertTrue(valid["ordered_coverage_complete"])

    def test_backtracking_cannot_increment_completed_frontier_twice(self):
        reference = np.array([[0, 0], [10, 0]], float)
        repeated = np.tile([[0, 0], [1, 0], [0, 0]], (30, 1))
        result = ordered_projection(reference, repeated, tolerance_mm=.1)
        self.assertLess(result["ordered_progress"][-1], .12)
        self.assertFalse(result["ordered_coverage_complete"])

    def test_output_difference_detects_clipping_hidden_by_internal_jerk(self):
        dt = .01
        feed = np.array([0, .1, .2, .3, .05, .0])
        positions = np.c_[np.cumsum(feed) * dt, np.zeros(len(feed))]
        kin = independent_kinematics(positions, np.arange(len(feed)) * dt,
                                    initial_velocity=[0, 0], initial_acceleration=[0, 0])
        self.assertTrue(np.allclose(kin["feed_mm_s"], feed))
        self.assertGreater(np.nanmax(np.abs(kin["feed_jerk_mm_s3"])), 1000)

    def test_semimplicit_euler_alignment_and_independent_axes(self):
        dt, count = .001, 30
        jerk = np.full(count, 100.0)
        acceleration = np.cumsum(jerk) * dt
        velocity = np.cumsum(acceleration) * dt
        x = np.cumsum(velocity) * dt
        p = np.c_[np.r_[0, x], np.zeros(count+1)]
        kin = independent_kinematics(p, np.arange(count + 1)*dt,
                                    initial_velocity=[0, 0], initial_acceleration=[0, 0])
        np.testing.assert_allclose(kin["feed_mm_s"][1:], velocity, atol=1e-10)
        np.testing.assert_allclose(kin["feed_acceleration_mm_s2"][1:], acceleration, atol=1e-9)
        np.testing.assert_allclose(kin["feed_jerk_mm_s3"][1:], jerk, atol=1e-7)
        self.assertTrue(np.allclose(kin["axis_velocity_mm_s"][:, 1], 0))

    def test_nonuniform_timestamps_and_invalid_failure_retained(self):
        t = np.array([0, .1, .3, .6])
        p = np.c_[2*t, 3*t]
        kin = independent_kinematics(p, t)
        np.testing.assert_allclose(kin["axis_velocity_mm_s"][1:], np.tile([2, 3], (3, 1)))
        bad = evaluate_trajectory(p, p, [0, .1, .1, .6], tolerance_mm=.1)
        self.assertFalse(bad["success"])
        summary = summarize_runs([bad, {"success": True, "path_completed": True, "elapsed_time_s": 1.2}])
        self.assertEqual(summary["failed_run_count"], 1)
        self.assertEqual(summary["successful_time_mean_s"], 1.2)

    def test_two_step_audit_exposes_output_difference_without_changing_planner(self):
        result = run_two_step_baseline(path_name="line_audit", reference_path=[[0, 0], [5, 0]],
            constraints=TwoStepConstraints(100, 2000, 62500), dt=.001, half_epsilon=.1, closed=False, max_steps=3000)
        self.assertEqual(result.metrics["termination_status"], "success")
        audit = result.metrics["output_command_audit"]
        self.assertTrue(audit["valid_trace"])
        self.assertFalse(result.metrics["angular_constraints_enforced"])
        self.assertGreater(audit["constraints"]["feed_jerk_mm_s3"]["peak"], 62500)
        self.assertEqual(result.metrics["max_relative_linear_jerk_exceedance"], 0)

    def test_macro_compression_cannot_delete_curved_parts_or_reverse_order(self):
        with tempfile.TemporaryDirectory() as temp:
            generate_suite(temp)
            reference = load_path("line_curve_validation", temp)
            smoothed = smooth_reference_path(reference, half_epsilon=1.25, closed=False)
            self.assertLess(smoothed.max_smoothed_path_error_mm, 1.25)
            self.assertFalse(smoothed.boundary_violation_flag)
        self.assertFalse(_compression_preserves_ordered_polyline(np.array([[0, 0], [3, 0], [2, 0], [4, 0]]), [0, 3], False))
        self.assertFalse(_compression_preserves_ordered_polyline(np.array([[0, 0], [-1, 0], [4, 0]]), [0, 2], False))
        # Exact straight-edge samples still compress and retain the old square behavior.
        square = np.vstack([np.c_[np.linspace(0, 10, 21), np.zeros(21)],
                            np.c_[np.full(20, 10), np.linspace(.5, 10, 20)],
                            np.c_[np.linspace(9.5, 0, 20), np.full(20, 10)],
                            np.c_[np.zeros(19), np.linspace(9.5, .5, 19)]])
        macro = _macro_vertices_for_corner_smoothing(square, True, SmoothingParams())
        self.assertEqual(len(macro), 4)


class MachineDataTests(unittest.TestCase):
    def test_offline_roundtrip_and_empty_template(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_templates(root)
            p = np.array([[0, 0], [.5, 0], [1, 0]])
            output = export_commands(root / "simulation.csv", p, [0, .5, 1], provenance={"data_kind": "simulation"})
            report = analyze_measurement(output, p, tolerance_mm=.1)
            self.assertEqual(report["data_kind"], "simulation")
            self.assertIsNone(report["feedback"])
            self.assertFalse(report["metadata"]["machine_execution_authorized"])
            with self.assertRaises(FileExistsError):
                export_commands(output, p, [0, .5, 1], provenance={"data_kind": "simulation"})
            # Recreating templates must preserve user-supplied platform metadata.
            meta_path = root / "measurement_metadata_template.json"
            customized = meta_path.read_text().replace("TO_BE_PROVIDED", "provided")
            meta_path.write_text(customized)
            create_templates(root)
            self.assertEqual(meta_path.read_text(), customized)
            with self.assertRaises(ValueError):
                analyze_measurement(root / "measurement_template.csv", p, tolerance_mm=.1,
                                    metadata_path=root / "measurement_metadata_template.json")
            with self.assertRaises(ValueError):
                export_commands(root / "fabricated.csv", p, [0, .5, 1], provenance={"data_kind": "machine_measurement"}, feedback=p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
