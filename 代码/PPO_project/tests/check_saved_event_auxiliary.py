"""Recompute saved auxiliary commands after their independent provenance gate.

No policy or planner execution. All failed records are audited and preserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project/tools"))
import preview_event_auxiliary_report as report
from check_saved_event_nnc import compare
from src.utils.revision_metrics import evaluate_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test", "regression"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve previous command audit")
    snapshot = report.collect(args.split)  # Full event freeze required before test results are read.
    if not snapshot["complete"]: raise ValueError("Wait for complete saved auxiliary results before final audit")
    checks = []
    for row in snapshot["rows"]:
        result_path = ROOT / row["result_file"]
        run = result_path.parent
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if row["method"] == "nnc_transfer":
            request = json.loads((run / "request.json").read_text(encoding="utf-8"))
            common = request["common"]
            spec = next(p for p in request["paths"] if p["id"] == row["path_id"])
            reference = np.loadtxt(spec["reference_file"], delimiter=",", skiprows=1)
            with np.load(run / (row["path_id"] + ".npz"), allow_pickle=False) as z:
                columns, data = z["columns"].tolist(), z["data"]
            get = lambda key: data[:, columns.index(key)]
            positions, timestamps = np.c_[get("x_mm"), get("y_mm")], get("timestamp_s")
            initial = dict(initial_velocity=[0., 0.], initial_acceleration=[0., 0.],
                initial_heading=float(get("heading_rad")[0]), initial_angular_velocity=0., initial_angular_acceleration=0.)
            tolerance, limits = common["epsilon"] / 2, common["limits"]
            rest_threshold = None
        else:
            protocol = json.loads((run.parent / "protocol.json").read_text(encoding="utf-8"))
            import yaml
            cfg = yaml.safe_load((run.parent / "common_config.yaml").read_text(encoding="utf-8"))
            reference = np.loadtxt(run / "reference.csv", delimiter=",", skiprows=1)
            data = np.loadtxt(run / "command_with_initial.csv", delimiter=",", skiprows=1)
            timestamps, positions = data[:, 0], data[:, 1:3]
            initial = {key: value for key, value in saved["initial_state"].items() if key.startswith("initial_")}
            tolerance, limits = protocol["one_sided_tolerance_mm"], cfg["kinematic_constraints"]
            rest_threshold = saved["terminal_feed_tolerance_mm_s"]
        metric = row["metrics"]
        independent = evaluate_trajectory(reference, positions, timestamps, tolerance_mm=tolerance, limits=limits,
            termination_reason=metric["termination_reason"], claimed_success=metric["claimed_success"],
            final_speed_tolerance_mm_s=rest_threshold, witness_spacing_mm=.01, **initial)
        mismatch = compare(metric, independent)
        if row["method"] == "two_step":
            geometric = evaluate_trajectory(reference, positions, timestamps, tolerance_mm=tolerance, limits=limits,
                termination_reason=metric["termination_reason"], claimed_success=metric["claimed_success"],
                witness_spacing_mm=.01, **initial)
            mismatch.extend(compare(saved["common_geometric_arrival_audit"], geometric, "geometric_arrival"))
        # Independent scalar finite differences do not use stored internal v/a/j.
        dt = np.diff(timestamps)
        velocity = np.vstack(([0., 0.], np.diff(positions, axis=0) / dt[:, None]))
        feed = np.linalg.norm(velocity, axis=1)
        acceleration = np.r_[0., np.diff(feed) / dt]
        jerk = np.diff(acceleration) / dt
        for name, values in [("feed_mm_s", feed), ("feed_acceleration_mm_s2", acceleration), ("feed_jerk_mm_s3", jerk)]:
            peak = float(np.max(np.abs(values)))
            if not np.isclose(peak, metric["constraints"][name]["peak"], rtol=1e-9, atol=1e-7): mismatch.append("manual_peak_" + name)
        checks.append(dict(method=row["method"], seed=row["seed"], path_id=row["path_id"],
            sample_count=len(timestamps), strict_success=metric["success"], failures=metric["failure_reasons"],
            terminal_rest_checked=metric["terminal_rest_checked"], mismatches=mismatch))
        print(row["method"], row["seed"], row["path_id"], "success=", metric["success"], "mismatches=", mismatch, flush=True)
    assert all(report.sha(ROOT / name) == expected for name, expected in snapshot["input_sha256"].items())
    result = dict(split=args.split, checks=checks, input_sha256=snapshot["input_sha256"],
        audit_script_sha256=report.sha(__file__), comparison_helper_sha256=report.sha(Path(__file__).with_name("check_saved_event_nnc.py")),
        metrics_source_sha256=report.sha(ROOT / "PPO_project/src/utils/revision_metrics.py"),
        command="python " + " ".join(sys.argv), policies_executed=False,
        mismatch_count=sum(bool(row["mismatches"]) for row in checks), failures=sum(not row["strict_success"] for row in checks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert result["mismatch_count"] == 0


if __name__ == "__main__": main()
