"""Offline event-study audit of the unchanged traditional two-step planner.

Validation can run before selection. Every test run first verifies the complete
event freeze. No trained policy, planner algorithm or numerical source is edited.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml

import preview_event_study as study
from preview_two_step_eval import (TwoStepConstraints, SmoothingParams,
                                   SchedulerParams, run_two_step_baseline)
from src.utils.revision_metrics import evaluate_trajectory


def require_event_test_freeze():
    """Check all registered models before loading any held-out geometry."""
    study.check_protocol()
    path = study.OUT / "freeze.json"
    if not path.exists():
        raise ValueError("Complete event freeze is required before test baseline evaluation")
    frozen = study.read(path)
    protocol = study.read(study.OUT / "protocol.json")
    required = {f"formal/{variant}_seed{seed}" for variant in protocol["formal_variants"]
                for seed in protocol["formal_seeds"]}
    if set(frozen["models"]) != required or set(frozen["configs"]) != required:
        raise ValueError("Event freeze must bind the exact complete model/configuration set")
    expected = {"source_sha256": study.source_state()["source_sha256"],
        "manifest_sha256": study.sha(study.DATA / "manifest.json"),
        "protocol_sha256": study.sha(study.OUT / "protocol.json"),
        "selection_sha256": study.sha(study.OUT / "selection.json")}
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("Event source, manifest, protocol or selection differs from freeze")
    for key in sorted(required):
        run = study.OUT / key
        if study.sha(run / "checkpoint.pt") != frozen["models"][key] or study.sha(run / "config.yaml") != frozen["configs"][key]:
            raise ValueError(f"Frozen event checkpoint/configuration mismatch: {key}")
        study.verify_training(run, protocol["formal_steps"])
        study.validation_rows(run)
    return frozen


def shared_config(regime):
    cfg = study.config_for("state_fixed", 101)
    if regime == "tight":
        cfg["environment"]["epsilon"] = 1.25
    elif regime == "low_jerk":
        cfg["kinematic_constraints"]["MAX_JERK"] = 31250.
    elif regime != "nominal":
        raise ValueError(regime)
    return cfg


def numerical_sources():
    names = ["PPO_project/src/algorithms/two_step_baseline.py",
             "PPO_project/src/utils/comparison_metrics.py", "PPO_project/src/utils/revision_metrics.py",
             "PPO_project/tools/preview_two_step_eval.py", "PPO_project/tools/preview_event_two_step.py"]
    return {name: study.sha(study.ROOT / name) for name in names}


def audit_output(reference, result, cfg):
    rows = result.trace_rows
    positions = np.vstack([result.smoothed.smoothed_path[0], [[r["x"], r["y"]] for r in rows]])
    timestamps = np.r_[0., [r["time"] for r in rows]]
    tangent = result.smoothed.smoothed_path[1] - result.smoothed.smoothed_path[0]
    initial = dict(initial_velocity=[0., 0.], initial_acceleration=[0., 0.],
        initial_heading=float(np.arctan2(tangent[1], tangent[0])),
        initial_angular_velocity=0., initial_angular_acceleration=0.)
    original = result.metrics["output_command_audit"]
    args = dict(tolerance_mm=cfg["environment"]["epsilon"] / 2,
        limits=cfg["kinematic_constraints"], termination_reason=original["termination_reason"],
        claimed_success=original["claimed_success"], witness_spacing_mm=.01, **initial)
    geometric = evaluate_trajectory(reference, positions, timestamps, **args)
    scheduler = SchedulerParams()
    rest_threshold = max(cfg["kinematic_constraints"]["MAX_VEL"] * scheduler.feedrate_safety_ratio
                         * scheduler.stop_velocity_ratio, 1e-5)
    native_rest = evaluate_trajectory(reference, positions, timestamps,
        final_speed_tolerance_mm_s=rest_threshold, **args)
    return positions, timestamps, initial, rest_threshold, geometric, native_rest


def evaluate(split, regime="nominal"):
    # This gate precedes configuration, path specs, output creation, or planner calls.
    if split == "test":
        require_event_test_freeze()
    elif split not in ("validation", "regression"):
        raise ValueError(split)
    study.check_protocol()
    cfg = shared_config(regime)
    limits = cfg["kinematic_constraints"]
    constraints = TwoStepConstraints(*(float(limits[k]) for k in
        ["MAX_VEL", "MAX_ACC", "MAX_JERK", "MAX_ANG_VEL", "MAX_ANG_ACC", "MAX_ANG_JERK"]))
    dt = float(cfg["environment"]["interpolation_period"])
    tolerance = float(cfg["environment"]["epsilon"]) / 2
    max_steps = int(cfg["revision"]["episode_max_steps"])
    destination = study.OUT / "two_step_event_v1" / f"{split}_{regime}"
    config_text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=True)
    protocol = dict(version="event-two-step-v1", numerical_sources=numerical_sources(),
        event_source=study.source_state(), manifest_sha256=study.sha(study.DATA / "manifest.json"),
        event_protocol_sha256=study.sha(study.OUT / "protocol.json"),
        freeze_sha256=study.sha(study.OUT / "freeze.json") if split == "test" else None,
        split=split, regime=regime, dt_s=dt, max_steps=max_steps, one_sided_tolerance_mm=tolerance,
        ordered_witness_spacing_mm=.01, constraints=asdict(constraints),
        smoothing=asdict(SmoothingParams()), scheduler=asdict(SchedulerParams()),
        common_config="event.config_for(state_fixed,101) with identical named PPO regime overrides; no baseline parameter tuning",
        algorithm="Existing preview_two_step_eval.py call and two_step_baseline.py, unchanged",
        initial_state="Reference start; smoothed initial tangent; zero feed/acceleration/angular rates; explicit t=0",
        terminal_state="Original scheduler requests rest; PPO geometric arrival does not. Both native-rest and common geometric audits are saved; no same-terminal-state time superiority claim.",
        constraints_scope="Scalar and angular output audit; original scheduler does not enforce angular or Cartesian axis limits",
        failure_policy="Retain full traces, all constraint failures, original internal metrics and independent recalculation; no failure deletion",
        machine_execution_authorized=False, data_kind="simulation", trained=False)
    if (destination / "protocol.json").exists():
        if study.read(destination / "protocol.json") != protocol or (destination / "common_config.yaml").read_text(encoding="utf-8") != config_text:
            raise ValueError("Preserve prior baseline protocol/configuration; use a new version")
    else:
        study.write(destination / "protocol.json", protocol)
        (destination / "common_config.yaml").write_text(config_text, encoding="utf-8")
        shutil.copyfile(__file__, destination / "preview_event_two_step_snapshot.py")
    if split == "regression":
        historical = yaml.safe_load((study.ROOT / "论文项目/generated/configs/full_method_snapshot.yaml").read_text(encoding="utf-8"))
        paths = [(p["name"], study.backend._build_path(p)) for p in historical["training"]["path_curriculum"]["paths"]]
    else:
        paths = [(p["id"], study.points(p["id"])) for p in study.manifest()["paths"] if p["split"] == split]
    if not paths:
        raise ValueError("Empty baseline split")
    records = []
    for path_id, reference in paths:
        run = destination / path_id
        output = run / "result.json"
        if output.exists():
            saved = study.read(output)
            if saved["protocol_sha256"] != study.sha(destination / "protocol.json"):
                raise ValueError("Cached baseline protocol changed")
            for name, expected in saved["artifact_sha256"].items():
                if study.sha(run / name) != expected:
                    raise ValueError(f"Cached baseline artifact changed: {path_id}/{name}")
            records.append(saved)
            continue
        if run.exists() and any(run.iterdir()):
            raise FileExistsError("Preserve incomplete baseline output for diagnosis")
        run.mkdir(parents=True, exist_ok=True)
        np.savetxt(run / "reference.csv", reference, delimiter=",", header="x_mm,y_mm", comments="")
        started = time.perf_counter()
        result = run_two_step_baseline(path_name=path_id, reference_path=reference, constraints=constraints,
            dt=dt, half_epsilon=tolerance, closed=bool(np.linalg.norm(reference[0] - reference[-1]) < 1e-9), max_steps=max_steps)
        with (run / "original_trace.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result.trace_rows[0]))
            writer.writeheader()
            writer.writerows(result.trace_rows)
        np.savetxt(run / "smoothed_path.csv", result.smoothed.smoothed_path, delimiter=",", header="x_mm,y_mm", comments="")
        positions, timestamps, initial, rest_threshold, geometric, native_rest = audit_output(reference, result, cfg)
        np.savetxt(run / "command_with_initial.csv", np.c_[timestamps, positions], delimiter=",",
                   header="timestamp_s,x_mm,y_mm", comments="")
        reference_tangent = reference[1] - reference[0]
        reference_heading = float(np.arctan2(reference_tangent[1], reference_tangent[0]))
        record = dict(path_id=path_id, split=split, regime=regime, data_kind="simulation",
            protocol_sha256=study.sha(destination / "protocol.json"),
            common_config_sha256=study.sha(destination / "common_config.yaml"),
            artifact_sha256={name: study.sha(run / name) for name in
                ("reference.csv", "original_trace.csv", "smoothed_path.csv", "command_with_initial.csv")},
            initial_state={"timestamp_s": 0., "position_mm": positions[0].tolist(), **initial},
            reference_initial_heading_rad=reference_heading,
            initial_heading_difference_from_reference_rad=float(np.arctan2(np.sin(initial["initial_heading"]-reference_heading), np.cos(initial["initial_heading"]-reference_heading))),
            terminal_feed_tolerance_mm_s=rest_threshold, common_geometric_arrival_audit=geometric,
            native_terminal_rest_audit=native_rest, original_planner_metrics=result.metrics,
            terminal_state_comparable_to_ppo=False,
            planning_and_audit_wall_s=time.perf_counter()-started, command="python " + " ".join(sys.argv))
        study.write(output, record)
        records.append(record)
        print(f"{path_id}: native={native_rest['success']}, geometric={geometric['success']}, failures={native_rest['failure_reasons']}", flush=True)
    study.write(destination / "summary.json", dict(rows=records,
        common_geometric_summary=study.backend.summarize_runs([r["common_geometric_arrival_audit"] for r in records]),
        native_terminal_rest_summary=study.backend.summarize_runs([r["native_terminal_rest_audit"] for r in records]),
        note="Descriptive baseline diagnostics; terminal boundary conditions differ from PPO. All paths retained."))
    if protocol["numerical_sources"] != numerical_sources():
        raise ValueError("Numerical sources changed during baseline evaluation; preserve outputs for audit")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "regression"), required=True)
    parser.add_argument("--regime", choices=("nominal", "tight", "low_jerk"), default="nominal")
    args = parser.parse_args()
    evaluate(args.split, args.regime)


if __name__ == "__main__":
    main()
