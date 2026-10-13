"""Offline-only command export and machine feedback analysis. No device interface.

All exports are analysis files and carry simulation/unreviewed provenance. A CSV
is not a machine-executable program, and this tool never creates G-code.
"""
from __future__ import annotations

import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PPO_ROOT = Path(__file__).resolve().parents[1]
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))
from src.utils.revision_metrics import evaluate_trajectory, independent_kinematics

FIELDS = ["timestamp_s", "cmd_x_mm", "cmd_y_mm", "feedback_x_mm", "feedback_y_mm"]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_command_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read exported CSV or the experiment runner's named-column NPZ trace."""
    path = Path(path)
    names = ("timestamp_s", "x_mm", "y_mm")
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            columns = archive['columns'].tolist()
            data = archive['data']
            values = {name: data[:, columns.index(name)] for name in names}
    else:
        raw = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
        values = {name: raw[name] for name in names}
    return np.c_[values['x_mm'], values['y_mm']], values['timestamp_s']


def export_commands(output_path: str | Path, positions: Any, timestamps: Any, *, provenance: dict[str, Any],
                    feedback: Any = None) -> Path:
    """Export synchronized XY commands, optionally paired real feedback, for review."""
    p, t = np.asarray(positions, float), np.asarray(timestamps, float)
    kin = independent_kinematics(p, t)  # validates finite samples and increasing time
    if provenance.get("data_kind") not in {"simulation", "machine_measurement"}:
        raise ValueError("Provenance must declare data_kind=simulation or machine_measurement")
    if provenance["data_kind"] == "machine_measurement" and not provenance.get("source_measurement_file"):
        raise ValueError("Real machine data requires source_measurement_file provenance")
    if feedback is not None:
        feedback = np.asarray(feedback, float)
        if feedback.shape != p.shape or not np.isfinite(feedback).all():
            raise ValueError("Feedback must match command shape and be finite")
        if provenance["data_kind"] != "machine_measurement":
            raise ValueError("Do not put synthetic feedback into a real-measurement export")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".metadata.json").exists():
        raise FileExistsError("Preserve existing exports; choose a new output filename")
    fields = FIELDS + ["cmd_feed_mm_s", "cmd_vx_mm_s", "cmd_vy_mm_s"]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, (point, stamp) in enumerate(zip(p, t)):
            row = {"timestamp_s": stamp, "cmd_x_mm": point[0], "cmd_y_mm": point[1],
                   "feedback_x_mm": "" if feedback is None else feedback[i, 0],
                   "feedback_y_mm": "" if feedback is None else feedback[i, 1]}
            for key, value in [("cmd_feed_mm_s", kin["feed_mm_s"][i]),
                               ("cmd_vx_mm_s", kin["axis_velocity_mm_s"][i, 0]),
                               ("cmd_vy_mm_s", kin["axis_velocity_mm_s"][i, 1])]:
                row[key] = value if np.isfinite(value) else ""
            writer.writerow(row)
    _write_json(output.with_suffix(".metadata.json"), {**provenance,
        "schema_version": "machine-data-xy-v1", "plane": "XY", "coordinate_unit": "mm", "time_unit": "s",
        "csv_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "feedback_present": feedback is not None,
        "machine_execution_authorized": False, "human_safety_review": "pending",
        "purpose": "offline_analysis_only", "controller_replanning": provenance.get("controller_replanning", "unknown")})
    return output


def create_templates(directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Header only: no fabricated experimental observations.
    csv_template = directory / "measurement_template.csv"
    if not csv_template.exists():
        csv_template.write_text(",".join(FIELDS) + "\n", encoding="utf-8")
    measurement_metadata = {
        "data_kind": "machine_measurement", "source_measurement_file": "TO_BE_PROVIDED",
        "machine_model": "TO_BE_PROVIDED", "controller_model_firmware": "TO_BE_PROVIDED",
        "command_transport": "TO_BE_PROVIDED: timestamped position streaming or ordinary G-code",
        "controller_replanning": "TO_BE_PROVIDED: secondary interpolation, smoothing, speed limiting and settings",
        "acquisition_device": "TO_BE_PROVIDED", "sample_period_s": None,
        "clock_synchronization": "TO_BE_PROVIDED", "command_feedback_latency_s": None,
        "coordinate_frame_and_offsets": "TO_BE_PROVIDED", "axis_constraints": "TO_BE_PROVIDED",
        "initial_final_state": "TO_BE_PROVIDED", "path_id": "TO_BE_PROVIDED", "configuration_hash": "TO_BE_PROVIDED",
        "material": "TO_BE_PROVIDED_IF_CUTTING", "tool": "TO_BE_PROVIDED_IF_CUTTING",
        "surface_measurement_method_and_data": "TO_BE_PROVIDED_IF_SURFACE_CLAIM",
        "human_safety_review": "pending", "machine_execution_authorized": False}
    cam_metadata = {"source_description": "TO_BE_PROVIDED",
        "cam_software": "TO_BE_PROVIDED", "export_settings": "TO_BE_PROVIDED", "units": "mm",
        "coordinate_system": "TO_BE_PROVIDED", "discretization_error_mm": "unknown"}
    for name, template in [("measurement_metadata_template.json", measurement_metadata), ("cam_metadata_template.json", cam_metadata)]:
        if not (directory / name).exists():
            _write_json(directory / name, template)


def analyze_measurement(csv_path: str | Path, reference: Any, *, tolerance_mm: float,
                        limits: dict[str, Any] | None = None, metadata_path: str | Path | None = None) -> dict[str, Any]:
    _validate_analysis_parameters(tolerance_mm)
    path = Path(csv_path)
    metadata_file = Path(metadata_path) if metadata_path else path.with_suffix(".metadata.json")
    if not metadata_file.exists():
        raise ValueError("A provenance metadata file is required for offline analysis")
    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    if meta.get("data_kind") not in {"simulation", "machine_measurement"}:
        raise ValueError("Metadata data_kind must distinguish simulation and real measurement")
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if "csv_sha256" in meta:
        expected = meta["csv_sha256"]
        if not isinstance(expected, str) or expected.lower() != source_sha256:
            raise ValueError("CSV content does not match metadata csv_sha256; preserve and verify the source export")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV columns: {FIELDS}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("At least two actual samples are required; the empty template contains no results")
    t = np.array([float(row["timestamp_s"]) for row in rows])
    commands = np.array([[float(row["cmd_x_mm"]), float(row["cmd_y_mm"])] for row in rows])
    result = {"data_kind": meta["data_kind"], "source_file": str(path),
              "source_sha256": source_sha256, "metadata": meta,
              "command": evaluate_trajectory(reference, commands, t, tolerance_mm=tolerance_mm, limits=limits),
              "controller_replanning": meta.get("controller_replanning", "unknown"),
              "feedback": None, "surface_quality_conclusion": "No conclusion without cutting and metrology data."}
    supplied = [bool(row["feedback_x_mm"].strip()) and bool(row["feedback_y_mm"].strip()) for row in rows]
    any_feedback = any(row["feedback_x_mm"].strip() or row["feedback_y_mm"].strip() for row in rows)
    if any_feedback and not all(supplied):
        raise ValueError("Partial feedback/missing samples need explicit preprocessing; no silent deletion or interpolation")
    if all(supplied):
        if meta["data_kind"] != "machine_measurement":
            raise ValueError("Feedback analysis requires declared machine measurement provenance")
        feedback = np.array([[float(row["feedback_x_mm"]), float(row["feedback_y_mm"])] for row in rows])
        result["feedback"] = evaluate_trajectory(reference, feedback, t, tolerance_mm=tolerance_mm, limits=limits)
        tracking = np.linalg.norm(feedback - commands, axis=1)
        result["command_feedback_tracking_error_mm"] = {"max": float(tracking.max()), "rmse": float(np.sqrt(np.mean(tracking ** 2)))}
        result["alignment_note"] = "Uses supplied timestamps without lag fitting; clock synchronization and controller delay must be measured separately."
    return result


def _validate_analysis_parameters(tolerance_mm: float, witness_spacing_mm: float | None = None) -> None:
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("One-sided tolerance must be finite and positive")
    if witness_spacing_mm is not None and (
            not np.isfinite(witness_spacing_mm) or not 0 < witness_spacing_mm < 2 * tolerance_mm):
        raise ValueError("Witness spacing must be finite, positive, and smaller than twice the one-sided tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("templates")
    template.add_argument("--output-dir", type=Path, default=PPO_ROOT / "data/revision_20260908/machine_templates")
    export = sub.add_parser("export")
    export.add_argument("trace_csv", type=Path, help="CSV or named-column NPZ with timestamp_s,x_mm,y_mm (initial sample required)")
    export.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("csv_path", type=Path)
    analyze.add_argument("--reference", type=Path, required=True, help="CSV with x_mm,y_mm")
    analyze.add_argument("--metadata", type=Path)
    analyze.add_argument("--one-sided-tolerance-mm", type=float, required=True)
    analyze.add_argument("--limits-json", type=Path)
    analyze.add_argument("--witness-spacing-mm", type=float,
                         help="Explicit ordered coverage resolution, e.g. 0.01 for the final simulation study")
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "templates":
        create_templates(args.output_dir)
        print(args.output_dir)
    elif args.command == "export":
        p, timestamps = read_command_trace(args.trace_csv)
        export_commands(args.output, p, timestamps, provenance={"data_kind": "simulation",
                        "source_trace": str(args.trace_csv), "source_trace_sha256": hashlib.sha256(args.trace_csv.read_bytes()).hexdigest(),
                        "export_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print(args.output)
    else:
        _validate_analysis_parameters(args.one_sided_tolerance_mm, args.witness_spacing_mm)
        raw = np.genfromtxt(args.reference, delimiter=",", names=True)
        reference = np.c_[raw["x_mm"], raw["y_mm"]]
        limits = json.loads(args.limits_json.read_text(encoding="utf-8")) if args.limits_json else None
        # CLI-scoped adapter: the public analysis function and historical
        # default remain unchanged. Always restore the module on failure too.
        from src.utils import revision_metrics
        previous_projection = revision_metrics.ordered_projection
        try:
            if args.witness_spacing_mm is not None:
                revision_metrics.ordered_projection = partial(previous_projection,
                    witness_spacing_mm=args.witness_spacing_mm)
            result = analyze_measurement(args.csv_path, reference, tolerance_mm=args.one_sided_tolerance_mm,
                                         limits=limits, metadata_path=args.metadata)
        finally:
            revision_metrics.ordered_projection = previous_projection
        result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["requested_witness_spacing_mm"] = args.witness_spacing_mm
        result["analysis_command"] = "python " + " ".join(sys.argv)
        _write_json(args.output, result)
        print(args.output)


if __name__ == "__main__":
    main()
