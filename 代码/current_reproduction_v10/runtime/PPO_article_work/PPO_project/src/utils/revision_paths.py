"""Frozen, synthetic 2-D paths for the 2026-09-08 revision (units: mm).

Run from the repository root::
    python PPO_project/src/utils/revision_paths.py generate
    python PPO_project/src/utils/revision_paths.py import-cam input.csv --metadata source.json

CAM import requires provenance and never relabels generated paths as CAM data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "revision_20260908"
SCHEMA_VERSION = "revision-paths-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def path_statistics(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Expected at least two finite x/y reference points")
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("Adjacent repeated reference points are not allowed")
    direction = delta / lengths[:, None]
    angles = np.arccos(np.clip(np.sum(direction[:-1] * direction[1:], axis=1), -1, 1))
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    corners = [{"point_index": int(i + 1), "s_mm": float(cumulative[i + 1]),
                "turn_deg": float(np.degrees(a)), "xy_mm": points[i + 1].tolist()}
               for i, a in enumerate(angles) if a >= math.radians(8.0)]
    return {"point_count": len(points), "segment_count": len(lengths),
            "path_length_mm": float(lengths.sum()),
            "closed": bool(np.linalg.norm(points[-1] - points[0]) < 1e-9),
            "segment_lengths_mm": lengths.tolist(),
            "segment_length_quantiles_mm": dict(zip(["min", "q25", "median", "q75", "max"],
                                                     np.quantile(lengths, [0, .25, .5, .75, 1]).tolist())),
            "major_corners_threshold_deg": 8.0, "major_corners": corners,
            "all_turn_angles_deg": np.degrees(angles).tolist(),
            "bounds_mm": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]}


def _sample_lines(vertices: np.ndarray, max_spacing: float) -> np.ndarray:
    out = [vertices[0]]
    for a, b in zip(vertices[:-1], vertices[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / max_spacing)))
        out.extend(a + t * (b - a) for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(out)


def _distance_to_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    vector = b - a
    denominator = float(vector @ vector)
    t = np.clip((points - a) @ vector / denominator, 0, 1) if denominator > 1e-20 else np.zeros(len(points))
    return np.linalg.norm(points - a - t[:, None] * vector, axis=1)


def _cubic(control: np.ndarray, chord_error: float, max_spacing: float, depth: int = 0) -> np.ndarray:
    # Convex-hull distance to the chord bounds the entire Bezier subcurve.
    flatness = float(_distance_to_segment(control[1:3], control[0], control[-1]).max())
    if flatness <= chord_error and np.linalg.norm(control[-1] - control[0]) <= max_spacing:
        return np.vstack([control[0], control[-1]])
    if depth >= 30:
        raise ValueError("Bezier subdivision did not converge")
    q = (control[:-1] + control[1:]) / 2
    r = (q[:-1] + q[1:]) / 2
    m = (r[0] + r[1]) / 2
    left = _cubic(np.vstack([control[0], q[0], r[0], m]), chord_error, max_spacing, depth + 1)
    right = _cubic(np.vstack([m, r[1], q[2], control[3]]), chord_error, max_spacing, depth + 1)
    return np.vstack([left, right[1:]])


def synthetic_definitions() -> list[dict[str, Any]]:
    """Explicit split definitions; variants differ in corner sequence/topology."""
    definitions: list[dict[str, Any]] = []
    mixed = {
        "train": [[0, 0], [9, 0], [9, .7], [10, .7], [10, 7], [10.6, 7], [10.6, 8], [2, 8]],
        "validation": [[0, 0], [6, 0], [6, 1], [5.4, 1], [5.4, 8], [12, 8], [12.7, 8.7], [13.4, 8], [18, 8]],
        "test": [[0, 0], [11, 0], [11.6, .6], [11, 1.2], [11, 6], [4, 6], [3.5, 6.8], [4.3, 7.5],
                 [4.3, 12], [12, 12], [12.5, 11.2], [13.3, 11.2], [17, 14]],
    }
    for split, vertices in mixed.items():
        definitions.append({"id": f"mixed_lengths_{split}", "family": "mixed_lengths", "split": split,
                            "parameters": {"vertices_mm": vertices, "max_spacing_mm": .6}})
    dense = {
        "train": {"blocks": 6, "motif_deg": [48, -52, 32, -35], "length_pattern_mm": [.65, .9, .8, .55]},
        "validation": {"blocks": 7, "motif_deg": [65, -30, -60, 25, 0], "length_pattern_mm": [.8, .65, 1.0, .7, 1.2]},
        "test": {"blocks": 8, "motif_deg": [25, 70, -55, -20, 45, -65], "length_pattern_mm": [.6, 1.0, .7, .9, .55, .8]},
    }
    for split, parameters in dense.items():
        definitions.append({"id": f"dense_corners_{split}", "family": "dense_corners", "split": split,
                            "parameters": parameters})
    curves = {
        "train": [dict(line_to=[7, 0]), dict(cubic=[[10, 0], [8, 6], [13, 6]]),
                  dict(line_to=[17, 6]), dict(cubic=[[21, 6], [16, -2], [23, 0]])],
        "validation": [dict(line_to=[4, 0]), dict(cubic=[[8, 0], [4, 7], [9, 7]]),
                       dict(cubic=[[12, 7], [11, 2], [14, 2]]), dict(line_to=[18, 2]),
                       dict(cubic=[[20, 2], [19, 8], [24, 5]])],
        "test": [dict(line_to=[8, 0]), dict(cubic=[[12, 0], [9, -5], [14, -5]]),
                 dict(line_to=[18, -5]), dict(cubic=[[21, -5], [20, 4], [15, 4]]),
                 dict(cubic=[[10, 4], [12, 10], [17, 9]]), dict(line_to=[22, 9]),
                 dict(cubic=[[25, 9], [23, 3], [28, 6]])],
    }
    for split, primitives in curves.items():
        definitions.append({"id": f"line_curve_{split}", "family": "line_curve", "split": split,
                            "parameters": {"start_mm": [0, 0], "primitives": primitives,
                                           "max_spacing_mm": .6, "chord_error_bound_mm": .005}})
    return definitions


def generate_path(definition: dict[str, Any]) -> np.ndarray:
    family, p = definition["family"], definition["parameters"]
    if family == "mixed_lengths":
        return _sample_lines(np.asarray(p["vertices_mm"], float), p["max_spacing_mm"])
    if family == "dense_corners":
        angles, lengths = [], []
        for block in range(p["blocks"]):
            for i, angle in enumerate(p["motif_deg"]):
                angles.append(math.radians(angle + 5 * math.sin(block * 1.7)))
                lengths.append(p["length_pattern_mm"][i] * (1 + .08 * math.sin(block + i)))
        delta = np.c_[np.cos(angles), np.sin(angles)] * np.asarray(lengths)[:, None]
        return np.vstack([[0, 0], np.cumsum(delta, axis=0)])
    if family == "line_curve":
        out = np.asarray([p["start_mm"]], float)
        for primitive in p["primitives"]:
            if "line_to" in primitive:
                piece = _sample_lines(np.vstack([out[-1], primitive["line_to"]]), p["max_spacing_mm"])
            else:
                piece = _cubic(np.vstack([out[-1], primitive["cubic"]]),
                               p["chord_error_bound_mm"], p["max_spacing_mm"])
            out = np.vstack([out, piece[1:]])
        return out
    raise ValueError(f"Unknown family: {family}")


def _save_path(directory: Path, definition: dict[str, Any], points: np.ndarray, source: dict[str, Any]) -> dict[str, Any]:
    path_id = definition["id"]
    csv_path = directory / "paths" / f"{path_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["x_mm", "y_mm"])
    writer.writerows([[format(x, ".12g"), format(y, ".12g")] for x, y in points])
    content = stream.getvalue()
    if csv_path.exists() and csv_path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Refusing to overwrite different frozen path: {csv_path}")
    csv_path.write_text(content, encoding="utf-8")
    # Statistics are computed from the saved precision, exactly as evaluation reads it.
    saved = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    metadata = {**definition, "source": source, "coordinate_unit": "mm", "plane": "XY",
                "reference_sha256": _sha(csv_path), "statistics": path_statistics(saved),
                "reference_file": f"paths/{path_id}.csv"}
    _write_json(directory / "paths" / f"{path_id}.json", metadata)
    return metadata


def generate_suite(output_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    directory = Path(output_dir)
    records = []
    for definition in synthetic_definitions():
        parameters = definition["parameters"]
        source = {"kind": "synthetic", "generator": "revision_paths.py", "generator_sha256": _sha(Path(__file__)),
                  "cam_export": False, "analytic_to_polyline_deviation_bound_mm": parameters.get("chord_error_bound_mm", 0.0),
                  "discretization_definition": "Bezier convex-hull chord bound; exact original straight segments otherwise. Independent of planner tolerance."}
        records.append(_save_path(directory, definition, generate_path(definition), source))
    manifest = {"schema_version": SCHEMA_VERSION, "split_frozen_date": "2026-09-08",
                "test_usage": "Run only after model/configuration freeze. Any test used for tuning becomes development data.",
                "split_rationale": "Distinct corner sequences, notch layouts, primitive counts and curvature reversals; no rotation/scale copies across splits.",
                "historical_regression_paths": ["square", "circle", "butterfly"],
                "historical_regression_note": "Previously used during development; generate from archived historical configuration, not counted as unseen evidence.",
                "units": {"distance": "mm", "time": "s", "angle": "rad"},
                "tolerance_definition": {"planner_band_width_mm": "epsilon in environment", "one_sided_allowed_deviation_mm": "epsilon/2", "cam_discretization_error_mm": "separate source metadata; not added to planner tolerance"},
                "paths": records}
    _write_json(directory / "manifest.json", manifest)
    return manifest


def load_manifest(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    return json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_path(path_id: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> np.ndarray:
    directory = Path(data_dir)
    entries = load_manifest(directory)["paths"]
    entry = next((entry for entry in entries if entry["id"] == path_id), None)
    if entry is None:
        raise KeyError(path_id)
    path = directory / entry["reference_file"]
    if _sha(path) != entry["reference_sha256"]:
        raise ValueError(f"Reference hash mismatch: {path_id}")
    return np.genfromtxt(path, delimiter=",", skip_header=1, ndmin=2)


def import_cam(csv_path: str | Path, metadata_path: str | Path, output_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """Import declared real CAM data into a separate pending-development manifest."""
    source_path, meta_path = Path(csv_path), Path(metadata_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    required = ["source_description", "cam_software", "export_settings", "units", "coordinate_system", "discretization_error_mm"]
    if any(key not in metadata or metadata[key] in (None, "") or
           (isinstance(metadata[key], str) and metadata[key].startswith("TO_BE_PROVIDED")) for key in required):
        raise ValueError(f"CAM metadata must supply {required}; unknown values must explicitly say 'unknown'")
    if metadata["units"] != "mm":
        raise ValueError("Explicitly convert input CAM coordinates to mm and record preprocessing before import")
    raw = np.genfromtxt(source_path, delimiter=",", names=True)
    if not {"x_mm", "y_mm"}.issubset(raw.dtype.names or ()):
        raise ValueError("CAM CSV requires x_mm,y_mm columns in execution order")
    points = np.column_stack([raw["x_mm"], raw["y_mm"]])
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10]
    points = points[keep]
    source = {**metadata, "kind": "user_declared_cam_export", "source_filename": source_path.name,
              "source_sha256": _sha(source_path), "metadata_sha256": _sha(meta_path),
              "preprocessing": {"removed_adjacent_duplicates": int((~keep).sum()), "smoothing": False, "reordering": False}}
    definition = {"id": f"cam_{_sha(source_path)[:12]}_{_sha(meta_path)[:8]}", "family": "imported_cam", "split": "pending_development", "parameters": {}}
    directory = Path(output_dir) / "cam_imports"
    record = _save_path(directory, definition, points, source)
    # Preserve the raw export verbatim for provenance; never inject CAM paths into frozen test set.
    (directory / f"{definition['id']}_original.csv").write_bytes(source_path.read_bytes())
    _write_json(directory / f"{definition['id']}_import.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    cam = sub.add_parser("import-cam")
    cam.add_argument("csv_path", type=Path)
    cam.add_argument("--metadata", type=Path, required=True)
    cam.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    if args.command == "generate":
        manifest = generate_suite(args.output_dir)
        print(json.dumps({"manifest": str(args.output_dir / "manifest.json"), "paths": len(manifest["paths"])}, ensure_ascii=False))
    else:
        print(json.dumps(import_cam(args.csv_path, args.metadata, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
