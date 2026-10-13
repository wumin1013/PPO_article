"""Freeze V2 holdout geometry without modifying V1 paths or frozen runtime."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

PPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PPO_ROOT))
from src.utils.revision_paths import load_manifest, generate_path, _save_path


def main():
    source = PPO_ROOT / "data/revision_20260908"
    destination = PPO_ROOT / "data/preview_v2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "paths").mkdir(exist_ok=True)
    original = load_manifest(source)
    manifest = copy.deepcopy(original)
    manifest.update({"schema_version": "revision-paths-v2", "split_frozen_date": "2026-09-08",
        "test_usage": "V2 holdouts below have not been evaluated. Freeze before V2 optimization. V1 former tests are development only.",
        "split_rationale": "Reuse previously exposed V1 train/validation; archive V1 tests as development; new holdout vertex/primitive sequences with different reentrant layout, short-block bursts, and curvature-reversal composition.",
        "generator": "PPO_project/tools/generate_preview_v2_paths.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "paths": []})
    ledger = {"version": "v2-exposure-ledger", "date": "2026-09-08", "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        "reason": "V1 results were inspected and informed continued method development; V1 test data no longer support fresh holdout claims.", "records": []}
    for old in original["paths"]:
        item = copy.deepcopy(old)
        old_split = item["split"]
        item["previous_split"] = old_split
        item["previous_version"] = "revision_20260908"
        item["exposure_status"] = "used_for_development"
        if old_split == "test":
            item["split"] = "development"
        source_csv = source / old["reference_file"]
        target_csv = destination / old["reference_file"]
        if target_csv.exists() and target_csv.read_bytes() != source_csv.read_bytes():
            raise FileExistsError(f"Refusing to overwrite different reference: {target_csv}")
        shutil.copyfile(source_csv, target_csv)
        (destination / "paths" / f"{item['id']}.json").write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["paths"].append(item)
        ledger["records"].append({"path_id": item["id"], "v1_split": old_split, "v2_split": item["split"], "exposure": "used_for_development", "reference_sha256": item["reference_sha256"]})
    definitions = [
        {"id": "mixed_lengths_v2_test", "family": "mixed_lengths", "split": "test", "parameters": {
            "vertices_mm": [[0,0],[6,0],[6,.5],[6.8,.5],[9.5,4],[8.8,4.5],[8.8,5.3],[4,8],[0,8],
                            [0,7.3],[-.8,7.3],[-4,11],[-4,11.7],[-3.2,11.7],[1,14]], "max_spacing_mm": .6}},
        {"id": "line_curve_v2_test", "family": "line_curve", "split": "test", "parameters": {
            "start_mm": [0,0], "primitives": [
                {"line_to": [5,0]}, {"cubic": [[8,0],[9,5],[5,5]]},
                {"line_to": [2,5]}, {"cubic": [[-1,5],[-1,10],[3,10]]},
                {"line_to": [7,10]}, {"cubic": [[11,10],[12,4],[15,7]]}],
            "max_spacing_mm": .6, "chord_error_bound_mm": .005}},
    ]
    headings = [[60,-40,20,-70,35,-20],[15,55,-60,0,-30,70],
                [45,-65,-10,35,65,-50],[0,-45,75,-25,50,-70],[25,60,-35,0,-65,40]]
    lengths = [[1.0,1.4,.8,1.3,.9,1.1],[1.5,.85,1.2,.95,1.4,.8],
               [1.1,.75,1.35,1.0,.9,1.3],[1.4,1.1,.8,1.3,1.0,.9],[.95,1.25,1.1,1.4,.8,1.2]]
    theta = np.radians(np.asarray(headings).ravel())
    increments = np.c_[np.cos(theta), np.sin(theta)] * np.asarray(lengths).reshape(-1,1)
    dense_points = np.vstack([[0,0], np.cumsum(increments, axis=0)])
    definitions.insert(1, {"id": "dense_corners_v2_test", "family": "dense_corners", "split": "test", "parameters": {
        "heading_blocks_deg": headings, "segment_length_blocks_mm": lengths,
        "generation": "Explicit non-repeating ordered heading/length bursts; cumulative Cartesian increments."}})
    for definition in definitions:
        points = dense_points if definition["id"] == "dense_corners_v2_test" else generate_path(definition)
        source_meta = {"kind": "synthetic", "cam_export": False, "generator": "generate_preview_v2_paths.py",
            "generator_sha256": manifest["generator_sha256"],
            "analytic_to_polyline_deviation_bound_mm": definition["parameters"].get("chord_error_bound_mm", 0.0),
            "discretization_definition": "Exact defining polyline or Bezier convex-hull chord bound; independent of planner tolerance."}
        definition["exposure_status"] = "unevaluated_holdout"
        item = _save_path(destination, definition, points, source_meta)
        if not 25 <= item["statistics"]["path_length_mm"] <= 50:
            raise ValueError("V2 holdout length outside frozen 25-50 mm design range")
        manifest["paths"].append(item)
        ledger["records"].append({"path_id": item["id"], "v2_split": "test", "exposure": "unevaluated_holdout", "reference_sha256": item["reference_sha256"]})
    for name, data in [("manifest.json", manifest), ("exposure_ledger.json", ledger)]:
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        target = destination / name
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to modify an existing V2 freeze: {target}")
        target.write_text(text, encoding="utf-8")
    print(destination / "manifest.json")
    print("manifest_sha256", hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest())
    for item in manifest["paths"]:
        if item["split"] == "test":
            print(item["id"], item["statistics"]["path_length_mm"], item["statistics"]["segment_count"], item["reference_sha256"])


if __name__ == "__main__":
    main()
