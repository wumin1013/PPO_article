"""Offline re-audit of preserved NNC validation commands; no policy imports.

Run explicitly with --output pointing to a new JSON file. All source results
remain read-only, including the historical adapter version they actually used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
from src.utils.revision_metrics import evaluate_trajectory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(actual, expected, prefix=""):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [prefix]
        return [problem for key, value in expected.items()
                for problem in compare(actual.get(key), value, prefix + "." + key)]
    if isinstance(expected, float):
        return [] if actual is not None and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7) else [prefix]
    return [] if actual == expected else [prefix]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recheck-dir", type=Path, help="Optional new seed11 validation run to compare against preserved output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior audit")
    base = ROOT / "论文项目/revision_20260909_event_preview/study/nnc_transfer/validation"
    replay = ROOT / "论文项目/revision_20260908/frozen_replay_explore"
    manifest_path = ROOT / "PPO_project/data/preview_event_v1/manifest.json"
    allowed = {p["id"] for p in read(manifest_path)["paths"] if p["split"] == "validation"}
    checked, hashes = [], {}
    for seed in [11, 22, 33, 44, 55]:
        run = base / f"nnc_seed{seed}"
        request_path = run / "request.json"
        request = read(request_path)
        assert request["split"] == "validation" and request["seed"] == seed
        assert request["manifest_sha256"] == sha(manifest_path)
        assert {p["id"] for p in request["paths"]} == allowed and len(request["paths"]) == len(allowed)
        adapter = run / "evaluation_adapter.py"
        assert sha(adapter) == request["script_sha256"]
        common = request["common"]
        summary_path = run / "summary.json"
        summary = read(summary_path)
        assert {r["path_id"] for r in summary["rows"]} == allowed
        assert summary["request_sha256"] == sha(request_path)
        assert summary["script_sha256"] == sha(adapter)
        for spec in request["paths"]:
            path_id = spec["id"]
            reference_file = Path(spec["reference_file"])
            assert reference_file.resolve().is_relative_to(run.resolve())
            assert sha(reference_file) == spec["reference_sha256"]
            reference = np.loadtxt(reference_file, delimiter=",", skiprows=1)
            # Confirm copied coordinates belong to the registered validation path.
            registered = np.loadtxt(ROOT / "PPO_project/data/preview_event_v1/paths" / (path_id + ".csv"), delimiter=",", skiprows=1)
            assert np.array_equal(reference, registered)
            worker_path = run / (path_id + ".worker.json")
            result_path = run / (path_id + ".json")
            trace_path = run / (path_id + ".npz")
            worker, result = read(worker_path), read(result_path)
            assert worker["request_sha256"] == result["request_sha256"] == sha(request_path)
            assert worker["trace_sha256"] == result["trace_sha256"] == sha(trace_path)
            assert worker["script_sha256"] == request["script_sha256"] == sha(adapter)
            assert result["worker_sha256"] == sha(worker_path)
            assert worker["reference_sha256"] == sha(reference_file)
            source_files = worker["original_training_provenance"]["source_files"]
            assert all(sha(replay / name) == expected for name, expected in source_files.items())
            with np.load(trace_path, allow_pickle=False) as z:
                columns, trace = z["columns"].tolist(), z["data"]
            get = lambda name: trace[:, columns.index(name)]
            metric = evaluate_trajectory(reference, np.c_[get("x_mm"), get("y_mm")], get("timestamp_s"),
                tolerance_mm=common["epsilon"] / 2, limits=common["limits"],
                termination_reason=worker["reason"], claimed_success=worker["reason"] == "success",
                initial_velocity=[0., 0.], initial_acceleration=[0., 0.],
                initial_heading=float(get("heading_rad")[0]), initial_angular_velocity=0.,
                initial_angular_acceleration=0., witness_spacing_mm=.01)
            differences = compare(result["metrics"], metric)
            summary_row = next(r for r in summary["rows"] if r["path_id"] == path_id)
            differences += compare(summary_row, result, "summary_row")
            checked.append(dict(seed=seed, path_id=path_id, sample_count=len(trace), success=metric["success"],
                failure_reasons=metric["failure_reasons"], elapsed_time_s=metric["elapsed_time_s"],
                contour_error_max_mm=metric["contour_error_max_mm"], metric_mismatches=differences,
                recorded_adapter_sha256=sha(adapter), archived_source_files_verified=len(source_files),
                extra_training_steps=result["new_training_steps"]))
            for file in [request_path, adapter, summary_path, reference_file, worker_path, result_path, trace_path]:
                hashes[file.relative_to(ROOT).as_posix()] = sha(file)
            print(f"seed={seed} {path_id}: success={metric['success']}, mismatches={differences}", flush=True)
    assert len(checked) == 15
    snapshot_comparisons = []
    if args.recheck_dir is not None:
        recheck = args.recheck_dir.resolve()
        original = base / "nnc_seed11"
        old_request = read(original / "request.json")
        new_request = read(recheck / "request.json")
        assert new_request["seed"] == 11 and new_request["split"] == "validation"
        assert new_request["common"] == old_request["common"]
        assert sha(recheck / "evaluation_adapter.py") == new_request["script_sha256"]
        for path_id in sorted(allowed):
            old_trace = original / (path_id + ".npz")
            new_trace = recheck / (path_id + ".npz")
            with np.load(old_trace, allow_pickle=False) as a, np.load(new_trace, allow_pickle=False) as b:
                assert np.array_equal(a["columns"], b["columns"])
                assert a["data"].shape == b["data"].shape
                columns = a["columns"].tolist()
                differing = [name for i, name in enumerate(columns)
                             if not np.array_equal(a["data"][:, i], b["data"][:, i], equal_nan=True)]
                count = len(a["data"])
            old_result = read(original / (path_id + ".json"))
            new_result_path = recheck / (path_id + ".json")
            new_result = read(new_result_path)
            new_worker_path = recheck / (path_id + ".worker.json")
            new_worker = read(new_worker_path)
            old_worker = read(original / (path_id + ".worker.json"))
            assert new_worker["script_sha256"] == new_request["script_sha256"]
            assert new_worker["request_sha256"] == sha(recheck / "request.json")
            assert new_worker["trace_sha256"] == new_result["trace_sha256"] == sha(new_trace)
            assert new_result["worker_sha256"] == sha(new_worker_path)
            for key in ["checkpoint_sha256", "original_config_sha256", "source_archive_sha256", "original_training_provenance", "effective_config"]:
                assert old_worker[key] == new_worker[key], key
            snapshot_comparisons.append(dict(path_id=path_id, sample_count=count, columns_checked=len(columns),
                different_columns=differing, metric_mismatches=compare(new_result["metrics"], old_result["metrics"]),
                old_adapter_sha256=old_request["script_sha256"], new_adapter_sha256=new_request["script_sha256"]))
            for file in [new_trace, new_worker_path, new_result_path, recheck / "request.json", recheck / "evaluation_adapter.py"]:
                hashes[file.relative_to(ROOT).as_posix()] = sha(file)
            print(f"snapshot replay {path_id}: {count} samples x {len(columns)} columns; differences={differing}", flush=True)
    report = dict(scope="15 saved NNC validation trajectories only; no policy execution, no test geometry",
        rows=checked, input_sha256=hashes, audited_numeric_source_sha256=sha(ROOT / "PPO_project/src/utils/revision_metrics.py"),
        audit_script_sha256=sha(__file__), command="python " + " ".join(sys.argv),
        new_snapshot_comparisons=snapshot_comparisons,
        mismatch_count=sum(bool(r["metric_mismatches"]) for r in checked), failure_count=sum(not r["success"] for r in checked))
    assert all(sha(ROOT / name) == expected for name, expected in hashes.items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert report["mismatch_count"] == 0
    assert all(not row["different_columns"] and not row["metric_mismatches"] for row in snapshot_comparisons)


if __name__ == "__main__":
    main()
