"""Hash-bound old evidence and explicit v9 runs; no implicit directory scan.

All writes are confined to the sibling audit directory. Original v5 evidence and
v9 run files are read only. New traces receive the unchanged full-interval audit;
old traces retain their bound geometry audits and receive arithmetic rechecks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


AUDIT = Path(__file__).resolve().parent
OLD = Path("D:/PPO_article_work/JNN_dominance_20261011_v5")
LIMITS = [100.0, 2000.0, 62500.0]
METHODS = ("JNN", "FIXED", "S2", "S2R", "SCMO", "COPP", "NNC")
CASES = ("original", "dense", "piecewise_curvature_coupon_scale10", "asymmetric_neck_contour_scale10")
METRIC_KEYS = ("success", "reason", "complete_time_s", "elapsed_prefix_s", "samples", "vector_peaks",
    "vector_limit_ratios", "kinematic_pass", "rest_pass", "terminal_rest_s", "geometry",
    "executed_length_mm", "max_polyline_deviation_mm", "rms_polyline_deviation_mm")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def write(path, obj):
    path = Path(path).resolve()
    require(path.is_relative_to(AUDIT), "Audit output must remain inside audit/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def bound(path, expected):
    path = Path(path).resolve()
    require(path.is_file() and sha(path) == expected, "Missing or changed evidence: " + str(path))
    return str(path)


def compact(metric):
    return {key: metric.get(key) for key in METRIC_KEYS}


def arithmetic(row, definition):
    metric, dt = row["metrics"], row["dt_s"]
    require(metric["success"] == (metric["complete_time_s"] is not None), "Failure/completion-time mismatch")
    if not row.get("trace"):
        require(not metric["success"], "Successful record lacks a trace")
        return {"scope": "No trace; failed planning/execution retained", "trace_available": False}
    bound(row["trace"], row["trace_sha256"])
    with np.load(row["trace"], allow_pickle=False) as data:
        positions, reference = data["positions"], data["reference"]
    expected = np.loadtxt(definition["reference"], delimiter=",", skiprows=1)
    require(np.array_equal(reference, expected), "Trace uses a different complete reference")
    require(positions.ndim == 2 and positions.shape[1] == 2 and len(positions) >= 1,
            "Invalid position samples")
    require(np.isfinite(positions).all() and np.array_equal(positions[0], reference[0]), "Invalid trace/start")
    if len(positions) < 2:
        require(not metric["success"], "Empty successful trace")
        return {"scope": "Zero command intervals; no kinematic samples", "trace_available": True}
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(np.vstack((np.zeros(2), velocity)), axis=0) / dt
    jerk = np.diff(np.vstack((np.zeros(2), acceleration)), axis=0) / dt
    norms = [np.linalg.norm(x, axis=1) for x in (velocity, acceleration, jerk)]
    peaks = [float(x.max()) for x in norms]
    elapsed = len(velocity) * dt
    length = float(norms[0].sum() * dt)
    deficit = float((LIMITS[0] - norms[0]).sum() * dt)
    rest = (norms[0] <= .001) & (norms[1] <= .01)
    nonrest = np.flatnonzero(~rest)
    rest_s = (len(rest) - (int(nonrest[-1]) + 1 if len(nonrest) else 0)) * dt
    require(np.allclose(peaks, metric["vector_peaks"], rtol=1e-8, atol=2e-4), "Vector peaks differ")
    require(abs(elapsed - metric["elapsed_prefix_s"]) < 1e-8, "Recorded elapsed time differs")
    require(abs(length - metric["executed_length_mm"]) < 1e-6, "Recorded executed length differs")
    require(abs(rest_s - metric["terminal_rest_s"]) < 1e-10, "Recorded final rest differs")
    require(abs(elapsed - (length + deficit) / LIMITS[0]) < 1e-8, "Time/length/deficit identity differs")
    if metric["success"]:
        require(abs(elapsed - metric["complete_time_s"]) < 1e-8, "Completion time excludes recorded commands")
        require(metric["geometry"]["geometry_complete"] and metric["kinematic_pass"] and metric["rest_pass"],
                "Successful record violates common acceptance")
    return dict(scope="All recorded command positions; zero initial velocity/acceleration, backward differences",
        trace_available=True, intervals=len(velocity), vector_peaks=peaks, terminal_rest_s=rest_s,
        executed_length_mm=length, speed_deficit_mm=deficit, elapsed_prefix_s=elapsed,
        endpoint_error_mm=float(np.linalg.norm(positions[-1] - reference[-1])))


def build_old():
    definitions, rows, receipts = {}, [], {}
    frozen = OLD / "evidence/frozen_main_v6"
    manifest_path = frozen / "manifest.json"
    manifest = read(manifest_path)
    receipts[str(manifest_path)] = sha(manifest_path)
    for case in CASES[:2]:
        reference = f"programs/{case}/reference.csv"
        geometry = f"programs/{case}/geometry.json"
        found = next(r for r in manifest["runs"] if r["case"] == case)
        effective = found["metrics"]["tolerance_mm"]
        definitions[case] = dict(reference=bound(frozen / reference, manifest["files"][reference]),
            reference_sha256=manifest["files"][reference], geometry=bound(frozen / geometry, manifest["files"][geometry]),
            geometry_sha256=manifest["files"][geometry], nominal_tolerance_mm=1.25,
            evaluation_tolerance_mm=effective, arc_chord_error_bound_mm=1.25 - effective,
            split="previously_observed_development", independent_test=False)
    for item in manifest["runs"]:
        rows.append(dict(run_id=item["id"], cohort="frozen_v6", case=item["case"], method=item["method"],
            dt_s=item["dt_s"], record=bound(frozen / item["source_record"], item["source_record_sha256"]),
            record_sha256=item["source_record_sha256"], trace=bound(frozen / item["trace"], item["trace_sha256"]),
            trace_sha256=item["trace_sha256"], metrics=compact(item["metrics"]),
            identity=item.get("identity"), geometry_audit="Inherited from hash-bound frozen manifest; full original command intervals",
            metric_authority=str(manifest_path)))
    global_path = OLD / "evidence/scaled_representative_v8_registration.json"
    global_reg = read(global_path)
    receipts[str(global_path)] = sha(global_path)
    for item in global_reg["cases"]:
        definitions[item["id"]] = {key: item[key] for key in ("nominal_tolerance_mm", "evaluation_tolerance_mm", "arc_chord_error_bound_mm", "physical_time_budget_s")}
        for key in ("reference", "geometry"):
            definitions[item["id"]][key] = bound(OLD / item[key], item[key + "_sha256"])
            definitions[item["id"]][key + "_sha256"] = item[key + "_sha256"]
        definitions[item["id"]].update(split="previously_observed_direct_variant", independent_test=False)
    for relative, collection in (("results/representative_scaled_v8/traditional_final/summary.json", "records"),
                                 ("results/representative_scaled_v8/learned/trace_summary.json", "rows")):
        path = OLD / relative
        summary, registration = read(path), read(path.parent / "registration.json")
        require(summary["registration_sha256"] == sha(path.parent / "registration.json"), "Substudy registration changed")
        require(registration["global_registration_sha256"] == sha(global_path), "Global study changed")
        receipts[str(path)] = sha(path)
        receipts[str(path.parent / "registration.json")] = sha(path.parent / "registration.json")
        for item in summary[collection]:
            record = bound(OLD / item["record"], item["record_sha256"])
            raw = read(record)
            metric = item.get("metrics", raw)
            rows.append(dict(run_id=raw.get("run_id", Path(record).stem), cohort="scaled_v8",
                case=item["case"], method=item["method"], dt_s=item["dt_s"], record=record,
                record_sha256=item["record_sha256"], trace=bound(OLD / item["trace"], item["trace_sha256"]) if item.get("trace") else None,
                trace_sha256=item.get("trace_sha256"), metrics=compact(metric),
                identity=item.get("method_configuration", item.get("model_sha256")),
                geometry_audit="Inherited from hash-bound original full-interval result", metric_authority=str(path),
                cost_accounting=item.get("cost_unit", "Keep source-specific cost boundary; do not aggregate unlike clocks")))
    keys = {(r["case"], r["method"], r["dt_s"]) for r in rows}
    require(len(rows) == len(keys) == 56, "Expected all 56 old case/method/resolution cells")
    for row in rows:
        row["arithmetic_audit"] = arithmetic(row, definitions[row["case"]])
    return dict(schema_version=1, source_receipts=receipts, cases=definitions, rows=rows,
        methods=list(METHODS), expected_old_cells=56, statistical_unit="One complete deterministic policy/planner execution per program and resolution; no seed replication",
        geometry_audit_scope="Old geometry acceptance is inherited from SHA-bound complete-interval evidence; arithmetic is independently recomputed here")


def audit_new(index_path, inventory):
    index_path = Path(index_path).resolve()
    index = read(index_path)
    rows, missing = [], []
    sys.path.insert(0, str(OLD / "src"))
    from metrics_exact_index_v5 import metrics
    seen = set()
    for item in index["runs"]:
        key = item["run_id"]
        require(key not in seen, "Duplicate indexed run: " + key)
        seen.add(key)
        path = Path(item["record"])
        path = path.resolve() if path.is_absolute() else (index_path.parent / path).resolve()
        if not path.exists():
            missing.append({"run_id": key, "record": str(path), "status": "not_run_or_not_saved"})
            continue
        raw = read(path)
        declared = raw.get("metrics", raw)
        require(isinstance(declared.get("success"), bool), "Run needs explicit success status")
        require(declared["success"] == (declared.get("complete_time_s") is not None), "Invalid declared completion time")
        case, dt = item["case"], float(item["dt_s"])
        require(item["method"] in ("JNN", "FIXED") and dt in (.001, .0005), "Unsupported new method/period")
        if "dt_s" in declared:
            require(abs(float(declared["dt_s"]) - dt) < 1e-12, "Indexed command period differs")
        if case in inventory["cases"]:
            definition = inventory["cases"][case]
        else:
            definition = {key: item[key] for key in ("reference", "reference_sha256", "nominal_tolerance_mm", "arc_chord_error_bound_mm", "physical_time_budget_s")}
            definition["reference"] = bound(definition["reference"], definition["reference_sha256"])
            definition["evaluation_tolerance_mm"] = definition["nominal_tolerance_mm"] - definition["arc_chord_error_bound_mm"]
        tolerance = definition["evaluation_tolerance_mm"]
        require(definition["nominal_tolerance_mm"] == 1.25 and 0 < tolerance <= 1.25, "Outside nominal v9 study")
        for name in ("nominal_tolerance_mm", "evaluation_tolerance_mm", "arc_chord_error_bound_mm"):
            if name in item:
                require(abs(float(item[name]) - definition[name]) < 1e-12, "Indexed geometry budget differs")
        for name in ("tolerance_mm", "evaluation_tolerance_mm"):
            if name in declared:
                require(abs(float(declared[name]) - tolerance) < 1e-12, "Changed tolerance: " + key)
        if "limits" in declared:
            require(declared["limits"] == LIMITS, "Changed vector limits: " + key)
        trace_name = item.get("trace", raw.get("trace"))
        trace = Path(trace_name) if trace_name else None
        if trace is not None:
            trace = trace.resolve() if trace.is_absolute() else (index_path.parent / trace).resolve()
            require(trace.is_file(), "Indexed trace missing: " + str(trace))
            with np.load(trace, allow_pickle=False) as archive:
                positions, reference = archive["positions"], archive["reference"]
            expected = np.loadtxt(definition["reference"], delimiter=",", skiprows=1)
            require(np.array_equal(reference, expected) and np.array_equal(positions[0], reference[0]), "Changed reference/start: " + key)
            metric = metrics(reference, positions, dt=dt, tolerance=tolerance, limits=LIMITS,
                declared_success=declared["success"], reason=declared.get("reason"))
            require(metric["success"] == declared["success"], "Independent audit rejects recorded status: " + key)
            if metric["success"]:
                require(abs(metric["complete_time_s"] - declared["complete_time_s"]) < 1e-8, "Changed completion definition: " + key)
                if "physical_time_budget_s" in definition:
                    require(metric["complete_time_s"] <= definition["physical_time_budget_s"] + dt, "Time budget exceeded")
        else:
            require(not declared["success"], "Success without full trace")
            metric = declared
        row = {key: item.get(key) for key in ("run_id", "case", "method", "dt_s", "split", "variant", "omega", "seed", "formal_new_experiment")}
        row.update(cohort="v9", record=str(path), record_sha256=sha(path), trace=str(trace) if trace else None,
            trace_sha256=sha(trace) if trace else None, metrics=compact(metric),
            geometry_audit="Recomputed every command interval and every original ordered gate",
            model_sha256=raw.get("model_sha256"), runtime_version=raw.get("runtime_version"),
            configuration_sha256=raw.get("configuration_sha256"), selection_plan_sha256=raw.get("selection_plan_sha256"))
        row["arithmetic_audit"] = arithmetic(row, definition)
        rows.append(row)
    return rows, missing, dict(path=str(index_path), sha256=sha(index_path))


def compare(rows, inventory):
    old = {(r["case"], r["method"], r["dt_s"]): r for r in inventory["rows"]}
    comparisons = []
    for row in rows:
        if row["case"] not in CASES:
            continue
        for method in METHODS:
            baseline = old[row["case"], method, row["dt_s"]]
            ours, theirs = row["metrics"], baseline["metrics"]
            formal = row.get("formal_new_experiment") is True
            both = bool(ours["success"] and theirs["success"] and formal)
            comparisons.append(dict(run_id=row["run_id"], case=row["case"], method=row["method"],
                comparator=method, dt_s=row["dt_s"], variant=row["variant"],
                new_success=ours["success"], old_success=theirs["success"],
                new_time_s=ours["complete_time_s"], old_time_s=theirs["complete_time_s"],
                comparable_complete_executions=both,
                time_difference_s=ours["complete_time_s"] - theirs["complete_time_s"] if both else None,
                saving_percent=100 * (theirs["complete_time_s"] - ours["complete_time_s"]) / theirs["complete_time_s"] if both else None,
                outcome="diagnostic_only" if not formal else "both_complete" if both else "no_complete_time_comparison"))
    return comparisons


def selection_review(rows, missing):
    candidates = sorted({r["omega"] for r in rows if r.get("omega") is not None})
    included = [r for r in rows if r["split"] in ("train", "validation") and r.get("formal_new_experiment") is True]
    forbidden = [r["run_id"] for r in rows if r["case"] in CASES and r["split"] in ("train", "validation")]
    family = {}
    for row in included:
        family.setdefault(row["omega"], set()).add((row["case"], row["method"], row["dt_s"], row.get("seed")))
    same = bool(family) and all(keys == next(iter(family.values())) for keys in family.values())
    scores = []
    for omega in candidates:
        group = [r for r in included if r["omega"] == omega]
        validation = [r for r in group if r["split"] == "validation"]
        complete = bool(group) and all(r["metrics"]["success"] for r in group)
        methods = {r["method"] for r in group}
        unique = len(group) == len(family.get(omega, set()))
        eligible = (complete and same and unique and bool(validation) and methods == {"JNN", "FIXED"}
                    and candidates == [40, 60, 80] and not forbidden and not missing)
        scores.append(dict(omega=omega, cells=len(group), complete_cells=sum(r["metrics"]["success"] for r in group),
            eligible_on_available_index=eligible,
            validation_geometric_mean_complete_time_s=math.exp(sum(math.log(r["metrics"]["complete_time_s"]) for r in validation) / len(validation)) if eligible else None))
    return dict(candidates_observed=candidates, intended_candidates=[40, 60, 80], same_selection_grid=same,
        representative_cases_mislabeled_for_selection=forbidden, scores=scores, selected_omega=None,
        automatic_selection=False,
        reason="This audit checks available evidence; it does not invent or retrospectively certify a prospective selection rule. Parent must bind the complete candidate grid and rule before outcome-dependent selection.",
        recommended_rule="Freeze one common grid for omega40/60/80 and both JNN/Fixed. Require every selection cell complete and feasible, then choose one global omega using equally weighted log completion times across the prespecified validation cells; break exact ties by the smallest omega. Training checks gate eligibility. No per-program omega selection or representative-case selection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-index", type=Path)
    parser.add_argument("--output", default="comparison.json", help="Filename under audit/ only")
    args = parser.parse_args()
    inventory_file = AUDIT / "baseline_inventory.json"
    inventory = build_old()
    if inventory_file.exists():
        require(read(inventory_file) == inventory, "Old evidence changed; preserve previous inventory and investigate")
    else:
        write(inventory_file, inventory)
    rows, missing, receipt = audit_new(args.new_index, inventory) if args.new_index else ([], [], None)
    result = dict(schema_version=1, generator_sha256=sha(__file__), baseline_inventory_sha256=sha(inventory_file),
        old_cells=56, new_index=receipt, new_cells=len(rows), missing_registered_runs=missing,
        status="audited" if rows and not missing else "incomplete_new_grid" if args.new_index else "baseline_ready_no_new_runs",
        acceptance=dict(limits=LIMITS, rest_s=.020, geometric_rule="Every original gate, every whole command interval, original complete reference, effective tolerance after chord-error reservation",
                        numerical_scope="Simulated piecewise-linear commands and vector backward differences, not physical axis tracking or continuous servo dynamics"),
        scientific_boundaries=["Four representative programs were previously observed; they are development/detail comparisons, not an independent held-out test.",
            "One frozen policy and deterministic replays do not establish five-seed training reproducibility.",
            "Two command periods assess numerical resolution and are not independent statistical replicates.",
            "No success-only average; failures retain null completion time and cannot produce a speed superiority claim.",
            "Changing omega changes the execution law. Report a new method version and apply the same law to Fixed before attributing effects to preview adaptation.",
            "Different paths are not exchangeable training seeds; do not label path variation as training standard deviation or confidence intervals.",
            "Offline terminal-prefix splices and diagnostic runs are not whole-policy executions; only an explicitly authorized run index is read."],
        rows=rows, comparisons=compare(rows, inventory), selection_audit=selection_review(rows, missing))
    write(AUDIT / args.output, result)
    print(json.dumps(dict(output=str((AUDIT / args.output).resolve()), old_cells=56, new_cells=len(rows),
        missing_runs=len(missing), status=result["status"]), indent=2))


if __name__ == "__main__":
    main()
