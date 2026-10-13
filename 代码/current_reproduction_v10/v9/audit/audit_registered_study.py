"""Independently inventory one registered tuning grid and reproduce its rule.

Only explicitly supplied study/results/tuning is inspected. The later straight
study and offline terminal research are not automatically traversed.
"""
import argparse
from pathlib import Path
import json

import numpy as np

from compare_results import AUDIT, OLD, read, sha, require, write


def audit(study, label):
    study = Path(study).resolve()
    registration_path = study / "registration.json"
    selection_path = study / "selection.json"
    registration, selection = read(registration_path), read(selection_path)
    registration_sha = sha(registration_path)
    require(selection["registration_sha256"] == registration_sha, "Selection is not bound to registration")
    source_receipts = {}
    for prefix, root in (("source_sha256", OLD), ("own_source_sha256", study)):
        for relative, expected in registration.get(prefix, {}).items():
            path = root / relative
            require(sha(path) == expected, "Executed source changed: " + str(path))
            source_receipts[str(path)] = expected
    cases = [c for c in registration["cases"] if c["historical_split"] in ("train", "validation")]
    methods, omegas, dt = registration["methods"], registration["candidates"], registration["tuning_dt_s"]
    records, receipts, index, expected = {}, [], [], set()
    for omega in omegas:
        for case in cases:
            reference = OLD / case["reference"]
            require(sha(reference) == case["reference_sha256"], "Changed tuning reference")
            points = np.loadtxt(reference, delimiter=",", skiprows=1)
            budget = 5. + float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / 5.
            for method in methods:
                stem = study / "results/tuning" / f"omega{omega:g}" / f"{case['id']}_{method}_dt{round(dt * 1e6)}us"
                record, trace, log = stem.with_suffix(".json"), stem.with_suffix(".npz"), stem.with_suffix(".log")
                expected.add(record)
                require(record.exists() and trace.exists() and log.exists(), "Missing attempted-run evidence: " + str(stem))
                raw = read(record)
                for key, value in (("case", case["id"]), ("method", method), ("omega", omega), ("dt_s", dt),
                                   ("registration_sha256", registration_sha), ("reference_sha256", case["reference_sha256"])):
                    require(raw[key] == value, "Run identity mismatch: " + str(record) + ":" + key)
                require(raw["model_sha256"] == registration["model_sha256"][method], "Changed policy checkpoint")
                require(sha(trace) == raw["trace_sha256"], "Changed trajectory")
                require(raw["success"] == (raw["complete_time_s"] is not None), "Failure assigned a completion time")
                records[omega, case["id"], method] = raw
                receipts.append(dict(record=str(record), record_sha256=sha(record), trace=str(trace),
                    trace_sha256=sha(trace), log=str(log), log_sha256=sha(log)))
                index.append(dict(run_id=f"{label}_{case['id']}_{method}_omega{omega:g}_dt{round(dt * 1e6)}us",
                    case=case["id"], method=method, omega=omega, dt_s=dt, split=case["historical_split"],
                    variant=registration["version"] + f"_omega{omega:g}", formal_new_experiment=True,
                    record=str(record), trace=str(trace), reference=str(reference),
                    reference_sha256=case["reference_sha256"], nominal_tolerance_mm=case["nominal_tolerance_mm"],
                    evaluation_tolerance_mm=case["evaluation_tolerance_mm"], arc_chord_error_bound_mm=case["arc_chord_error_bound_mm"],
                    physical_time_budget_s=budget))
    actual = set((study / "results/tuning").glob("omega*/*.json"))
    unexpected = sorted(str(path) for path in actual - expected)
    require(not unexpected, "Unindexed tuning records exist: " + str(unexpected))
    baseline = {(case, method): result for (omega, case, method), result in records.items() if omega == 40.}
    validation = [(case["id"], "JNN") for case in cases if case["historical_split"] == "validation" and baseline[case["id"], "JNN"]["success"]]
    candidate_rows = []
    for omega in omegas:
        lost = [list(key) for key, result in baseline.items() if result["success"] and not records[(omega,) + key]["success"]]
        failures = {method: sum(not records[omega, case["id"], method]["success"] for case in cases) for method in methods}
        eligible = not lost and all(failures[method] <= sum(not result["success"] for key, result in baseline.items() if key[1] == method) for method in methods)
        score = sum(records[(omega,) + key]["complete_time_s"] for key in validation) if eligible else None
        candidate_rows.append(dict(omega=omega, eligible=eligible, lost_baseline_complete_cells=lost,
            failure_counts=failures, validation_sum_s=score))
    selected = min((row for row in candidate_rows if row["eligible"]), key=lambda row: (row["validation_sum_s"], row["omega"]))
    require(selection["selected_omega"] == selected["omega"], "Selection does not implement its registered ranking")
    require(selection["adoptable"] == (selected["omega"] != 40.), "Adoption decision differs")
    require(selection["candidates"] == candidate_rows, "Candidate selection table differs")
    require(selection["validation_cells"] == [list(key) for key in validation], "Validation scoring cells differ")
    copied = {(row["omega"], row["case"], row["method"]): row for row in selection["records"]}
    require(len(copied) == len(selection["records"]) == len(records) and set(copied) == set(records), "Selection omitted/duplicated runs")
    for key, result in records.items():
        require(copied[key]["success"] == result["success"] and copied[key]["time_s"] == result["complete_time_s"], "Selection altered an outcome")
    index_path = AUDIT / f"{label}_run_index.json"
    write(index_path, dict(schema_version=1, study_root=str(study), registration_sha256=registration_sha, runs=index))
    output = dict(study_root=str(study), registration_sha256=registration_sha, selection_sha256=sha(selection_path),
        index=str(index_path), index_sha256=sha(index_path), source_sha256=source_receipts,
        expected_cells=len(omegas) * len(cases) * len(methods), record_cells=len(records), trace_cells=len(receipts),
        log_cells=len(receipts), complete_cells=sum(row["success"] for row in records.values()),
        failed_cells=sum(not row["success"] for row in records.values()), missing_cells=[], unexpected_tuning_records=unexpected,
        selection_recomputed=True, selected_omega=selected["omega"], adoptable=selection["adoptable"],
        candidates=candidate_rows, validation_score_cells=[list(key) for key in validation],
        selection_interpretation="No-regression selection relative to omega40, not a requirement that every tuning path completes. Previously unsuccessful baseline cells remain failures and do not contribute a fabricated time.",
        scope_cautions=["The chosen score is registered JNN validation total, with Fixed guarded against lost successes/increased failures; it is not a symmetric joint-method timing objective.",
            "Baseline-complete validation cells alone determine the time score; all other outcomes remain visible. The score is not proof of reliability over the whole validation set.",
            "Rejection of omega60/80 on lost baseline-complete cells remains valid under the stricter all-cells-must-complete rule.",
            "These are deterministic runs from existing checkpoints; no independent-seed or held-out-test inference follows."],
        evidence=receipts, full_interval_audit_entry=f"compare_results.py --new-index {index_path} --output {label}_comparison.json")
    write(AUDIT / f"{label}_registration_audit.json", output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=AUDIT.parent)
    parser.add_argument("--label", default="global_tracking")
    args = parser.parse_args()
    result = audit(args.study_root, args.label)
    print(json.dumps({key: result[key] for key in ("expected_cells", "record_cells", "complete_cells", "failed_cells", "selected_omega", "adoptable", "index")}, indent=2))
