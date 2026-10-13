"""Check explicit final-study receipts; never discover or execute holdout runs.

This checks the formal target and evidence bindings. Whole-trajectory geometry
must already have been checked by the independently bound trajectory auditor.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def assess(rules, evidence=None, *, artifact_root=None, rules_sha256=None, allow_synthetic=False):
    issues, comparisons, stronger, cells, summaries = [], [], [], [], []

    def issue(level, code, detail):
        issues.append(dict(level=level, code=code, detail=detail))

    def require(condition, code, detail, level="invalid"):
        if not condition:
            issue(level, code, detail)
        return bool(condition)

    def artifact(item, label, *, parse=True):
        if not isinstance(item, dict) or not hash_value(item.get("sha256")) or not item.get("path"):
            issue("pending", "missing_artifact_binding", label)
            return None
        if artifact_root is None:
            issue("pending", "missing_artifact_root", label)
            return None
        root = Path(artifact_root).resolve()
        file = (root / item["path"]).resolve()
        if not file.is_relative_to(root):
            issue("invalid", "artifact_outside_explicit_root", label)
            return None
        if not file.is_file():
            issue("pending", "missing_artifact", label)
            return None
        data = file.read_bytes()
        if digest(data) != item["sha256"]:
            issue("invalid", "artifact_hash_mismatch", label)
            return None
        if not parse:
            return data
        try:
            return json.loads(data)
        except (UnicodeError, ValueError):
            issue("invalid", "invalid_json_artifact", label)
            return None

    def finish():
        levels = {x["level"] for x in issues}
        status = next((x for x in ("invalid", "fail", "pending") if x in levels), "pass")
        synthetic = evidence is not None and evidence.get("evidence_kind") == "synthetic_fixture"
        return dict(schema_version=1, rule_id=rules.get("rule_id"), status=status,
            evidence_kind="none" if evidence is None else evidence.get("evidence_kind"),
            synthetic_fixture_only=synthetic,
            scientific_target_passed=status == "pass" and not synthetic and evidence is not None,
            submission_readiness="not_determined_by_this_validator",
            rule_scope="Complete registered grid, paired-seed strict time dominance; stronger envelope separately reported.",
            no_model_or_planner_calls=True, no_holdout_directory_discovery=True,
            issues=issues, cells=cells, paired_comparisons=comparisons,
            stronger_envelope_review=stronger, summaries=summaries,
            statistical_unit="training seed within complete program; resolutions are separate numerical checks",
            limitation="A pass assumes the SHA-bound independent trajectory-audit receipts are valid; this tool does not rerun geometry, training, or journal checks.")

    if not require(rules.get("schema_version") == 1 and rules.get("kind") == "formal_acceptance_rules",
                   "unsupported_rules", "Use the version-1 formal acceptance schema."):
        return finish()
    study = rules.get("study", {})
    require(study.get("status") == "frozen", "study_not_frozen", "Final study registration remains pending.", "pending")
    for key in ("registration_sha256", "path_registry_sha256", "learning_fairness_audit_sha256",
                "model_selection_rule_sha256", "family_separation_audit_sha256"):
        require(hash_value(study.get(key)), "pending_study_binding", key, "pending")
    require(study.get("frozen_at_utc") is not None, "missing_freeze_time", "Final registration time is not set.", "pending")
    require(study.get("final_training_complete") is True, "final_training_pending", "Five final runs are not yet bound.", "pending")
    contract = study.get("common_contract", {})
    require(hash_value(contract.get("contract_sha256")), "pending_contract", "Common completion/auditor contract hash.", "pending")
    seeds = study.get("seed_ids", [])
    require(len(seeds) == 5 and len(set(seeds)) == 5 and all(type(s) is int for s in seeds),
            "invalid_seed_set", "Exactly five distinct predeclared integer training seeds are required.")
    paths = study.get("paths", [])
    require(len(paths) >= 5 and len({p.get("id") for p in paths}) == len(paths),
            "invalid_path_set", "At least five distinct complete programs are required.")
    for path in paths:
        require(all(hash_value(path.get(k)) for k in ("reference_sha256", "geometry_sha256")),
                "invalid_path_binding", str(path.get("id")))
        nominal, reserve, allowance = (path.get(k) for k in
            ("nominal_tolerance_mm", "arc_chord_error_bound_mm", "evaluation_tolerance_mm"))
        require(all(finite(x) for x in (nominal, reserve, allowance)) and nominal == contract.get("nominal_tolerance_mm")
                and reserve >= 0 and allowance > 0 and abs(nominal - reserve - allowance) <= 1e-12,
                "invalid_arc_budget", str(path.get("id")))
    grid = study.get("seed_grid_by_dt_us", {})
    require(set(grid) == {"1000", "500"}, "missing_resolution_grid", "Both 1 ms and 0.5 ms must be registered.", "pending")
    require(grid.get("1000") == seeds, "nominal_seed_grid", "The nominal grid must include all five seeds.")
    for dt, members in grid.items():
        require(isinstance(members, list) and len(members) > 0,
                "pending_resolution_seed_grid", dt, "pending")
        if isinstance(members, list):
            require(len(set(members)) == len(members) and set(members) <= set(seeds),
                    "invalid_resolution_seed_grid", dt)
    main = study.get("main_method", {})
    inventory = study.get("comparators", [])
    require(main.get("selection") == "selected" and main.get("kind") == "learned",
            "main_identity_pending", "Final main-method identity remains pending.", "pending")
    for method in inventory:
        require(method.get("selection") in ("selected", "excluded"), "comparator_selection_pending", method.get("id"), "pending")
        if method.get("selection") == "excluded":
            require(bool(method.get("exclusion_reason")), "unexplained_comparator_exclusion", method.get("id"))
    comparators = [m for m in inventory if m.get("selection") == "selected"]
    require(bool(comparators), "no_predeclared_comparators", "No selected comparator; an empty comparison never passes.", "pending")
    methods = [main] + comparators
    require(len({m.get("id") for m in methods}) == len(methods), "duplicate_method_id", "Method IDs must be unique.")
    for method in methods:
        require(method.get("kind") in ("learned", "deterministic"), "invalid_method_kind", method.get("id"))
        require(hash_value(method.get("method_identity_sha256")), "method_identity_pending", method.get("id"), "pending")
        if method.get("kind") == "learned":
            for field in ("training_recipe_sha256", "common_learning_sha256"):
                require(hash_value(method.get(field)), "training_identity_pending", f"{method.get('id')}: {field}", "pending")
            require(isinstance(method.get("training_budget"), dict) and bool(method["training_budget"]),
                    "training_budget_pending", method.get("id"), "pending")
            if method.get("id") == "fixed":
                require(method.get("common_learning_sha256") == main.get("common_learning_sha256")
                        and method.get("training_budget") == main.get("training_budget"),
                        "unequal_fixed_preview_learning_contract", "Observations, model capacity, data, executor, selection and full budget must match.")
    if evidence is None:
        issue("pending", "no_explicit_final_evidence", "No final receipt manifest supplied; no final results were read.")
        return finish()
    synthetic = evidence.get("evidence_kind") == "synthetic_fixture"
    require(evidence.get("evidence_kind") in ("experimental", "synthetic_fixture"), "invalid_evidence_kind", "Unknown evidence scope.")
    require(not synthetic or allow_synthetic, "synthetic_not_authorized", "Use --allow-synthetic-fixtures for software tests only.")
    require(hash_value(rules_sha256) and evidence.get("rules_sha256") == rules_sha256,
            "rules_hash_mismatch", "Evidence must bind the exact frozen rules bytes.")
    # Pending registration must never cause result files to be opened prematurely.
    if any(x["level"] in ("invalid", "pending") for x in issues):
        issue("pending", "evidence_not_opened", "Resolve/freeze study metadata before supplying final receipts.")
        return finish()
    ledger = artifact(evidence.get("ledger_artifact"), "complete attempt ledger")
    if ledger is not None:
        for flag in ("all_attempts_retained", "closed", "no_final_test_selection", "source_identity_audit_passed",
                     "learning_fairness_audit_passed", "family_separation_audit_passed", "cost_accounting_complete"):
            require(ledger.get(flag) is True, "ledger_not_accepted", flag, "pending")
        require(ledger.get("registration_sha256") == study["registration_sha256"]
                and ledger.get("rules_sha256") == rules_sha256, "ledger_identity_mismatch", "Ledger registration/rules mismatch.")
        require(ledger.get("evidence_kind") == evidence["evidence_kind"], "ledger_scope_mismatch", "Experimental and synthetic scopes cannot be mixed.")
        for key in ("learning_fairness_audit_sha256", "family_separation_audit_sha256"):
            require(ledger.get(key) == study[key], "ledger_audit_identity_mismatch", key)
        require(ledger.get("fixed_preview_target_conversion_audited") is True or not any(m["id"] == "fixed" for m in comparators),
                "fixed_target_conversion_pending", "Fixed-preview steering targets must use the registered conversion.", "pending")
    training, runs = [], []
    for index, item in enumerate(evidence.get("training_artifacts", [])):
        row = artifact(item, f"training receipt {index}")
        if row is not None:
            training.append(row)
    for index, item in enumerate(evidence.get("run_artifacts", [])):
        row = artifact(item, f"run receipt {index}")
        if row is not None:
            runs.append(row)
    ids = [r.get("attempt_id") for r in runs]
    require(all(isinstance(x, str) and x for x in ids) and len(set(ids)) == len(ids), "duplicate_or_missing_attempt_id", "Every actual attempt needs a unique ID.")
    if ledger is not None:
        require(len(ledger.get("attempt_ids", [])) == len(ids) and set(ledger.get("attempt_ids", [])) == set(ids),
                "ledger_attempts_missing_or_extra", "Include every ledger attempt, including failures.")
    learned = [m for m in methods if m["kind"] == "learned"]
    training_lookup = {}
    for method in learned:
        for seed in seeds:
            matches = [r for r in training if (r.get("method"), r.get("seed")) == (method["id"], seed)]
            if not require(len(matches) == 1, "missing_or_duplicate_training_receipt", f"{method['id']}/{seed}", "pending" if not matches else "invalid"):
                continue
            row = matches[0]
            for key in ("method_identity_sha256", "training_recipe_sha256", "common_learning_sha256", "training_budget"):
                require(row.get(key) == method[key], "training_identity_or_budget_mismatch", f"{method['id']}/{seed}: {key}")
            for flag in ("complete", "independent_seed_initialization_verified", "final_checkpoint_selected_by_registered_rule",
                         "no_final_test_selection", "all_training_attempts_retained", "cost_accounting_complete"):
                require(row.get(flag) is True, "incomplete_training_evidence", f"{method['id']}/{seed}: {flag}", "pending")
            require(row.get("evidence_kind") == evidence["evidence_kind"], "training_scope_mismatch", str(seed))
            require(row.get("actual_training_budget") == method["training_budget"], "incomplete_training_budget", f"{method['id']}/{seed}", "fail")
            require(row.get("model_selection_rule_sha256") == study["model_selection_rule_sha256"], "model_selection_mismatch", str(seed))
            artifact(row.get("model_artifact"), f"trained model {method['id']}/{seed}", parse=False)
            training_lookup[method["id"], seed] = row
    require(len(training) == len(learned) * len(seeds), "extra_training_receipts", "Final receipt set must match the frozen method/seed set; prior development is separate.")
    expected = {(p["id"], m["id"], seed, int(dt)) for p in paths for dt, members in grid.items()
                for m in methods for seed in (members if m["kind"] == "learned" else [None])}
    indexed = {}
    for row in runs:
        key = (row.get("path"), row.get("method"), row.get("seed"), row.get("dt_us"))
        if not require(key in expected, "unregistered_run", str(key)):
            continue
        if not require(key not in indexed, "duplicate_cell_requires_prefrozen_attempt_policy", str(key)):
            continue
        indexed[key] = row
    for key in sorted(expected, key=str):
        if key not in indexed:
            issue("pending", "missing_run", str(key))
            cells.append(dict(path=key[0], method=key[1], seed=key[2], dt_us=key[3], status="missing"))
    accepted = {}
    path_lookup, method_lookup = {p["id"]: p for p in paths}, {m["id"]: m for m in methods}
    flags = ("initial_rest", "ordered_complete", "all_command_intervals_inside", "braking_complete", "final_rest", "planar_vector_limits")
    for key, row in indexed.items():
        path, method = path_lookup[key[0]], method_lookup[key[1]]
        before = len(issues)
        for field, value in (("reference_sha256", path["reference_sha256"]), ("geometry_sha256", path["geometry_sha256"]),
            ("method_identity_sha256", method["method_identity_sha256"]), ("contract_sha256", contract["contract_sha256"]),
            ("registration_sha256", study["registration_sha256"]), ("evidence_kind", evidence["evidence_kind"]),
            ("limits", contract["limits"]), ("nominal_tolerance_mm", path["nominal_tolerance_mm"]),
            ("arc_chord_error_bound_mm", path["arc_chord_error_bound_mm"]), ("evaluation_tolerance_mm", path["evaluation_tolerance_mm"])):
            require(row.get(field) == value, "run_identity_mismatch", f"{key}: {field}")
        require(isinstance(row.get("started_utc"), str) and row["started_utc"] > study["frozen_at_utc"],
                "run_precedes_freeze", str(key))
        for flag in ("independent_full_trace_audit", "all_command_intervals_accounted", "action_sources_verified", "cost_accounting_complete"):
            require(row.get(flag) is True, "run_audit_incomplete", f"{key}: {flag}", "pending")
        if method["kind"] == "learned":
            receipt = training_lookup.get((method["id"], key[2]), {})
            require(row.get("model_sha256") == receipt.get("model_artifact", {}).get("sha256") and hash_value(row.get("model_sha256")),
                    "run_model_mismatch", str(key))
        status, time_s = row.get("status"), row.get("complete_time_s")
        require(status in ("completed", "failed", "unresolved"), "invalid_outcome", str(key))
        if status == "completed":
            artifact(row.get("trace_artifact"), f"complete trace {key}", parse=False)
            passed = row.get("constraints", {})
            require(all(passed.get(f) is True for f in flags), "false_completion", str(key), "fail")
            ratios = row.get("vector_limit_ratios", [])
            require(len(ratios) == 3 and all(finite(x) and 0 <= x <= 1 + contract["vector_numerical_relative_tolerance"] for x in ratios),
                    "vector_limit_violation", str(key), "fail")
            require(finite(row.get("terminal_rest_s")) and row["terminal_rest_s"] >= contract["terminal_rest_minimum_s"] - 1e-10,
                    "terminal_rest_incomplete", str(key), "fail")
            require(finite(time_s) and time_s > 0 and type(row.get("command_intervals")) is int
                    and row["command_intervals"] > 0 and abs(time_s - row["command_intervals"] * key[3] / 1e6) <= 1e-9,
                    "invalid_complete_time", str(key), "fail")
            require(row.get("time_includes_start_braking_final_rest") is True, "time_scope_incomplete", str(key), "fail")
        else:
            require(time_s is None, "failed_prefix_used_as_completion_time", str(key))
            require(bool(row.get("reason")), "missing_failure_reason", str(key))
            if row.get("trace_artifact") is not None:
                artifact(row["trace_artifact"], f"failed prefix {key}", parse=False)
            if status == "unresolved":
                issue("pending", "unresolved_run", str(key))
            elif method["id"] == main["id"]:
                issue("fail", "main_run_incomplete", str(key))
        eligible = status == "completed" and len(issues) == before
        accepted[key] = eligible
        cells.append(dict(path=key[0], method=key[1], seed=key[2], dt_us=key[3], status=status,
                          eligible_complete_time=eligible, complete_time_s=time_s if eligible else None, reason=row.get("reason")))
    for path in paths:
        for dt, members in grid.items():
            dt = int(dt)
            for seed in members:
                key = (path["id"], main["id"], seed, dt)
                feasible = []
                for method in comparators:
                    other = (path["id"], method["id"], seed if method["kind"] == "learned" else None, dt)
                    if accepted.get(other):
                        feasible.append(other)
                        if accepted.get(key):
                            delta = indexed[other]["complete_time_s"] - indexed[key]["complete_time_s"]
                            win = delta > rules["objective"]["strict_time_margin_s"]
                            comparisons.append(dict(path=path["id"], dt_us=dt, seed=seed, comparator=method["id"],
                                comparator_seed=other[2], comparator_minus_main_s=delta, strictly_faster=win))
                            require(win, "paired_dominance_not_met", f"{key} vs {method['id']}", "fail")
                require(bool(feasible), "no_feasible_comparator_for_cell", str(key), "pending")
            main_times = [indexed[k]["complete_time_s"] for k in expected if k[0] == path["id"] and k[1] == main["id"] and k[3] == dt and accepted.get(k)]
            for method in comparators:
                other_times = [indexed[k]["complete_time_s"] for k in expected if k[0] == path["id"] and k[1] == method["id"] and k[3] == dt and accepted.get(k)]
                stronger.append(dict(path=path["id"], dt_us=dt, comparator=method["id"], required_for_acceptance=False,
                    eligible=len(main_times) == len(members) and bool(other_times),
                    main_slowest_s=max(main_times) if main_times else None,
                    comparator_fastest_s=min(other_times) if other_times else None,
                    passed=max(main_times) < min(other_times) if len(main_times) == len(members) and other_times else None))
            for method in methods:
                rows = [c for c in cells if c["path"] == path["id"] and c["method"] == method["id"] and c["dt_us"] == dt]
                times = [c["complete_time_s"] for c in rows if c.get("eligible_complete_time")]
                total = len(members) if method["kind"] == "learned" else 1
                summaries.append(dict(path=path["id"], dt_us=dt, method=method["id"], expected=total,
                    completed=len(times), failed=sum(c["status"] == "failed" for c in rows),
                    unresolved_or_missing=sum(c["status"] in ("unresolved", "missing") for c in rows),
                    mean_s=statistics.mean(times) if len(times) == total else None,
                    sample_sd_s=statistics.stdev(times) if len(times) == total and total > 1 else None,
                    conditional_success_only_mean_not_reported=True))
    return finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-synthetic-fixtures", action="store_true")
    args = parser.parse_args()
    rules_bytes = args.rules.read_bytes()
    rules = json.loads(rules_bytes)
    evidence = json.loads(args.evidence.read_bytes()) if args.evidence else None
    result = assess(rules, evidence, artifact_root=args.artifact_root,
                    rules_sha256=digest(rules_bytes), allow_synthetic=args.allow_synthetic_fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "evidence_kind", "scientific_target_passed")}))
    raise SystemExit(0 if result["status"] == "pass" else 2 if result["status"] == "pending" else 1)


if __name__ == "__main__":
    main()
