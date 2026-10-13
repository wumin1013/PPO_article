"""Read-only summaries of completed preview-study validation.

Never load a policy, run an environment, or read local-corner live checkpoints.
Immutable stepN snapshots are correlated checkpoints, not independent trials. Missing completion
records remain pending. All report output directories must be new.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "论文项目/revision_20260910_continued"
REPORTS = BASE / "reports"
MANIFEST = ROOT / "PPO_project/data/preview_event_v1/manifest.json"
SPECS = {"positive": ("positive_feed_v1", "pilot"), "local": ("local_corner_v1", "development"),
         "full_batch": ("full_batch_v1", "development"), "target_tracking": ("target_tracking_v1", "development")}
SNAPSHOT_STUDIES = {"full_batch", "target_tracking"}
INPUTS = {}


def bytes_read(path):
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while reading; retry with a new report: {path}")
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
    return content


def bind(path, expected=None):
    value = hashlib.sha256(bytes_read(path)).hexdigest()
    assert expected is None or value == expected, f"SHA mismatch: {path}"
    return value


def read(path):
    return json.loads(bytes_read(path).decode("utf-8"))


def safe_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(count=len(values), minimum=float(values.min()) if len(values) else None,
                mean=float(values.mean()) if len(values) else None, maximum=float(values.max()) if len(values) else None)


def trace_diagnostics(path, expected):
    bind(path, expected)
    with np.load(path, allow_pickle=False) as archive:
        c = dict(zip(archive["columns"].astype(str), archive["data"].T))
    for name in ("timestamp_s", "feed_logit", "feed_executed_action", "macro_decision"):
        assert name in c, f"Required positive-feed field absent: {path}:{name}"
    t = c["timestamp_s"]
    terminal = np.nan_to_num(c.get("terminal_braking", np.zeros(len(t))), nan=0.) > .5
    executed = (~terminal) & np.isfinite(c["feed_logit"]) & np.isfinite(c["feed_executed_action"])
    decisions = executed & (np.nan_to_num(c["macro_decision"], nan=0.) > .5)
    logits, action = c["feed_logit"][executed], c["feed_executed_action"][executed]
    sigmoid = np.exp(-np.logaddexp(0., -logits))
    assert np.allclose(action, sigmoid, rtol=1e-12, atol=1e-12), f"Executed sigmoid mismatch: {path}"
    tail_indices = np.flatnonzero(terminal)
    if len(tail_indices):
        first = int(tail_indices[0])
        assert first > 0 and np.all(terminal[first:]), "Terminal tail must be a contiguous suffix"
        tail_start, tail_s = float(t[first-1]), float(t[-1]-t[first-1])
    else:
        tail_start = tail_s = None
    return dict(terminal_tail_s=tail_s, terminal_tail_start_s=tail_start, terminal_tail_samples=int(terminal.sum()),
                policy_command_samples=int(executed.sum()), macro_decisions_observed=int(decisions.sum()),
                feed_logit_at_decisions=safe_stats(c["feed_logit"][decisions]),
                feed_executed_action=safe_stats(action),
                executed_zero_actions=int((action == 0.).sum()), executed_one_actions=int((action == 1.).sum()),
                negative_feed_logits=int((logits < 0).sum()),
                zero_physical_feed_policy_samples=int(((c["v_mm_s"] == 0.) & executed).sum()),
                note="feed_logit is the Gaussian coordinate; negative logits are valid. feed_executed_action is sigmoid(logit). Terminal braking is excluded; latent_1 is never used. Validation actions are deterministic, not training exploration samples.")


def episode_diagnostic(live_run, training):
    path = live_run / "episodes.jsonl"
    if not path.exists():
        return dict(status="unavailable", scope="Training-only diagnostic; not performance selection")
    # Logs may grow while a later milestone trains. One consistent byte snapshot
    # is enough; a partial final line is ignored and never treated as a sample.
    content = path.read_bytes()
    records, malformed = {}, 0
    for line in content.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if row["episode"] < training["episodes"] and row["decisions"] <= training["decisions"]:
            records[row["episode"]] = row
    return dict(status="complete" if len(records) == training["episodes"] else "partial",
                log_path=path.relative_to(ROOT).as_posix(), byte_snapshot_sha256=hashlib.sha256(content).hexdigest(),
                completed_episodes=len(records), expected_completed_episodes=training["episodes"],
                successes=sum(bool(r["result"]["success"]) for r in records.values()),
                failures=dict(Counter(reason for r in records.values() for reason in r["result"].get("failure_reasons", []))),
                ignored_partial_or_invalid_lines=malformed,
                scope="Training diagnosis only; never selected or pooled with deterministic validation performance")


def prior_optimizer_cost(folder, decisions):
    path = folder/"updates.jsonl"
    if not path.exists():
        return dict(optimizer_updates=None, adam_steps_per_network=None,
                    optimizer_cost_scope="Historical update log unavailable; missing is not zero")
    records = [json.loads(line) for line in bytes_read(path).splitlines() if line.strip()]
    assert sum(r["transition_count"] for r in records) == decisions
    assert records[-1]["decisions"] == decisions
    return dict(optimizer_updates=len(records),
                adam_steps_per_network=sum(len(r["epochs"]) for r in records),
                optimizer_cost_scope="Exact completed update and per-network epoch counts from historical updates.jsonl")


def process_checkpoint(study, folder, live_run, step, protocol, expected_paths):
    prefix = dict(study=study, method=live_run.name.removeprefix("preview_time_").rsplit("_seed", 1)[0],
                  seed=int(live_run.name.rsplit("_seed", 1)[1]), additional_decisions=step,
                  checkpoint_directory=folder.relative_to(ROOT).as_posix(), live_run=live_run.relative_to(ROOT).as_posix(),
                  initial_diagnostic=bool(study in SNAPSHOT_STUDIES and step == 0),
                  selection_eligible=not (study in SNAPSHOT_STUDIES and step == 0))
    required = [folder/"training_summary.json", folder/"validation/summary.json", folder/"validation/binding.json",
                folder/"config.yaml", folder/"provenance.json", folder/"source.zip"]
    if study in SNAPSHOT_STUDIES:
        required.append(folder/"snapshot_manifest.json")
    missing = [p.relative_to(folder).as_posix() for p in required if not p.exists()]
    if missing:
        # Critically, the checkpoint file is not touched while completion is absent.
        return dict(**prefix, status="pending", missing=missing), []
    try:
        training, evaluation = read(required[0]), read(required[1])
        binding, provenance = read(required[2]), read(folder/"provenance.json")
    except (json.JSONDecodeError, RuntimeError):
        return dict(**prefix, status="pending", missing=["completion metadata still being written"]), []
    expected_status = "registered_development_milestone_complete" if study in SNAPSHOT_STUDIES else "registered_development_budget_complete"
    assert training["status"] == expected_status
    assert training["decisions"] == step and training["method"] == prefix["method"] and training["seed"] == prefix["seed"]
    if study in SNAPSHOT_STUDIES:
        assert training["optimized_decisions"] == step and training["pending_buffer_decisions"] == 0
        assert training["zero_step_initial_diagnostic"] == (step == 0)
        assert training["optimizer_updates"] == step // protocol["batch_macro_decisions"]
        snapshot_manifest = read(folder/"snapshot_manifest.json")
        assert snapshot_manifest["decisions"] == step
        expected_members = {"checkpoint.pt", "config.yaml", "provenance.json", "source.zip", "training_summary.json",
                            "updates.jsonl", "episodes.jsonl", "training_final_partial.npz"}
        assert expected_members <= set(snapshot_manifest["files"]), "Incomplete snapshot manifest"
        for name, expected in snapshot_manifest["files"].items():
            member = (folder/name).resolve()
            assert member.is_relative_to(folder.resolve()), "Snapshot member escapes its directory"
            bind(member, expected)
    rows = evaluation["rows"]
    assert len(rows) == len(expected_paths) and {r["path_id"] for r in rows} == expected_paths
    assert binding["split"] == "validation"
    assert binding["source_sha256"] == provenance["source_sha256"] == protocol["source"]["source_sha256"]
    assert provenance["source_files"] == protocol["source"]["source_files"]
    source_digest = hashlib.sha256(json.dumps(provenance["source_files"], sort_keys=True).encode()).hexdigest()
    assert source_digest == binding["source_sha256"]
    protocol_key = (BASE/SPECS[study][0]/"protocol.json").relative_to(ROOT).as_posix()
    assert provenance["protocol_sha256"] == INPUTS[protocol_key]
    checkpoint_sha = bind(folder/"checkpoint.pt", training["checkpoint_sha256"])
    assert binding["checkpoint_sha256"] == checkpoint_sha
    config_sha = bind(folder/"config.yaml", binding["config_sha256"])
    assert config_sha == provenance["config_sha256"]
    bind(MANIFEST, binding["manifest_sha256"])
    bind(folder/"source.zip")
    with zipfile.ZipFile(folder/"source.zip") as archive:
        for name, expected in provenance["source_files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
            bind(ROOT/name, expected)
    cfg = yaml.safe_load((folder/"config.yaml").read_text(encoding="utf-8"))
    prior_path = ROOT/cfg["revision"]["init_checkpoint"]
    prior_sha = bind(prior_path, cfg["revision"]["init_checkpoint_sha256"])
    prior = read(prior_path.parent/"training_summary.json")
    assert prior["checkpoint_sha256"] == prior_sha
    assert prior["decisions"] == cfg["revision"]["pretraining_macro_decisions"]
    complete_rows = []
    for row in rows:
        assert row["checkpoint_sha256"] == checkpoint_sha and row["source_sha256"] == binding["source_sha256"]
        assert row["config_sha256"] == config_sha and row["training_decisions"] == step
        assert row["method"] == prefix["method"] and row["seed"] == prefix["seed"]
        raw = read(folder/"validation"/f"{row['path_id']}.json")
        assert raw == row, "Validation summary differs from saved per-path result"
        trace = trace_diagnostics(folder/"validation"/f"{row['path_id']}.npz", row["trace_sha256"])
        complete_rows.append({**prefix, **row, "trace_diagnostics": trace})
    # Guard against a changed positive-feed checkpoint during report assembly.
    bind(folder/"checkpoint.pt", checkpoint_sha)
    return dict(**prefix, status="complete", checkpoint_sha256=checkpoint_sha, config_sha256=config_sha,
        source_sha256=binding["source_sha256"], training=training,
        prior=dict(checkpoint_sha256=prior_sha, path=prior_path.relative_to(ROOT).as_posix(),
                   decisions=prior["decisions"], physics_steps=prior["physics_steps"], command_steps=prior["command_steps"], training_wall_s=prior["training_wall_s"],
                   **prior_optimizer_cost(prior_path.parent, prior["decisions"])),
        attributed_total_decisions=prior["decisions"]+step,
        training_diagnostic=episode_diagnostic(folder if study in SNAPSHOT_STUDIES else live_run, training)), complete_rows


def registered_steps(study, protocol):
    if study == "positive":
        return [protocol["additional_macro_decisions"]]
    if study == "local":
        return protocol["additional_validation_milestones"]
    return protocol["validation_milestones"]


def latest_completed(checkpoints):
    latest = {}
    for c in checkpoints:
        if c["status"] == "complete" and (c["live_run"] not in latest or c["additional_decisions"] > latest[c["live_run"]]["additional_decisions"]):
            latest[c["live_run"]] = c
    return latest


def cumulative_cost(checkpoints):
    latest = latest_completed(checkpoints)
    priors = {c["prior"]["checkpoint_sha256"]: c["prior"] for c in latest.values()}
    return dict(completed_run_count=len(latest), cumulative_new_decisions=sum(c["training"]["decisions"] for c in latest.values()),
        cumulative_new_physics_steps=sum(c["training"]["physics_steps"] for c in latest.values()),
        cumulative_new_command_steps=sum(c["training"]["command_steps"] for c in latest.values()),
        cumulative_reported_training_wall_s=sum(c["training"]["training_wall_s"] for c in latest.values()),
        cumulative_recorded_recovery_wall_s=sum(c["training"].get("environment_replay_wall_s", 0.) for c in latest.values()),
        recovery_wall_scope="Latest completed snapshot per chain, not the sum of milestones. Includes checkpoint/optimizer restoration, trace verification and environment replay; unsaved early-return invocation overhead is absent.",
        cumulative_recorded_optimizer_updates=sum(c["training"].get("optimizer_updates", 0) for c in latest.values()),
        cumulative_recorded_adam_steps_per_network=sum(c["training"].get("adam_steps_per_network", 0) for c in latest.values()),
        optimizer_cost_scope="Full_batch and target_tracking summaries record optimizer counts; missing earlier counts are not a zero-cost claim.",
        unique_prior_models=list(priors.values()), unique_prior_decisions=sum(p["decisions"] for p in priors.values()),
        unique_prior_recorded_optimizer_updates=sum(p.get("optimizer_updates") or 0 for p in priors.values()),
        unique_prior_recorded_adam_steps_per_network=sum(p.get("adam_steps_per_network") or 0 for p in priors.values()),
        training_wall_scope="Recorded training-loop cumulative wall time only; excludes preparation, agent construction and final snapshot copy, so this is not whole-study compute time.",
        scope="Latest completed, evaluated snapshot per live run within the report budget only; milestone snapshots are not added twice. Shared prior copies are counted once. In-flight training, later excluded milestones and smoke budgets are not included and are not assumed zero.")


def recorded_selections(studies, checkpoints, budget):
    """Bind existing selections without choosing a checkpoint in this reporter."""
    complete = {c["checkpoint_directory"]: c for c in checkpoints if c["status"] == "complete"}
    latest = {key: c["additional_decisions"] for key, c in latest_completed(checkpoints).items()}
    selections = []
    for study in studies:
        root = BASE/SPECS[study][0]
        for path in sorted(root.glob("selection_budget*.json")):
            named_budget = int(path.stem.removeprefix("selection_budget"))
            if budget is not None and named_budget > budget:
                continue
            record = read(path)
            assert record["budget"] == named_budget
            if study in SNAPSHOT_STUDIES:
                assert record["zero_step_excluded"] is True
            selected_rows = []
            for method_record in record["records"]:
                chosen = method_record["selected"]
                for entry in method_record["all_checkpoints"]:
                    assert not (study in SNAPSHOT_STUDIES and entry["step"] == 0), "Step0 cannot enter trained model selection"
                    checkpoint = complete[entry["checkpoint"]]
                    assert checkpoint["additional_decisions"] == entry["step"] <= named_budget
                    assert checkpoint["checkpoint_sha256"] == entry["checkpoint_sha256"]
                    bind(ROOT/entry["checkpoint"]/"validation/summary.json", entry["validation_sha256"])
                assert chosen in method_record["all_checkpoints"]
                c = complete[chosen["checkpoint"]]
                selected_rows.append(dict(study=study, method=chosen["method"], seed=c["seed"],
                    declared_available_budget=named_budget, selected_model_additional_decisions=chosen["step"],
                    latest_observed_additional_decisions=latest[c["live_run"]],
                    selected_model_attributed_decisions=c["prior"]["decisions"]+chosen["step"],
                    actual_chain_attributed_decisions=c["prior"]["decisions"]+latest[c["live_run"]],
                    selected_full_successes=chosen["successes"], expected_paths=chosen["count"],
                    checkpoint_directory=chosen["checkpoint"], checkpoint_sha256=chosen["checkpoint_sha256"],
                    selection_file=path.relative_to(ROOT).as_posix(),
                    uniform_fixed=(chosen == record["uniform_fixed"])))
            assert sum(r["uniform_fixed"] for r in selected_rows) == 1
            selections.extend(selected_rows)
    return selections


def summary_group(rows):
    good = [r for r in rows if r["success"]]
    return dict(count=len(rows), successes=len(good), failures=len(rows)-len(good),
        complete_success_time_sum_s=sum(r["elapsed_time_s"] for r in good) if good else None,
        complete_success_time_mean_s=float(np.mean([r["elapsed_time_s"] for r in good])) if good else None,
        successful_path_ids=[r["path_id"] for r in good],
        geometry_successes=sum(bool(r["geometry_feasible"]) for r in rows),
        ordered_completions=sum(bool(r["path_completed"]) for r in rows),
        scalar_motion_successes=sum(all(r["constraints"][n]["satisfied"] for n in ("feed_mm_s", "feed_acceleration_mm_s2", "feed_jerk_mm_s3")) for r in rows),
        angular_motion_successes=sum(all(r["constraints"][n]["satisfied"] for n in ("angular_velocity_rad_s", "angular_acceleration_rad_s2", "angular_jerk_rad_s3")) for r in rows),
        rest_successes=sum(bool(r["full_rest_satisfied"]) for r in rows),
        failures_by_reason=dict(Counter(reason for r in rows for reason in r["failure_reasons"])),
        scope="One deterministic validation rollout per path and trained checkpoint; no interpolation point is an independent replicate")


def retained_initializer_results(checkpoints, rows, paths):
    """Keep known pretrained fixed baselines visible without altering selection."""
    full = [c for c in checkpoints if c["study"] in SNAPSHOT_STUDIES and c["status"] == "complete"]
    baselines = []
    for c in full:
        if c["additional_decisions"] == 0 and c["method"] != "adaptive":
            subset = [r for r in rows if r["checkpoint_directory"] == c["checkpoint_directory"]]
            baselines.append(dict(study=c["study"], baseline_id=f"{c['study']}_step0_{c['method']}", method=c["method"], seed=c["seed"],
                checkpoint_directory=c["checkpoint_directory"], checkpoint_sha256=c["checkpoint_sha256"], rows=subset,
                **summary_group(subset),
                role="Retained initialization diagnostic under the registered study configuration; excluded from frozen post-training selection, but not from scientific comparison"))
    priors = {(c["study"], c["prior"]["checkpoint_sha256"]): c["prior"] for c in full if c["method"] != "adaptive"}
    for (study, _), prior in priors.items():
        folder = (ROOT/prior["path"]).parent
        binding = read(folder/"validation/binding.json")
        provenance = read(folder/"provenance.json")
        bind(folder/"checkpoint.pt", binding["checkpoint_sha256"])
        assert binding["checkpoint_sha256"] == prior["checkpoint_sha256"]
        bind(folder/"config.yaml", binding["config_sha256"])
        bind(MANIFEST, binding["manifest_sha256"])
        assert binding["split"] == "validation"
        assert binding["source_sha256"] == provenance["source_sha256"]
        assert binding["config_sha256"] == provenance["config_sha256"]
        assert hashlib.sha256(json.dumps(provenance["source_files"], sort_keys=True).encode()).hexdigest() == binding["source_sha256"]
        bind(folder/"source.zip")
        with zipfile.ZipFile(folder/"source.zip") as archive:
            for name, expected in provenance["source_files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        subset = read(folder/"validation/summary.json")["rows"]
        assert len(subset) == len(paths) and {r["path_id"] for r in subset} == paths
        for r in subset:
            assert read(folder/"validation"/f"{r['path_id']}.json") == r
            assert r["checkpoint_sha256"] == binding["checkpoint_sha256"]
            assert r["config_sha256"] == binding["config_sha256"] and r["source_sha256"] == binding["source_sha256"]
            assert r["training_decisions"] == prior["decisions"] and r["method"] == "fixed3"
            bind(folder/"validation"/f"{r['path_id']}.npz", r["trace_sha256"])
        baselines.append(dict(study=study, baseline_id="sensor_original_fixed3", method="fixed3", seed=subset[0]["seed"],
            checkpoint_directory=folder.relative_to(ROOT).as_posix(), checkpoint_sha256=prior["checkpoint_sha256"],
            rows=subset, **summary_group(subset),
            role="Original pretrained fixed3 evaluation with original sensor state; known feasible historical development baseline, not a control isolating learned distance"))
    if any(c["study"] == "target_tracking" for c in full):
        existing = {(b["study"], b["baseline_id"]) for b in baselines}
        baselines.extend(b for b in retained_target_development_baselines(paths)
                         if (b["study"], b["baseline_id"]) not in existing)
    return baselines, compare_retained_initializers(full, rows, paths, baselines)


def retained_target_development_baselines(paths):
    """Bind previously audited baselines even when new fixed runs are pending."""
    directory = REPORTS/"full_batch_4096_report_v1"
    manifest = read(directory/"manifest.json")
    bind(directory/"report.json", manifest["outputs"]["report.json"])
    previous = read(directory/"report.json")
    sensor = next(b for b in previous["retained_initializer_baselines"] if b["baseline_id"] == "sensor_original_fixed3")
    baselines = [{**sensor, "study": "target_tracking"}]
    for selection in previous["recorded_selections"]:
        if selection["method"] != "adaptive" and not selection["uniform_fixed"]:
            continue
        subset = [r for r in previous["rows"] if r["checkpoint_directory"] == selection["checkpoint_directory"]]
        baselines.append(dict(study="target_tracking", baseline_id=f"full_batch_selected_{selection['method']}",
            method=selection["method"], seed=selection["seed"], checkpoint_directory=selection["checkpoint_directory"],
            checkpoint_sha256=selection["checkpoint_sha256"], rows=subset, **summary_group(subset),
            selected_model_attributed_decisions=selection["selected_model_attributed_decisions"],
            actual_chain_attributed_decisions=selection["actual_chain_attributed_decisions"],
            role="Previously selected full_batch development model; comparison diagnoses the common feed execution correction, not the independent contribution of learned preview"))
    for baseline in baselines:
        folder = ROOT/baseline["checkpoint_directory"]
        members = [folder/name for name in ("checkpoint.pt", "config.yaml", "provenance.json", "source.zip", "training_summary.json", "validation/binding.json", "validation/summary.json")]
        assert len(baseline["rows"]) == len(paths) and {r["path_id"] for r in baseline["rows"]} == paths
        for r in baseline["rows"]:
            members.extend([folder/"validation"/f"{r['path_id']}.json", folder/"validation"/f"{r['path_id']}.npz"])
        for member in members:
            bind(member, previous["input_sha256"][member.relative_to(ROOT).as_posix()])
        baseline["source_report"] = (directory/"report.json").relative_to(ROOT).as_posix()
        baseline["source_report_sha256"] = manifest["outputs"]["report.json"]
    return baselines


def compare_retained_initializers(full, rows, paths, baselines):
    comparisons = []
    for c in full:
        if c["method"] != "adaptive" or c["additional_decisions"] == 0:
            continue
        adaptive = [r for r in rows if r["checkpoint_directory"] == c["checkpoint_directory"]]
        for baseline in baselines:
            if baseline.get("study", "full_batch") != c.get("study", "full_batch"):
                continue
            a = {r["path_id"]: r for r in adaptive}
            f = {r["path_id"]: r for r in baseline["rows"]}
            assert set(a) == set(f) == paths
            for path in paths:
                assert a[path]["reference_length_mm"] == f[path]["reference_length_mm"]
                assert {k: v["limit"] for k, v in a[path]["constraints"].items()} == {k: v["limit"] for k, v in f[path]["constraints"].items()}
            common = sorted(k for k in paths if a[k]["success"] and f[k]["success"])
            ft = sum(f[k]["elapsed_time_s"] for k in common) if common else None
            at = sum(a[k]["elapsed_time_s"] for k in common) if common else None
            comparison = dict(study=c.get("study", "full_batch"), additional_decisions=c["additional_decisions"], seed=c["seed"],
                baseline_id=baseline["baseline_id"], adaptive_successes=sum(r["success"] for r in adaptive),
                fixed_successes=baseline["successes"], expected_paths=len(paths), common_success_path_ids=common,
                fixed_time_s=ft, adaptive_time_s=at, improvement_pct=100*(ft-at)/ft if common else None,
                full_suite_success=len(common) == len(paths),
                per_path=[dict(path_id=k, adaptive_success=a[k]["success"], fixed_success=f[k]["success"],
                    adaptive_time_s=a[k]["elapsed_time_s"], fixed_time_s=f[k]["elapsed_time_s"],
                    improvement_pct=100*(f[k]["elapsed_time_s"]-a[k]["elapsed_time_s"])/f[k]["elapsed_time_s"] if k in common else None) for k in sorted(paths)],
                scope="Retained known baselines, not frozen post-training selection. Different initial weights and/or shared changes preclude attribution solely to learned distance; same-actor locked-distance ablation is separate. Checkpoints on one seed are correlated, not independent repetitions.")
            if c.get("study") == "target_tracking":
                comparison["baseline_method"] = baseline.get("method", "fixed3")
                comparison["baseline_successes"] = comparison.pop("fixed_successes")
                comparison["baseline_time_s"] = comparison.pop("fixed_time_s")
                for path in comparison["per_path"]:
                    path["baseline_success"] = path.pop("fixed_success")
                    path["baseline_time_s"] = path.pop("fixed_time_s")
                comparison["scope"] += " Target-tracking feed correction is shared execution behavior, not a manuscript contribution or evidence attributable solely to preview learning. New fixed runs may still be pending."
            comparisons.append(comparison)
    return comparisons


def write_csv(path, records):
    if not records:
        path.write_text("status\npending\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def build(studies, output, budget):
    output = Path(output).resolve()
    if not output.is_relative_to(REPORTS.resolve()):
        raise ValueError("Use a new directory below 论文项目/revision_20260910_continued/reports")
    if output.exists():
        raise FileExistsError("Preserve previous reports; --output must be a new directory")
    INPUTS.clear()
    bind(Path(__file__))
    manifest = read(MANIFEST)
    paths = {p["id"] for p in manifest["paths"] if p["split"] == "validation"}
    checkpoints, rows, unregistered_studies = [], [], []
    for study in studies:
        directory, stage = SPECS[study]
        root = BASE/directory
        if not (root/"protocol.json").exists():
            unregistered_studies.append(study)
            continue
        protocol = read(root/"protocol.json")
        bind(MANIFEST, protocol["manifest_sha256"])
        steps = registered_steps(study, protocol)
        for method in protocol["methods"]:
            run = root/stage/f"preview_time_{method}_seed{protocol['seed']}"
            for step in steps:
                if budget is not None and step > budget:
                    continue
                folder = run if study == "positive" else run/"checkpoints"/f"step{step}"
                record, evaluations = process_checkpoint(study, folder, run, step, protocol, paths)
                checkpoints.append(record); rows.extend(evaluations)
    groups = []
    for checkpoint in checkpoints:
        if checkpoint["status"] != "complete":
            continue
        subset = [r for r in rows if r["checkpoint_directory"] == checkpoint["checkpoint_directory"]]
        groups.append(dict(study=checkpoint["study"], method=checkpoint["method"], seed=checkpoint["seed"],
                           additional_decisions=checkpoint["additional_decisions"], initial_diagnostic=checkpoint["initial_diagnostic"],
                           selection_eligible=checkpoint["selection_eligible"], **summary_group(subset)))
    pairs = []
    for candidate in [g for g in groups if g["method"] == "adaptive"]:
        others = [g for g in groups if g["study"] == candidate["study"] and g["additional_decisions"] == candidate["additional_decisions"] and g["seed"] == candidate["seed"] and g["method"] != "adaptive"]
        subset = [r for r in rows if r["study"] == candidate["study"] and r["additional_decisions"] == candidate["additional_decisions"] and r["seed"] == candidate["seed"]]
        for fixed in others:
            a = {r["path_id"]: r for r in subset if r["method"] == "adaptive"}
            f = {r["path_id"]: r for r in subset if r["method"] == fixed["method"]}
            common = sorted(k for k in a if a[k]["success"] and f[k]["success"])
            ft = sum(f[k]["elapsed_time_s"] for k in common) if common else None
            at = sum(a[k]["elapsed_time_s"] for k in common) if common else None
            pairs.append(dict(study=candidate["study"], additional_decisions=candidate["additional_decisions"], seed=candidate["seed"],
                initial_diagnostic=candidate["initial_diagnostic"], selection_eligible=candidate["selection_eligible"],
                fixed_method=fixed["method"], adaptive_successes=candidate["successes"], fixed_successes=fixed["successes"],
                expected_paths=len(paths), common_success_path_ids=common, fixed_time_s=ft, adaptive_time_s=at,
                improvement_pct=100*(ft-at)/ft if common else None, full_suite_success=bool(len(common)==len(paths)),
                scope="Common-success paths only; failures retained. Single development seed, not repeatability or independent-test evidence."))
    costs = cumulative_cost(checkpoints)
    costs.update(completed_validation_rollouts=len(rows),
                 completed_validation_wall_s=sum(r["evaluation_wall_s"] for r in rows),
                 validation_cost_scope="Every completed milestone including initial diagnostics is counted; wall time includes rollout, audit and NPZ storage, not pure inference. Validation runs are not independent training repetitions.")
    selections = recorded_selections(studies, checkpoints, budget)
    initializers, initializer_comparisons = retained_initializer_results(checkpoints, rows, paths)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), studies=studies, max_additional_budget=budget,
                  unregistered_studies=unregistered_studies, recorded_selections=selections,
                  pending=[c for c in checkpoints if c["status"] == "pending"], checkpoints=checkpoints,
                  rows=rows, groups=groups, pairs=pairs, training_cost=costs,
                  retained_initializer_baselines=initializers, retained_initializer_comparison=initializer_comparisons,
                  input_sha256=INPUTS, scope="Read-only development validation report; no policy loading, training, held-out evaluation, checkpoint selection from episodes, or paper update.")
    output.mkdir(parents=True)
    (output/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    flat = []
    for r in rows:
        d = {k: r[k] for k in ("study", "method", "seed", "additional_decisions", "initial_diagnostic", "selection_eligible", "path_id", "success", "elapsed_time_s", "contour_error_max_mm", "contour_error_rmse_mm", "failure_reasons", "checkpoint_directory", "checkpoint_sha256", "config_sha256", "source_sha256", "trace_sha256")}
        d.update(ordered_progress=r["ordered_progress"], geometry_feasible=r["geometry_feasible"], full_rest_satisfied=r["full_rest_satisfied"],
                 terminal_tail_s=r["trace_diagnostics"]["terminal_tail_s"], observed_rest_window_s=r["terminal_rest_window"]["observed_duration_s"],
                 trace_diagnostics=r["trace_diagnostics"], constraints=r["constraints"])
        flat.append(d)
    write_csv(output/"rows.csv", flat)
    write_csv(output/"training_budgets.csv", [dict(study=c["study"], method=c["method"], seed=c["seed"], additional_decisions=c["additional_decisions"],
        status=c["status"], checkpoint_directory=c["checkpoint_directory"], prior_decisions=c.get("prior", {}).get("decisions"),
        attributed_total_decisions=c.get("attributed_total_decisions"), physics_steps=c.get("training", {}).get("physics_steps"),
        command_steps=c.get("training", {}).get("command_steps"), training_wall_s=c.get("training", {}).get("training_wall_s"),
        initial_diagnostic=c["initial_diagnostic"], selection_eligible=c["selection_eligible"],
        optimizer_updates=c.get("training", {}).get("optimizer_updates"), optimized_decisions=c.get("training", {}).get("optimized_decisions"),
        adam_steps_per_network=c.get("training", {}).get("adam_steps_per_network"),
        pending_buffer_decisions=c.get("training", {}).get("pending_buffer_decisions"), environment_replay_wall_s=c.get("training", {}).get("environment_replay_wall_s"),
        wall_time_scope="training_wall_s is cumulative training loop; environment_replay_wall_s is recorded recovery overhead, not environment-only compute",
        checkpoint_sha256=c.get("checkpoint_sha256"), pending_files=c.get("missing")) for c in checkpoints])
    write_csv(output/"paired_comparisons.csv", pairs)
    write_csv(output/"recorded_selections.csv", selections)
    write_csv(output/"retained_initializer_comparison.csv", initializer_comparisons)
    write_csv(output/"retained_initializer_baselines.csv", [{k: v for k, v in r.items() if k != "rows"} for r in initializers])
    lines = ["# 后续前瞻研究：已完成检查点验证汇总", "", "仅统计完成的验证输出。训练回合只作诊断，不用于性能筛选；未完成项标pending，没有读取local正在写入的顶层checkpoint。", "",
             "|研究|预算|方法|完整成功|成功样本总时间/s|几何通过|标量通过|方向通过|停稳通过|", "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        elapsed = "—" if g["complete_success_time_sum_s"] is None else f"{g['complete_success_time_sum_s']:.3f}"
        lines.append(f"|{g['study']}|{g['additional_decisions']}|{g['method']}|{g['successes']}/{g['count']}|{elapsed}|{g['geometry_successes']}|{g['scalar_motion_successes']}|{g['angular_motion_successes']}|{g['rest_successes']}|")
    lines += ["", "各方法成功样本集合可能不同，时间优势只看paired_comparisons.csv中的共同成功路径；报告不会把失败耗时参与成功时间平均。rows.csv保留全部路径、失败原因、尾段时间、停稳观察窗和约束峰值。", "",
              f"当前pending检查点：{len(report['pending'])}，尚未生成协议的研究：{unregistered_studies}。所有明细见report.json。预算按每个训练运行的最高已完成且已验证检查点累计，较早stepN不重复相加；共同先验按checkpoint SHA去重，复制先验的归因成本另列。", "",
              "full_batch与target_tracking的step0仅为初始化诊断，不能进入已训练模型选择。recorded_selections.csv只转录并核验已有selection文件，分开列出登记可用预算、选中模型步数与该训练链实际已观测累计步数；选择早期模型不会抹去后续训练成本。没有selection文件时不由本报告擅自选择。", "",
              "retained_initializer_comparison.csv另列已知的step0固定基准与原sensor fixed3基准，不因其未进入后续训练选择而隐藏较好的结果。所有检查点属于同一种子训练链，不能统计为独立重复；固定和自适应先验不同，公共改动与前瞻独立贡献须分开说明。", "",
              "training_wall_s仅为训练循环累计耗时，未覆盖准备、构造模型和最终快照复制；environment_replay_wall_s为恢复开销，包括模型/优化器恢复、校验和环境重放。evaluation_wall_s包含轨迹运行、审计和存储，并非纯推理开销。先验优化器更新次数及每网络Adam步数由原updates.jsonl核验并单列。", "",
              "target_tracking_v1/invocations由preview_checkpoint_batch.py保存实际命令与完整进程墙时，是更完整的调用成本台账；本报告不重复导入。完整进程墙时已包含其内部训练、恢复、评价和存储，不能再与training_summary循环耗时相加；需同时保留失败与恢复调用。", "",
              "target_tracking可仅运行adaptive，协议中尚未运行的固定组透明标pending，不阻塞报告，也不触发额外训练。已有sensor fixed3及full_batch选中固定/自适应基准从冻结报告和原文件SHA独立核验；修正后的完整方法与原固定前瞻可报告版本之间的性能差异。新的进给修正属于公共执行修正，不能将其效果全部归因为前瞻学习，也不作为论文贡献；D的独立贡献由同actor锁定距离对照另行检验。", "",
              "动作诊断使用feed_logit与feed_executed_action，并排除terminal_braking；负logit是合法的sigmoid坐标。零物理速度不自动等于动作被裁零；终末制动产生的静止也不计入策略动作零值。旧latent_1没有用于该诊断。", "",
              "本报告不进行模型选择、推断重算或新测试；所有结论限于单种子开发验证。"]
    (output/"报告.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps(dict(inputs=INPUTS, outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), complete_checkpoints=len(groups), pending_checkpoints=len(report["pending"]), validation_rows=len(rows), pairs=pairs), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--budget", type=int, choices=(0, 256, 512, 1024, 2048, 4096, 8192, 16384))
    parser.add_argument("--output", type=Path, required=True, help="New directory below continued/reports")
    args = parser.parse_args()
    build(list(SPECS) if args.study == "all" else [args.study], args.output, args.budget)
