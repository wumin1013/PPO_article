"""Frozen ten-model holdout gate, deterministic evaluation and raw-data report.

No policy or test reference is loaded at import time. Every evaluation checks
the entire frozen study before reading the held-out manifest or coordinates.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
FORMAL = ROOT/"论文项目/revision_20260910_continued/formal_v1"
SEEDS = (11, 22, 33, 44, 55)
METHODS = ("fixed3", "adaptive")
STEPS = (0, 256, 512, 1024, 2048, 4096)
CORE_ARTIFACTS = {"checkpoint.pt", "config.yaml", "provenance.json", "source.zip", "training_summary.json"}


class FreezeError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise FreezeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f"Changed while reading: {path}")
    return digest


def relative_path(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Use repository-relative paths")
    path = (root/name).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes repository: {name}")
    return path


def bound_file(root, item):
    require(isinstance(item, dict) and set(item) >= {"file", "sha256"}, "Missing file/SHA binding")
    path = relative_path(root, item["file"])
    require(path.is_file() and sha(path) == item["sha256"], f"Missing or changed bound file: {path}")
    return path


def bound_artifacts(root, study_dir, entry, required):
    directory = relative_path(root, entry["checkpoint_dir"])
    require(directory.is_relative_to(study_dir), "Models must belong to the registered formal study")
    artifacts = entry["artifacts"]
    require(set(artifacts) >= required, "Incomplete artifact set")
    paths = {}
    for name, item in artifacts.items():
        expected = (directory/name).resolve()
        require(expected.is_relative_to(directory) and relative_path(root, item["file"]) == expected, "Artifact escapes checkpoint directory")
        paths[name] = bound_file(root, item)
    return directory, paths


def verify_source(root, paths, source=None):
    provenance = read_json(paths["provenance.json"])
    files = provenance["source_files"]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    require(digest == provenance["source_sha256"], "Inconsistent archived source digest")
    if source is not None:
        require(files == source["source_files"] and digest == source["source_sha256"], "Model source differs from frozen continuation source")
    with zipfile.ZipFile(paths["source.zip"]) as archive:
        for name, expected in files.items():
            path = relative_path(root, name)
            require(path.suffix == ".py" and name.startswith("PPO_project/"), "Unexpected source member")
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f"Archived source mismatch: {name}")
            require(sha(path) == expected, f"Live numerical source changed: {name}")
    require(provenance["config_sha256"] == sha(paths["config.yaml"]), "Configuration/provenance mismatch")
    return provenance


def verify_snapshot(directory, paths, step):
    manifest = read_json(paths["snapshot_manifest.json"])
    require(manifest["decisions"] == step, "Snapshot decision mismatch")
    require(set(manifest["files"]) >= CORE_ARTIFACTS | {"episodes.jsonl", "updates.jsonl", "training_final_partial.npz"}, "Snapshot logs missing")
    for name, expected in manifest["files"].items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory) and path.is_file(), "Invalid snapshot member")
        require(sha(path) == expected, f"Snapshot member changed: {path}")


def common_config(cfg, seed, method):
    require(cfg["seed"] == seed, "Config seed mismatch")
    revision = cfg["revision"]
    require(bool(revision["adaptive_preview"]) == (method == "adaptive"), "Method/config mismatch")
    require(cfg["environment"]["interpolation_period"] == .001 and revision["macro_period_steps"] == 20, "Sampling model must remain1ms/20ms")
    require(revision["fixed_distance_mm"] == 3. and revision["initial_preview_mm"] == 3., "Use one frozen3mm fixed distance and common initial distance")
    require("feed_tracking" not in revision, "Target-tracking correction was not adopted")
    require(revision["active_action_dims"] == ([1, 2] if method == "adaptive" else [1]), "Unexpected active policy actions")
    return dict(environment=cfg["environment"], constraints=cfg["kinematic_constraints"],
                macro_period_steps=revision["macro_period_steps"], terminal_definition=revision["terminal_definition"])


def validate_freeze(freeze_path, root=ROOT):
    """Validate every model first; never open the holdout manifest here."""
    root = Path(root).resolve()
    freeze_path = Path(freeze_path).resolve()
    require(freeze_path.is_file() and freeze_path.is_relative_to(root), "All ten models must be frozen before test access")
    digest = sha(freeze_path)
    freeze = read_json(freeze_path)
    require(freeze["version"] == "formal-preview-continuation-v1", "Unknown freeze schema")
    require(freeze["seeds"] == list(SEEDS) and freeze["methods"] == list(METHODS), "Freeze requires five independent seeds and both methods")
    records = freeze["records"]
    require(len(records) == 10 and {(r["seed"], r["method"]) for r in records} == {(s, m) for s in SEEDS for m in METHODS}, "Missing/duplicate frozen model")
    protocol_path = bound_file(root, freeze["protocol"])
    protocol = read_json(protocol_path)
    require(protocol["sources"]["continuation"] == freeze["source"], "Formal protocol/source mismatch")
    require(protocol.get("seeds") == list(SEEDS) and protocol.get("methods") == list(METHODS), "Protocol seed/method mismatch")
    require(freeze.get("all_training_complete") is True, "Formal training has not completed")
    evaluation_sources = protocol["evaluation_sources"]
    require(hashlib.sha256(json.dumps(evaluation_sources["source_files"], sort_keys=True).encode()).hexdigest() == evaluation_sources["source_sha256"], "Invalid frozen evaluator digest")
    require(set(evaluation_sources["source_files"]) >= {"PPO_project/tools/preview_formal_holdout.py", "PPO_project/tools/preview_formal_statistics.py"}, "Evaluator and statistical protocol were not frozen")
    for name, expected in evaluation_sources["source_files"].items():
        require(sha(relative_path(root, name)) == expected, f"Frozen evaluator changed: {name}")
    # Merely validate the descriptor. Manifest bytes may include geometric
    # definitions and are deliberately not opened until this entire gate passes.
    require(set(freeze["holdout_manifest"]) >= {"file", "sha256"}, "No frozen holdout identity")
    require(freeze["holdout_manifest"] == protocol["holdout_manifest"], "Holdout descriptor differs from registered protocol")
    holdout_path = relative_path(root, freeze["holdout_manifest"]["file"])
    require(holdout_path == root/"PPO_project/data/preview_continuation_v1/manifest.json", "Unexpected holdout dataset")
    study_dir = freeze_path.parent
    models, priors, config_signature = [], set(), None
    for record in records:
        seed, method, step = record["seed"], record["method"], record["selected_step"]
        require(step in STEPS, "Unregistered selected step")
        require(record["available_pretraining_decisions"] == 4096 and record["available_additional_decisions"] == 4096, "Unequal available training budget")
        required = CORE_ARTIFACTS | {"snapshot_manifest.json", "validation/summary.json"}
        directory, paths = bound_artifacts(root, study_dir, record, required)
        provenance = verify_source(root, paths, freeze["source"])
        require(record["source_sha256"] == freeze["source"]["source_sha256"], "Record source mismatch")
        verify_snapshot(directory, paths, step)
        training = read_json(paths["training_summary.json"])
        require(training["seed"] == seed and training["method"] == method and training["decisions"] == step, "Selected training metadata mismatch")
        require(training["checkpoint_sha256"] == sha(paths["checkpoint.pt"]), "Selected checkpoint mismatch")
        require(training["optimized_decisions"] == step and training["pending_buffer_decisions"] == 0, "Selected snapshot has unoptimized pending transitions")
        cfg = yaml.safe_load(paths["config.yaml"].read_text(encoding="utf-8"))
        signature = common_config(cfg, seed, method)
        require(config_signature is None or signature == config_signature, "Reference evaluation constraints differ across frozen models")
        config_signature = signature
        prior = record["pretraining"]
        prior_dir, prior_paths = bound_artifacts(root, study_dir, prior, CORE_ARTIFACTS)
        verify_source(root, prior_paths, protocol["sources"]["pretraining"])
        prior_summary = read_json(prior_paths["training_summary.json"])
        prior_cfg = yaml.safe_load(prior_paths["config.yaml"].read_text(encoding="utf-8"))
        require(prior["decisions"] == prior_summary["decisions"] == 4096, "Incomplete independent pretraining")
        require(prior_summary["seed"] == prior_cfg["seed"] == seed and prior_summary["method"] == method, "Pretraining reused another seed/method")
        require(prior_summary.get("pretraining_steps", 0) == 0, "Pretraining must start from scratch")
        require("init_checkpoint" not in prior_cfg["revision"], "Pretraining must not import a shared prior")
        prior_sha = sha(prior_paths["checkpoint.pt"])
        require(prior_summary["checkpoint_sha256"] == prior_sha and prior_sha not in priors, "Pretraining checkpoint reused across formal models")
        priors.add(prior_sha)
        require(relative_path(root, cfg["revision"]["init_checkpoint"]) == prior_paths["checkpoint.pt"] and cfg["revision"]["init_checkpoint_sha256"] == prior_sha, "Continuation does not use its own independent pretraining")
        selection_path = bound_file(root, record["selection"])
        selection = read_json(selection_path)
        require(selection["seed"] == seed and selection["method"] == method, "Selection seed/method mismatch")
        candidates = selection["all_checkpoints"]
        require(len(candidates) == len(STEPS) and {c["step"] for c in candidates} == set(STEPS), "Every model must receive all registered validation opportunities")
        for candidate in candidates:
            snapshot = relative_path(root, candidate["checkpoint_dir"])
            require(snapshot.is_relative_to(study_dir), "Selection candidate outside formal study")
            for name, key in (("checkpoint.pt", "checkpoint_sha256"), ("training_summary.json", "training_summary_sha256"), ("snapshot_manifest.json", "snapshot_manifest_sha256"), ("validation/summary.json", "validation_sha256")):
                require(sha(snapshot/name) == candidate[key], f"Validation candidate changed: {snapshot}/{name}")
            observed = read_json(snapshot/"validation/summary.json")["rows"]
            require(len(observed) == 3 and {r["path_id"] for r in observed} == set(protocol["validation_path_ids"]), "Validation paths differ across candidates")
            require(all(r["seed"] == seed and r["method"] == method and r["checkpoint_sha256"] == candidate["checkpoint_sha256"] for r in observed), "Validation/model identity mismatch")
            successes = sum(bool(r["success"]) for r in observed)
            elapsed = sum(r["elapsed_time_s"] for r in observed) if successes == 3 else None
            require(candidate["successes"] == successes and candidate["count"] == 3 and candidate["complete_validation_time_s"] == elapsed, "Saved selection statistics changed")
        chosen = selection["selected"]
        selected_by_rule = min(candidates, key=lambda c: (-c["successes"], c["complete_validation_time_s"] if c["successes"] == 3 else float("inf"), c["step"]))
        require(chosen == selected_by_rule and selection["step0_eligible"] is True, "Selection does not follow the common frozen validation rule")
        require(chosen["step"] == step and chosen["checkpoint_sha256"] == sha(paths["checkpoint.pt"]), "Freeze differs from saved validation selection")
        completion = bound_file(root, record["additional_budget_completion"])
        require(completion.is_relative_to(study_dir), "Budget completion belongs to another study")
        final_training = read_json(completion)
        require(final_training["seed"] == seed and final_training["method"] == method and final_training["decisions"] == 4096, "Full additional training budget not completed")
        require(final_training["optimized_decisions"] == 4096 and final_training["pending_buffer_decisions"] == 0, "Final budget has unoptimized transitions")
        final_manifest = bound_file(root, record["additional_budget_snapshot_manifest"])
        require(final_manifest.parent == completion.parent, "Final budget snapshot mismatch")
        verify_snapshot(completion.parent, {"snapshot_manifest.json": final_manifest}, 4096)
        models.append(dict(record=record, cfg=cfg, checkpoint=paths["checkpoint.pt"], directory=directory))
    require(sha(freeze_path) == digest, "Freeze changed during validation")
    return dict(freeze=freeze, freeze_path=freeze_path, freeze_sha256=digest, models=models, common_configuration=config_signature)


def frozen_manifest(gate, root=ROOT):
    path = bound_file(Path(root), gate["freeze"]["holdout_manifest"])
    manifest = read_json(path)
    items = manifest["paths"]
    require(len(items) == 6 and len({p["id"] for p in items}) == 6, "Expected six distinct holdouts")
    require(all(p["split"] == "test" for p in items), "Only registered held-out paths are allowed")
    require({family: sum(p["family"] == family for p in items) for family in ("mixed_lengths", "dense_corners", "line_curve")} == {"mixed_lengths": 2, "dense_corners": 2, "line_curve": 2}, "Holdout family split changed")
    return manifest, path.parent


def load_frozen_references(freeze_path, root=ROOT, loader=None):
    gate = validate_freeze(freeze_path, root)
    manifest, data = frozen_manifest(gate, root)
    if loader is None:
        from generate_preview_continuation_paths import load_path
        loader = load_path
    references = {item["id"]: loader(item["id"], data) for item in manifest["paths"]}
    return gate, manifest, references


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv_new(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def terminal_tail(trace):
    indices = [i for i, row in enumerate(trace) if row.get("terminal_braking", 0.) == 1.]
    if not indices:
        return None
    require(indices[0] > 0 and indices == list(range(indices[0], len(trace))), "Noncontiguous terminal braking suffix")
    return float(trace[-1]["timestamp_s"]-trace[indices[0]-1]["timestamp_s"])


def evaluate(freeze_path, seed, method):
    gate = validate_freeze(freeze_path)
    require(seed in SEEDS and method in METHODS, "Choose a frozen method/seed")
    chosen = next(m for m in gate["models"] if m["record"]["seed"] == seed and m["record"]["method"] == method)
    output = gate["freeze_path"].parent/"holdout"/f"{method}_seed{seed}"
    identity = dict(freeze_sha256=gate["freeze_sha256"], seed=seed, method=method,
                    checkpoint_sha256=sha(chosen["checkpoint"]), config_sha256=chosen["record"]["artifacts"]["config.yaml"]["sha256"],
                    source_sha256=gate["freeze"]["source"]["source_sha256"], holdout_manifest_sha256=gate["freeze"]["holdout_manifest"]["sha256"])
    if (output/"binding.json").exists():
        require(read_json(output/"binding.json") == identity, "Cannot resume with different freeze or model")
    else:
        write_json_new(output/"binding.json", identity)
    invocation_id = uuid.uuid4().hex
    invocation = gate["freeze_path"].parent/"holdout_invocations"/invocation_id
    write_json_new(invocation/"started.json", dict(**identity, created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], role="First/further frozen independent holdout exposure; subsequent test-informed tuning invalidates independent-test status", machine_execution=False))
    # Runtime imports and reference loading are strictly after the ten-model gate.
    import torch
    import preview_full_batch_study as full
    from src.environment import cnc_env, kinematics
    from generate_preview_continuation_paths import load_path
    require(cnc_env.apply_kinematic_constraints is kinematics.apply_kinematic_constraints, "Unexpected installed execution correction")
    require(full.core.plan_terminal_brake.__module__ == "terminal_braking", "Original v2 terminal planner is required")
    torch.set_num_threads(1)
    manifest, data = frozen_manifest(gate)
    payload = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
    rows = []
    process_started = time.perf_counter()
    for item in manifest["paths"]:
        path_id = item["id"]
        result_path = output/f"{path_id}.json"
        if result_path.exists():
            row = read_json(result_path)
            require(all(row[k] == v for k, v in identity.items()), "Saved result binding differs")
            bound_file(ROOT, row["trace"])
            bound_file(ROOT, row["inference_samples"])
            rows.append(row)
            continue
        validate_freeze(freeze_path)
        reference = load_path(path_id, data)
        cfg = copy.deepcopy(chosen["cfg"])
        env = full.local.LocalCornerEnvironment(cfg, path_id, reference=reference)
        agent = full.local.make_agent(cfg, env)
        agent.actor.load_state_dict(payload["actor"])
        agent.actor.eval()
        timings = []
        started = time.perf_counter()
        while not env.done:
            state = env.state
            inference_start = time.perf_counter()
            action = agent.take_action(state, deterministic=True)
            timings.append(time.perf_counter()-inference_start)
            env.step(action)
        trajectory_wall_s = time.perf_counter()-started
        attempt = output/"attempts"/invocation_id
        attempt.mkdir(parents=True, exist_ok=True)
        trace_path = attempt/f"{path_id}.npz"
        full.core.old.save_trace(trace_path, env.trace)
        samples_path = attempt/f"{path_id}_inference.json"
        write_json_new(samples_path, dict(seconds=timings, unit="s", measured_call="deterministic actor.take_action including Python call, excluding state retrieval; no warmup", cpu_threads=1))
        row = dict(env.result, **identity, path_id=path_id, family=item["family"],
            selected_step=chosen["record"]["selected_step"], terminal_tail_s=terminal_tail(env.trace),
            policy_decisions=len(timings), inference_wall_s=float(sum(timings)),
            inference_mean_s=float(np.mean(timings)), inference_p50_s=float(np.quantile(timings, .5)), inference_p95_s=float(np.quantile(timings, .95)),
            cpu_threads=1, trajectory_wall_s=trajectory_wall_s, evaluation_wall_s=time.perf_counter()-started,
            trace=dict(file=trace_path.relative_to(ROOT).as_posix(), sha256=sha(trace_path)),
            inference_samples=dict(file=samples_path.relative_to(ROOT).as_posix(), sha256=sha(samples_path)),
            reference_sha256=item["reference_sha256"], data_role="frozen independent holdout; no test-path-specific configuration/model selection",
            timing_scope="Actor call timing is separate; trajectory includes policy/environment/terminal audit; evaluation additionally includes trace and timing storage. Physical timestamps never depend on wall time.")
        write_json_new(result_path, row)
        rows.append(row)
        print(json.dumps(dict(seed=seed, method=method, path=path_id, success=row["success"], elapsed_time_s=row["elapsed_time_s"])), flush=True)
    if not (output/"summary.json").exists():
        write_json_new(output/"summary.json", dict(**identity, rows=rows))
    write_json_new(invocation/"completed.json", dict(**identity, rows=len(rows), completed_utc=datetime.now(timezone.utc).isoformat(), wall_s=time.perf_counter()-process_started))
    return rows


def build_report(freeze_path, output):
    gate = validate_freeze(freeze_path)
    manifest, _ = frozen_manifest(gate)
    output = Path(output).resolve()
    require(output.is_relative_to(gate["freeze_path"].parent/"reports") and not output.exists(), "Choose a new formal/reports directory")
    from preview_formal_statistics import summarize
    rows, inputs = [], {gate["freeze_path"].relative_to(ROOT).as_posix(): gate["freeze_sha256"]}
    for model in gate["models"]:
        r = model["record"]
        directory = gate["freeze_path"].parent/"holdout"/f"{r['method']}_seed{r['seed']}"
        for item in manifest["paths"]:
            path = directory/f"{item['id']}.json"
            require(path.is_file(), f"All60 path outcomes, including failures, are required: {path}")
            row = read_json(path)
            require(row["freeze_sha256"] == gate["freeze_sha256"] and row["checkpoint_sha256"] == r["artifacts"]["checkpoint.pt"]["sha256"], "Result/model freeze mismatch")
            require(row["config_sha256"] == r["artifacts"]["config.yaml"]["sha256"] and row["source_sha256"] == gate["freeze"]["source"]["source_sha256"] and row["holdout_manifest_sha256"] == gate["freeze"]["holdout_manifest"]["sha256"], "Result config/source/data freeze mismatch")
            require(row["seed"] == r["seed"] and row["method"] == r["method"] and row["path_id"] == item["id"] and row["family"] == item["family"], "Result identity mismatch")
            for key in ("trace", "inference_samples"):
                member = bound_file(ROOT, row[key])
                inputs[member.relative_to(ROOT).as_posix()] = sha(member)
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
            rows.append(row)
    statistics = summarize(rows, list(SEEDS), {p["id"]: p["family"] for p in manifest["paths"]})
    write_json_new(output/"report.json", dict(freeze_sha256=gate["freeze_sha256"], rows=rows, statistics=statistics,
        scope="All60 frozen model/path outcomes including failures; independent unit is training seed, with path pairing and whole-seed clustered confidence intervals. No trajectory sample is an independent training replicate."))
    write_csv_new(output/"raw_metrics.csv", rows)
    method_rows, pairs = [], []
    groups = [("overall", "all", statistics["overall"])]
    groups.extend((scope, label, value) for scope in ("by_family", "by_seed", "by_path") for label, value in statistics[scope].items())
    for scope, label, value in groups:
        method_rows.extend(dict(scope=scope, label=label, method=method, **value[method]) for method in METHODS)
        pairs.append(dict(scope=scope, label=label, **value["paired"]))
    write_csv_new(output/"method_statistics.csv", method_rows)
    write_csv_new(output/"paired_statistics.csv", pairs)
    write_json_new(output/"manifest.json", dict(inputs=inputs, outputs={name: sha(output/name) for name in ("report.json", "raw_metrics.csv", "method_statistics.csv", "paired_statistics.csv")}))
    return dict(output=str(output), rows=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("check", "evaluate", "report"):
        p = sub.add_parser(command)
        p.add_argument("--freeze", type=Path, default=FORMAL/"freeze.json")
        if command == "evaluate":
            p.add_argument("--seed", type=int, choices=SEEDS, required=True)
            p.add_argument("--method", choices=METHODS, required=True)
        if command == "report":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        result = validate_freeze(args.freeze)
        print(json.dumps(dict(freeze_sha256=result["freeze_sha256"], frozen_models=len(result["models"]), test_manifest_or_coordinates_read=False)))
    elif args.command == "evaluate":
        evaluate(args.freeze, args.seed, args.method)
    else:
        print(json.dumps(build_report(args.freeze, args.output)))
