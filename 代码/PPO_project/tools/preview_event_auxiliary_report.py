"""Read saved auxiliary baselines; never import or execute a policy runner.

NNC transfer and the deterministic two-step planner remain separate from the
matched PPO effect analysis. All recorded failures stay in descriptive summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "论文项目"
STUDY = PAPER / "revision_20260909_event_preview/study"
DATA = ROOT / "PPO_project/data/preview_event_v1"
REPLAY = PAPER / "revision_20260908/frozen_replay_explore"
SEEDS = (11, 22, 33, 44, 55)
CHECKS = ("feed_mm_s", "feed_acceleration_mm_s2", "feed_jerk_mm_s3",
          "angular_velocity_rad_s", "angular_acceleration_rad_s2", "angular_jerk_rad_s3")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Inputs:
    def __init__(self): self.hashes = {}
    def file(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected: raise ValueError(f"Artifact hash mismatch: {path}")
        self.hashes[path.relative_to(ROOT).as_posix()] = actual
        return path
    def read(self, path, expected=None): return json.loads(self.file(path, expected).read_text(encoding="utf-8"))


def verify_sources(base, source_files, expected, inputs, archive=None):
    if digest(source_files) != expected: raise ValueError("Source-map digest mismatch")
    for name, value in source_files.items(): inputs.file(base / name, value)
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            for name, value in source_files.items():
                if hashlib.sha256(z.read(name)).hexdigest() != value: raise ValueError("Source ZIP member differs from provenance")


def require_complete_freeze(inputs):
    """Independently inspect all formal artifacts before reading test results."""
    protocol = inputs.read(STUDY / "protocol.json")
    frozen = inputs.read(STUDY / "freeze.json")
    manifest = inputs.read(DATA / "manifest.json", frozen["manifest_sha256"])
    inputs.file(STUDY / "protocol.json", frozen["protocol_sha256"])
    inputs.file(STUDY / "selection.json", frozen["selection_sha256"])
    expected = {f"formal/{method}_seed{seed}" for method in protocol["formal_variants"] for seed in protocol["formal_seeds"]}
    if set(frozen["models"]) != expected or set(frozen["configs"]) != expected: raise ValueError("Complete formal freeze roster required")
    if frozen["source_sha256"] != protocol["source_sha256"]: raise ValueError("Frozen protocol source differs")
    validation_ids = {p["id"] for p in manifest["paths"] if p["split"] == "validation"}
    import torch  # Data-only checkpoint inspection; no application modules imported.
    for key in sorted(expected):
        run = STUDY / key
        model = inputs.file(run / "checkpoint.pt", frozen["models"][key])
        config_path = inputs.file(run / "config.yaml", frozen["configs"][key])
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        summary = inputs.read(run / "training_summary.json")
        provenance = inputs.read(run / "provenance.json")
        if summary["checkpoint_sha256"] != sha(model) or summary["steps"] != protocol["formal_steps"]:
            raise ValueError("Formal training summary/model budget mismatch")
        if torch.load(model, map_location="cpu", weights_only=False)["steps"] != protocol["formal_steps"]:
            raise ValueError("Actual checkpoint budget incomplete")
        if provenance["config_sha256"] != sha(config_path) or provenance["manifest_sha256"] != sha(DATA / "manifest.json"):
            raise ValueError("Formal provenance configuration/path mismatch")
        if provenance["source_sha256"] != frozen["source_sha256"]: raise ValueError("Formal source mismatch")
        verify_sources(ROOT, provenance["source_files"], frozen["source_sha256"], inputs, inputs.file(run / "source.zip"))
        rows = inputs.read(run / "evaluation_validation_nominal/summary.json")["rows"]
        if len(rows) != len(validation_ids) or {r["path_id"] for r in rows} != validation_ids: raise ValueError("Formal validation path set incomplete")
        for row in rows:
            binding = dict(variant=cfg["revision"]["variant"], seed=cfg["seed"], split="validation", regime="nominal",
                training_steps=protocol["formal_steps"], checkpoint_sha256=sha(model), config_sha256=sha(config_path),
                source_sha256=frozen["source_sha256"], manifest_sha256=sha(DATA / "manifest.json"))
            if any(row.get(k) != v for k, v in binding.items()): raise ValueError("Formal validation binding changed")
    return frozen


def stats(values):
    values = list(values)
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=len(a), missing_n=len(values)-len(a), mean=float(a.mean()) if len(a) else None,
        std=float(a.std(ddof=1)) if len(a)>1 else None, maximum=float(a.max()) if len(a) else None)


def summarize(rows):
    return dict(n=len(rows), successes=sum(r["metrics"].get("success") is True for r in rows),
        failures=sum(r["metrics"].get("success") is not True for r in rows),
        invalid_traces=sum(r["metrics"].get("valid_trace") is False for r in rows),
        failure_reasons=dict(Counter(reason for r in rows for reason in r["metrics"].get("failure_reasons", []))),
        elapsed_time_all_s=stats(r["metrics"].get("elapsed_time_s") for r in rows),
        peak_error_all_mm=stats(r["metrics"].get("contour_error_max_mm") for r in rows),
        terminal_feed_all_mm_s=stats(r["metrics"].get("terminal_feed_mm_s") for r in rows),
        constraints={key: dict(failed_runs=sum(r["metrics"].get("constraints", {}).get(key, {}).get("satisfied") is False for r in rows),
            unknown_runs=sum(r["metrics"].get("constraints", {}).get(key, {}).get("satisfied") is None for r in rows),
            peaks=stats(r["metrics"].get("constraints", {}).get(key, {}).get("peak") for r in rows),
            violating_samples=(sum(r["metrics"].get("constraints", {}).get(key, {}).get("violating_samples") for r in rows)
                if all(r["metrics"].get("constraints", {}).get(key, {}).get("violating_samples") is not None for r in rows) else None),
            sample_count_role="Within-trace diagnostic count, not independent statistical replicates") for key in CHECKS})


def nnc_rows(split, paths, inputs, protocol, pending, common_cfg):
    rows = []
    old_base = REPLAY / "论文项目/revision_20260908/preview_v2/nnc_auxiliary"
    for seed in SEEDS:
        run = STUDY / "nnc_transfer" / split / f"nnc_seed{seed}"
        if not (run / "request.json").exists():
            pending.extend(dict(method="nnc_transfer", seed=seed, path_id=p, reason="request_pending") for p in paths)
            continue
        request = inputs.read(run / "request.json")
        if request["split"] != split or request["seed"] != seed or {p["id"] for p in request["paths"]} != set(paths) or len(request["paths"]) != len(paths):
            raise ValueError("NNC split, seed or path set differs")
        if request["event_source_sha256"] != protocol["source_sha256"]: raise ValueError("NNC event source differs")
        path_manifest = inputs.read(DATA / "manifest.json", request["manifest_sha256"])
        registered = {p["id"]: p for p in path_manifest["paths"] if p["split"] == split}
        common = request["common"]
        expected_common = dict(epsilon=common_cfg["environment"]["epsilon"], dt_s=common_cfg["environment"]["interpolation_period"],
            max_steps=common_cfg["revision"]["episode_max_steps"], limits=common_cfg["kinematic_constraints"])
        if common != expected_common: raise ValueError("NNC nominal conditions differ from registered common configuration")
        if split == "test": inputs.file(STUDY / "freeze.json", request["event_freeze_sha256"])
        inputs.file(run / "evaluation_adapter.py", request["script_sha256"])
        summary = inputs.read(run / "summary.json") if (run / "summary.json").exists() else None
        if summary is None: pending.append(dict(method="nnc_transfer", seed=seed, reason="run_summary_pending"))
        elif summary["request_sha256"] != sha(run / "request.json") or summary["script_sha256"] != request["script_sha256"]:
            raise ValueError("NNC run summary binding differs")
        for spec in request["paths"]:
            path_id = spec["id"]
            if not (run / (path_id + ".json")).exists():
                pending.append(dict(method="nnc_transfer", seed=seed, path_id=path_id, reason="result_pending")); continue
            result = inputs.read(run / (path_id + ".json"))
            worker = inputs.read(run / (path_id + ".worker.json"), result["worker_sha256"])
            inputs.file(run / (path_id + ".npz"), result["trace_sha256"])
            copied_reference = inputs.file(Path(spec["reference_file"]), spec["reference_sha256"])
            if split != "regression":
                saved_reference = inputs.file(DATA / registered[path_id]["reference_file"], registered[path_id]["reference_sha256"])
                if not np.array_equal(np.loadtxt(copied_reference, delimiter=",", skiprows=1), np.loadtxt(saved_reference, delimiter=",", skiprows=1)):
                    raise ValueError("NNC copied reference differs from registered geometry")
            if result["path_id"] != path_id or result["seed"] != seed or result["new_training_steps"] != 0:
                raise ValueError("NNC result identity or cost differs")
            if worker["request_sha256"] != sha(run / "request.json") or result["request_sha256"] != worker["request_sha256"] or worker["trace_sha256"] != result["trace_sha256"] or worker["script_sha256"] != request["script_sha256"]:
                raise ValueError("NNC worker/request/trace/snapshot binding differs")
            if worker["reference_sha256"] != spec["reference_sha256"]: raise ValueError("NNC reference binding differs")
            if summary is not None and [r for r in summary["rows"] if r["path_id"] == path_id] != [result]: raise ValueError("NNC result differs from stored summary")
            old_run = old_base / "runs" / f"nnc_seed{seed}"
            old_freeze = inputs.read(old_base / "freeze.json", worker["old_freeze_sha256"])
            old_key = f"runs/nnc_seed{seed}"
            inputs.file(old_run / "checkpoint.pt", worker["checkpoint_sha256"])
            inputs.file(old_run / "config.yaml", worker["original_config_sha256"])
            archive = inputs.file(old_run / "source.zip", worker["source_archive_sha256"])
            provenance = inputs.read(old_run / "provenance.json")
            training = inputs.read(old_run / "training_summary.json")
            if provenance != worker["original_training_provenance"] or training != worker["original_training_summary"] or training["steps"] != 98304:
                raise ValueError("NNC original training record differs")
            if old_freeze["models"][old_key] != worker["checkpoint_sha256"] or old_freeze["configs"][old_key] != worker["original_config_sha256"] or training["checkpoint_sha256"] != worker["checkpoint_sha256"]:
                raise ValueError("NNC original model freeze differs")
            if old_freeze["source_sha256"] != provenance["source_sha256"] or yaml.safe_load((old_run/"config.yaml").read_text(encoding="utf-8")) != worker["original_config"]:
                raise ValueError("NNC original source/configuration record differs")
            verify_sources(REPLAY, provenance["source_files"], provenance["source_sha256"], inputs, archive)
            common, effective = request["common"], worker["effective_config"]
            if effective["environment"]["epsilon"] != common["epsilon"] or effective["environment"]["interpolation_period"] != common["dt_s"] or effective["kinematic_constraints"] != common["limits"] or effective["revision"]["episode_max_steps"] != common["max_steps"]:
                raise ValueError("NNC effective constraints differ from request")
            rows.append(dict(method="nnc_transfer", seed=seed, path_id=path_id, metrics=result["metrics"],
                original_training_steps=98304, new_training_steps=0, terminal_rest_required=False,
                result_file=(run/(path_id+".json")).relative_to(ROOT).as_posix(), result_sha256=sha(run/(path_id+".json"))))
    return rows


def two_step_rows(split, paths, inputs, protocol, pending, common_cfg):
    run = STUDY / "two_step_event_v1" / f"{split}_nominal"
    if not (run / "protocol.json").exists():
        pending.extend(dict(method="two_step", seed=None, path_id=p, reason="protocol_pending") for p in paths)
        return []
    saved = inputs.read(run / "protocol.json")
    if saved["split"] != split or saved["regime"] != "nominal" or saved["ordered_witness_spacing_mm"] != .01 or saved["event_source"]["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Two-step common protocol differs")
    inputs.file(DATA / "manifest.json", saved["manifest_sha256"])
    inputs.file(STUDY / "protocol.json", saved["event_protocol_sha256"])
    if split == "test": inputs.file(STUDY / "freeze.json", saved["freeze_sha256"])
    local_cfg = yaml.safe_load(inputs.file(run/"common_config.yaml").read_text(encoding="utf-8"))
    if local_cfg["environment"]["epsilon"] != common_cfg["environment"]["epsilon"] or local_cfg["environment"]["interpolation_period"] != common_cfg["environment"]["interpolation_period"] or local_cfg["kinematic_constraints"] != common_cfg["kinematic_constraints"]:
        raise ValueError("Two-step nominal conditions differ from registered common configuration")
    verify_sources(ROOT, saved["event_source"]["source_files"], protocol["source_sha256"], inputs)
    for name, expected in saved["numerical_sources"].items():
        file = run / "preview_event_two_step_snapshot.py" if name.endswith("/preview_event_two_step.py") else ROOT / name
        inputs.file(file, expected)
    summary = inputs.read(run / "summary.json") if (run / "summary.json").exists() else None
    if summary is None: pending.append(dict(method="two_step", reason="run_summary_pending"))
    rows = []
    for path_id in paths:
        path = run / path_id / "result.json"
        if not path.exists(): pending.append(dict(method="two_step", seed=None, path_id=path_id, reason="result_pending")); continue
        result = inputs.read(path)
        inputs.file(run / "protocol.json", result["protocol_sha256"])
        inputs.file(run / "common_config.yaml", result["common_config_sha256"])
        for name, expected in result["artifact_sha256"].items(): inputs.file(path.parent / name, expected)
        if result["path_id"] != path_id or result["split"] != split or result["regime"] != "nominal" or result["terminal_state_comparable_to_ppo"] is not False:
            raise ValueError("Two-step identity/terminal declaration differs")
        if summary is not None and [r for r in summary["rows"] if r["path_id"] == path_id] != [result]: raise ValueError("Two-step summary row differs")
        rows.append(dict(method="two_step", seed=None, path_id=path_id, metrics=result["native_terminal_rest_audit"],
            common_geometric_arrival_audit=result["common_geometric_arrival_audit"], original_training_steps=0,
            new_training_steps=0, terminal_rest_required=True, result_file=path.relative_to(ROOT).as_posix(), result_sha256=sha(path)))
    return rows


def collect(split):
    inputs = Inputs()
    if split == "test": require_complete_freeze(inputs)
    protocol = inputs.read(STUDY / "protocol.json")
    manifest = inputs.read(DATA / "manifest.json")
    common_run = STUDY / "pilot/state_fixed_seed101"
    common_provenance = inputs.read(common_run / "provenance.json")
    common_cfg = yaml.safe_load(inputs.file(common_run / "config.yaml", common_provenance["config_sha256"]).read_text(encoding="utf-8"))
    if common_provenance["source_sha256"] != protocol["source_sha256"]: raise ValueError("Registered common configuration source differs")
    if split == "regression":
        cfg = yaml.safe_load(inputs.file(PAPER / "generated/configs/full_method_snapshot.yaml").read_text(encoding="utf-8"))
        paths = [p["name"] for p in cfg["training"]["path_curriculum"]["paths"]]
    else: paths = [p["id"] for p in manifest["paths"] if p["split"] == split]
    if not paths: raise ValueError("No registered paths")
    pending = []
    rows = nnc_rows(split, paths, inputs, protocol, pending, common_cfg) + two_step_rows(split, paths, inputs, protocol, pending, common_cfg)
    for row in rows:
        if row["metrics"].get("valid_trace") and row["metrics"].get("witness_spacing_mm") != .01: raise ValueError("Baseline uses another coverage certificate")
    summaries = [dict(method=m, path_id=p, **summarize([r for r in rows if r["method"] == m and r["path_id"] == p]))
                 for p in paths for m in ["nnc_transfer", "two_step"]]
    return dict(split=split, regime="nominal", paths=paths, rows=rows, summaries=summaries, pending=pending,
        expected_trajectories=6*len(paths), complete=len(rows)==6*len(paths) and not pending,
        input_sha256=inputs.hashes,
        roles={"nnc_transfer": "Five independent preserved NNC seeds, 98,304 training steps each; zero new training, auxiliary transfer only",
               "two_step": "One deterministic run per path; native terminal-rest criterion differs from PPO geometric arrival"},
        statistical_scope="Separate per-path descriptive summaries. Times include failed runs and unknown fields remain missing. No superiority inference or PPO paired effect statistic.")


def table(snapshot):
    def fmt(summary):
        if summary["mean"] is None: return "--"
        return f"{summary['mean']:.3f}" + (f" $\\pm$ {summary['std']:.3f}" if summary["std"] is not None else "")
    lines = ["% Generated from verified saved auxiliary artifacts; see report_manifest.json.", r"\begin{table}[htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\caption{Auxiliary " + snapshot["split"] + r" diagnostics. Time includes failed trajectories; mean $\pm$ sample SD for five preserved NNC seeds (98,304 training steps each), one deterministic run for two-step. Terminal conditions differ; these times do not establish superiority.}",
        r"\label{tab:event-auxiliary-" + snapshot["split"] + "}", r"\begin{tabular}{llrrrr}\toprule",
        r"Path & Method & Success & Time (s) & Peak error (mm) & Constraint failures\\\midrule"]
    for row in snapshot["summaries"]:
        name = row["path_id"].replace("_event_v1_test", "").replace("_validation", "").replace("_", " ")
        label = "NNC transfer" if row["method"] == "nnc_transfer" else "Two-step"
        failed = "/".join(str(row["constraints"][key]["failed_runs"]) for key in CHECKS)
        lines.append(f"{name} & {label} & {row['successes']}/{row['n']} & {fmt(row['elapsed_time_all_s'])} & {fmt(row['peak_error_all_mm'])} & {failed} " + r"\\")
    lines += [r"\bottomrule\end{tabular}", r"\par\footnotesize Constraint counts follow $v,a,j,\omega,\dot{\omega},\ddot{\omega}$; missing numeric fields are excluded only from numeric summaries."]
    if not snapshot["complete"]: lines.append(r"\par\textbf{Incomplete: registered auxiliary results remain pending.}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test", "regression"], default="validation")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--table-output", type=Path)
    args = parser.parse_args()
    result = collect(args.split)
    if not result["complete"] and not args.allow_partial: raise ValueError("Auxiliary results incomplete; --allow-partial labels pending work explicitly")
    out = (args.output_dir or STUDY / "reports" / f"preview_event_auxiliary_{args.split}").resolve()
    tex = (args.table_output or PAPER / "generated" / f"preview_event_auxiliary_{args.split}.tex").resolve()
    # Manifest keys are repository-relative, including user-supplied CLI paths.
    out.relative_to(ROOT)
    tex.relative_to(ROOT)
    if out.exists(): raise FileExistsError("Preserve old report; choose a new output directory")
    content = table(result)
    if tex.exists() and tex.read_text(encoding="utf-8") != content: raise FileExistsError("Preserve old table; choose a new table output")
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fields = ["method", "seed", "path_id", "success", "elapsed_time_s", "contour_error_max_mm", "terminal_feed_mm_s", "failure_reasons", "original_training_steps", "new_training_steps", "terminal_rest_required", "result_file", "result_sha256"]
    fields += [key+"__"+field for key in CHECKS for field in ("peak", "limit", "violating_samples", "satisfied")]
    with (out / "source_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in result["rows"]:
            metric = row["metrics"]
            flat = {key: row.get(key, metric.get(key)) for key in fields if "__" not in key}
            flat["failure_reasons"] = json.dumps(metric.get("failure_reasons", []))
            for key in CHECKS:
                for field in ("peak", "limit", "violating_samples", "satisfied"):
                    flat[key+"__"+field] = metric.get("constraints", {}).get(key, {}).get(field)
            writer.writerow(flat)
    tex.parent.mkdir(parents=True, exist_ok=True)
    if not tex.exists(): tex.write_text(content, encoding="utf-8")
    outputs = [out / "report.json", out / "source_data.csv", tex]
    manifest = dict(command="python " + " ".join(sys.argv), script_sha256=sha(__file__),
        input_sha256=result["input_sha256"], output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
        complete=result["complete"], pending_count=len(result["pending"]), policies_executed=False)
    (out / "report_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(rows=len(result["rows"]), complete=result["complete"], pending=len(result["pending"]), report=str(out)), ensure_ascii=False))


if __name__ == "__main__": main()
