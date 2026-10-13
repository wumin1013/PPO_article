"""Read-only audit of all registered validation snapshots and selections.

This is development validation, never an independent holdout evaluation. A new
output directory receives only progress.json until all ten selections and all
180 candidate rows are complete and verified. No actor is loaded or executed.
"""
from __future__ import annotations

import argparse
import copy
import csv
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import yaml

from preview_formal_holdout import (CORE_ARTIFACTS, METHODS, ROOT, SEEDS, STEPS,
    common_config, relative_path, require, sha, verify_source)

DEFAULT_STUDY = ROOT / "论文项目/revision_20260910_continued/formal_v1"
DATA_ROLE = "development validation used for checkpoint selection; not independent test evidence"


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs = {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "Input escapes repository")
        require("preview_continuation_v1" not in path.parts, "Holdout access is forbidden")
        digest = sha(path)
        require(expected is None or digest == expected, f"Checksum mismatch: {path}")
        name = path.relative_to(self.root).as_posix()
        require(name not in self.inputs or self.inputs[name] == digest, f"Input changed: {path}")
        self.inputs[name] = digest
        return digest

    def json(self, path, expected=None):
        self.file(path, expected)
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        self.file(path)
        return value

    def manifest(self, folder, name, required):
        manifest = self.json(folder/name)
        require(set(manifest["files"]) >= required, f"Missing {name} members")
        for member, expected in manifest["files"].items():
            path = (folder/member).resolve()
            require(path.is_relative_to(folder.resolve()), "Manifest member escapes folder")
            self.file(path, expected)
        return manifest

    def source(self, folder, expected):
        paths = {name: folder/name for name in CORE_ARTIFACTS}
        for path in paths.values():
            self.file(path)
        provenance = verify_source(self.root, paths, expected)
        for name, digest in provenance["source_files"].items():
            self.file(relative_path(self.root, name), digest)
        return provenance

    def unchanged(self):
        for name, expected in list(self.inputs.items()):
            self.file(self.root/name, expected)


def checkpoint_dir(study, method, seed, step):
    return study/f"continuation/formal/preview_time_{method}_seed{seed}/checkpoints/step{step}"


def config_at(folder, audit):
    audit.file(folder/"config.yaml")
    return yaml.safe_load((folder/"config.yaml").read_text(encoding="utf-8"))


def prior_at(study, method, seed, protocol, audit):
    folder = study/f"pretraining/formal/preview_time_{method}_seed{seed}"
    audit.source(folder, protocol["sources"]["pretraining"])
    training = audit.json(folder/"training_summary.json")
    cfg = config_at(folder, audit)
    require(training["seed"] == cfg["seed"] == seed and training["method"] == method,
            "Independent pretraining seed/method mismatch")
    require(training["decisions"] == 4096 and training.get("pretraining_steps", 0) == 0,
            "Incomplete or shared pretraining")
    require("init_checkpoint" not in cfg["revision"], "Pretraining did not start fresh")
    digest = audit.file(folder/"checkpoint.pt", training["checkpoint_sha256"])
    costs = audit.json(folder/"costs.json")
    require(costs["decisions"] == training["decisions"], "Pretraining cost budget mismatch")
    return dict(folder=folder, digest=digest, training=training, costs=costs)


def verify_checkpoint(study, method, seed, step, protocol, prior, audit):
    folder = checkpoint_dir(study, method, seed, step)
    required = CORE_ARTIFACTS | {"episodes.jsonl", "updates.jsonl", "training_final_partial.npz"}
    snapshot = audit.manifest(folder, "snapshot_manifest.json", required)
    require(snapshot["decisions"] == step, "Snapshot step mismatch")
    provenance = audit.source(folder, protocol["sources"]["continuation"])
    cfg = config_at(folder, audit)
    signature = common_config(cfg, seed, method)
    require(relative_path(audit.root, cfg["revision"]["init_checkpoint"]) == prior["folder"]/"checkpoint.pt"
            and cfg["revision"]["init_checkpoint_sha256"] == prior["digest"], "Wrong independent initialization")
    training = audit.json(folder/"training_summary.json")
    require(training["seed"] == seed and training["method"] == method and training["decisions"] == step,
            "Training metadata mismatch")
    require(training["optimized_decisions"] == step and training["pending_buffer_decisions"] == 0,
            "Incomplete optimized checkpoint budget")
    digest = audit.file(folder/"checkpoint.pt", training["checkpoint_sha256"])
    path_ids = protocol["validation_path_ids"]
    members = {"summary.json", "binding.json"} | {f"{p}.{ext}" for p in path_ids for ext in ("json", "npz")}
    delivery = audit.manifest(folder/"validation", "delivery_manifest.json", members)
    binding = audit.json(folder/"validation/binding.json")
    expected_binding = dict(checkpoint_sha256=digest,
        config_sha256=audit.file(folder/"config.yaml"), source_sha256=provenance["source_sha256"])
    require(all(delivery[k] == binding[k] == value for k, value in expected_binding.items()),
            "Validation source/config/model binding mismatch")
    require(binding["split"] == "validation", "Unexpected data split")
    # Read only the identity of the old development manifest, never new holdout.
    descriptor = protocol["training_validation_manifest"]
    development_manifest = audit.root/"PPO_project/data/preview_event_v1/manifest.json"
    require(relative_path(audit.root, descriptor["file"]) == development_manifest
            and descriptor["sha256"] == binding["manifest_sha256"], "Wrong development manifest binding")
    audit.file(development_manifest, binding["manifest_sha256"])
    summary = audit.json(folder/"validation/summary.json")
    rows = summary["rows"]
    require(len(rows) == 3 and {r["path_id"] for r in rows} == set(path_ids), "Missing/duplicate validation path")
    for row in rows:
        raw = audit.json(folder/f"validation/{row['path_id']}.json")
        require(raw == row, "Raw JSON differs from validation summary")
        require(row["seed"] == seed and row["method"] == method and row["training_decisions"] == step,
                "Validation row identity mismatch")
        require(all(row[k] == value for k, value in expected_binding.items()), "Validation row binding mismatch")
        audit.file(folder/f"validation/{row['path_id']}.npz", row["trace_sha256"])
        require(type(row["success"]) is bool and isinstance(row["failure_reasons"], list), "Invalid success schema")
        require(row["success"] == (len(row["failure_reasons"]) == 0), "Inconsistent failure reasons")
        require(math.isfinite(row["elapsed_time_s"]) and row["elapsed_time_s"] > 0, "Invalid elapsed time")
    successes = sum(r["success"] for r in rows)
    candidate = dict(step=step, successes=successes, count=3,
        complete_validation_time_s=sum(r["elapsed_time_s"] for r in rows) if successes == 3 else None,
        checkpoint_dir=folder.relative_to(audit.root).as_posix(), checkpoint_sha256=digest,
        validation=dict(file=(folder/"validation/summary.json").relative_to(audit.root).as_posix(),
                        sha256=audit.file(folder/"validation/summary.json")),
        validation_sha256=audit.file(folder/"validation/summary.json"),
        training_summary_sha256=audit.file(folder/"training_summary.json"),
        snapshot_manifest_sha256=audit.file(folder/"snapshot_manifest.json"),
        validation_evaluation_wall_s=sum(r["evaluation_wall_s"] for r in rows))
    return dict(rows=rows, candidate=candidate, signature=signature, training=training)


def selection_key(candidate):
    return (-candidate["successes"], candidate["complete_validation_time_s"]
            if candidate["successes"] == 3 else math.inf, candidate["step"])


def collect(study, root=ROOT):
    root, study = Path(root).resolve(), Path(study).resolve()
    require(study.is_relative_to(root), "Study must be inside repository")
    audit = Audit(root)
    protocol = audit.json(study/"protocol.json")
    require(protocol["version"] == "formal-preview-continuation-v1", "Unknown protocol")
    require(protocol["seeds"] == list(SEEDS) and protocol["methods"] == list(METHODS), "Wrong method/seed registry")
    require(protocol["validation_milestones"] == list(STEPS) and protocol["selection_milestones"] == list(STEPS),
            "Unequal validation opportunities")
    require(protocol["pretraining_decisions"] == protocol["additional_decisions"] == 4096, "Unequal training budget")
    require(len(protocol["validation_path_ids"]) == len(set(protocol["validation_path_ids"])) == 3,
            "Exactly three development validation paths required")
    progress, all_rows, selected_rows, selections, priors, signature = [], [], [], [], set(), None
    for seed in SEEDS:
        for method in METHODS:
            selection_path = study/f"selections/{method}_seed{seed}.json"
            selection_exists = selection_path.is_file()
            steps = [s for s in STEPS if (checkpoint_dir(study, method, seed, s)/"validation/delivery_manifest.json").is_file()]
            progress.append(dict(seed=seed, method=method, selection_present=selection_exists,
                                 completed_validation_steps=steps, verified_rows=len(steps)*3))
            require(not selection_exists or steps == list(STEPS), "Selection exists before all validation is sealed")
            if not steps:
                continue
            prior = prior_at(study, method, seed, protocol, audit)
            require(prior["digest"] not in priors, "Pretraining weights reused across seed/method")
            priors.add(prior["digest"])
            snapshots = []
            for step in steps:
                snap = verify_checkpoint(study, method, seed, step, protocol, prior, audit)
                require(signature is None or signature == snap["signature"], "Shared motion constraints/config differ")
                signature = snap["signature"]
                snapshots.append(snap)
                all_rows.extend(copy.deepcopy(snap["rows"]))
            if not selection_exists:
                continue
            selection = audit.json(selection_path)
            require(selection["seed"] == seed and selection["method"] == method, "Selection identity mismatch")
            require(selection["available_pretraining_decisions"] == selection["available_additional_decisions"] == 4096,
                    "Selection budget mismatch")
            require(selection["step0_eligible"] is True and selection["rule"] == protocol["selection_rule"],
                    "Selection rule changed")
            candidates = [s["candidate"] for s in snapshots]
            require(selection["all_checkpoints"] == candidates, "Selection candidates differ from verified raw records")
            require(selection["selected"] == min(candidates, key=selection_key), "Checkpoint selection violates frozen rule")
            final = snapshots[-1]["training"]
            require(final["decisions"] == 4096 and selection["costs"]["continuation"] == final,
                    "Full continuation budget not expended or costs changed")
            require(selection["costs"]["pretraining"] == prior["costs"]
                    and selection["costs"]["validation_evaluation_wall_s"] == sum(c["validation_evaluation_wall_s"] for c in candidates),
                    "Pretraining/validation cost record mismatch")
            chosen_step = selection["selected"]["step"]
            selected_rows.extend(copy.deepcopy(next(s["rows"] for s in snapshots if s["candidate"]["step"] == chosen_step)))
            selections.append(selection)
    audit.unchanged()
    complete = len(selections) == 10 and len(all_rows) == 180 and len(selected_rows) == 30
    return dict(complete=complete, status="complete" if complete else "incomplete", data_role=DATA_ROLE,
        selection_count=len(selections), expected_selections=10, verified_candidate_rows=len(all_rows), expected_candidate_rows=180,
        progress=progress, all_rows=all_rows, selected_rows=selected_rows, selections=selections,
        validation_path_ids=protocol["validation_path_ids"], input_sha256=audit.inputs,
        protocol_sha256=audit.file(study/"protocol.json"), holdout_access=False, actor_execution=False)


def summarize(rows, path_ids):
    expected = {(s, m, p) for s in SEEDS for m in METHODS for p in path_ids}
    require(len(rows) == 30 and {(r["seed"], r["method"], r["path_id"]) for r in rows} == expected,
            "Selected summary needs all thirty unique validation rows")
    per_seed, pairs = [], []
    index = {(r["seed"], r["method"], r["path_id"]): r for r in rows}
    for seed in SEEDS:
        chosen = {m: [index[seed, m, p] for p in path_ids] for m in METHODS}
        common = [p for p in path_ids if all(index[seed, m, p]["success"] for m in METHODS)]
        totals = {m: sum(index[seed, m, p]["elapsed_time_s"] for p in common) if common else None for m in METHODS}
        per_seed.append(dict(seed=seed, fixed3_successes=sum(r["success"] for r in chosen["fixed3"]),
            adaptive_successes=sum(r["success"] for r in chosen["adaptive"]), paths_per_method=3,
            fixed3_selected_step=chosen["fixed3"][0]["training_decisions"],
            adaptive_selected_step=chosen["adaptive"][0]["training_decisions"], common_success_paths=common,
            paired_count=len(common), fixed3_common_time_s=totals["fixed3"], adaptive_common_time_s=totals["adaptive"],
            common_time_gain_fraction=1-totals["adaptive"]/totals["fixed3"] if common else None))
        for path in path_ids:
            f, a = index[seed, "fixed3", path], index[seed, "adaptive", path]
            both = f["success"] and a["success"]
            pairs.append(dict(seed=seed, path_id=path, fixed3_success=f["success"], adaptive_success=a["success"],
                common_success=both, fixed3_time_s=f["elapsed_time_s"], adaptive_time_s=a["elapsed_time_s"],
                time_gain_fraction=1-a["elapsed_time_s"]/f["elapsed_time_s"] if both else None))
    common_pairs = [p for p in pairs if p["common_success"]]
    ft = sum(p["fixed3_time_s"] for p in common_pairs)
    at = sum(p["adaptive_time_s"] for p in common_pairs)
    methods = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        successes = sum(r["success"] for r in subset)
        methods[method] = dict(runs=15, successes=successes, failures=15-successes,
            success_fraction=successes/15, failure_fraction=(15-successes)/15,
            all_three_success_seeds=[s for s in SEEDS if all(index[s, method, p]["success"] for p in path_ids)],
            failure_reasons=dict(Counter(reason for r in subset for reason in r["failure_reasons"])))
    return dict(data_role=DATA_ROLE, methods=methods, by_seed=per_seed, paired_rows=pairs,
        common_success=dict(pairs=len(common_pairs), candidate_pairs=15,
            contributing_seeds=sorted({p["seed"] for p in common_pairs}),
            fixed3_total_time_s=ft if common_pairs else None, adaptive_total_time_s=at if common_pairs else None,
            time_gain_fraction=1-at/ft if common_pairs else None),
        statistical_units="Five independent training seeds and three paths; interpolation samples and six correlated checkpoints are not independent replicates.",
        uncertainty="Descriptive validation only: checkpoint selection used these paths and times; no independent-test confidence or generalization claim.")


def raw_csv(rows):
    fields = ("seed", "method", "path_id", "training_decisions", "success", "elapsed_time_s", "ordered_progress",
              "contour_error_max_mm", "contour_error_mean_mm", "contour_error_rmse_mm", "termination_reason",
              "terminal_rest_satisfied", "full_rest_satisfied", "sample_count", "checkpoint_sha256", "config_sha256", "source_sha256", "trace_sha256")
    output = []
    for r in rows:
        row = {key: r.get(key) for key in fields}
        for key in ("failure_reasons", "constraints", "axis_diagnostics", "terminal_rest_checks"):
            row[key] = json.dumps(r.get(key), ensure_ascii=False, sort_keys=True)
        row["data_role"] = DATA_ROLE
        output.append(row)
    return output


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(study, output, root=ROOT):
    output = Path(output).resolve()
    require(output.is_relative_to(Path(root).resolve()), "Report output must be inside repository")
    require(not output.exists(), "Preserve previous report; use a new output directory")
    data = collect(study, root)
    output.mkdir(parents=True)
    if not data["complete"]:
        progress = {k: v for k, v in data.items() if k not in ("all_rows", "selected_rows", "selections")}
        write_json(output/"progress.json", progress)
        return progress
    summary = summarize(data["selected_rows"], data["validation_path_ids"])
    write_csv(output/"all_checkpoints_180.csv", raw_csv(data["all_rows"]))
    write_csv(output/"selected_validation_30.csv", raw_csv(data["selected_rows"]))
    seed_rows = copy.deepcopy(summary["by_seed"])
    for row in seed_rows:
        row["common_success_paths"] = ";".join(row["common_success_paths"])
    write_csv(output/"per_seed.csv", seed_rows)
    write_csv(output/"paired_validation_15.csv", summary["paired_rows"])
    write_json(output/"summary.json", summary)
    lines = ["# 五种子训练的开发验证记录", "",
        "全部十条独立训练链、六个验证快照已完成并核验，保留 180 条候选记录与 30 条选定记录。验证路径用于选择 checkpoint，以下是开发验证结果，不能当作独立测试或泛化结论。", "",
        "|种子|固定前瞻成功数|自适应成功数|固定所选步|自适应所选步|共同成功路径数|共同成功时间收益|", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["by_seed"]:
        gain = row["common_time_gain_fraction"]
        lines.append(f"|{row['seed']}|{row['fixed3_successes']}/3|{row['adaptive_successes']}/3|"
            f"{row['fixed3_selected_step']}|{row['adaptive_selected_step']}|{row['paired_count']}|"
            + (f"{gain*100:.2f}%|" if gain is not None else "无共同成功样本|"))
    lines += ["", "时间收益为同一种子、同一路径且两方法均成功时的 1−ΣT自适应/ΣT固定；失败轨迹的时间仅在原始表保留，不作为有效完成时间。所有配置的失败均保留在 15 条/方法的分母中。", "",
        "统计单位是独立训练种子与路径。六个 checkpoint 属于同一训练链，插补采样点也不是独立重复。每条链均完成 4096 次独立预训练和 4096 次追加训练；选中 step0 仍计完整追加成本。", "",
        f"固定前瞻：{summary['methods']['fixed3']['successes']}/15 成功，三条路径全部成功的种子为 {summary['methods']['fixed3']['all_three_success_seeds']}。",
        f"自适应前瞻：{summary['methods']['adaptive']['successes']}/15 成功，三条路径全部成功的种子为 {summary['methods']['adaptive']['all_three_success_seeds']}。", "",
        "本程序没有加载模型或访问 holdout 文件，没有改变原成功/失败标签。完整原始约束、停稳检查、所有失败原因及配置/模型/trace SHA 均保留在 CSV；verification.json 记录输入 SHA。", ""]
    (output/"验证说明.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output/"verification.json", dict(status="complete", candidate_rows=180, selected_rows=30,
        selection_count=10, data_role=DATA_ROLE, protocol_sha256=data["protocol_sha256"],
        input_sha256=data["input_sha256"], holdout_access=False, actor_execution=False,
        output_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.study, args.output)
    print(json.dumps({k: result[k] for k in ("status", "selection_count", "verified_candidate_rows", "methods", "common_success") if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
