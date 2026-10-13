"""Create standalone paper tables from a completed, SHA-bound formal report.

Only saved JSON logs and byte hashes are read. No environment, model, reference
coordinates, or trajectory arrays are loaded. Existing reports are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import preview_formal_statistics as formal_stats


ROOT = Path(__file__).resolve().parents[2]
STEPS = (0, 256, 512, 1024, 2048, 4096)
REPORT_OUTPUTS = {"report.json", "raw_metrics.csv", "method_statistics.csv", "paired_statistics.csv"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Source changed during hashing: {path}")
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_file(root, name):
    require(isinstance(name, str) and not Path(name).is_absolute(), "Expected a relative source path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), f"Source path escapes root: {name}")
    return path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bind(root, descriptor, inputs):
    require(isinstance(descriptor, dict) and {"file", "sha256"}.issubset(descriptor),
            "Missing file/SHA source binding")
    path = relative_file(root, descriptor["file"])
    require(path.is_file() and sha(path) == descriptor["sha256"], f"Bound source SHA mismatch: {path}")
    name = path.relative_to(root).as_posix()
    require(name not in inputs or inputs[name] == descriptor["sha256"], "Conflicting source bindings")
    inputs[name] = descriptor["sha256"]
    return path


def _training_records(freeze, root, inputs):
    require(freeze.get("version") == "formal-preview-continuation-v1"
            and freeze.get("all_training_complete") is True, "A completed formal freeze is required")
    require(freeze.get("seeds") == list(formal_stats.FORMAL_SEEDS)
            and freeze.get("methods") == list(formal_stats.METHODS), "Wrong formal seed/method design")
    records = freeze.get("records", [])
    expected = {(seed, method) for seed in formal_stats.FORMAL_SEEDS for method in formal_stats.METHODS}
    require(len(records) == 10 and {(r["seed"], r["method"]) for r in records} == expected,
            "All ten unique frozen models are required")
    protocol = read_json(bind(root, freeze["protocol"], inputs))
    require(protocol["sources"]["continuation"] == freeze["source"], "Freeze/protocol source mismatch")
    evaluation = protocol["evaluation_sources"]
    require(hashlib.sha256(json.dumps(evaluation["source_files"], sort_keys=True).encode()).hexdigest()
            == evaluation["source_sha256"], "Malformed frozen evaluator source digest")
    stats_name = "PPO_project/tools/preview_formal_statistics.py"
    require(stats_name in evaluation["source_files"]
            and sha(formal_stats.__file__) == evaluation["source_files"][stats_name],
            "Loaded statistics implementation differs from frozen evaluator")
    for name, digest in evaluation["source_files"].items():
        bind(root, {"file": name, "sha256": digest}, inputs)
    require(freeze["holdout_manifest"] == protocol["holdout_manifest"], "Holdout identity differs from protocol")
    # Hash the manifest as a bound artifact; never parse reference geometry.
    bind(root, freeze["holdout_manifest"], inputs)
    result = {}
    for record in records:
        seed, method, step = record["seed"], record["method"], record["selected_step"]
        require(step in STEPS, "Selected step is outside the registered grid")
        require(record["available_pretraining_decisions"] == 4096
                and record["available_additional_decisions"] == 4096,
                "Both methods require the complete 4096 + 4096 available budget")
        require(record["source_sha256"] == freeze["source"]["source_sha256"], "Model/source mismatch")
        artifacts = record["artifacts"]
        for name in ("checkpoint.pt", "config.yaml"):
            bind(root, artifacts[name], inputs)
        selected = read_json(bind(root, artifacts["training_summary.json"], inputs))
        prior = read_json(bind(root, record["pretraining"]["artifacts"]["training_summary.json"], inputs))
        final = read_json(bind(root, record["additional_budget_completion"], inputs))
        selection = read_json(bind(root, record["selection"], inputs))
        for log, decisions in ((selected, step), (prior, 4096), (final, 4096)):
            require(log["seed"] == seed and log["method"] == method and log["decisions"] == decisions,
                    "Training log seed/method/budget mismatch")
        for log, decisions in ((selected, step), (final, 4096)):
            require(log["optimized_decisions"] == decisions and log["pending_buffer_decisions"] == 0,
                    "Training log contains an incomplete optimizer budget")
        require(record["pretraining"]["decisions"] == prior["decisions"] == 4096,
                "Independent pretraining budget mismatch")
        require(selection["seed"] == seed and selection["method"] == method
                and selection["selected"]["step"] == step
                and selection["selected"]["checkpoint_sha256"] == artifacts["checkpoint.pt"]["sha256"]
                and selected["checkpoint_sha256"] == artifacts["checkpoint.pt"]["sha256"],
                "Selected checkpoint differs from the frozen training logs")
        result[seed, method] = record
    return result


def verify_report(report_path, freeze_path, *, root=ROOT, manifest_path=None):
    """Verify original report bytes, every original input, and all 60 outcomes."""
    root = Path(root).resolve()
    report_path, freeze_path = Path(report_path).resolve(), Path(freeze_path).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path else report_path.parent / "manifest.json"
    require(report_path.name == "report.json", "Use the completed formal report.json")
    require(all(path.is_relative_to(root) and path.is_file()
                for path in (report_path, freeze_path, manifest_path)), "Missing report, manifest, or freeze within root")
    require(manifest_path.parent == report_path.parent, "Report manifest must accompany report.json")
    manifest_digest = sha(manifest_path)
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get("inputs"), dict) and isinstance(manifest.get("outputs"), dict)
            and REPORT_OUTPUTS.issubset(manifest["outputs"]), "Legacy or incomplete formal report manifest")
    for name, digest in manifest["outputs"].items():
        output = relative_file(report_path.parent, name)
        require(output.is_file() and sha(output) == digest, f"Report output SHA mismatch: {name}")
    inputs = {}
    for name, digest in manifest["inputs"].items():
        bind(root, {"file": name, "sha256": digest}, inputs)
    freeze_name = freeze_path.relative_to(root).as_posix()
    require(manifest["inputs"].get(freeze_name) == sha(freeze_path), "Report is not bound to this freeze")
    report, freeze = read_json(report_path), read_json(freeze_path)
    require(report.get("freeze_sha256") == sha(freeze_path), "Report/freeze SHA mismatch")
    require(isinstance(report.get("rows"), list) and len(report["rows"]) == 60
            and isinstance(report.get("statistics"), dict), "The completed report requires all 60 outcomes")
    records = _training_records(freeze, root, inputs)
    expected_paths = report["statistics"].get("expected_paths")
    expected_inputs = {freeze_name}
    for row in report["rows"]:
        key = (row["seed"], row["method"])
        require(key in records, "Result has an unknown seed/method")
        record = records[key]
        source = freeze_path.parent / "holdout" / f"{row['method']}_seed{row['seed']}" / f"{row['path_id']}.json"
        require(source.resolve().is_relative_to(freeze_path.parent), "Result identity escapes the study")
        name = source.relative_to(root).as_posix()
        require(name in manifest["inputs"], "Original outcome missing from report manifest")
        require(canonical(read_json(source)) == canonical(row), "Report row differs from its original outcome")
        expected_inputs.add(name)
        require(row["freeze_sha256"] == report["freeze_sha256"]
                and row["checkpoint_sha256"] == record["artifacts"]["checkpoint.pt"]["sha256"]
                and row["config_sha256"] == record["artifacts"]["config.yaml"]["sha256"]
                and row["source_sha256"] == freeze["source"]["source_sha256"]
                and row["holdout_manifest_sha256"] == freeze["holdout_manifest"]["sha256"]
                and row["selected_step"] == record["selected_step"], "Outcome differs from the frozen model binding")
        for field in ("trace", "inference_samples"):
            artifact = row[field]
            require(manifest["inputs"].get(artifact["file"]) == artifact["sha256"],
                    f"Outcome {field} is not bound by the report manifest")
            expected_inputs.add(artifact["file"])
    require(expected_inputs == set(manifest["inputs"]), "Report input manifest has missing or unrelated raw outcomes")
    recomputed = formal_stats.summarize(report["rows"], freeze["seeds"], expected_paths)
    require(canonical(recomputed) == canonical(report["statistics"]), "Saved statistics differ from recomputation")
    inputs[report_path.relative_to(root).as_posix()] = manifest["outputs"]["report.json"]
    inputs[manifest_path.relative_to(root).as_posix()] = manifest_digest
    return {"rows": report["rows"], "statistics": recomputed, "models": records, "inputs": inputs}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def table_data(verified):
    rows, stats, models = verified["rows"], verified["statistics"], verified["models"]
    main_rows = []
    for index, path_id in enumerate(sorted(stats["expected_paths"]), 1):
        common = stats["by_path"][path_id]["paired"]["common_success_pairs"]
        for method in formal_stats.METHODS:
            times = [pair[f"{method}_time_s"] for pair in common]
            runs = [row for row in rows if row["path_id"] == path_id and row["method"] == method]
            peak = [row["contour_error_max_mm"] for row in runs]
            rmse = [row["contour_error_rmse_mm"] for row in runs]
            main_rows.append({
                "path_label": f"P{index}", "path_id": path_id, "family": stats["expected_paths"][path_id],
                "method": method, "successes": sum(row["success"] for row in runs), "total_seeds": 5,
                "common_success_n": len(common), "common_success_seeds": [pair["seed"] for pair in common],
                "common_time_mean_s": statistics.fmean(times) if times else None,
                "common_time_sd_s": statistics.stdev(times) if len(times) > 1 else None,
                "all_runs_peak_error_worst_mm": max(peak) if all(map(finite, peak)) else None,
                "all_runs_rmse_mean_mm": statistics.fmean(rmse) if all(map(finite, rmse)) else None,
                "peak_error_finite_n": sum(map(finite, peak)), "rmse_finite_n": sum(map(finite, rmse)),
            })
    seed_rows = []
    for seed in formal_stats.FORMAL_SEEDS:
        item = {"seed": seed}
        for method in formal_stats.METHODS:
            model = models[seed, method]
            item[method] = {
                "selected_step": model["selected_step"],
                "available_pretraining_decisions": model["available_pretraining_decisions"],
                "available_additional_decisions": model["available_additional_decisions"],
                "test_successes": stats["by_seed"][str(seed)][method]["n_success"], "test_runs": 6,
            }
        seed_rows.append(item)
    return {"main_rows": main_rows, "seed_rows": seed_rows,
            "overall": stats["overall"], "statistical_protocol": stats["statistical_protocol"]}


def tex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(text))


def number(value, digits=3):
    return f"{value:.{digits}f}" if finite(value) else r"\mathrm{NA}"


def render_main(data):
    lines = [r"% Generated from verified formal results; do not edit numeric cells.",
             r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{六条独立测试路径的结果。耗时仅比较两种方法共同完整成功的 seed。}",
             r"\label{tab:formal-holdout}", r"\begin{tabular}{llcccrr}", r"\hline",
             r"路径 & 方法 & 成功 & $T_{\cap}$ (s) & $n_{\cap}$ & $e_{\max}$ (mm) & RMSE (mm) \\", r"\hline"]
    for row in data["main_rows"]:
        label = "固定 3 mm" if row["method"] == "fixed3" else "自适应"
        time = (f"${number(row['common_time_mean_s'])} \\pm {number(row['common_time_sd_s'])}$"
                if row["common_success_n"] else r"$\mathrm{NA}$")
        lines.append(f"{row['path_label']} & {label} & {row['successes']}/5 & {time} & "
                     f"{row['common_success_n']} & ${number(row['all_runs_peak_error_worst_mm'])}$ & "
                     f"${number(row['all_runs_rmse_mean_mm'])}$" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
              r"成功数计入全部 5 个 seed；$T_{\cap}$ 为共同成功集合上的均值 $\pm$ 样本标准差。",
              r"$e_{\max}$ 为全部运行的最坏峰值，RMSE 为全部运行的均值；失败耗时不计入 $T_{\cap}$。",
              r"共同成功数为 0 时耗时记为 NA；为 1 时标准差记为 NA。误差存在非有限值时对应单元格记为 NA。",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_seeds(data):
    lines = [r"% Selected checkpoints and available budgets only.", r"\begin{table}[t]", r"\centering\small",
             r"\caption{各独立训练 seed 的选中检查点与完整测试成功数。}", r"\label{tab:formal-seeds}",
             r"\begin{tabular}{rrrrcc}", r"\hline",
             r"Seed & 固定 step & 自适应 step & 各方法可用预算 & 固定成功 & 自适应成功 \\", r"\hline"]
    for row in data["seed_rows"]:
        fixed, adaptive = row["fixed3"], row["adaptive"]
        budget = f"{fixed['available_pretraining_decisions']}+{fixed['available_additional_decisions']}"
        lines.append(f"{row['seed']} & {fixed['selected_step']} & {adaptive['selected_step']} & {budget} & "
                     f"{fixed['test_successes']}/6 & {adaptive['test_successes']}/6" + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
              r"预算按独立预训练决策数加后续训练决策数计；两种方法均完成全部可用预算后依据共同验证规则选模。",
              r"选中 step 为后续训练阶段的检查点位置，不代表实际只训练到该 step。",
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_macros(data):
    paired = data["overall"]["paired"]
    interval = paired["time_reduction_ci95"]
    valid_interval = (isinstance(interval, (list, tuple)) and len(interval) == 2
                      and all(map(finite, interval))
                      and paired["time_reduction_bootstrap"]["ci_unavailable_reason"] is None)
    gain = paired["time_reduction_fraction"]
    values = {
        "FormalOverallTimeReductionPercent": f"{100 * gain:.2f}\\%" if finite(gain) else "NA",
        "FormalOverallTimeReductionCI": f"[{100 * interval[0]:.2f}\\%, {100 * interval[1]:.2f}\\%]" if valid_interval else "NA",
        "FormalOverallCommonSuccessPairs": str(paired["n_common_success"]),
        "FormalBootstrapValidReplicates": str(paired["time_reduction_bootstrap"]["bootstrap_valid_resamples"]),
        "FormalBootstrapTotalReplicates": str(paired["time_reduction_bootstrap"]["bootstrap_total_resamples"]),
    }
    return "% Overall time reduction is conditional on joint success; CI clusters whole seeds.\n" + "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()) + "\n"


def render_notes(data):
    paired = data["overall"]["paired"]
    bootstrap = paired["time_reduction_bootstrap"]
    lines = ["# 正式结果表说明", "", "本目录为独立生成产物，未接入现用论文、模板或图。",
             "", "- formal_results.tex：6 条路径 × 2 方法，共 12 行。成功数使用全部 5 个 seed。",
             "- formal_seeds.tex：5 个 seed 的选中 step、每方法 4096+4096 可用预算、测试完整成功数。",
             "- formal_macros.tex：共同成功集合上的总体耗时下降及 95% CI；不可用值写 NA。",
             "- paper_table_data.json：未四舍五入的表格数值、共同成功 seed 集合与统计口径。",
             "- manifest.json：原始 report/manifest/freeze、原始结果、训练日志、生成源码及输出 SHA-256。", "",
             "主表 T 为同一路径两种方法共同完整成功 seed 的均值±样本标准差（ddof=1），不是置信区间。",
             "失败运行保留在成功率分母；其耗时不进入成功耗时。峰值误差取全部运行中的最坏值，RMSE取全部运行均值。",
             "若误差包含缺失或非有限数，对应表格单元格为 NA，并在 JSON 保留有效数量，不能用有限子集冒充全部运行。", "",
             "总体耗时下降 = 1 - ΣT_adaptive / ΣT_fixed3，仅在相同 seed/path 的共同完整成功集合上计算。",
             "95% bootstrap CI 以独立训练 seed 为整簇重采样，保留该 seed 的全部路径和两方法配对；仅有5个独立seed，CI稳定性有限。",
             f"Bootstrap 共 {bootstrap['bootstrap_total_resamples']} 次，有效 {bootstrap['bootstrap_valid_resamples']} 次，"
             f"无共同集合的重采样 {bootstrap['bootstrap_undefined_resamples']} 次。",
             "若部分重采样无共同集合，区间条件于非空共同集合；无有效区间或不足两个贡献seed时显示NA。", "",
             "路径编号："]
    for row in data["main_rows"]:
        if row["method"] == "fixed3":
            lines.append(f"- {row['path_label']} = {row['path_id']}（{row['family']}）")
    return "\n".join(lines) + "\n"


def build_tables(report_path, freeze_path, *, root=ROOT, manifest_path=None):
    report_path, root = Path(report_path).resolve(), Path(root).resolve()
    output = report_path.parent / "paper_tables_v1"
    require(not output.exists(), f"Preserve existing paper tables: {output}")
    verified = verify_report(report_path, freeze_path, root=root, manifest_path=manifest_path)
    data = table_data(verified)
    contents = {"formal_results.tex": render_main(data), "formal_seeds.tex": render_seeds(data),
                "formal_macros.tex": render_macros(data), "README.md": render_notes(data),
                "paper_table_data.json": json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    # Recheck all bound sources after rendering and before creating any output.
    for name, digest in verified["inputs"].items():
        require(sha(relative_file(root, name)) == digest, f"Bound source changed during rendering: {name}")
    output.mkdir()
    for name, content in contents.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    sources = {str(Path(path).resolve()): sha(path) for path in (__file__, formal_stats.__file__)}
    manifest = {"version": "formal-paper-tables-v1", "inputs": verified["inputs"], "sources": sources,
                "outputs": {name: sha(output / name) for name in contents},
                "statistics_recomputed_equal": True, "n_verified_outcomes": 60,
                "main_table_rows": 12, "seed_table_rows": 5}
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"output": str(output), "main_rows": 12, "seed_rows": 5,
            "manifest_sha256": sha(output / "manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_tables(arguments.report, arguments.freeze, manifest_path=arguments.manifest), ensure_ascii=False))
