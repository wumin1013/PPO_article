"""Generate V1 revision figures/tables strictly from saved runs; no training.

Contract: descriptive diagnostics under a bounded budget; all failed runs remain.
Python/matplotlib only, 183-mm vector figures, editable text. The existing paper's
Times font and restrained method colors are retained. No external precedent is
claimed: network retrieval was outside this offline task; bundled figure notes
support aligned quantitative panels, direct labels and shared legends only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "论文项目"
DEFAULT_REVISION = PAPER / "revision_20260908"
FIGURES = PAPER / "figures/generated"
GENERATED = PAPER / "generated"
METHODS = ["coupled", "decoupled", "fixed_short"]
LABEL = {"coupled": "Coupled", "decoupled": "Decoupled", "fixed_short": "Fixed short",
         "fixed_mid": "Fixed medium", "fixed_long": "Fixed long", "nnc": "NNC auxiliary"}
COLOR = {"coupled": "#1f77b4", "decoupled": "#d78332", "fixed_short": "#60666c", "nnc": "#33956d"}
FAMILIES = ["mixed_lengths", "dense_corners", "line_curve"]
FAMILY_LABEL = {"mixed_lengths": "Mixed lengths", "dense_corners": "Dense corners", "line_curve": "Line + curve"}
SOURCES: dict[str, str] = {}


def read_json(path: Path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def style():
    fonts = {item.name for item in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Times New Roman" if "Times New Roman" in fonts else "DejaVu Sans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
        "lines.linewidth": 1.2, "legend.frameon": False})


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(FIGURES / f"revision_{name}.{extension}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_sd(values):
    x = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else None) if len(x) else (None, None)


def stat_text(values, decimals=3):
    mean, sd = mean_sd(values)
    if mean is None:
        return "--"
    return f"${mean:.{decimals}f}\\pm{sd:.{decimals}f}$" if sd is not None else f"${mean:.{decimals}f}$"


def escaped(text):
    return str(text).replace("_", r"\_").replace("%", r"\%")


def table(caption, label, headers, rows):
    n = len(headers)
    return [r"\begin{table}[htbp]", r"\centering\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            r"\setlength{\tabcolsep}{4pt}", r"\resizebox{0.99\linewidth}{!}{%",
            "\\begin{tabular}{" + "l" * min(2, n) + "r" * max(0, n-2) + "}", r"\toprule",
            " & ".join(headers) + r"\\", r"\midrule", *[" & ".join(map(str, row)) + r"\\" for row in rows],
            r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]


def figure(name, caption, label, width="0.98"):
    return [r"\begin{figure}[htbp]", r"\centering", f"\\includegraphics[width={width}\\linewidth]{{figures/generated/revision_{name}.pdf}}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{figure}", ""]


def discover_runs(revision: Path):
    result = []
    for stage in ["screen", "main", "nnc_auxiliary/runs"]:
        for summary in sorted((revision / "experiments" / stage).glob("*/evaluation_*/summary.json")):
            document = read_json(summary)
            for item in document.get("rows", []):
                row = dict(item)
                row.update(stage=stage, evaluation=summary.parent.name, source=summary.relative_to(ROOT).as_posix(), run_dir=str(summary.parent.parent))
                result.append(row)
    return result


def selected(rows, stage="main", evaluation="evaluation_test_nominal", method=None):
    return [r for r in rows if r["stage"] == stage and r["evaluation"] == evaluation and (method is None or r["variant"] == method)]


def common_success(rows, methods=METHODS):
    successful = [{(r["seed"], r["path_id"]) for r in rows if r["variant"] == method and r["success"]} for method in methods]
    return set.intersection(*successful) if successful else set()


def path_figure(manifest):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.15), constrained_layout=True)
    source_dir = ROOT / "PPO_project/data/revision_20260908"
    for i, family in enumerate(FAMILIES):
        for j, split in enumerate(["train", "validation", "test"]):
            row = next(item for item in manifest["paths"] if item["family"] == family and item["split"] == split)
            path = source_dir / row["reference_file"]
            SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            p = np.genfromtxt(path, delimiter=",", skip_header=1)
            ax = axes[i, j]
            ax.plot(p[:, 0], p[:, 1], color="#1f77b4", lw=1.1)
            ax.scatter(p[0, 0], p[0, 1], s=14, color="#333333", zorder=3)
            ax.scatter(p[-1, 0], p[-1, 1], s=18, facecolors="white", edgecolors="#333333", zorder=3)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=.18, linestyle=":")
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            stats = row["statistics"]
            ax.set_title(f"{FAMILY_LABEL[family]}\n{stats['path_length_mm']:.1f} mm; {stats['segment_count']} segments", fontsize=8)
            ax.text(-.16, 1.02, chr(97 + i*3+j), transform=ax.transAxes, weight="bold")
    fig.suptitle("V1 synthetic paths: train | validation | initially held out, now development", fontsize=10)
    save(fig, "paths_v1")


def diagnostic_figure(revision):
    diagnosis = read_json(revision / "environment_diagnostics/environment_diagnostics.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    versions = list(diagnosis["kcm"])
    bounds = [diagnosis["constraints"]["max_jerk"], diagnosis["constraints"]["max_angular_jerk"]]
    for offset, version in enumerate(versions):
        vals = diagnosis["kcm"][version]
        ratio = [vals["jerk"]/bounds[0], vals["angular_jerk"]/bounds[1]]
        axes[0].bar(np.arange(2)+(offset-.5)*.32, ratio, width=.30,
                    color=["#a8a8a8", "#1f77b4"][offset], label=["Historical", "Corrected"][offset])
    axes[0].axhline(1, color="#333333", ls="--", lw=.8)
    axes[0].set_xticks([0,1], ["Feed jerk", "Angular jerk"])
    axes[0].set_ylabel("Peak / prescribed limit")
    axes[0].set_title("Output consistency probe")
    axes[0].legend()
    for distance in sorted({r["reference_distance"] for r in diagnosis["geometry"]}, key=lambda x: -1 if x is None else x):
        subset = [r for r in diagnosis["geometry"] if r["reference_distance"] == distance]
        label = "Coupled reward scale" if distance is None else "Fixed reward scale"
        color = "#60666c" if distance is None else "#d78332"
        x = [r["active_distance"] for r in subset]
        axes[1].plot(x, [r["reward_components"]["w_track_dyn"] for r in subset], "o-", color=color, label=label)
        axes[2].plot(x, [r["reward"] for r in subset], "o-", color=color, label=label)
    axes[1].set_ylabel("Tracking weight")
    axes[2].set_ylabel("Reward at identical physical state")
    axes[1].set_title("Reward-scale coupling")
    axes[2].set_title("Immediate reward effect")
    for i, ax in enumerate(axes):
        ax.text(-.13, 1.06, chr(97+i), transform=ax.transAxes, weight="bold")
        if i:
            ax.set_xlabel("Active preview (mm)")
        ax.grid(axis="y", alpha=.15, ls=":")
    axes[2].legend(loc="lower right", fontsize=6)
    save(fig, "mechanism_v1")
    return diagnosis


def performance_figure(rows):
    data = selected(rows)
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    for j, method in enumerate(METHODS):
        for col, metric in enumerate(["ordered_progress", "contour_error_max_mm", "success"]):
            means, deviations = [], []
            for family in FAMILIES:
                subset = [r for r in data if r["variant"] == method and r["path_id"] == family+"_test"]
                mean, sd = mean_sd([float(r[metric]) for r in subset])
                means.append(mean if mean is not None else np.nan)
                deviations.append(sd or 0)
            axes[col].errorbar(np.arange(3)+(j-1)*.15, means, yerr=deviations if col < 2 else None,
                marker=["o","s","^"][j], ms=4, capsize=2, ls="none", color=COLOR[method], label=LABEL[method])
    for j, ax in enumerate(axes):
        ax.set_xticks(range(3), ["Mixed\nlengths", "Dense\ncorners", "Line +\ncurve"])
        ax.grid(axis="y", ls=":", alpha=.2)
        ax.text(-.13, 1.04, chr(97+j), transform=ax.transAxes, weight="bold")
    axes[0].set_ylabel("Ordered progress")
    axes[0].set_ylim(-.03, 1.03)
    axes[1].set_ylabel("Peak observed error (mm)")
    axes[2].set_ylabel("Success fraction")
    axes[2].set_ylim(-.03, 1.03)
    axes[0].legend(loc="upper right", fontsize=6.5)
    n_common = len(common_success(data))
    fig.suptitle(f"V1 limited-budget diagnostics; common successful seed/path pairs: {n_common}", fontsize=9)
    save(fig, "performance_v1")


def time_series_figure(rows):
    data = selected(rows, method="decoupled")
    if not data:
        return
    chosen_seed = min(r["seed"] for r in data)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.0), sharex=True, constrained_layout=True)
    family_colors = ["#1f77b4", "#d78332", "#60666c"]
    for family, color in zip(FAMILIES, family_colors):
        record = next(r for r in data if r["seed"] == chosen_seed and r["path_id"] == family+"_test")
        trace = ROOT / Path(record["source"]).parent / f"{record['path_id']}.npz"
        SOURCES[trace.relative_to(ROOT).as_posix()] = hashlib.sha256(trace.read_bytes()).hexdigest()
        z = np.load(trace)
        trace_data = dict(zip(z["columns"].tolist(), z["data"].T))
        for ax, metric in zip(axes, ["lookahead_mm", "v_mm_s", "kcm_intervention"]):
            ax.plot(trace_data["timestamp_s"], trace_data[metric], color=color, label=FAMILY_LABEL[family])
    for i, (ax, label) in enumerate(zip(axes, ["Preview (mm)", "Feed (mm/s)", "KCM intervention"])):
        ax.set_ylabel(label)
        ax.grid(alpha=.2, ls=":")
        ax.text(-.08, 1.03, chr(97+i), transform=ax.transAxes, weight="bold")
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Elapsed simulation time (s)")
    fig.suptitle(f"Decoupled preview, seed {chosen_seed}; all failed traces retained", fontsize=9)
    save(fig, "traces_v1")


def original_ablation_values():
    path = GENERATED / "ablation_table.tex"
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result, current = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "{J-NNC}" in line:
            current = "full_method_snapshot"
        elif "{Fixed Adaptive Preview}" in line:
            current = "abl_fixed_lookahead"
        elif "\\multirow" in line:
            current = None
        if current and "&" in line:
            parts = [s.strip().rstrip("\\") for s in line.split("&")]
            if len(parts) == 7 and parts[1] in {"square","circle","butterfly"}:
                result[(current, parts[1])] = {"time": float(parts[-1]), "error": float(parts[-3])}
    return result


def generate_tex(rows, revision, manifest, diagnosis):
    out = ["% Automatically generated by PPO_project/tools/revision_plot.py. Do not hand-edit numerical results.",
        r"\section{Revision experiments and implementation audit}", r"\label{sec:revision_results}", "",
        "This section reports the first bounded-budget revision (V1) separately from the historical manuscript results. Its initially held-out paths were evaluated only after that configuration was frozen. Because these outcomes subsequently informed continued development, all V1 test paths are now designated development data. A separate V2 exposure ledger and newly generated, unevaluated holdouts preserve the next evaluation boundary. V1 is an intermediate diagnostic study; it does not establish a final preview advantage.", "",
        r"\subsection{Synthetic paths and evaluation protocol}",
        "Three explicit geometric families cover mixed long and short segments, consecutive short-block corners, and straight--curve mixtures. Each has distinct training, validation and initial test geometries; no cross-split example is a uniform scaling or rotation of another. All paths are synthetic, and none is presented as an industrial CAM export. Exact reference coordinates, generating parameters, segment-length distributions, corner angles and SHA-256 identifiers are retained. Cubic Bezier primitives use a separate analytic-to-polyline deviation bound; this discretization error is not added to the planning tolerance.", ""]
    lengths = [r["statistics"]["path_length_mm"] for r in manifest["paths"]]
    bounds = [r["source"]["analytic_to_polyline_deviation_bound_mm"] for r in manifest["paths"]]
    out += [f"The V1 paths span ${min(lengths):.3f}$--${max(lengths):.3f}$ mm, with a maximum curve discretization bound of ${max(bounds):.3f}$ mm.", ""]
    out += figure("paths_v1", "The nine synthetic V1 geometries. Columns show training, validation and the initially held-out set, which is now development data. Filled and open markers indicate start and endpoint. Axes show physical millimetres; curves were not rescaled to a common shape.", "fig:revision_paths")
    training = []
    for path in sorted((revision / "experiments/main").glob("*/training_summary.json")):
        training.append(read_json(path))
    cfg = read_yaml(Path(next(r["run_dir"] for r in rows if r["stage"] == "main")) / "config.yaml")
    env, constraints = cfg["environment"], cfg["kinematic_constraints"]
    budgets = sorted({r["steps"] for r in training})
    seeds = sorted({r["seed"] for r in training})
    out += [f"The primary V1 configurations use {len(seeds)} independent training seeds ({', '.join(map(str,seeds))}) and {', '.join(map(str,budgets))} environment steps per seed. Total primary training comprises {sum(r['steps'] for r in training):,} environment steps. No primary run uses pretraining; convergence was not established at this budget. The interpolation period is ${env['interpolation_period']:g}$ s and the full tolerance-band width is ${env['epsilon']:g}$ mm (one-sided allowance ${env['epsilon']/2:g}$ mm). The scalar feed, acceleration and jerk limits are ${constraints['MAX_VEL']:g}$ mm/s, ${constraints['MAX_ACC']:g}$ mm/s$^2$ and ${constraints['MAX_JERK']:g}$ mm/s$^3$.", "",
        "Coupled preview uses a reward scale that depends on the action. Decoupled preview fixes only that scale. Its observation and guidance distance remain learned. Fixed preview uses the same implementation corrections and fixed reward scale. It is trained independently, with the inactive preview dimension excluded from PPO likelihood and entropy calculations. This control separates the candidate preview contribution from shared fixes.", "",
        "Completion requires ordered reference coverage and endpoint arrival. V1 does not require zero terminal speed; terminal feed is saved for every rollout. It is therefore an arrival task, and its times are not directly comparable with the traditional scheduler's zero-speed termination. All geometric, scalar-feed and position-derived angular diagnostics are retained. Position-derived heading is poorly identified during rest or extremely small displacements; raw angular flags are preserved and require trace-level interpretation, rather than being silently discarded or taken alone as proof of physical angular-limit violations. Cartesian-axis values are diagnostic and no unimplemented axis guarantee is asserted.", "",
        r"\subsection{Mechanism checks and fixed-distance screening}"]
    cost_rows=[]
    for stage in ["screen","main","nnc_auxiliary/runs"]:
        budgets_stage=[read_json(path) for path in sorted((revision/"experiments"/stage).glob("*/training_summary.json"))]
        if budgets_stage:
            cost_rows.append([stage.replace("nnc_auxiliary/runs","NNC auxiliary"),len(budgets_stage),
                len({r["seed"] for r in budgets_stage}),sum(r["steps"] for r in budgets_stage),
                sum(r.get("pretraining_steps",0) for r in budgets_stage),
                f"{sum(r['training_wall_s'] for r in budgets_stage):.1f}"])
    out += table("Recorded V1 training cost, including screening and auxiliary training. Wall time sums recorded training durations and excludes evaluation and plotting; independent seeds are counted within each stage, not interpolation samples.",
        "tab:revision_budget",["Stage","Training runs","Distinct seeds","Environment steps","Pretrain steps","Training wall/s"],cost_rows)
    before, after = diagnosis["kcm"]["before_d8ff2b5"], diagnosis["kcm"]["corrected"]
    out += [f"A deterministic KCM stress test of {before['samples']} samples recorded {before['violation_samples']} violating samples before the correction and {after['violation_samples']} afterward under its stated test limits. This verifies the discrete internal/output consistency correction on that probe. Holding the physical state fixed while changing preview also changed reward weights in the coupled design; a fixed reward reference removed that immediate dependence. These checks establish implementation behavior, not trained-policy efficiency.", ""]
    out += figure("mechanism_v1", "Deterministic implementation probes. (a) Peak output feed and angular jerk normalized by the probe limits before and after KCM correction. (b,c) Tracking weight and immediate reward under identical physical state while preview changes. The fixed reward scale removes the immediate reward-scale dependence; these are mechanism probes, not independent performance repetitions.", "fig:revision_mechanism")
    screen_rows = []
    for method in ["fixed_short","fixed_mid","fixed_long"]:
        subset = selected(rows,"screen","evaluation_validation_nominal",method)
        if not subset:
            continue
        config = read_yaml(Path(subset[0]["run_dir"])/"config.yaml")
        distance = config["reward_weights"]["lookahead"]["distance"]
        screen_rows.append([f"{distance:g}", len({r['seed'] for r in subset}), subset[0]["training_steps"],
                           f"{sum(r['success'] for r in subset)}/{len(subset)}", stat_text([r["ordered_progress"] for r in subset]),
                           stat_text([r["contour_error_max_mm"] for r in subset]), "--" if not any(r["success"] for r in subset) else stat_text([r["elapsed_time_s"] for r in subset if r["success"]])])
    out += table("Independent fixed-distance screening on the three validation paths. Dispersion here is across paths for the single screening seed, not across independent training seeds. All failures remain; no successful time is reported when the successful set is empty.", "tab:revision_fixed_screen", ["Distance/mm","Seeds","Train steps","Success","Progress","Peak error/mm","Success time/s"], screen_rows)
    out += ["The short fixed distance was selected using the prespecified validation ordered-progress fallback because the screening stage produced no successful configuration. This is a failure-diagnostic ranking under limited search, not evidence of adequate hyperparameter optimization. No test-path-specific fixed distance was selected. Directly overwriting the output distance of a trained adaptive policy is stored separately as a diagnostic intervention and is not counted as an independently trained fixed baseline.", "",
            r"\subsection{Five-seed V1 results and retained failures}"]
    replacement_rows=[]
    for evaluation in sorted({r["evaluation"] for r in rows if r["stage"]=="main" and r["evaluation"].endswith("_diagnostic")}):
        subset=selected(rows,"main",evaluation)
        configuration=read_yaml(ROOT/Path(subset[0]["source"]).parent/"effective_config.yaml")
        distance=configuration["reward_weights"]["lookahead"]["distance"]
        replacement_rows.append([LABEL.get(subset[0]["variant"],subset[0]["variant"]),f"{distance:g}",
            ",".join(map(str,sorted({r["seed"] for r in subset}))),len(subset),
            f"{sum(r['success'] for r in subset)}/{len(subset)}",stat_text([r["ordered_progress"] for r in subset])])
    if replacement_rows:
        out += table("Direct preview-output replacement on the validation set using an already trained adaptive policy. These are interventions on one checkpoint, with no independent retraining. Mean and SD summarize paths, not independent model repetitions.",
            "tab:revision_replacement",["Source policy","Forced distance/mm","Seed","Paths","Success","Progress"],replacement_rows)
    main = selected(rows)
    common = common_success(main)
    result_rows = []
    for method in METHODS:
        for family in FAMILIES:
            subset = [r for r in main if r["variant"] == method and r["path_id"] == family+"_test"]
            if not subset:
                continue
            matched = [r["elapsed_time_s"] for r in subset if (r["seed"],r["path_id"]) in common]
            result_rows.append([LABEL[method], FAMILY_LABEL[family], len(subset),
                f"{sum(r['path_completed'] for r in subset)}/{len(subset)}", f"{sum(r['success'] for r in subset)}/{len(subset)}",
                stat_text([r["ordered_progress"] for r in subset]), stat_text([r["contour_error_max_mm"] for r in subset]), stat_text(matched)])
    out += table("Primary V1 results on the initially held-out paths (subsequently development). Means and sample standard deviations are across independent training seeds within each path; all rollouts, including failures, enter progress and observed-error summaries. Time uses only seed/path pairs successful under all three methods.", "tab:revision_primary", ["Method","Path","n","Arrival","Success","Progress","Peak error/mm","Common time/s"], result_rows)
    out += [f"The primary nominal set contains {len(main)} rollouts, with {sum(r['success'] for r in main)} independently accepted successes and {sum(r['path_completed'] for r in main)} ordered endpoint arrivals. There are {len(common)} common successful seed/path pairs. Failed stopping times are not reported as execution-efficiency improvements. Error statistics summarize only the observed part of each terminated trajectory and must be interpreted together with progress and failure rate.", ""]
    out += figure("performance_v1", "V1 progress, peak observed contour error and success fraction on each initial holdout family. Points and error bars denote the mean and sample standard deviation over independent training seeds; success is a fraction. All failures remain. Truncated low-progress traces can have small observed error without constituting accurate task completion.", "fig:revision_performance")
    failure_rows = []
    for method in METHODS:
        subset = [r for r in main if r["variant"] == method]
        if not subset:
            continue
        count = lambda predicate: sum(predicate(r) for r in subset)
        failure_rows.append([LABEL[method],len(subset),count(lambda r:not r["success"]),count(lambda r:not r["path_completed"]),
            count(lambda r:not r["geometry_feasible"]),count(lambda r:r["termination_reason"]=="stall"),
            count(lambda r:any(c["satisfied"] is False for k,c in r["constraints"].items() if k.startswith("feed"))),
            count(lambda r:any(c["satisfied"] is False for k,c in r["constraints"].items() if k.startswith("angular"))),
            stat_text([r.get("inference_median_us") for r in subset],1)])
    out += table("Failure audit for the primary nominal set. Flags overlap and must not be summed as disjoint categories. Inference values are the mean and sample SD of per-rollout median inference latency on the local CPU, and are not controller-cycle guarantees. Angular columns retain all raw position-derived flags, including low-speed observability issues.","tab:revision_failures",["Method","n","Failed","Incomplete","Geometry","Stall","Feed flags","Angular flags","Inference/$\\mu$s"],failure_rows)
    out += figure("traces_v1", "Preview distance, scalar feed and KCM intervention along the complete saved failed traces of the lowest-numbered decoupled-preview seed. Time axes end at each failure. Preview variability illustrates mechanism only and does not establish performance.", "fig:revision_traces")
    aux_rows = []
    for evaluation in ["evaluation_regression_nominal","evaluation_test_tight","evaluation_test_low_jerk"]:
        for method in METHODS:
            subset = selected(rows,"main",evaluation,method)
            if subset:
                aux_rows.append([evaluation.removeprefix("evaluation_").replace("_", " "), LABEL[method],len(subset),
                    len({r["seed"] for r in subset}),f"{sum(r['path_completed'] for r in subset)}/{len(subset)}",f"{sum(r['success'] for r in subset)}/{len(subset)}"])
    nnc = selected(rows,"nnc_auxiliary/runs")
    if nnc:
        aux_rows.append(["NNC auxiliary nominal","No KCM",len(nnc),len({r["seed"] for r in nnc}),f"{sum(r['path_completed'] for r in nnc)}/{len(nnc)}",f"{sum(r['success'] for r in nnc)}/{len(nnc)}"])
    out += table("Additional V1 regression, sensitivity and auxiliary NNC outcomes. Sparse sensitivity seeds are diagnostic; they do not substitute for the primary five-seed comparison.","tab:revision_additional",["Evaluation","Method","Rollouts","Seeds","Arrival","Success"],aux_rows)
    for evaluation in ["evaluation_test_tight","evaluation_test_low_jerk"]:
        found=selected(rows,"main",evaluation)
        if found:
            sensitivity_cfg=read_yaml(ROOT/Path(found[0]["source"]).parent/"effective_config.yaml")
            out += [f"The {escaped(evaluation.removeprefix('evaluation_test_'))} diagnostic uses full band width ${sensitivity_cfg['environment']['epsilon']:g}$ mm and scalar jerk limit ${sensitivity_cfg['kinematic_constraints']['MAX_JERK']:g}$ mm/s$^3$."]
    out += ["",r"\subsection{Historical reproduction boundary}"]
    old = original_ablation_values()
    history_rows=[]
    for name in ["full_method_snapshot","abl_fixed_lookahead","baseline_policy"]:
        for path in sorted((revision/"historical"/name).glob("rerun_*/result.json")):
            item=read_json(path)
            original=old.get((name,item["path_id"]))
            history_rows.append([{"full_method_snapshot":"Historical full","abl_fixed_lookahead":"Historical fixed","baseline_policy":"Historical NNC"}[name],item["path_id"],
                "--" if original is None else f"{original['time']:.3f}",f"{item['execution_time_s']:.3f}",
                "--" if original is None else f"{original['error']:.3f}",f"{item['max_contour_error_mm']:.3f}",f"{item['half_epsilon']:g}",escaped(item["done_reason"])])
    out += table("Historical model reruns with the archived main implementation, separate from V1 learning repetitions. Original values are read directly from the existing ablation table. The final column is the archived environment termination label, not the new independent success criterion.","tab:revision_historical",["Archive","Path","Original time/s","Rerun time/s","Original error/mm","Rerun error/mm","Half-band/mm","Status"],history_rows)
    out += ["The historical fixed model has its own recovered checkpoint and training configuration. Its reruns are therefore distinguished from output-replacement diagnostics. The recovered historical NNC configuration uses a different tolerance, shown explicitly in the table, so its rerun is not a fair same-tolerance baseline for V1. Discrepancies from original tables are retained; no statistical definition or recorded value was altered to match the manuscript. The broader serial-baseline audit also found velocity/endpoint clipping differences between internal jerk and output-position derivatives; the corrected geometric compression check does not resolve that scheduler's remaining dynamic violations.","",
        "The V1 budget establishes an executable repeated-experiment workflow and identifies failure mechanisms. It does not yet establish an efficiency or stability advantage for adaptive preview. Further development uses these now-exposed V1 paths, while the separate V2 holdout remains reserved for a later frozen configuration. Simulation diagnostics do not replace real machine tracking or cutting measurements.",""]
    return "\n".join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-dir",type=Path,default=DEFAULT_REVISION)
    args=parser.parse_args()
    style()
    manifest=read_json(ROOT/"PPO_project/data/revision_20260908/manifest.json")
    rows=discover_runs(args.revision_dir)
    path_figure(manifest)
    diagnosis=diagnostic_figure(args.revision_dir)
    performance_figure(rows)
    time_series_figure(rows)
    text=generate_tex(rows,args.revision_dir,manifest,diagnosis)
    GENERATED.mkdir(parents=True,exist_ok=True)
    (GENERATED/"revision_20260908_results.tex").write_text(text,encoding="utf-8")
    fields=["stage","evaluation","variant","seed","path_id","success","path_completed","ordered_progress","elapsed_time_s","contour_error_max_mm","contour_error_rmse_mm","inference_median_us","source"]
    with (FIGURES/"revision_source_data_v1.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    main_rows=selected(rows)
    summary={"row_count":len(rows),"main_nominal_rows":len(main_rows),"main_nominal_successes":sum(r["success"] for r in main_rows),
        "main_nominal_arrivals":sum(r["path_completed"] for r in main_rows),"common_successful_pairs":sorted(common_success(main_rows)),
        "source_hashes":SOURCES,"generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "v1_exposure":"Initially held-out paths were exposed and are now development only; V2 ledger preserves the new boundary.",
        "statistical_contract":"Per-path means and sample SD across independent training seeds. All failures retained. Successful times require common seed/path pairs. No inferential test.",
        "figure_contract":{"paths":"Spatial quantitative 3x3 grid: verify distinct geometry and real units.","mechanism":"Constraint ratios and paired state-controlled reward diagnostics, no efficiency inference.","performance":"Parallel seed-level mean/SD diagnostics plus complete success fractions.","traces":"Synchronized descriptive traces of every family for the preselected lowest seed, no successful-case selection."},
        "visual_precedent":"Existing paper style retained per task. No external figure or ranking verification claimed; offline bundled observations inform shared axes and labels only.",
        "validation":"Vector PDF/SVG plus Python PNG previews; source hashes included. Raw flags unchanged."}
    (FIGURES/"revision_summary_v1.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k in ["row_count","main_nominal_rows","main_nominal_successes","main_nominal_arrivals","common_successful_pairs"]}))
    print(GENERATED/"revision_20260908_results.tex")


if __name__=="__main__":
    main()
