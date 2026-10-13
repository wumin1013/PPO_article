"""Read-only statistics and optional reference figures from saved event logs.

No environment, policy, torch, runner or performance-report module is imported.
Formal mechanism examples are fixed in advance to seed 11 and every registered
path. Pilot summaries include every registered seed-101 configuration/path.
Figure generation requires --figures; default exports data only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "论文项目"
STUDY = PAPER / "revision_20260909_event_preview/study"
DATA = ROOT / "PPO_project/data/preview_event_v1"
PRESELECTED_FORMAL_SEED = 11
BLUE, LIGHT_BLUE, ORANGE, GRAY = "#2467a4", "#628cac", "#d58a43", "#818181"
LABELS = {"legacy_direct": "Original direct", "state_direct": "Corrected instant",
          "selected_fixed": "Trained fixed", "selected_hold": "Selected held preview"}
FIELDS = {
    "time": "timestamp_s", "distance": "lookahead_mm", "speed": "v_mm_s",
    "q": "control_status__direct_observation_q",
    "turn_range": "control_status__direct_observation_range_over_pi",
    "cornerness": "reward_components__cornerness", "kcm": "kcm_intervention",
    "progress": "reward_components__ordered_progress", "event": "preview_event",
    "coverage": "reward_components__ordered_coverage_complete",
}


class Inputs:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.records = {}
        self.log_snapshots = {}

    def name(self, path):
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def bytes(self, path):
        path = Path(path)
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        self.records[self.name(path)] = digest
        if path.suffix == ".jsonl":
            self.log_snapshots[digest] = value
        return value

    def sha(self, path):
        return hashlib.sha256(self.bytes(path)).hexdigest()

    def json(self, path):
        return json.loads(self.bytes(path).decode("utf-8"))

    def lines(self, path):
        raw = self.bytes(path)
        lines = raw.splitlines(keepends=True)
        # A writer may be in the middle of its last append. Preserve these
        # exact bytes, expose the omitted tail, and never invent a full row.
        partial = bool(lines and not lines[-1].endswith((b"\n", b"\r")))
        if partial:
            lines = lines[:-1]
        return [json.loads(line) for line in lines if line.strip()], partial

    def trace(self, path):
        with np.load(io.BytesIO(self.bytes(path)), allow_pickle=False) as value:
            columns, data = value["columns"].tolist(), value["data"]
        if data.ndim != 2 or data.shape[1] != len(columns) or len(set(columns)) != len(columns):
            raise ValueError(f"Invalid trace layout: {path}")
        return {str(column): data[:, index] for index, column in enumerate(columns)}


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def finite_stats(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=None, std=None, min=None, max=None)
    return dict(mean=float(values.mean()), std=float(values.std()), min=float(values.min()), max=float(values.max()))


def trajectory_statistics(trace, result, cfg):
    missing = sorted(set(FIELDS.values()) - set(trace))
    if missing:
        raise ValueError("Required trace fields missing: " + ", ".join(missing))
    time_values = np.asarray(trace[FIELDS["time"]], float)
    if len(time_values) < 2 or not np.isfinite(time_values).all() or time_values[0] != 0 or np.any(np.diff(time_values) <= 0):
        raise ValueError("Expected initial row at t=0 followed by increasing physical timestamps")
    post = {key: np.asarray(value)[1:] for key, value in trace.items()}
    distance = post[FIELDS["distance"]]
    event = post[FIELDS["event"]]
    if not np.isfinite(distance).all() or not np.isin(event, (0., 1.)).all():
        raise ValueError("Distance/event trace contains invalid values")
    ordered = post[FIELDS["progress"]]
    if not np.isfinite(ordered).all():
        raise ValueError("Ordered progress after the initial row must be finite")
    if np.any(np.diff(ordered) < -1e-10):
        raise ValueError("Ordered progress must be monotone; do not replace it with nearest distance")
    crossing = np.flatnonzero(ordered >= .99)
    entry_time = float(time_values[1:][crossing[0]]) if len(crossing) else None
    observed_tail = float(time_values[-1] - entry_time) if entry_time is not None else None
    complete = bool(result.get("path_completed", False)) and post[FIELDS["coverage"]][-1] == 1.
    revision = cfg["revision"]
    period = int(revision["preview_period_steps"])
    expected_events = (len(distance) + period - 1) // period if revision["adaptive_preview"] else 0
    event_count = int(event.sum())
    if event_count != expected_events:
        raise ValueError(f"Recorded events {event_count} disagree with scheduled count {expected_events}")
    row = dict(variant=result["variant"], seed=result["seed"], path_id=result["path_id"],
               split=result["split"], success=bool(result["success"]), path_completed=bool(result["path_completed"]),
               failure_reasons=result.get("failure_reasons", []), total_time_s=float(time_values[-1]),
               post_initial_samples=len(distance), distance_range_mm=float(np.ptp(distance)),
               distance_std_mm=float(distance.std()), distance_min_mm=float(distance.min()),
               distance_max_mm=float(distance.max()), distance_mean_mm=float(distance.mean()),
               preview_event_count=event_count, preview_period_steps=period,
               ordered_progress_final=float(ordered[-1]), first_sample_at_99pct_s=entry_time,
               observed_time_after_99pct_s=observed_tail,
               last_1pct_completion_time_s=observed_tail if complete else None,
               last_1pct_status="completed" if complete else ("threshold_not_reached" if entry_time is None else "terminated_without_completion"),
               kcm_score_mean=float(np.mean(post[FIELDS["kcm"]])),
               kcm_score_max=float(np.max(post[FIELDS["kcm"]])),
               kcm_nonzero_fraction=float(np.mean(post[FIELDS["kcm"]] > 1e-9)),
               initial_row_excluded=True)
    for key in ("speed", "q", "turn_range", "cornerness"):
        row.update({key + "_" + suffix: value for suffix, value in finite_stats(post[FIELDS[key]]).items()})
    return row


def export_trace(path, trace):
    # Export every recorded reward component and the mechanism inputs. Preserve
    # NaNs in the initial row; they are missing measurements, not zero rewards.
    columns = list(dict.fromkeys([*FIELDS.values(), "preview_held_u", "preview_phase", "reward",
                                 "x_mm", "y_mm", "omega_rad_s", "a_mm_s2", "j_mm_s3",
                                 *(key for key in trace if key.startswith("reward_components__"))]))
    columns = [key for key in columns if key in trace]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["is_initial_row", *columns])
        for index in range(len(trace[FIELDS["time"]])):
            writer.writerow([int(index == 0), *(trace[key][index] for key in columns)])


def verify_source_archive(run, provenance, inputs):
    declared = hashlib.sha256(json.dumps(provenance["source_files"], sort_keys=True).encode()).hexdigest()
    if declared != provenance["source_sha256"]:
        raise ValueError("Source member manifest does not match its aggregate hash")
    with zipfile.ZipFile(io.BytesIO(inputs.bytes(run / "source.zip"))) as archive:
        if set(archive.namelist()) != set(provenance["source_files"]):
            raise ValueError("Archived source member list differs")
        for name, expected in provenance["source_files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Archived source member differs: " + name)


def check_test_freeze(study, protocol, inputs):
    frozen_path = study / "freeze.json"
    if not frozen_path.exists():
        raise ValueError("Test mechanisms require the completed formal freeze; no test artifacts were read")
    frozen = inputs.json(frozen_path)
    for key, expected in (("source_sha256", protocol["source_sha256"]),
                          ("protocol_sha256", inputs.sha(study / "protocol.json")),
                          ("selection_sha256", inputs.sha(study / "selection.json")),
                          ("manifest_sha256", protocol["manifest_sha256"])):
        if frozen.get(key) != expected:
            raise ValueError("Test freeze binding differs: " + key)
    expected_keys = {f"formal/{method}_seed{seed}" for method in protocol["formal_variants"] for seed in protocol["formal_seeds"]}
    if set(frozen.get("models", {})) != expected_keys or set(frozen.get("configs", {})) != expected_keys:
        raise ValueError("Freeze must bind every registered formal model and seed")
    # Validate all training completions and immutable bindings before opening
    # even a seed-11 test result. A partial freeze is not sufficient.
    for key in sorted(expected_keys):
        run = study / key
        summary = inputs.json(run / "training_summary.json")
        provenance = inputs.json(run / "provenance.json")
        model, config = inputs.sha(run / "checkpoint.pt"), inputs.sha(run / "config.yaml")
        if summary["steps"] != protocol["formal_steps"] or summary["checkpoint_sha256"] != model:
            raise ValueError("Frozen formal training is incomplete: " + key)
        if frozen["models"][key] != model or frozen["configs"][key] != config:
            raise ValueError("Frozen model/config changed: " + key)
        if provenance["source_sha256"] != protocol["source_sha256"] or provenance["config_sha256"] != config or provenance["manifest_sha256"] != protocol["manifest_sha256"]:
            raise ValueError("Frozen provenance changed: " + key)
        verify_source_archive(run, provenance, inputs)
    return frozen


def learning_statistics(method, seed, updates, cfg, summary, episodes):
    epochs = [epoch for update in updates for epoch in update["epochs"]]
    steps = [update["step"] for update in updates]
    if len(set(steps)) != len(steps) or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Duplicate or non-increasing update steps")
    transitions = sum(update["transition_count"] for update in updates)
    if steps and transitions != steps[-1]:
        raise ValueError("Update transition totals do not match logged steps")
    row = dict(variant=method, seed=seed, status="complete" if summary else "training_in_progress",
               training_steps=steps[-1] if steps else 0, interaction_s=transitions * cfg["environment"]["interpolation_period"],
               updates=len(updates), optimizer_epochs=len(epochs),
               preview_decisions=sum(update["preview_decision_count"] for update in updates),
               preview_decision_fraction=sum(update["preview_decision_count"] for update in updates) / max(transitions, 1),
               successful_episodes=sum(ep.get("done_reason") == "success" and ep.get("done", False) for ep in episodes),
               failed_episodes=sum(ep.get("done", False) and ep.get("done_reason") != "success" for ep in episodes),
               budget_partials=sum(ep.get("budget_truncation", False) for ep in episodes),
               std_mode=cfg["revision"]["std_mode"], preview_period_steps=cfg["revision"]["preview_period_steps"],
               training_wall_s=summary.get("training_wall_s") if summary else None)
    if epochs:
        last = epochs[-1]
        exposure_weights = np.array([update["transition_count"] for update in updates for _ in update["epochs"]], float)
        weighted = lambda key: float(np.average([epoch[key] for epoch in epochs], weights=exposure_weights))
        row.update(last_angular_std=last["std_by_action_mean"][0], last_feed_std=last["std_by_action_mean"][1],
                   last_preview_std_u=last["preview_std_mean"], last_preview_std_mm=3.8 * last["preview_std_mean"],
                   preview_gradient_ratio_median=float(np.median([e["preview_mu_head_gradient_norm"] / max(1e-30, np.hypot(e["angular_mu_head_gradient_norm"], e["feed_mu_head_gradient_norm"])) for e in epochs])),
                   global_clip_scale_median=float(np.median([e["actor_gradient_clip_scale"] for e in epochs])),
                   surrogate_clipped_fraction_mean=weighted("surrogate_clipped_fraction"),
                   ratio_outside_clip_fraction_mean=weighted("ratio_outside_clip_fraction"),
                   old_new_kl_masked_mean=weighted("old_new_kl_masked_mean"),
                   last_preview_mu_min_mm=.8 + 3.8 * last["preview_mu_min"], last_preview_mu_max_mm=.8 + 3.8 * last["preview_mu_max"])
    return row


def collect(stage, split, study=STUDY, data=DATA, root=ROOT, allow_partial=False):
    study, data = Path(study), Path(data)
    inputs = Inputs(root)
    protocol, manifest = inputs.json(study / "protocol.json"), inputs.json(data / "manifest.json")
    if inputs.sha(data / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Registered path manifest changed")
    if split == "test":
        if stage != "formal":
            raise ValueError("Pilot test figures are forbidden")
        check_test_freeze(study, protocol, inputs)
    methods = protocol[stage + "_variants"]
    seeds = [protocol["pilot_seed"]] if stage == "pilot" else protocol["formal_seeds"]
    if stage == "formal" and PRESELECTED_FORMAL_SEED not in seeds:
        raise ValueError("The preselected mechanism seed 11 is absent from the protocol")
    paths = [entry for entry in manifest["paths"] if entry["split"] == split]
    if not paths:
        raise ValueError("No registered paths for split")
    learning, histories, epoch_rows, statistics, bundles, pending = [], [], [], [], [], []
    budget = protocol[stage + "_steps"]
    for method in methods:
        for seed in seeds:
            run = study / stage / f"{method}_seed{seed}"
            if not (run / "config.yaml").exists():
                pending.append(dict(variant=method, seed=seed, reason="training_not_started")); continue
            cfg = yaml.safe_load(inputs.bytes(run / "config.yaml").decode("utf-8"))
            provenance = inputs.json(run / "provenance.json")
            if provenance["source_sha256"] != protocol["source_sha256"] or provenance["config_sha256"] != inputs.sha(run / "config.yaml") or provenance["manifest_sha256"] != protocol["manifest_sha256"]:
                raise ValueError("Run provenance/config mismatch: " + run.name)
            verify_source_archive(run, provenance, inputs)
            updates, partial_log = inputs.lines(run / "updates.jsonl") if (run / "updates.jsonl").exists() else ([], False)
            episodes, partial_episode = inputs.lines(run / "episodes.jsonl") if (run / "episodes.jsonl").exists() else ([], False)
            summary = inputs.json(run / "training_summary.json") if (run / "training_summary.json").exists() else None
            if summary:
                model_hash = inputs.sha(run / "checkpoint.pt")
                if summary["steps"] != budget or summary["checkpoint_sha256"] != model_hash or not updates or updates[-1]["step"] != budget or partial_log or partial_episode:
                    raise ValueError("Completed training record is inconsistent: " + run.name)
            else:
                model_hash = None
                pending.append(dict(variant=method, seed=seed, reason="training_in_progress", partial_jsonl_tail=partial_log or partial_episode))
            learning.append(learning_statistics(method, seed, updates, cfg, summary, episodes))
            for update in updates:
                item = {key: value for key, value in update.items() if key != "epochs"}
                histories.append(dict(variant=method, seed=seed, **item))
                epoch_rows.extend(dict(variant=method, seed=seed, step=update["step"], **epoch) for epoch in update["epochs"])
            if not summary or (stage == "formal" and seed != PRESELECTED_FORMAL_SEED):
                continue
            method_paths = []
            for entry in paths:
                path_id = entry["id"]
                result_path = run / f"evaluation_{split}_nominal" / (path_id + ".json")
                trace_path = result_path.with_suffix(".npz")
                if not result_path.exists() or not trace_path.exists():
                    pending.append(dict(variant=method, seed=seed, path_id=path_id, reason="evaluation_pending")); continue
                result = inputs.json(result_path)
                expected = dict(variant=method, seed=seed, path_id=path_id, split=split, regime="nominal",
                                training_steps=budget, checkpoint_sha256=model_hash,
                                config_sha256=provenance["config_sha256"], source_sha256=protocol["source_sha256"],
                                manifest_sha256=protocol["manifest_sha256"])
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError("Saved evaluation binding differs: " + str(result_path))
                trace = inputs.trace(trace_path)
                stat = trajectory_statistics(trace, result, cfg)
                stat.update(result_file=inputs.name(result_path), trace_file=inputs.name(trace_path), checkpoint_sha256=model_hash)
                statistics.append(stat)
                method_paths.append(dict(path_id=path_id, trace=trace, result=result, statistics=stat))
            bundles.append(dict(variant=method, seed=seed, config=cfg, paths=method_paths))
    if pending and not allow_partial:
        raise ValueError(f"{len(pending)} registered records pending; use --allow-partial only for development progress")
    return dict(stage=stage, split=split, methods=methods, paths=[entry["id"] for entry in paths],
                learning=learning, histories=histories, epochs=epoch_rows, statistics=statistics,
                bundles=bundles, pending=pending, inputs=inputs, protocol=protocol)


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7., "axes.titlesize": 7.,
                        "axes.labelsize": 6.8, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
                        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
                        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def export_figure(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("pdf", "svg", "png"):
        path = stem.with_suffix("." + extension)
        fig.savefig(path, dpi=240, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def family_color(method):
    if method.startswith("hold") or method == "selected_hold":
        return ORANGE
    return GRAY if "fixed" in method else BLUE


def learning_figure(collected, figure_dir):
    rows = collected["learning"]
    if not rows:
        return []
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(183/25.4, max(110, 6*len(rows)+24)/25.4))
    fields = [("last_angular_std", r"Last logged $\sigma_\theta$"), ("last_preview_std_mm", r"Last logged $\sigma_D$ (mm)"),
              ("preview_gradient_ratio_median", "Preview / physical\nmean-head gradient"),
              ("surrogate_clipped_fraction_mean", "Surrogate clipped\nsample fraction")]
    for index, row in enumerate(rows):
        for ax, (field, _) in zip(axes, fields):
            value = row.get(field)
            if value is not None:
                ax.scatter(value, index, s=16, facecolors=family_color(row["variant"]) if row["status"] == "complete" else "none",
                           edgecolors=family_color(row["variant"]), linewidths=.8)
    for index, (ax, (_, title)) in enumerate(zip(axes, fields)):
        ax.set_title(chr(97+index) + "  " + title, loc="left")
        ax.grid(axis="x", alpha=.18, linestyle=":")
        ax.set_ylim(len(rows)-.5, -.5)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), [row["variant"] + f" / {row['seed']}" for row in rows])
    fig.suptitle(f"{collected['stage'].capitalize()} PPO diagnostics: all registered runs", fontsize=8.5)
    fig.text(.02, .012, "One row = one training seed; values describe optimizer logs, not independent trajectory-point replicates.\nOpen marker = unfinished budget. Gray: fixed; blue: instant; orange: held preview.", fontsize=6)
    fig.tight_layout(rect=[0, .052, 1, .965], w_pad=1.)
    return export_figure(fig, figure_dir / f"revision_event_{collected['stage']}_learning_diagnostics")


def pilot_summary_figure(collected, figure_dir):
    if collected["stage"] != "pilot" or not collected["statistics"]:
        return []
    methods, paths = collected["methods"], collected["paths"]
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(183/25.4, 151/25.4))
    fields = [("distance_range_mm", "Post-initial D range (mm)"),
              ("observed_time_after_99pct_s", "Time after ordered 99% (s)"),
              ("kcm_score_mean", "Mean KCM intervention score")]
    markers = ("o", "s", "^")
    for row in collected["statistics"]:
        y = methods.index(row["variant"]) + (paths.index(row["path_id"])-1)*.18
        for ax, (field, _) in zip(axes, fields):
            value = row[field]
            if value is None:
                continue
            ax.scatter(value, y, s=15, marker=markers[paths.index(row["path_id"]) % len(markers)],
                       facecolors=family_color(row["variant"]) if row["success"] else "none",
                       edgecolors=family_color(row["variant"]), linewidths=.7)
    for index, (ax, (_, title)) in enumerate(zip(axes, fields)):
        ax.set_title(chr(97+index) + "  " + title, loc="left")
        ax.set_ylim(len(methods)-.6, -.6); ax.grid(axis="x", alpha=.18, linestyle=":")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(methods)), methods)
    for index, path in enumerate(paths):
        axes[-1].scatter([], [], marker=markers[index % 3], s=15, color=GRAY,
                         label=path.replace("_validation", "").replace("_", " "))
    axes[-1].legend(loc="upper center", bbox_to_anchor=(.5, -.045), fontsize=6, ncol=1)
    fig.suptitle(f"Pilot validation mechanisms: every configuration, seed {collected['protocol']['pilot_seed']}", fontsize=8.5)
    fig.text(.02, .012, "Three paths per configuration; descriptive points only, no confidence intervals. Open = strict failure.\nMissing 99% entry stays blank in the figure and is identified in trajectory_statistics.csv. No test results used.", fontsize=6)
    fig.tight_layout(rect=[0, .085, 1, .965], w_pad=1.2)
    return export_figure(fig, figure_dir / "revision_event_pilot_validation_mechanism_summary")


def mechanism_figure(bundle, split, figure_dir):
    if bundle["seed"] != PRESELECTED_FORMAL_SEED:
        raise ValueError("Formal mechanism figures must use preselected seed 11")
    paths = bundle["paths"]
    if not paths:
        return []
    fig, axes = plt.subplots(6, len(paths), sharex="col", squeeze=False,
                             figsize=(183/25.4, 223/25.4),
                             gridspec_kw={"height_ratios": [1.15, .85, .85, .85, 1.05, .6]})
    labels = ["D (mm)", "Feed (mm/s)", "Geometry (a.u.)", "KCM score", "Reward / tick", "Terminal reward"]
    for column, item in enumerate(paths):
        trace, stat = item["trace"], item["statistics"]
        t = trace[FIELDS["time"]][1:]
        get = lambda key: trace[key][1:]
        axes[0, column].step(t, get(FIELDS["distance"]), where="post", color=BLUE, linewidth=.8)
        bounds = bundle["config"]["revision"]["preview_bounds_mm"]
        axes[0, column].set_ylim(min(bounds[0], stat["distance_min_mm"]), max(bounds[1], stat["distance_max_mm"]))
        axes[0, column].text(.97, .95, f"range {stat['distance_range_mm']:.4g} mm", ha="right", va="top", transform=axes[0, column].transAxes, fontsize=5.8)
        axes[1, column].plot(t, get(FIELDS["speed"]), color=BLUE, linewidth=.65)
        for name, label, color, line in (("q", "q", BLUE, "-"), ("turn_range", "turn range / pi", ORANGE, "--"), ("cornerness", "corner intensity", GRAY, ":")):
            axes[2, column].plot(t, get(FIELDS[name]), color=color, linestyle=line, linewidth=.65, label=label)
        axes[3, column].plot(t, get(FIELDS["kcm"]), color=ORANGE, linewidth=.65)
        for key, label, color, line in (("r_progress", "progress", BLUE, "-"), ("r_track", "tracking", ORANGE, "-"),
                                        ("r_smooth", "smoothness", GRAY, "--"), ("r_time", "time", LIGHT_BLUE, ":")):
            axes[4, column].plot(t, get("reward_components__"+key), color=color, linestyle=line, linewidth=.55, label=label)
        terminal = get("reward_components__r_terminal")
        axes[4, column].plot(t, get("reward")-terminal, color="#333333", linestyle="-.", linewidth=.55, label="total minus terminal")
        axes[5, column].plot(t, terminal, color=GRAY, linewidth=.65)
        if stat["first_sample_at_99pct_s"] is not None:
            for ax in axes[:, column]:
                ax.axvspan(stat["first_sample_at_99pct_s"], t[-1], color="#dddddd", alpha=.45, zorder=-5)
        title = item["path_id"].replace("_validation", "").replace("_", " ")
        axes[0, column].set_title(title + ("\ncomplete" if stat["success"] else "\nstrict failure"), fontsize=6.7)
        for row, ax in enumerate(axes[:, column]):
            ax.text(-.13, 1.015, chr(97 + row*len(paths)+column), transform=ax.transAxes, fontsize=7, fontweight="bold")
            ax.grid(alpha=.13, linestyle=":"); ax.set_xlim(0, t[-1])
            if column == 0:
                ax.set_ylabel(labels[row])
        axes[-1, column].set_xlabel("Time (s)")
    axes[2, 0].legend(fontsize=5.2, loc="upper left")
    axes[4, 0].legend(fontsize=5.1, loc="upper left", ncol=1)
    method = LABELS.get(bundle["variant"], bundle["variant"])
    period = bundle["config"]["revision"]["preview_period_steps"]
    fig.suptitle(f"{method}; seed 11; {split}; preview period {period} ms", fontsize=8.2)
    fig.text(.02, .01, "Seed 11 and all registered paths were specified in advance. Initial row excluded; raw series are not smoothed.\nGray band: after first sampled ordered progress >=99%; mechanism traces alone do not establish a performance gain.\nAll recorded reward components, including additional terms, are retained in the companion per-trajectory CSV.", fontsize=5.8)
    fig.tight_layout(rect=[0, .054, 1, .972], h_pad=.75, w_pad=.7)
    return export_figure(fig, figure_dir / f"revision_event_mechanism_{split}_{bundle['variant']}_seed11")


def build(collected, output, figure_dir, generate_figures=False):
    output, figure_dir = Path(output), Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = collected["inputs"]
    previous_manifest = output / "mechanism_manifest.json"
    previous_figure_record = None
    if previous_manifest.exists():
        original = previous_manifest.read_bytes()
        original_record = json.loads(original)
        previous_figure_record = original_record.get("previous_reference_figure_manifest")
        if original_record.get("figures"):
            archive = output / "reference_drafts" / (hashlib.sha256(original).hexdigest() + "_mechanism_manifest.json")
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                archive.write_bytes(original)
            previous_figure_record = str(archive.resolve())
    write_csv(output / "learning_summary.csv", collected["learning"])
    write_csv(output / "learning_updates.csv", collected["histories"])
    write_csv(output / "learning_epochs.csv", collected["epochs"])
    write_csv(output / "trajectory_statistics.csv", collected["statistics"])
    for bundle in collected["bundles"]:
        for item in bundle["paths"]:
            export_trace(output / "timeseries" / f"{bundle['variant']}_seed{bundle['seed']}_{item['path_id']}.csv", item["trace"])
    figures = []
    if generate_figures:
        style()
        figures = learning_figure(collected, figure_dir) + pilot_summary_figure(collected, figure_dir)
        if collected["stage"] == "formal":
            for bundle in collected["bundles"]:
                if len(bundle["paths"]) == len(collected["paths"]):
                    figures.extend(mechanism_figure(bundle, collected["split"], figure_dir))
    snapshot_dir = output / "input_log_snapshots"; snapshot_dir.mkdir(exist_ok=True)
    for digest, payload in inputs.log_snapshots.items():
        target = snapshot_dir / (digest + ".jsonl")
        if not target.exists():
            target.write_bytes(payload)
    contract = PAPER / "revision_20260909_event_preview/figure_contract.md"
    if contract.exists():
        inputs.sha(contract)
    write_json(output / "figure_notes.json", dict(
        figure_generation_requested=generate_figures,
        figure_status="Reference drafts for user drawing only; not approved manuscript artwork" if generate_figures else "Data only; existing reference drafts are unchanged",
        claim="Show observed distance decisions, motion, geometry and reward timing; do not infer superiority from distance variation",
        archetype="Synchronized quantitative time-series grid; pilot all-configuration descriptive point summaries",
        design="Existing manuscript blue/orange/gray palette and typography; user requested continuity, no external visual precedent claimed",
        dimensions="Mechanism: 183 x 223 mm; editable SVG/PDF text; PNG at 240 dpi for visual QA",
        backend="Python matplotlib only", initial_row="excluded from D statistics, event counts and plotted traces; preserved and marked in CSV",
        last_one_percent="First sampled reward_components__ordered_progress >=0.99 to recorded termination; resolution is one interpolation step; completion-time field is null for incomplete trajectories",
        kcm_definition="abs(feed_output-feed_intent)/MAX_VEL + abs(omega_output-omega_intent)/MAX_ANG_VEL; may exceed one; not a percentage",
        reward_display="Four core per-tick terms and total minus terminal; terminal separately; every raw reward component exported without replacing missing values",
        statistics="Training seed and registered path are the units; one predetermined seed in each mechanism example; time points are not independent replicates",
        learning_summary="Last logged std is measured before the final optimizer step; clipping/KL fractions are weighted by transition count per epoch, gradient ratios use the median over epochs",
        image_integrity="No smoothing, resampling, peak clipping or selective failure removal; fixed allowed D bounds shown and expanded if any observed value exceeds them",
        pending=collected["pending"]))
    write_json(output / "mechanism_manifest.json", dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        command="python " + " ".join(sys.argv), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage=collected["stage"], split=collected["split"], formal_example_seed=PRESELECTED_FORMAL_SEED,
        registered_methods=collected["methods"], registered_paths=collected["paths"],
        learning_run_count=len(collected["learning"]), trajectory_count=len(collected["statistics"]),
        pending=collected["pending"], input_sha256=inputs.records,
        figure_generation_requested=generate_figures,
        figure_status="reference_drafts_only" if generate_figures else "data_only",
        previous_reference_figure_manifest=previous_figure_record,
        figures=[dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in figures],
        no_policy_or_environment_execution=True))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Opt in to reference draft figures; default exports statistics and CSV only")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=PAPER / "figures/generated")
    args = parser.parse_args()
    collected = collect(args.stage, args.split, allow_partial=args.allow_partial)
    output = args.output or PAPER / "revision_20260909_event_preview/mechanism" / args.stage / args.split
    figures = build(collected, output, args.figure_dir, generate_figures=args.figures)
    print(json.dumps(dict(learning_runs=len(collected["learning"]), trajectories=len(collected["statistics"]),
                         pending=len(collected["pending"]), exported_figures=len(figures), output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
