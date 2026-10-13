"""Analyse saved formal mechanism records; data only, no policy execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "论文项目/revision_20260909_event_preview/mechanism/formal"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)


def stats(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError("Post-initial statistics require finite recorded values")
    return dict(mean=float(values.mean()), std=float(values.std()), minimum=float(values.min()),
                maximum=float(values.max()), sum=float(values.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(); directory = BASE / args.split
    manifest_path = directory / "mechanism_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["stage"] != "formal" or manifest["split"] != args.split or manifest["pending"]:
        raise ValueError("Complete formal mechanism data required")
    inputs = {manifest_path.relative_to(ROOT).as_posix(): sha(manifest_path)}

    def checked(path, raw=False):
        path = Path(path); name = path.relative_to(ROOT).as_posix(); digest = sha(path)
        if raw and manifest["input_sha256"].get(name) != digest:
            raise ValueError("Raw input differs from verified mechanism manifest: " + name)
        inputs[name] = digest
        return path

    # Test inputs have already passed the mechanism exporter's all-model gate.
    # Recheck its saved freeze binding before reading any test trajectory.
    if args.split == "test":
        freeze = ROOT / "论文项目/revision_20260909_event_preview/study/freeze.json"
        checked(freeze, raw=True)
    with checked(directory / "trajectory_statistics.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != 12 or any(int(row["seed"]) != 11 for row in summary):
        raise ValueError("Expected all four methods and three paths at preselected seed 11")
    trajectory_rows, reward_rows = [], []
    for row in summary:
        with np.load(checked(ROOT / row["trace_file"], raw=True), allow_pickle=False) as archive:
            columns, data = archive["columns"].tolist(), archive["data"]
            trace = {name: data[1:, i] for i, name in enumerate(columns)}
        result = json.loads(checked(ROOT / row["result_file"], raw=True).read_text(encoding="utf-8"))
        if result["seed"] != 11 or result["split"] != args.split:
            raise ValueError("Wrong trajectory binding")
        time_values = trace["timestamp_s"]; distance = trace["lookahead_mm"]
        ordered = trace["reward_components__ordered_progress"]
        events = trace["preview_event"].astype(bool)
        after_first = np.r_[False, np.ones(len(events)-1, bool)]
        changes = np.r_[0., np.diff(distance)]
        threshold = np.flatnonzero(ordered >= .99)
        last_time = float(time_values[-1] - time_values[threshold[0]]) if len(threshold) else None
        if not np.isclose(float(row["distance_range_mm"]), np.ptp(distance), atol=1e-12):
            raise ValueError("Exported D statistics differ")
        if last_time is not None and not np.isclose(float(row["observed_time_after_99pct_s"]), last_time, atol=1e-12):
            raise ValueError("Exported tail statistics differ")
        details = dict(variant=row["variant"], seed=11, path_id=row["path_id"], success=result["success"],
                       total_time_s=float(time_values[-1]), last_1pct_observed_time_s=last_time,
                       last_1pct_time_fraction=last_time / time_values[-1] if last_time is not None else None,
                       preview_event_count=int(events.sum()), post_initial_steps=len(distance),
                       distance_range_mm=float(np.ptp(distance)), distance_mean_mm=float(distance.mean()),
                       distance_std_mm=float(distance.std()), max_event_distance_jump_mm=float(np.abs(changes[events]).max()) if events.any() else 0.,
                       non_event_distance_change_max_mm=float(np.abs(changes[~events & after_first]).max()) if (~events & after_first).any() else None,
                       speed_mean_mm_s=float(trace["v_mm_s"].mean()), speed_std_mm_s=float(trace["v_mm_s"].std()),
                       speed_below_1_mm_s_fraction=float(np.mean(trace["v_mm_s"] < 1.)),
                       kcm_score_mean=float(trace["kcm_intervention"].mean()),
                       kcm_score_max=float(trace["kcm_intervention"].max()),
                       kcm_score_nonzero_fraction=float(np.mean(trace["kcm_intervention"] > 1e-9)))
        reward_fields = [name for name in trace if name.startswith("reward_components__r_")]
        reconstructed = sum(trace[name] for name in reward_fields)
        details["reward_component_sum_max_abs_error"] = float(np.max(np.abs(reconstructed - trace["reward"])))
        # Sum only actual reward components, never diagnostics such as weights,
        # corner intensity or the original unscaled per-tick total.
        for field in reward_fields:
            reward_rows.append(dict(variant=row["variant"], seed=11, path_id=row["path_id"], component=field,
                                    **stats(trace[field])))
            details[field.removeprefix("reward_components__") + "_sum"] = float(trace[field].sum())
        for field in ("reward", "reward_components__w_track_dyn", "reward_components__w_smooth_dyn",
                      "reward_components__cornerness", "reward_components__running_cost_rate"):
            details[field.removeprefix("reward_components__") + "_mean"] = float(trace[field].mean())
        trajectory_rows.append(details)

    learning_rows = []
    with checked(directory / "learning_summary.csv").open(encoding="utf-8", newline="") as stream:
        learning = list(csv.DictReader(stream))
    for row in learning:
        run = ROOT / f"论文项目/revision_20260909_event_preview/study/formal/{row['variant']}_seed{row['seed']}"
        updates = [json.loads(line) for line in checked(run / "updates.jsonl", raw=True).read_text(encoding="utf-8").splitlines()]
        n = sum(update["transition_count"] for update in updates)
        event_n = sum(update["preview_decision_count"] for update in updates)
        if n != 262144 or n != sum(update["physical_decision_count"] for update in updates):
            raise ValueError("Physical decision count differs from registered budget")
        fixed = row["variant"] == "selected_fixed"
        expected_dims = [0, 1] if fixed else [0, 1, 2]
        if any(update["active_action_dims"] != expected_dims for update in updates):
            raise ValueError("Active dimensions differ")
        if fixed and event_n != 0:
            raise ValueError("Fixed preview scored an event")
        if row["variant"] in ("legacy_direct", "state_direct") and event_n != n:
            raise ValueError("Instant preview omitted decisions")
        epochs = [(epoch, update["transition_count"], update["preview_decision_count"])
                  for update in updates for epoch in update["epochs"]]
        weighted = lambda field: float(np.average([ep[field] for ep, _, _ in epochs], weights=[count for _, count, _ in epochs]))
        event_weighted = lambda field: float(np.average([ep[field] for ep, _, _ in epochs], weights=[events for _, _, events in epochs])) if event_n else None
        audit = dict(variant=row["variant"], seed=int(row["seed"]), physical_steps=n, preview_decisions=event_n,
                     event_fraction=event_n/n, latent_preview_execution_clip_fraction=sum(u["preview_latent_clipped_fraction"]*u["preview_decision_count"] for u in updates)/event_n if event_n else None,
                     rollout_update_log_prob_max_error=max(u["old_log_probability_max_absolute_error"] for u in updates),
                     masked_kl_identity_max_error=max(abs(ep["old_new_kl_masked_mean"]-ep["old_new_kl_physics_mean"]-events/count*ep["old_new_kl_preview_event_mean"]) for ep, count, events in epochs),
                     surrogate_clipped_fraction=weighted("surrogate_clipped_fraction"),
                     event_surrogate_clipped_fraction=event_weighted("event_surrogate_clipped_fraction"),
                     numerical_log_ratio_guard_fraction=weighted("log_ratio_guard_fraction"),
                     physics_kl_mean=weighted("old_new_kl_physics_mean"),
                     preview_event_kl_mean=event_weighted("old_new_kl_preview_event_mean"),
                     angular_std_gradient_median=float(np.median([ep["angular_std_head_gradient_norm"] for ep, _, _ in epochs])),
                     feed_std_gradient_median=float(np.median([ep["feed_std_head_gradient_norm"] for ep, _, _ in epochs])))
        for field in ("last_angular_std", "last_feed_std", "last_preview_std_mm", "preview_gradient_ratio_median", "global_clip_scale_median"):
            audit[field] = float(row[field])
        learning_rows.append(audit)

    pair_rows = []
    for path_id in manifest["registered_paths"]:
        selected = {row["variant"]: row for row in trajectory_rows if row["path_id"] == path_id}
        hold, fixed = selected["selected_hold"], selected["selected_fixed"]
        paired_success = hold["success"] and fixed["success"]
        dt = hold["total_time_s"] - fixed["total_time_s"] if paired_success else None
        d_tail = hold["last_1pct_observed_time_s"] - fixed["last_1pct_observed_time_s"] if paired_success and hold["last_1pct_observed_time_s"] is not None and fixed["last_1pct_observed_time_s"] is not None else None
        pair_rows.append(dict(path_id=path_id, seed=11, both_strict_success=paired_success,
                              held_minus_fixed_total_s=dt, held_minus_fixed_last1pct_s=d_tail,
                              held_minus_fixed_before99pct_s=dt-d_tail if dt is not None and d_tail is not None else None,
                              held_minus_fixed_feed_mean_mm_s=hold["speed_mean_mm_s"]-fixed["speed_mean_mm_s"],
                              held_minus_fixed_kcm_mean=hold["kcm_score_mean"]-fixed["kcm_score_mean"]))
    # The full mechanism examples remain seed 11. Separately audit distance
    # variation across every seed, as requested, without selecting examples.
    protocol_path = ROOT / "论文项目/revision_20260909_event_preview/study/protocol.json"
    protocol = json.loads(checked(protocol_path, raw=True).read_text(encoding="utf-8"))
    preview_rows = []
    for method in protocol["formal_variants"]:
        for seed in protocol["formal_seeds"]:
            run = ROOT / f"论文项目/revision_20260909_event_preview/study/formal/{method}_seed{seed}"
            model_key = (run / "checkpoint.pt").relative_to(ROOT).as_posix()
            config_key = (run / "config.yaml").relative_to(ROOT).as_posix()
            for path_id in manifest["registered_paths"]:
                result_path = run / f"evaluation_{args.split}_nominal/{path_id}.json"
                result = json.loads(checked(result_path).read_text(encoding="utf-8"))
                expected = dict(variant=method, seed=seed, path_id=path_id, split=args.split, regime="nominal",
                                checkpoint_sha256=manifest["input_sha256"][model_key],
                                config_sha256=manifest["input_sha256"][config_key],
                                source_sha256=protocol["source_sha256"], manifest_sha256=protocol["manifest_sha256"],
                                training_steps=protocol["formal_steps"])
                if any(result.get(key) != value for key, value in expected.items()):
                    raise ValueError("All-seed evaluation binding differs: " + str(result_path))
                with np.load(checked(result_path.with_suffix(".npz")), allow_pickle=False) as archive:
                    columns, values = archive["columns"].tolist(), archive["data"][1:]
                    distance = values[:, columns.index("lookahead_mm")]
                if not np.isfinite(distance).all():
                    raise ValueError("Non-finite post-initial distance")
                preview_rows.append(dict(variant=method, seed=seed, path_id=path_id, split=args.split,
                    success=result["success"], initial_row_excluded=True,
                    distance_min_mm=float(distance.min()), distance_max_mm=float(distance.max()),
                    distance_range_mm=float(np.ptp(distance)), distance_std_mm=float(distance.std()),
                    distance_mean_mm=float(distance.mean()), contour_error_max_mm=result["contour_error_max_mm"],
                    result_file=result_path.relative_to(ROOT).as_posix()))
    output = directory / "analysis"; output.mkdir(exist_ok=True)
    products = []
    for name, rows in (("trajectory_diagnostics.csv", trajectory_rows), ("reward_components.csv", reward_rows),
                       ("learning_mask_audit.csv", learning_rows), ("seed11_held_fixed_pairs.csv", pair_rows),
                       ("all_seed_preview_statistics.csv", preview_rows)):
        path = output / name; write_csv(path, rows); products.append(path)
    record = dict(command="python " + " ".join(sys.argv), script_sha256=sha(__file__),
                  input_sha256=inputs, output_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in products},
                  training_runs=len(learning_rows), mechanism_trajectories=len(trajectory_rows), split=args.split,
                  all_seed_preview_trajectories=len(preview_rows),
                  mechanism_seed=11, initial_rows_excluded=True, figures_generated=False,
                  scope="Descriptive validation or frozen-test mechanism records, not a causal attribution or independent performance test")
    (output / "analysis_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(training_runs=len(learning_rows), trajectories=len(trajectory_rows), figures=0, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
