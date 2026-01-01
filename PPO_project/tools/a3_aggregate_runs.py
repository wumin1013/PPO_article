from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd = Path.cwd()
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return (cwd / path).resolve()
    cwd_candidate = (cwd / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _find_run_bundles(roots: Sequence[Path]) -> List[Path]:
    bundles: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for manifest_path in root.rglob("manifest.json"):
            bundle_dir = manifest_path.parent.resolve()
            if bundle_dir in seen:
                continue
            run1 = bundle_dir / "eval" / "run1" / "summary.json"
            run2 = bundle_dir / "eval" / "run2" / "summary.json"
            if run1.exists() or run2.exists():
                bundles.append(bundle_dir)
                seen.add(bundle_dir)
    return sorted(bundles)


def _relpath(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path.resolve())


def _pick_summary(bundle_dir: Path) -> Tuple[dict, Optional[Path]]:
    candidates = [
        bundle_dir / "eval" / "run2" / "summary.json",
        bundle_dir / "eval" / "run1" / "summary.json",
        bundle_dir / "eval" / "run2" / "summary_raw.json",
        bundle_dir / "eval" / "run1" / "summary_raw.json",
    ]
    for path in candidates:
        if path.exists():
            data = _json_load(path)
            if isinstance(data, dict):
                if "global" in data and isinstance(data["global"], dict):
                    return data["global"], path
                if "summary" in data and isinstance(data["summary"], dict):
                    return data["summary"], path
            return data if isinstance(data, dict) else {}, path
    return {}, None


def _pick_raw_episodes(bundle_dir: Path) -> Tuple[List[dict], Optional[Path]]:
    candidates = [
        bundle_dir / "eval" / "run2" / "summary_raw.json",
        bundle_dir / "eval" / "run1" / "summary_raw.json",
    ]
    for path in candidates:
        if path.exists():
            data = _json_load(path)
            episodes = data.get("episodes") if isinstance(data, dict) else None
            if isinstance(episodes, list):
                return [ep for ep in episodes if isinstance(ep, dict)], path
    return [], None


def _pick_smoke_summary(bundle_dir: Path) -> Tuple[dict, Optional[Path]]:
    for path in [bundle_dir / "smoke" / "summary.json", bundle_dir / "smoke" / "summary_raw.json"]:
        if path.exists():
            data = _json_load(path)
            if isinstance(data, dict):
                if "global" in data and isinstance(data["global"], dict):
                    return data["global"], path
                if "summary" in data and isinstance(data["summary"], dict):
                    return data["summary"], path
            return data if isinstance(data, dict) else {}, path
    return {}, None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compute_step_stats(episodes: List[dict]) -> Tuple[Optional[float], Optional[int]]:
    steps: List[int] = []
    for ep in episodes:
        step = ep.get("steps")
        if isinstance(step, (int, float)) and math.isfinite(float(step)):
            steps.append(int(step))
    if not steps:
        return None, None
    mean_steps = float(sum(steps)) / float(len(steps))
    return mean_steps, max(steps)


def _compute_trace_stats(trace_path: Path) -> Dict[str, Optional[float]]:
    if not trace_path.exists():
        return {}
    sums = {
        "err_sq": 0.0,
        "abs_err": 0.0,
        "vel": 0.0,
        "jerk": 0.0,
        "kcm": 0.0,
    }
    count = 0
    max_abs_err = None
    with trace_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            count += 1
            err = _safe_float(row.get("contour_error"))
            vel = _safe_float(row.get("velocity"))
            jerk = _safe_float(row.get("jerk"))
            kcm = _safe_float(row.get("kcm_intervention"))
            if err is not None and math.isfinite(err):
                sums["err_sq"] += err * err
                abs_err = abs(err)
                sums["abs_err"] += abs_err
                if max_abs_err is None or abs_err > max_abs_err:
                    max_abs_err = abs_err
            if vel is not None and math.isfinite(vel):
                sums["vel"] += vel
            if jerk is not None and math.isfinite(jerk):
                sums["jerk"] += abs(jerk)
            if kcm is not None and math.isfinite(kcm):
                sums["kcm"] += kcm
    if count == 0:
        return {}
    rmse_err = math.sqrt(sums["err_sq"] / float(count)) if count else None
    return {
        "trace_steps": float(count),
        "trace_rmse_error": rmse_err,
        "trace_max_abs_error": max_abs_err,
        "trace_mean_velocity": sums["vel"] / float(count),
        "trace_mean_jerk": sums["jerk"] / float(count),
        "trace_mean_kcm_intervention": sums["kcm"] / float(count),
    }


def _extract_config_fields(config: dict) -> Dict[str, Any]:
    exp_cfg = config.get("experiment", {}) if isinstance(config.get("experiment"), dict) else {}
    reward_cfg = config.get("reward_weights", {}) if isinstance(config.get("reward_weights"), dict) else {}
    smooth_weight = None
    for key in ("w_action_smooth", "w_smooth"):
        if key in reward_cfg:
            smooth_weight = _safe_float(reward_cfg.get(key))
            break
    return {
        "mode": exp_cfg.get("mode"),
        "enable_kcm": exp_cfg.get("enable_kcm"),
        "smooth_weight": smooth_weight,
    }


def _build_ablation_key(row: dict) -> str:
    mode = row.get("mode") or "unknown"
    enable_kcm = row.get("enable_kcm")
    smooth_weight = row.get("smooth_weight")
    smooth_enabled = None
    if smooth_weight is not None:
        smooth_enabled = smooth_weight > 0.0
    parts = [str(mode)]
    if enable_kcm is not None:
        parts.append(f"kcm={bool(enable_kcm)}")
    if smooth_enabled is not None:
        parts.append(f"smooth={bool(smooth_enabled)}")
    return "|".join(parts)


def _write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate(roots: Sequence[str], out_dir: str) -> dict:
    root_paths = [_resolve_root(r) for r in roots]
    bundles = _find_run_bundles(root_paths)
    rows: List[dict] = []
    trace_rows: List[dict] = []
    for bundle_dir in bundles:
        manifest_path = bundle_dir / "manifest.json"
        manifest = _json_load(manifest_path) if manifest_path.exists() else {}
        summary, summary_path = _pick_summary(bundle_dir)
        episodes, raw_path = _pick_raw_episodes(bundle_dir)
        smoke, smoke_path = _pick_smoke_summary(bundle_dir)
        verdict_path = bundle_dir / "a1_verdict.json"
        verdict = _json_load(verdict_path) if verdict_path.exists() else {}
        config_path = bundle_dir / "config.yaml"
        config = _load_yaml(config_path) if config_path.exists() else {}
        cfg_fields = _extract_config_fields(config)
        mean_steps, max_steps = _compute_step_stats(episodes)

        trace_path = bundle_dir / "rollout_det" / "trace.csv"
        trace_stats = _compute_trace_stats(trace_path) if trace_path.exists() else {}
        if trace_path.exists():
            trace_rows.append(
                {
                    "run_id": manifest.get("run_id") or bundle_dir.name,
                    "bundle_path": _relpath(bundle_dir),
                    "trace_path": _relpath(trace_path),
                    **trace_stats,
                }
            )

        row = {
            "run_id": manifest.get("run_id") or bundle_dir.name,
            "bundle_path": _relpath(bundle_dir),
            "mode": cfg_fields.get("mode"),
            "enable_kcm": cfg_fields.get("enable_kcm"),
            "smooth_weight": cfg_fields.get("smooth_weight"),
            "baseline_ref": manifest.get("baseline_ref"),
            "git_hash": manifest.get("git_hash"),
            "config_hash": manifest.get("config_hash"),
            "checkpoint_hash": manifest.get("checkpoint_hash"),
            "eval_pass": summary.get("passed"),
            "smoke_pass": smoke.get("passed") if smoke else None,
            "verdict_pass": verdict.get("pass"),
            "episodes": summary.get("episodes"),
            "seed_eval": summary.get("seed_eval"),
            "episode_set": summary.get("episode_set"),
            "success_rate": summary.get("success_rate"),
            "stall_rate": summary.get("stall_rate"),
            "mean_progress_final": summary.get("mean_progress_final"),
            "max_abs_contour_error": summary.get("max_abs_contour_error"),
            "has_non_finite": summary.get("has_non_finite"),
            "half_epsilon": summary.get("half_epsilon"),
            "mean_steps": mean_steps,
            "max_steps": max_steps,
            "summary_path": _relpath(summary_path) if summary_path else None,
            "summary_raw_path": _relpath(raw_path) if raw_path else None,
            "smoke_path": _relpath(smoke_path) if smoke_path else None,
            "trace_path": _relpath(trace_path) if trace_path.exists() else None,
        }
        row.update(trace_stats)
        row["ablation_key"] = _build_ablation_key(row)
        rows.append(row)

    rows = sorted(rows, key=lambda r: str(r.get("run_id") or ""))
    out_path = _resolve_root(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    main_fields = [
        "run_id",
        "bundle_path",
        "mode",
        "ablation_key",
        "baseline_ref",
        "eval_pass",
        "smoke_pass",
        "verdict_pass",
        "episodes",
        "seed_eval",
        "episode_set",
        "success_rate",
        "stall_rate",
        "mean_progress_final",
        "max_abs_contour_error",
        "half_epsilon",
        "mean_steps",
        "max_steps",
        "trace_rmse_error",
        "trace_max_abs_error",
        "trace_mean_velocity",
        "trace_mean_jerk",
        "trace_mean_kcm_intervention",
        "has_non_finite",
        "git_hash",
        "config_hash",
        "checkpoint_hash",
        "summary_path",
        "summary_raw_path",
        "smoke_path",
        "trace_path",
    ]
    _write_csv(out_path / "main_table.csv", rows, main_fields)
    _json_dump({"rows": rows, "count": len(rows)}, out_path / "main_table.json")

    trace_fields = [
        "run_id",
        "bundle_path",
        "trace_path",
        "trace_steps",
        "trace_rmse_error",
        "trace_max_abs_error",
        "trace_mean_velocity",
        "trace_mean_jerk",
        "trace_mean_kcm_intervention",
    ]
    _write_csv(out_path / "trace_index.csv", trace_rows, trace_fields)

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("ablation_key"))].append(row)

    ablation_rows: List[dict] = []
    metric_fields = [
        "success_rate",
        "stall_rate",
        "mean_progress_final",
        "max_abs_contour_error",
        "mean_steps",
        "trace_rmse_error",
        "trace_mean_velocity",
        "trace_mean_jerk",
        "trace_mean_kcm_intervention",
    ]
    for key, group in grouped.items():
        summary_row = {"ablation_key": key, "count": len(group)}
        for metric in metric_fields:
            values = [
                float(v)
                for v in (row.get(metric) for row in group)
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            ]
            if values:
                mean_val = sum(values) / float(len(values))
                variance = sum((v - mean_val) ** 2 for v in values) / float(len(values))
                summary_row[f"{metric}_mean"] = mean_val
                summary_row[f"{metric}_std"] = math.sqrt(variance)
            else:
                summary_row[f"{metric}_mean"] = None
                summary_row[f"{metric}_std"] = None
        ablation_rows.append(summary_row)

    ablation_fields = ["ablation_key", "count"]
    for metric in metric_fields:
        ablation_fields.append(f"{metric}_mean")
        ablation_fields.append(f"{metric}_std")
    _write_csv(out_path / "ablation_table.csv", ablation_rows, ablation_fields)
    _json_dump({"rows": ablation_rows, "count": len(ablation_rows)}, out_path / "ablation_table.json")

    manifest = {
        "roots": [str(p) for p in root_paths],
        "bundle_count": len(rows),
        "bundles": [r.get("bundle_path") for r in rows],
    }
    _json_dump(manifest, out_path / "aggregation_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Run Bundles into paper-ready tables.")
    parser.add_argument(
        "--root",
        action="append",
        default=["artifacts"],
        help="Run Bundle root (repeatable)",
    )
    parser.add_argument("--out", default="artifacts/aggregation", help="Output directory")
    args = parser.parse_args()
    manifest = aggregate(args.root, args.out)
    print(f"[A3] bundles={manifest['bundle_count']} out={_resolve_root(args.out)}")


if __name__ == "__main__":
    main()
