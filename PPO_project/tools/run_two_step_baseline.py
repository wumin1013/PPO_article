from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PPO_ROOT.parent
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.algorithms.two_step_baseline import (  # noqa: E402
    SchedulerParams,
    SmoothingParams,
    TwoStepConstraints,
    run_two_step_baseline,
)
from src.utils.comparison_metrics import as_point_array  # noqa: E402
from src.utils.path_generator import get_path_by_name  # noqa: E402
from sync_two_step_to_paper import (  # noqa: E402
    PATHS,
    load_jnnc_square_trace_rows,
    save_two_step_comparison_figure,
    write_paper_tables_and_summary,
)


RESULTS_DIR = PPO_ROOT / "results" / "two_step_baseline"
TRACES_DIR = RESULTS_DIR / "traces"
DEFAULT_CONFIG_CANDIDATES = [
    REPO_ROOT / "论文项目" / "generated" / "configs" / "current_best.yaml",
    PPO_ROOT / "configs" / "default.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the traditional two-step CNC baseline and sync paper outputs.")
    parser.add_argument("--config", type=str, default="", help="Config YAML. Defaults to paper current_best, then PPO default.")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR), help="Output directory for two-step results.")
    parser.add_argument("--sync-paper", action="store_true", help="Generate paper tables, figure, summary, and update main.tex.")
    parser.add_argument("--paths", nargs="*", default=list(PATHS), choices=list(PATHS), help="Path names to evaluate.")
    parser.add_argument("--max-steps", type=int, default=0, help="Override maximum scheduler steps per path.")
    return parser.parse_args()


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, PPO_ROOT / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def resolve_config(path_arg: str) -> tuple[Path, dict[str, Any]]:
    if path_arg:
        path = resolve_path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        return path, load_yaml(path)
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate, load_yaml(candidate)
    raise FileNotFoundError("No default config found.")


def nested_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def path_config_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section_name in ("path_curriculum", "training"):
        section = nested_dict(config.get(section_name))
        if section_name == "training":
            section = nested_dict(section.get("path_curriculum"))
        paths = section.get("paths", [])
        if isinstance(paths, list):
            for item in paths:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("type") or "").strip()
                if name in PATHS and name not in out:
                    out[name] = dict(item)

    direct = nested_dict(config.get("path"))
    direct_name = str(direct.get("name") or direct.get("type") or "").strip()
    if direct_name in PATHS and direct_name not in out:
        out[direct_name] = dict(direct)

    defaults = {
        "square": {
            "name": "square",
            "type": "square",
            "scale": 100.0,
            "num_points": 1200,
            "closed": True,
            "square": {"start_offset_ratio": 0.2},
        },
        "circle": {
            "name": "circle",
            "type": "circle",
            "scale": 100.0,
            "num_points": 960,
            "closed": True,
            "circle": {"closed": True},
        },
        "butterfly": {
            "name": "butterfly",
            "type": "butterfly",
            "scale": 28.0,
            "num_points": 720,
            "closed": True,
            "butterfly": {"closed": True, "cross_ratio": 0.1, "long_ratio": 1.45, "style": "academic", "wing_ratio": 1.0},
        },
    }
    for name, item in defaults.items():
        out.setdefault(name, item)
    return out


def build_path_points(path_cfg: dict[str, Any]) -> np.ndarray:
    path_type = str(path_cfg.get("type") or path_cfg.get("name")).strip()
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    kwargs = nested_dict(path_cfg.get(path_type))
    if "closed" in path_cfg:
        kwargs.setdefault("closed", bool(path_cfg.get("closed")))
    return as_point_array(get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs))


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "time",
        "x",
        "y",
        "v",
        "a",
        "j",
        "contour_error",
        "progress",
        "v_over_vmax",
        "abs_j_over_jmax",
        "active_accdec_window",
        "s_smoothed",
        "v_cap",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize(row.get(key, "")) for key in fieldnames})


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    config_path, config = resolve_config(args.config)
    env_cfg = nested_dict(config.get("environment"))
    kin_cfg = nested_dict(config.get("kinematic_constraints"))
    dt = float(env_cfg.get("interpolation_period", 0.001))
    epsilon = float(env_cfg.get("epsilon", 2.5))
    half_epsilon = epsilon / 2.0
    max_steps_default = int(env_cfg.get("max_steps", 60000) or 60000)
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else max_steps_default
    constraints = TwoStepConstraints(
        max_vel=float(kin_cfg.get("MAX_VEL", 100.0)),
        max_acc=float(kin_cfg.get("MAX_ACC", 2000.0)),
        max_jerk=float(kin_cfg.get("MAX_JERK", 62500.0)),
        max_ang_vel=float(kin_cfg.get("MAX_ANG_VEL", 0.0)),
        max_ang_acc=float(kin_cfg.get("MAX_ANG_ACC", 0.0)),
        max_ang_jerk=float(kin_cfg.get("MAX_ANG_JERK", 0.0)),
    )
    path_map = path_config_map(config)
    results_dir = resolve_path(args.results_dir)
    traces_dir = results_dir / "traces"
    warnings: list[str] = []
    two_step_metrics: dict[str, dict[str, Any]] = {}
    two_step_trace_paths: dict[str, str] = {}
    two_step_rows: list[dict[str, Any]] = []
    path_points: dict[str, np.ndarray] = {}

    for path_name in args.paths:
        path_cfg = path_map[path_name]
        ref = build_path_points(path_cfg)
        path_points[path_name] = ref
        closed = bool(path_cfg.get(path_name, {}).get("closed", path_cfg.get("closed", True)))
        result = run_two_step_baseline(
            path_name=path_name,
            reference_path=ref,
            constraints=constraints,
            dt=dt,
            half_epsilon=half_epsilon,
            closed=closed,
            max_steps=max_steps,
            smoothing_params=SmoothingParams(),
            scheduler_params=SchedulerParams(),
        )
        trace_path = traces_dir / f"{path_name}_trajectory.csv"
        write_trace_csv(trace_path, result.trace_rows)
        two_step_trace_paths[path_name] = repo_relative(trace_path)
        metrics = dict(result.metrics)
        metrics.update(
            {
                "path": path_name,
                "method": "Traditional two-step",
                "config_path": repo_relative(config_path),
                "path_config": path_cfg,
                "epsilon": epsilon,
                "half_epsilon": half_epsilon,
                "interpolation_period": dt,
                "MAX_VEL": constraints.max_vel,
                "MAX_ACC": constraints.max_acc,
                "MAX_JERK": constraints.max_jerk,
                "MAX_ANG_VEL": constraints.max_ang_vel,
                "MAX_ANG_ACC": constraints.max_ang_acc,
                "MAX_ANG_JERK": constraints.max_ang_jerk,
                "smoothing": {
                    "transition_ratio": result.smoothed.transition_ratio,
                    "safety_factor": result.smoothed.safety_factor,
                    "retry_count": result.smoothed.retry_count,
                    "corner_count": len(result.smoothed.corners),
                    "corners": result.smoothed.corners,
                    "fixed_smoothed_path": True,
                },
            }
        )
        if result.smoothed.boundary_violation_flag:
            warnings.append(f"{path_name}: smoothed path exceeds half_epsilon.")
        two_step_metrics[path_name] = metrics
        two_step_rows.append(metrics)

    two_step_csv = results_dir / "two_step_results.csv"
    two_step_json = results_dir / "two_step_results.json"
    comparison_csv = results_dir / "comparison_metrics.csv"
    comparison_json = results_dir / "comparison_metrics.json"
    write_results_csv(two_step_csv, two_step_rows)
    write_json(
        two_step_json,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": repo_relative(config_path),
            "paths": two_step_metrics,
            "trace_paths": two_step_trace_paths,
            "warnings": warnings,
        },
    )
    write_results_csv(comparison_csv, two_step_rows)
    write_json(
        comparison_json,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "paper_sync_enabled": bool(args.sync_paper),
            "note": "Without --sync-paper, this file contains only the Traditional two-step metrics and does not modify paper tables.",
            "rows": two_step_rows,
        },
    )

    generated_files = {
        "two_step_results_csv": repo_relative(two_step_csv),
        "two_step_results_json": repo_relative(two_step_json),
        "comparison_metrics_csv": repo_relative(comparison_csv),
        "comparison_metrics_json": repo_relative(comparison_json),
    }
    if args.sync_paper and "square" in path_points and "square" in args.paths:
        figure_files = save_two_step_comparison_figure(
            square_reference=path_points["square"],
            square_trace_rows=[
                row for row in csv.DictReader((traces_dir / "square_trajectory.csv").open("r", encoding="utf-8"))
            ],
            jnnc_trace_rows=load_jnnc_square_trace_rows(),
            half_epsilon=half_epsilon,
            constraints={
                "MAX_VEL": constraints.max_vel,
                "MAX_ACC": constraints.max_acc,
                "MAX_JERK": constraints.max_jerk,
                "DT": dt,
            },
        )
        generated_files.update(figure_files)

    paper_payload = {}
    if args.sync_paper:
        paper_payload = write_paper_tables_and_summary(
            results_dir=results_dir,
            two_step_metrics=two_step_metrics,
            two_step_trace_paths=two_step_trace_paths,
            generated_files=generated_files,
            warnings=warnings,
        )

    return {
        "config_path": config_path,
        "results_dir": results_dir,
        "two_step_metrics": two_step_metrics,
        "trace_paths": two_step_trace_paths,
        "generated_files": generated_files,
        "paper_payload": paper_payload,
        "warnings": warnings,
    }


def print_summary(payload: dict[str, Any]) -> None:
    print("\nTraditional two-step baseline summary")
    for path_name in PATHS:
        metrics = payload["two_step_metrics"].get(path_name)
        if not metrics:
            continue
        print(
            "  {path}: status={status}, progress={progress:.3f}, max_error={max_err:.3f}, "
            "mean_error={mean_err:.3f}, jerk_exceed={jerk_exceed:.3f}, time={time_s:.3f}, "
            "mean_feed={mean_feed:.3f}, active_jerk_engagement={jerk_engagement}".format(
                path=path_name,
                status=metrics.get("termination_status", "unknown"),
                progress=float(metrics.get("final_progress") or 0.0),
                max_err=float(metrics.get("max_contour_error_mm") or 0.0),
                mean_err=float(metrics.get("mean_contour_error_mm") or 0.0),
                jerk_exceed=float(metrics.get("max_relative_linear_jerk_exceedance") or 0.0),
                time_s=float(metrics.get("termination_time_s") or 0.0),
                mean_feed=float(metrics.get("mean_feedrate_utilization") or 0.0),
                jerk_engagement=(
                    "N/A"
                    if metrics.get("jerk_reach_rate_80_active") is None
                    or not math.isfinite(float(metrics.get("jerk_reach_rate_80_active")))
                    else f"{float(metrics.get('jerk_reach_rate_80_active')):.3f}"
                ),
            )
        )
    print("\nGenerated files")
    for key, path in sorted(payload["generated_files"].items()):
        print(f"  {key}: {path}")
    for path in payload.get("trace_paths", {}).values():
        print(f"  trace: {path}")
    paper_files = payload.get("paper_payload", {}).get("files", {}) if payload.get("paper_payload") else {}
    for key, path in sorted(paper_files.items()):
        print(f"  {key}: {path}")
    if payload.get("warnings"):
        print("\nWarnings")
        for item in payload["warnings"]:
            print(f"  {item}")


def main() -> None:
    args = parse_args()
    payload = run_all(args)
    print_summary(payload)


if __name__ == "__main__":
    main()
