from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
ACCEPTANCE_SCRIPT = PROJECT_ROOT / "tools" / "acceptance_suite.py"
EXPORT_SCRIPT = PROJECT_ROOT / "tools" / "phase32_export_best_trajectories.py"
SWEEP_ROOT = PROJECT_ROOT / "artifacts" / "night_sweeps"


PATH_LIBRARY: Dict[str, Dict[str, Any]] = {
    "square": {
        "name": "square",
        "type": "square",
        "closed": True,
        "scale": 22.0,
        "num_points": 260,
        "square": {"start_offset_ratio": 0.20},
    },
    "s_shape": {
        "name": "s_shape",
        "type": "s_shape",
        "scale": 26.0,
        "num_points": 280,
        "s_shape": {"amplitude": 6.0, "periods": 1.5},
    },
    "butterfly": {
        "name": "butterfly",
        "type": "butterfly",
        "scale": 40.0,
        "num_points": 420,
        "butterfly": {
            "style": "academic",
            "wing_ratio": 1.00,
            "long_ratio": 1.45,
            "cross_ratio": 0.10,
            "closed": True,
        },
    },
    "trapezoid": {
        "name": "trapezoid",
        "type": "trapezoid",
        "scale": 24.0,
        "num_points": 260,
        "trapezoid": {
            "top_ratio": 0.45,
            "height_ratio": 0.80,
            "start_offset_ratio": 0.50,
            "closed": True,
        },
    },
    "circle": {
        "name": "circle",
        "type": "circle",
        "scale": 24.0,
        "num_points": 280,
        "circle": {"closed": True},
    },
}


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid yaml config: {path}")
    return data


def _load_config_from_checkpoint(checkpoint_path: Path) -> dict:
    with checkpoint_path.open("rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint does not contain config: {checkpoint_path}")
    return copy.deepcopy(config)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _run_command(cmd: Sequence[str], *, cwd: Path) -> None:
    print(f"[RUN] {_format_cmd(cmd)}")
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = cfg
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _get_nested(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _parse_csv_names(raw: str) -> List[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("paths list is empty")
    return names


def _build_selected_paths(path_names: Sequence[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for name in path_names:
        spec = PATH_LIBRARY.get(str(name))
        if spec is None:
            available = ", ".join(sorted(PATH_LIBRARY))
            raise ValueError(f"unknown path '{name}', available: {available}")
        selected.append(copy.deepcopy(spec))
    return selected


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, float(value))))


def _mul_nested(cfg: dict, dotted_key: str, factor: float, *, low: float | None = None, high: float | None = None) -> None:
    value = _get_nested(cfg, dotted_key)
    if value is None:
        return
    new_value = float(value) * float(factor)
    if low is not None:
        new_value = max(float(low), new_value)
    if high is not None:
        new_value = min(float(high), new_value)
    _set_nested(cfg, dotted_key, new_value)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    apply: Callable[[dict], None]


def _candidate_specs() -> List[CandidateSpec]:
    def baseline(cfg: dict) -> None:
        return

    def lookahead_aggressive(cfg: dict) -> None:
        _mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", 1.12, low=0.35, high=0.90)
        _mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", 1.12, low=1.4, high=4.5)
        _mul_nested(cfg, "reward_weights.lookahead_reward.corner_target", 1.08, low=0.45, high=0.95)

    def lookahead_conservative(cfg: dict) -> None:
        _mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", 0.88, low=0.25, high=0.90)
        _mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", 1.12, low=0.8, high=3.0)
        _mul_nested(cfg, "reward_weights.lookahead_reward.w_straight", 1.20, low=0.1, high=2.5)

    def smoother_corner(cfg: dict) -> None:
        _mul_nested(cfg, "reward_weights.cornerness.w_track_min", 1.30, low=0.2, high=8.0)
        _mul_nested(cfg, "reward_weights.cornerness.w_smooth0", 1.25, low=0.2, high=8.0)
        _mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", 1.15, low=0.05, high=0.30)

    def stall_strict(cfg: dict) -> None:
        _mul_nested(cfg, "reward_weights.p4.stall_steps", 0.75, low=200.0, high=5000.0)
        _mul_nested(cfg, "reward_weights.p4.stall_progress_eps", 2.00, low=1e-8, high=1e-2)
        _mul_nested(cfg, "reward_weights.p4.stall_v_eps", 1.50, low=1e-4, high=0.5)

    def stall_relaxed(cfg: dict) -> None:
        _mul_nested(cfg, "reward_weights.p4.stall_steps", 1.25, low=200.0, high=5000.0)
        _mul_nested(cfg, "reward_weights.p4.stall_progress_eps", 0.60, low=1e-8, high=1e-2)
        _mul_nested(cfg, "reward_weights.p4.stall_v_eps", 0.80, low=1e-4, high=0.5)

    def exploration_low(cfg: dict) -> None:
        _mul_nested(cfg, "ppo.ent_coef", 0.50, low=0.0, high=0.05)
        epochs = int(_get_nested(cfg, "ppo.epochs", 6))
        _set_nested(cfg, "ppo.epochs", max(4, epochs - 2))
        _mul_nested(cfg, "reward_weights.control_authority.tangent_blend", 1.20, low=0.0, high=0.30)

    def exploration_high(cfg: dict) -> None:
        _mul_nested(cfg, "ppo.ent_coef", 2.00, low=0.0, high=0.05)
        epochs = int(_get_nested(cfg, "ppo.epochs", 6))
        _set_nested(cfg, "ppo.epochs", min(14, epochs + 2))
        _mul_nested(cfg, "reward_weights.control_authority.tangent_blend", 0.85, low=0.0, high=0.30)

    return [
        CandidateSpec("baseline", "保留基线配置，仅切到多路径夜跑模式", baseline),
        CandidateSpec("lookahead_aggressive", "更积极的转角前瞻与角区目标", lookahead_aggressive),
        CandidateSpec("lookahead_conservative", "更保守的直线前瞻与直线奖励", lookahead_conservative),
        CandidateSpec("smoother_corner", "提升角区平滑与跟踪下限", smoother_corner),
        CandidateSpec("stall_strict", "更早判定 stall，抑制卡顿拖尾", stall_strict),
        CandidateSpec("stall_relaxed", "放宽 stall 触发，保留恢复空间", stall_relaxed),
        CandidateSpec("exploration_low", "减小探索，偏向快速收敛", exploration_low),
        CandidateSpec("exploration_high", "增大探索，争取跳出局部策略", exploration_high),
    ]


def _load_checkpoint_episode(checkpoint_path: Path) -> int:
    with checkpoint_path.open("rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    return int(checkpoint.get("episode", -1))


def _resolve_total_episodes(resume_path: Optional[Path], extra_episodes: int) -> int:
    extra = max(0, int(extra_episodes))
    base_total = (_load_checkpoint_episode(resume_path) + 1) if resume_path is not None else 0
    total = base_total + extra
    if resume_path is not None:
        return max(base_total, total)
    return max(1, total)


def _prepare_candidate_config(
    base_config: dict,
    *,
    candidate: CandidateSpec,
    candidate_label: str,
    path_specs: Sequence[Dict[str, Any]],
    stage_total_episodes: int,
    episodes_per_path: int,
) -> dict:
    cfg = copy.deepcopy(base_config)
    if not isinstance(cfg.get("training"), dict):
        cfg["training"] = {}
    if not isinstance(cfg.get("experiment"), dict):
        cfg["experiment"] = {}

    training_cfg = cfg["training"]
    experiment_cfg = cfg["experiment"]

    training_cfg["num_episodes"] = int(stage_total_episodes)
    training_cfg["enable_final_visualization"] = False
    training_cfg["enable_latest_trajectory"] = False
    training_cfg["traj_write_interval_steps"] = 0
    training_cfg["step_log_interval_steps"] = int(max(10, int(training_cfg.get("step_log_interval_steps", 10) or 10)))
    training_cfg["path_curriculum"] = {
        "enabled": True,
        "mode": "round_robin",
        "episodes_per_path": int(max(1, episodes_per_path)),
        "seed": int(cfg.get("seed", experiment_cfg.get("seed", 42))),
        "paths": copy.deepcopy(list(path_specs)),
    }

    cfg["path"] = copy.deepcopy(path_specs[0])
    experiment_cfg["category"] = "night_sweep"
    experiment_cfg["name"] = candidate_label

    candidate.apply(cfg)

    mix_gain = _get_nested(cfg, "reward_weights.lookahead_control.mix_gain", 0.6)
    _set_nested(cfg, "reward_weights.lookahead_control.mix_gain", _clamp(float(mix_gain), 0.10, 0.95))
    corner_target = _get_nested(cfg, "reward_weights.lookahead_reward.corner_target", 0.8)
    _set_nested(cfg, "reward_weights.lookahead_reward.corner_target", _clamp(float(corner_target), 0.05, 0.98))

    return cfg


def _find_model_checkpoint(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_model.pth",
        run_dir / "checkpoints" / "tracking_model_final.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no model checkpoint found under {run_dir / 'checkpoints'}")


def _find_latest_checkpoint(run_dir: Path) -> Path:
    candidate = run_dir / "checkpoints" / "latest_checkpoint.pth"
    if not candidate.exists():
        raise FileNotFoundError(f"latest checkpoint missing: {candidate}")
    return candidate


def _train_candidate(
    *,
    config_path: Path,
    run_dir: Path,
    resume_path: Optional[Path],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--mode",
        "train",
        "--config",
        str(config_path),
        "--experiment_dir",
        str(run_dir),
    ]
    if resume_path is not None:
        cmd.extend(["--resume", str(resume_path)])
    _run_command(cmd, cwd=PROJECT_ROOT)


def _build_eval_config(
    *,
    trained_config_path: Path,
    out_path: Path,
    path_cfg: Dict[str, Any],
) -> Path:
    cfg = _load_yaml(trained_config_path)
    cfg["path"] = copy.deepcopy(path_cfg)
    _write_yaml(out_path, cfg)
    return out_path


def _read_eval_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError(f"invalid summary payload: {summary_path}")
    return summary


def _evaluate_model_across_paths(
    *,
    trained_config_path: Path,
    model_path: Path,
    out_dir: Path,
    path_specs: Sequence[Dict[str, Any]],
    episodes: int,
    deterministic: bool,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_results: Dict[str, dict] = {}

    for path_cfg in path_specs:
        path_name = str(path_cfg.get("name") or path_cfg.get("type") or f"path_{len(path_results)}")
        eval_cfg_path = _build_eval_config(
            trained_config_path=trained_config_path,
            out_path=out_dir / "configs" / f"{path_name}.yaml",
            path_cfg=path_cfg,
        )
        eval_out_dir = out_dir / "eval" / path_name
        cmd = [
            sys.executable,
            str(ACCEPTANCE_SCRIPT),
            "--phase",
            "p0_eval",
            "--config",
            str(eval_cfg_path),
            "--model",
            str(model_path),
            "--episodes",
            str(int(episodes)),
            "--out",
            str(eval_out_dir),
            "--seed",
            str(int(seed)),
        ]
        if deterministic:
            cmd.append("--deterministic")
        _run_command(cmd, cwd=PROJECT_ROOT)
        path_results[path_name] = _read_eval_summary(eval_out_dir / "summary.json")

    return _aggregate_eval_results(path_results)


def _aggregate_eval_results(path_results: Dict[str, dict]) -> dict:
    if not path_results:
        return {
            "path_results": {},
            "aggregated": {
                "path_count": 0,
                "pass_count": 0,
                "mean_success_rate": 0.0,
                "mean_stall_rate": 1.0,
                "mean_progress_final": 0.0,
                "mean_error_ratio": 999.0,
                "max_error_ratio": 999.0,
                "score": float("-inf"),
            },
        }

    success_values: List[float] = []
    stall_values: List[float] = []
    progress_values: List[float] = []
    error_ratios: List[float] = []
    pass_count = 0

    for summary in path_results.values():
        success_values.append(float(summary.get("success_rate", 0.0)))
        stall_values.append(float(summary.get("stall_rate", 1.0)))
        progress_values.append(float(summary.get("mean_progress_final", 0.0)))
        pass_count += int(bool(summary.get("passed", False)))
        max_err = float(summary.get("max_abs_contour_error", 1e9))
        half_eps = max(float(summary.get("half_epsilon", 1e-6)), 1e-6)
        error_ratios.append(max_err / half_eps)

    mean_success = float(mean(success_values))
    mean_stall = float(mean(stall_values))
    mean_progress = float(mean(progress_values))
    mean_error_ratio = float(mean(error_ratios))
    max_error_ratio = float(max(error_ratios))
    score = (
        1000.0 * float(pass_count)
        + 120.0 * mean_success
        + 25.0 * mean_progress
        - 80.0 * mean_stall
        - 20.0 * mean_error_ratio
    )

    return {
        "path_results": path_results,
        "aggregated": {
            "path_count": len(path_results),
            "pass_count": int(pass_count),
            "mean_success_rate": mean_success,
            "mean_stall_rate": mean_stall,
            "mean_progress_final": mean_progress,
            "mean_error_ratio": mean_error_ratio,
            "max_error_ratio": max_error_ratio,
            "score": score,
        },
    }


def _summarize_stage_rows(stage_rows: Sequence[dict], *, stage_name: str) -> str:
    lines = [
        f"# {stage_name}",
        "",
        "| rank | candidate | score | pass_count | mean_success | mean_progress | mean_stall | max_error_ratio | run_dir |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(stage_rows, key=lambda row: float(row.get("score", float("-inf"))), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {candidate} | {score:.3f} | {pass_count} | {mean_success_rate:.4f} | "
            "{mean_progress_final:.4f} | {mean_stall_rate:.4f} | {max_error_ratio:.4f} | `{run_dir}` |".format(
                rank=idx,
                candidate=row.get("candidate", "-"),
                score=float(row.get("score", float("-inf"))),
                pass_count=int(row.get("pass_count", 0)),
                mean_success_rate=float(row.get("mean_success_rate", 0.0)),
                mean_progress_final=float(row.get("mean_progress_final", 0.0)),
                mean_stall_rate=float(row.get("mean_stall_rate", 0.0)),
                max_error_ratio=float(row.get("max_error_ratio", 0.0)),
                run_dir=row.get("run_dir", "-"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _export_best_rollouts(*, config_path: Path, run_dir: Path, out_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(EXPORT_SCRIPT),
        "--config",
        str(config_path),
        "--run_dir",
        str(run_dir),
        "--out",
        str(out_dir),
    ]
    _run_command(cmd, cwd=PROJECT_ROOT)
    return out_dir / "summary.json"


def _build_manifest(args: argparse.Namespace, *, base_config_path: Path, path_names: Sequence[str], sweep_dir: Path) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_dir": str(sweep_dir),
        "base_config": str(base_config_path),
        "resume": str(_resolve_path(args.resume)) if args.resume else None,
        "paths": list(path_names),
        "episodes_per_path": int(args.episodes_per_path),
        "stage1_episodes": int(args.stage1_episodes),
        "stage2_episodes": int(args.stage2_episodes),
        "top_k": int(args.top_k),
        "eval_episodes": int(args.eval_episodes),
        "seed_eval": int(args.eval_seed),
        "deterministic_eval": bool(args.deterministic_eval),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Night sweep runner for multi-path PPO experiments.")
    parser.add_argument(
        "--base-config",
        type=str,
        default=None,
        help="Optional base YAML config path. If omitted and --resume is set, config is loaded from checkpoint.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional latest_checkpoint.pth to branch from.",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="square,s_shape,butterfly,trapezoid,circle",
        help="Comma-separated path names from built-in library.",
    )
    parser.add_argument("--episodes-per-path", type=int, default=2, help="Round-robin episodes per path during training.")
    parser.add_argument("--stage1-episodes", type=int, default=80, help="Extra episodes for stage1 screening.")
    parser.add_argument("--stage2-episodes", type=int, default=180, help="Extra episodes for stage2 finalists.")
    parser.add_argument("--top-k", type=int, default=2, help="How many stage1 candidates advance to stage2.")
    parser.add_argument("--eval-episodes", type=int, default=8, help="Evaluation episodes per path.")
    parser.add_argument("--eval-seed", type=int, default=43, help="Shared evaluation seed.")
    parser.add_argument("--deterministic-eval", action="store_true", help="Use deterministic policy during eval.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional sweep run id.")
    args = parser.parse_args(argv)

    resume_path = _resolve_path(args.resume) if args.resume else None
    selected_path_names = _parse_csv_names(args.paths)
    path_specs = _build_selected_paths(selected_path_names)

    if args.base_config:
        base_config_path = _resolve_path(args.base_config)
        base_config = _load_yaml(base_config_path)
    elif resume_path is not None:
        base_config_path = resume_path
        base_config = _load_config_from_checkpoint(resume_path)
    else:
        base_config_path = _resolve_path("configs/default.yaml")
        base_config = _load_yaml(base_config_path)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = (SWEEP_ROOT / run_id).resolve()
    configs_dir = sweep_dir / "configs"
    runs_dir = sweep_dir / "runs"
    reports_dir = sweep_dir / "reports"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(args, base_config_path=base_config_path, path_names=selected_path_names, sweep_dir=sweep_dir)
    _write_json(sweep_dir / "manifest.json", manifest)

    stage1_total = _resolve_total_episodes(resume_path, args.stage1_episodes)
    stage2_rows: List[dict] = []
    stage1_results: List[dict] = []

    for candidate in _candidate_specs():
        candidate_label = candidate.name
        stage1_cfg = _prepare_candidate_config(
            base_config,
            candidate=candidate,
            candidate_label=f"{candidate_label}_stage1",
            path_specs=path_specs,
            stage_total_episodes=stage1_total,
            episodes_per_path=args.episodes_per_path,
        )
        stage1_cfg_path = configs_dir / candidate_label / "stage1.yaml"
        stage1_run_dir = runs_dir / candidate_label / "stage1"
        _write_yaml(stage1_cfg_path, stage1_cfg)

        result_row = {
            "candidate": candidate_label,
            "description": candidate.description,
            "run_dir": str(stage1_run_dir),
            "config_path": str(stage1_cfg_path),
            "status": "pending",
        }

        try:
            _train_candidate(config_path=stage1_cfg_path, run_dir=stage1_run_dir, resume_path=resume_path)
            model_path = _find_model_checkpoint(stage1_run_dir)
            eval_payload = _evaluate_model_across_paths(
                trained_config_path=stage1_run_dir / "config.yaml",
                model_path=model_path,
                out_dir=stage1_run_dir / "night_eval",
                path_specs=path_specs,
                episodes=args.eval_episodes,
                deterministic=args.deterministic_eval,
                seed=args.eval_seed,
            )
            aggregated = eval_payload["aggregated"]
            result_row.update(
                {
                    "status": "ok",
                    "model_path": str(model_path),
                    "score": float(aggregated["score"]),
                    "pass_count": int(aggregated["pass_count"]),
                    "mean_success_rate": float(aggregated["mean_success_rate"]),
                    "mean_stall_rate": float(aggregated["mean_stall_rate"]),
                    "mean_progress_final": float(aggregated["mean_progress_final"]),
                    "mean_error_ratio": float(aggregated["mean_error_ratio"]),
                    "max_error_ratio": float(aggregated["max_error_ratio"]),
                    "path_results": eval_payload["path_results"],
                }
            )
        except Exception as exc:
            result_row.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "score": float("-inf"),
                    "pass_count": 0,
                    "mean_success_rate": 0.0,
                    "mean_stall_rate": 1.0,
                    "mean_progress_final": 0.0,
                    "mean_error_ratio": 999.0,
                    "max_error_ratio": 999.0,
                }
            )
        stage1_results.append(result_row)
        _write_json(stage1_run_dir / "night_sweep_result.json", result_row)
        _write_json(reports_dir / "stage1_results.json", {"results": stage1_results})

    ranked_stage1 = sorted(
        [row for row in stage1_results if row.get("status") == "ok"],
        key=lambda row: float(row.get("score", float("-inf"))),
        reverse=True,
    )

    top_k = max(0, min(int(args.top_k), len(ranked_stage1)))
    finalists = ranked_stage1[:top_k]
    for finalist in finalists:
        candidate_label = str(finalist["candidate"])
        stage1_run_dir = Path(str(finalist["run_dir"]))
        stage1_latest = _find_latest_checkpoint(stage1_run_dir)
        stage1_total_effective = _resolve_total_episodes(stage1_latest, 0)
        stage2_total = max(stage1_total_effective + int(args.stage2_episodes), stage1_total_effective + 1)

        candidate_spec = next(spec for spec in _candidate_specs() if spec.name == candidate_label)
        stage2_cfg = _prepare_candidate_config(
            base_config,
            candidate=candidate_spec,
            candidate_label=f"{candidate_label}_stage2",
            path_specs=path_specs,
            stage_total_episodes=stage2_total,
            episodes_per_path=args.episodes_per_path,
        )
        stage2_cfg_path = configs_dir / candidate_label / "stage2.yaml"
        stage2_run_dir = runs_dir / candidate_label / "stage2"
        _write_yaml(stage2_cfg_path, stage2_cfg)

        stage2_row = {
            "candidate": candidate_label,
            "description": finalist.get("description", ""),
            "run_dir": str(stage2_run_dir),
            "config_path": str(stage2_cfg_path),
            "resume_from": str(stage1_latest),
            "status": "pending",
        }

        try:
            _train_candidate(config_path=stage2_cfg_path, run_dir=stage2_run_dir, resume_path=stage1_latest)
            model_path = _find_model_checkpoint(stage2_run_dir)
            eval_payload = _evaluate_model_across_paths(
                trained_config_path=stage2_run_dir / "config.yaml",
                model_path=model_path,
                out_dir=stage2_run_dir / "night_eval",
                path_specs=path_specs,
                episodes=args.eval_episodes,
                deterministic=args.deterministic_eval,
                seed=args.eval_seed,
            )
            export_summary = _export_best_rollouts(
                config_path=stage2_run_dir / "config.yaml",
                run_dir=stage2_run_dir,
                out_dir=stage2_run_dir / "best_rollouts",
            )
            aggregated = eval_payload["aggregated"]
            stage2_row.update(
                {
                    "status": "ok",
                    "model_path": str(model_path),
                    "score": float(aggregated["score"]),
                    "pass_count": int(aggregated["pass_count"]),
                    "mean_success_rate": float(aggregated["mean_success_rate"]),
                    "mean_stall_rate": float(aggregated["mean_stall_rate"]),
                    "mean_progress_final": float(aggregated["mean_progress_final"]),
                    "mean_error_ratio": float(aggregated["mean_error_ratio"]),
                    "max_error_ratio": float(aggregated["max_error_ratio"]),
                    "path_results": eval_payload["path_results"],
                    "best_rollouts_summary": str(export_summary),
                }
            )
        except Exception as exc:
            stage2_row.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "score": float("-inf"),
                    "pass_count": 0,
                    "mean_success_rate": 0.0,
                    "mean_stall_rate": 1.0,
                    "mean_progress_final": 0.0,
                    "mean_error_ratio": 999.0,
                    "max_error_ratio": 999.0,
                }
            )

        stage2_rows.append(stage2_row)
        _write_json(stage2_run_dir / "night_sweep_result.json", stage2_row)
        _write_json(reports_dir / "stage2_results.json", {"results": stage2_rows})

    stage1_md = _summarize_stage_rows(stage1_results, stage_name="Stage1 Ranking")
    _write_text(reports_dir / "stage1_ranking.md", stage1_md)

    if stage2_rows:
        stage2_md = _summarize_stage_rows(stage2_rows, stage_name="Stage2 Ranking")
        _write_text(reports_dir / "stage2_ranking.md", stage2_md)
    else:
        _write_text(reports_dir / "stage2_ranking.md", "# Stage2 Ranking\n\n未启用 Stage2 或无有效候选。\n")

    final_payload = {
        "manifest": manifest,
        "stage1": stage1_results,
        "stage2": stage2_rows,
    }
    _write_json(reports_dir / "summary.json", final_payload)

    final_rows = stage2_rows or stage1_results
    best_row = max(final_rows, key=lambda row: float(row.get("score", float("-inf"))))
    print(
        "[DONE] best_candidate={candidate} score={score:.3f} run_dir={run_dir}".format(
            candidate=best_row.get("candidate", "-"),
            score=float(best_row.get("score", float("-inf"))),
            run_dir=best_row.get("run_dir", "-"),
        )
    )
    print(f"[DONE] summary={reports_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
