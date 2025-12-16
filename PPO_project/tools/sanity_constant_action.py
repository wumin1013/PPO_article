"""P3.0 常量动作 sanity：验证 open 终点 success 与 closed lap_completed。"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.environment import Env  # noqa: E402
    from src.utils.path_generator import get_path_by_name  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python -m pip install -r PPO_project/requirements.txt")
    raise


CASES = ("open_line", "open_square", "s_shape", "square_closed")


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    reached_target: bool
    lap_completed: bool
    closed: bool
    steps: int
    final_progress: float
    final_contour_error: float
    final_end_distance: float
    final_dist_to_start: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    print(f"[SEED] seed={seed} (random/numpy)")


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pm_for_case(case: str, cfg: Dict) -> List[np.ndarray]:
    path_cfg = cfg.get("path", {})
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    if case == "open_line":
        line_cfg = path_cfg.get("line", {}) if isinstance(path_cfg.get("line", {}), dict) else {}
        angle = float(line_cfg.get("angle", 0.0))
        return get_path_by_name("line", scale=scale, num_points=num_points, angle=angle)

    if case in {"open_square", "square_closed"}:
        pm = get_path_by_name("square", scale=scale, num_points=num_points)
        if case == "open_square" and len(pm) > 2 and np.allclose(pm[0], pm[-1], atol=1e-6):
            return pm[:-1]
        if case == "square_closed" and len(pm) > 2 and not np.allclose(pm[0], pm[-1], atol=1e-6):
            return list(pm) + [np.asarray(pm[0], dtype=float)]
        return pm

    if case == "s_shape":
        s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
        return get_path_by_name(
            "s_shape",
            scale=scale,
            num_points=num_points,
            amplitude=float(s_cfg.get("amplitude", scale / 2.0)),
            periods=float(s_cfg.get("periods", 2.0)),
        )

    raise ValueError(f"unknown case: {case}")


def _build_env(cfg: Dict, pm: Sequence[np.ndarray]) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    return Env(
        device="cpu",
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=list(pm),
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=cfg.get("reward_weights", {}),
    )


def _run_episode(env: Env, action: np.ndarray) -> EpisodeResult:
    env.reset()
    done = False
    info: Dict = {}
    while not done:
        _, _, done, info = env.step(action)

    reached_target = bool(getattr(env, "reached_target", False))
    lap_completed = bool(getattr(env, "lap_completed", False))
    closed = bool(getattr(env, "closed", False))
    success = lap_completed if closed else reached_target

    final_progress = float(env.state[4]) if getattr(env, "state", None) is not None and len(env.state) > 4 else 0.0
    final_contour_error = float(info.get("contour_error", 0.0))
    end_distance = float(np.linalg.norm(env.current_position - np.array(env.Pm[-1])))
    dist_to_start = float(np.linalg.norm(env.current_position - np.array(env.Pm[0])))
    return EpisodeResult(
        success=success,
        reached_target=reached_target,
        lap_completed=lap_completed,
        closed=closed,
        steps=int(getattr(env, "current_step", 0)),
        final_progress=final_progress,
        final_contour_error=final_contour_error,
        final_end_distance=end_distance,
        final_dist_to_start=dist_to_start,
    )


def _run_case(
    case: str,
    cfg: Dict,
    *,
    theta: float,
    vel_ratio: float,
    episodes: int,
    seed: int,
    enable_episode_diagnostics: bool,
) -> bool:
    pm = _pm_for_case(case, cfg)

    env = _build_env(cfg, pm)
    env.enable_episode_diagnostics = bool(enable_episode_diagnostics)

    action = np.array([float(theta), float(vel_ratio)], dtype=float)
    ok = True
    for ep in range(episodes):
        _set_seed(seed + ep)
        result = _run_episode(env, action)
        tag = "PASS" if result.success else "FAIL"
        print(
            f"[{tag}] case={case} ep={ep+1}/{episodes} closed={result.closed} "
            f"reached_target={result.reached_target} lap_completed={result.lap_completed} "
            f"steps={result.steps} progress={result.final_progress:.4f} "
            f"end_dist={result.final_end_distance:.4f} start_dist={result.final_dist_to_start:.4f} "
            f"contour={result.final_contour_error:.4f}"
        )
        ok = ok and result.success
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="P3.0 常量动作 sanity（open 终点 / closed lap）")
    parser.add_argument("--case", type=str, default="all", choices=("all",) + CASES)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于构建环境与路径参数的 YAML（默认使用 original_configs/smoke.yaml，适合快速 sanity）。",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--vel-ratio", type=float, default=0.8)
    parser.add_argument(
        "--print-episode-end",
        action="store_true",
        help="开启 Env 的 episode_end 诊断打印（每回合只打印一次）。",
    )
    args = parser.parse_args()

    selected = list(CASES) if args.case == "all" else [args.case]
    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)

    all_ok = True
    for case in selected:
        all_ok = all_ok and _run_case(
            case,
            cfg,
            theta=args.theta,
            vel_ratio=args.vel_ratio,
            episodes=max(1, int(args.episodes)),
            seed=int(args.seed),
            enable_episode_diagnostics=bool(args.print_episode_end),
        )

    raise SystemExit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
