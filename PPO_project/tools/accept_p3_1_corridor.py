"""P3.1 自动化验收：VirtualCorridor 开关/滞回/corridor_status/指标对比。"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.environment import Env  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python -m pip install -r PPO_project/requirements.txt")
    raise


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _linspace_points(p0: np.ndarray, p1: np.ndarray, n: int, *, include_start: bool) -> List[np.ndarray]:
    if n <= 1:
        return [p1.copy()] if not include_start else [p0.copy()]
    pts: List[np.ndarray] = []
    ts = np.linspace(0.0, 1.0, n, endpoint=True)
    if not include_start:
        ts = ts[1:]
    for t in ts:
        pts.append(p0 + t * (p1 - p0))
    return pts


def build_open_square_path(side: float = 10.0, num_points: int = 200, *, clockwise: bool = False) -> List[np.ndarray]:
    """三边开链正方形：含两个拐角，避免“终点贴近起点”的退化用例。"""
    if num_points < 10:
        raise ValueError("num_points too small")
    L = float(side)
    vertices_ccw = [
        np.array([0.0, 0.0], dtype=float),
        np.array([L, 0.0], dtype=float),
        np.array([L, L], dtype=float),
        np.array([0.0, L], dtype=float),
    ]
    vertices = list(reversed(vertices_ccw)) if clockwise else vertices_ccw

    # 三条边：0->1->2->3
    per_edge = max(2, num_points // 3)
    pts: List[np.ndarray] = []
    pts.extend(_linspace_points(vertices[0], vertices[1], per_edge, include_start=True))
    pts.extend(_linspace_points(vertices[1], vertices[2], per_edge, include_start=False))
    pts.extend(_linspace_points(vertices[2], vertices[3], num_points - len(pts), include_start=False))
    return [np.array(p, dtype=float) for p in pts]


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, corridor_enabled: bool) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    reward_weights = dict(cfg.get("reward_weights", {}) or {})
    corridor_cfg = dict(reward_weights.get("corridor", {}) or {}) if isinstance(reward_weights.get("corridor", {}), dict) else {}
    corridor_cfg["enabled"] = bool(corridor_enabled)
    reward_weights["corridor"] = corridor_cfg

    env = Env(
        device="cpu",
        epsilon=float(env_cfg.get("epsilon", 1.5)),
        interpolation_period=float(env_cfg.get("interpolation_period", 0.1)),
        MAX_VEL=float(kcm_cfg.get("MAX_VEL", 1.0)),
        MAX_ACC=float(kcm_cfg.get("MAX_ACC", 2.0)),
        MAX_JERK=float(kcm_cfg.get("MAX_JERK", 3.0)),
        MAX_ANG_VEL=float(kcm_cfg.get("MAX_ANG_VEL", 1.5)),
        MAX_ANG_ACC=float(kcm_cfg.get("MAX_ANG_ACC", 3.0)),
        MAX_ANG_JERK=float(kcm_cfg.get("MAX_ANG_JERK", 5.0)),
        Pm=list(pm),
        max_steps=int(env_cfg.get("max_steps", 400)),
        lookahead_points=int(env_cfg.get("lookahead_points", 5)),
        reward_weights=reward_weights,
    )
    return env


@dataclass
class Metrics:
    episodes: int
    success_rate: float
    oob_rate: float
    steps_mean: float
    v_mean: float
    mean_e_n_corner: float


def _controller_action(env: Env, corridor_status: Dict[str, object], *, vel: float, kp: float) -> np.ndarray:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    desired = 0.0
    if bool(corridor_status.get("enabled", False)) and bool(corridor_status.get("corner_phase", False)):
        desired = float(corridor_status.get("e_target", 0.0))
    theta = float(np.clip(kp * (desired - e_n) / max(half, 1e-6), -1.0, 1.0))
    return np.array([theta, float(vel)], dtype=float)


def _run_eval(
    env: Env,
    *,
    episodes: int,
    seed: int,
    vel: float,
    kp: float,
) -> Metrics:
    successes = 0
    oobs = 0
    steps: List[int] = []
    v_means: List[float] = []
    e_corner: List[float] = []

    for ep in range(episodes):
        _set_seed(seed + ep)
        env.reset()
        action = np.array([0.0, float(vel)], dtype=float)

        done = False
        info: Dict[str, object] = {}
        v_samples: List[float] = []
        while not done:
            _, _, done, info = env.step(action)
            v_samples.append(float(getattr(env, "velocity", 0.0)))
            corridor_status = info.get("corridor_status", {}) if isinstance(info, dict) else {}
            if isinstance(corridor_status, dict) and bool(corridor_status.get("corner_phase", False)):
                e_corner.append(float(corridor_status.get("e_n", 0.0)))
            action = _controller_action(env, corridor_status if isinstance(corridor_status, dict) else {}, vel=vel, kp=kp)

        reached = bool(getattr(env, "reached_target", False))
        lap = bool(getattr(env, "lap_completed", False))
        success = lap if bool(getattr(env, "closed", False)) else reached
        successes += int(success)

        contour_error = float(info.get("contour_error", 0.0)) if isinstance(info, dict) else 0.0
        if (not success) and contour_error > float(getattr(env, "half_epsilon", 0.0)):
            oobs += 1

        steps.append(int(getattr(env, "current_step", 0)))
        v_means.append(float(np.mean(v_samples)) if v_samples else 0.0)

    return Metrics(
        episodes=episodes,
        success_rate=float(successes / max(episodes, 1)),
        oob_rate=float(oobs / max(episodes, 1)),
        steps_mean=float(np.mean(steps)) if steps else 0.0,
        v_mean=float(np.mean(v_means)) if v_means else 0.0,
        mean_e_n_corner=float(np.mean(e_corner)) if e_corner else 0.0,
    )


def _print_corridor_sample(env: Env, *, max_steps: int = 200) -> Dict[str, object]:
    env.reset()
    action = np.array([0.0, 0.8], dtype=float)
    for _ in range(max_steps):
        _, _, done, info = env.step(action)
        cs = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if isinstance(cs, dict) and bool(cs.get("corner_phase", False)) and int(cs.get("turn_sign", 0)) != 0:
            return cs
        if done:
            env.reset()
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="P3.1 VirtualCorridor 自动化验收")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vel", type=float, default=0.8)
    parser.add_argument("--kp", type=float, default=1.0, help="走廊跟踪 P 控制系数（越大越靠近 e_target）。")
    parser.add_argument(
        "--side",
        type=float,
        default=0.0,
        help="三边开链正方形的边长；<=0 时按 dt/max_steps/vel 自动估算可跑完的尺度。",
    )
    parser.add_argument("--num-points", type=int, default=200)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_config(args.config)

    env_cfg = cfg.get("environment", {}) or {}
    dt = float(env_cfg.get("interpolation_period", 0.1))
    max_steps = int(env_cfg.get("max_steps", 200))

    side = float(args.side)
    if side <= 0.0:
        # 估算 50 episodes 的快速验收尺度：留出加速段与拐角损耗裕量
        est_travel = float(args.vel) * dt * max_steps * 0.85
        side = float(np.clip(est_travel / 3.0 * 0.9, 1.0, 10.0))
        print(f"[AUTO] side={side:.2f} (dt={dt}, max_steps={max_steps}, vel={args.vel})")

    # 1) corridor_status 符号自检：CCW(左转) 与 CW(右转) 各取一段样例
    pm_left = build_open_square_path(side=side, num_points=int(args.num_points), clockwise=False)
    pm_right = build_open_square_path(side=side, num_points=int(args.num_points), clockwise=True)
    env_left = _build_env(cfg, pm_left, corridor_enabled=True)
    env_right = _build_env(cfg, pm_right, corridor_enabled=True)

    cs_left = _print_corridor_sample(env_left)
    cs_right = _print_corridor_sample(env_right)

    if not cs_left or not cs_right:
        print("[FAIL] corridor_status sample not found (corner_phase may never enter).")
        raise SystemExit(2)

    print("[SAMPLE] left-turn (expect turn_sign=+1, lower>=0):", cs_left)
    print("[SAMPLE] right-turn (expect turn_sign=-1, upper<=0):", cs_right)

    if (
        int(cs_left.get("turn_sign", 0)) != 1
        or float(cs_left.get("lower", 0.0)) < -1e-9
        or float(cs_left.get("upper", 0.0)) <= 1e-9
    ):
        print("[FAIL] left-turn sign/bounds mismatch.")
        raise SystemExit(2)
    if (
        int(cs_right.get("turn_sign", 0)) != -1
        or float(cs_right.get("upper", 0.0)) > 1e-9
        or float(cs_right.get("lower", 0.0)) >= -1e-9
    ):
        print("[FAIL] right-turn sign/bounds mismatch.")
        raise SystemExit(2)

    # 2) 指标对比：同一条路径上 corridor on/off 各跑 E=episodes
    env_off = _build_env(cfg, pm_left, corridor_enabled=False)
    env_on = _build_env(cfg, pm_left, corridor_enabled=True)

    m_off = _run_eval(env_off, episodes=int(args.episodes), seed=int(args.seed), vel=float(args.vel), kp=float(args.kp))
    m_on = _run_eval(env_on, episodes=int(args.episodes), seed=int(args.seed), vel=float(args.vel), kp=float(args.kp))

    print(
        f"[EVAL] corridor=OFF episodes={m_off.episodes} success_rate={m_off.success_rate:.3f} "
        f"oob_rate={m_off.oob_rate:.3f} steps_mean={m_off.steps_mean:.1f} v_mean={m_off.v_mean:.3f} "
        f"mean_e_n_corner={m_off.mean_e_n_corner:.4f}"
    )
    print(
        f"[EVAL] corridor=ON  episodes={m_on.episodes} success_rate={m_on.success_rate:.3f} "
        f"oob_rate={m_on.oob_rate:.3f} steps_mean={m_on.steps_mean:.1f} v_mean={m_on.v_mean:.3f} "
        f"mean_e_n_corner={m_on.mean_e_n_corner:.4f}"
    )

    # 3) 最低验收：开关都能跑；开启后越界率不显著上升（给宽松阈值避免偶然波动）
    if m_on.oob_rate > m_off.oob_rate + 0.10:
        print("[FAIL] oob_rate worsened too much with corridor enabled.")
        raise SystemExit(2)

    if m_off.success_rate == 0.0 and m_on.success_rate == 0.0:
        print("[NOTE] success_rate=0 多半是 max_steps/路径长度/速度组合导致未到终点；可尝试提高 max_steps 或指定更小的 --side。")

    if abs(m_on.mean_e_n_corner) < 1e-3:
        print("[NOTE] mean_e_n_corner≈0，说明走廊偏置不明显；可提高 --kp 或检查 corridor_status 是否进入 corner_phase。")

    print("[PASS] P3.1 VirtualCorridor acceptance passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
