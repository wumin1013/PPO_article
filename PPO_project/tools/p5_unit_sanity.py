"""P5.0 单位一致性自检：ratio→phys 映射 + cap 链路 + quick_eval 回归。

按 08_P5_0 文档要求提供两个场景：
- 场景 A：直线 + 固定动作 (theta_u=0, v_u=1)，验证 v_ratio_exec 能爬升到接近 1
- 场景 B：square 拐角附近 cap 生效，验证：
  - v_ratio_exec <= v_ratio_cap 恒成立
  - v_phys <= MAX_VEL * v_ratio_cap 恒成立
并附带 quick_eval（line/square/s_shape，各 E=5）。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_line(length: float, num_points: int, angle: float = 0.0) -> List[np.ndarray]:
    ts = np.linspace(0.0, 1.0, max(2, int(num_points)))
    dx = math.cos(float(angle))
    dy = math.sin(float(angle))
    return [np.array([float(length) * t * dx, float(length) * t * dy], dtype=float) for t in ts]


def build_open_square(side: float, num_points: int) -> List[np.ndarray]:
    if num_points < 10:
        raise ValueError("num_points too small")
    L = float(side)
    vertices = [
        np.array([0.0, 0.0], dtype=float),
        np.array([L, 0.0], dtype=float),
        np.array([L, L], dtype=float),
        np.array([0.0, L], dtype=float),
    ]
    per_edge = max(2, num_points // 3)

    def edge_points(p0: np.ndarray, p1: np.ndarray, n: int, *, include_start: bool) -> List[np.ndarray]:
        ts = np.linspace(0.0, 1.0, max(2, int(n)), endpoint=True)
        if not include_start:
            ts = ts[1:]
        return [p0 + t * (p1 - p0) for t in ts]

    pts: List[np.ndarray] = []
    pts.extend(edge_points(vertices[0], vertices[1], per_edge, include_start=True))
    pts.extend(edge_points(vertices[1], vertices[2], per_edge, include_start=False))
    pts.extend(edge_points(vertices[2], vertices[3], num_points - len(pts), include_start=False))
    return [np.array(p, dtype=float) for p in pts]


def build_s_shape(scale: float, num_points: int, amplitude: float, periods: float) -> List[np.ndarray]:
    t = np.linspace(0.0, 1.0, max(2, int(num_points)))
    x = float(scale) * t
    y = float(amplitude) * np.sin(2.0 * math.pi * float(periods) * t)
    return [np.array([float(x[i]), float(y[i])], dtype=float) for i in range(len(t))]


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, max_steps: int) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    return Env(
        device="cpu",
        epsilon=float(env_cfg.get("epsilon", 0.5)),
        interpolation_period=float(env_cfg.get("interpolation_period", 0.1)),
        MAX_VEL=float(kcm_cfg.get("MAX_VEL", 1.0)),
        MAX_ACC=float(kcm_cfg.get("MAX_ACC", 2.0)),
        MAX_JERK=float(kcm_cfg.get("MAX_JERK", 3.0)),
        MAX_ANG_VEL=float(kcm_cfg.get("MAX_ANG_VEL", 1.5)),
        MAX_ANG_ACC=float(kcm_cfg.get("MAX_ANG_ACC", 3.0)),
        MAX_ANG_JERK=float(kcm_cfg.get("MAX_ANG_JERK", 5.0)),
        Pm=list(pm),
        max_steps=int(max_steps),
        lookahead_points=int(env_cfg.get("lookahead_points", 5)),
        reward_weights=cfg.get("reward_weights", {}),
    )


def _p_controller_theta(env: Env, corridor_status: Dict[str, object], *, kp: float) -> float:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    return float(np.clip(kp * (0.0 - e_n) / max(half, 1e-6), -1.0, 1.0))


@dataclass
class EpisodeMetrics:
    success: bool
    oob: bool
    steps: int
    v_mean: float
    rmse: float


def _run_episode(env: Env, *, seed: int, kp_lat: float, v_policy: float | None = None) -> EpisodeMetrics:
    _set_seed(seed)
    env.reset()
    done = False
    info: Dict[str, object] = {}
    v_samples: List[float] = []
    e_samples: List[float] = []

    # 初始动作：使用当前 speed_target 或外部固定 v_policy
    p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
    v_u = float(v_policy) if v_policy is not None else float(p4_next.get("speed_target", 0.8))
    action = np.array([0.0, float(np.clip(v_u, 0.0, 1.0))], dtype=float)

    while not done:
        _, _, done, info = env.step(action)
        v_samples.append(float(getattr(env, "velocity", 0.0)))
        e_samples.append(float(info.get("contour_error", 0.0)) if isinstance(info, dict) else 0.0)

        corridor_status = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(corridor_status, dict):
            corridor_status = {}
        theta_u = _p_controller_theta(env, corridor_status, kp=float(kp_lat))

        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_u = float(v_policy) if v_policy is not None else float(p4_next.get("speed_target", 0.8))
        action = np.array([theta_u, float(np.clip(v_u, 0.0, 1.0))], dtype=float)

    contour_error = float(info.get("contour_error", 0.0)) if isinstance(info, dict) else float("inf")
    oob = bool(contour_error > float(getattr(env, "half_epsilon", 1.0)))
    success = bool(getattr(env, "reached_target", False))
    steps = int(getattr(env, "current_step", 0))
    v_mean = float(np.mean(v_samples)) if v_samples else 0.0
    rmse = float(np.sqrt(np.mean(np.square(e_samples)))) if e_samples else 0.0
    return EpisodeMetrics(success=success, oob=oob, steps=steps, v_mean=v_mean, rmse=rmse)


def _scenario_a(cfg: Dict, *, steps: int, seed: int) -> Tuple[bool, str]:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    dt = float(env_cfg.get("interpolation_period", 0.1))
    max_vel_yaml = float(kcm_cfg.get("MAX_VEL", 1.0))

    # 避免 smoke.yaml 的 MAX_VEL=1.0 掩盖“ratio→phys”错误：若 MAX_VEL 太小则临时放大做单位检验
    max_vel = float(max_vel_yaml)
    cfg_a = cfg
    if max_vel_yaml <= 1.05:
        max_vel = 3.0
        cfg_a = dict(cfg)
        cfg_a["kinematic_constraints"] = dict(kcm_cfg)
        cfg_a["kinematic_constraints"]["MAX_VEL"] = float(max_vel)
        print(f"[SANITY-A] override MAX_VEL for unit check: {max_vel_yaml} -> {max_vel}")

    # 保证不会在 steps 内跑到终点：长度按 MAX_VEL*dt*steps 放大
    length = max(float(env_cfg.get("epsilon", 0.5)) * 10.0, max_vel * dt * float(steps) * 2.0)
    pm = build_line(length=length, num_points=200, angle=0.0)
    env = _build_env(cfg_a, pm, max_steps=max(int(env_cfg.get("max_steps", 200)), int(steps) + 5))

    _set_seed(seed)
    env.reset()

    v_ratios: List[float] = []
    v_phys: List[float] = []
    omega_phys: List[float] = []

    action = np.array([0.0, 1.0], dtype=float)
    last_info: Dict[str, object] = {}
    for _ in range(int(steps)):
        _, _, done, info = env.step(action)
        last_info = info if isinstance(info, dict) else {}
        v_phys.append(float(getattr(env, "velocity", 0.0)))
        omega_phys.append(float(getattr(env, "angular_vel", 0.0)))
        v_ratios.append(float(v_phys[-1] / max(max_vel, 1e-6)))
        if done:
            # 若意外结束，重新 reset 继续收集（避免“太短路径”导致误判）
            env.reset()

    tail = v_ratios[-max(10, min(50, len(v_ratios))) :]
    v_ratio_tail = float(np.mean(tail)) if tail else 0.0
    v_phys_tail = float(np.mean(v_phys[-len(tail) :])) if tail else 0.0
    omega_phys_tail = float(np.mean(omega_phys[-len(tail) :])) if tail else 0.0

    exec_action = last_info.get("action_exec", None)
    if isinstance(exec_action, (list, tuple, np.ndarray)) and len(exec_action) >= 2:
        v_exec = float(exec_action[1])
    else:
        v_exec = float("nan")
    v_phys_last = float(v_phys[-1]) if v_phys else 0.0

    print(
        f"[SANITY-A] theta_u=0 v_u=1 -> v_ratio_exec_tail={v_ratio_tail:.3f} v_phys_tail={v_phys_tail:.3f}/{max_vel} "
        f"omega_phys_tail={omega_phys_tail:.3f} action_exec.v={v_exec:.3f}"
    )

    if v_ratio_tail < 0.90:
        return False, "v_ratio_exec did not rise close to 1.0"
    if not math.isfinite(v_exec) or abs(v_exec - v_phys_last) > 1e-6 * max(1.0, max_vel):
        return False, "action_exec[1] not consistent with v_phys"
    return True, "ok"


def _scenario_b(cfg: Dict, *, seed: int, kp_lat: float) -> Tuple[bool, str]:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    max_vel = float(kcm_cfg.get("MAX_VEL", 1.0))
    side = float(cfg.get("path", {}).get("scale", 10.0))
    num_points = int(cfg.get("path", {}).get("num_points", 200))

    pm = build_open_square(side=side, num_points=num_points)
    env = _build_env(cfg, pm, max_steps=max(300, int(env_cfg.get("max_steps", 200))))

    _set_seed(seed)
    env.reset()

    action = np.array([0.0, 1.0], dtype=float)
    done = False
    saw_cap_drop = False
    min_cap = 1.0

    for _ in range(int(getattr(env, "max_steps", 300))):
        _, _, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}
        p4 = info.get("p4_status", {})
        if isinstance(p4, dict):
            v_ratio_cap = float(p4.get("v_ratio_cap", 1.0))
            v_ratio_exec = float(p4.get("v_ratio_exec", 0.0))
            v_phys = float(p4.get("v_exec", getattr(env, "velocity", 0.0)))

            min_cap = min(min_cap, v_ratio_cap)
            if v_ratio_cap < 0.95:
                saw_cap_drop = True

            if v_ratio_exec > v_ratio_cap + 1e-6:
                return False, "v_ratio_exec exceeded v_ratio_cap"
            if v_phys > max_vel * v_ratio_cap + 1e-6 * max(1.0, max_vel):
                return False, "v_phys exceeded MAX_VEL * v_ratio_cap"

            if v_ratio_cap < 0.95:
                print(
                    f"[SANITY-B] v_u_policy=1 v_ratio_cap={v_ratio_cap:.3f} v_ratio_exec={v_ratio_exec:.3f} v_phys={v_phys:.3f} MAX_VEL={max_vel}"
                )

        corridor_status = info.get("corridor_status", {})
        if not isinstance(corridor_status, dict):
            corridor_status = {}
        theta_u = _p_controller_theta(env, corridor_status, kp=float(kp_lat))
        action = np.array([theta_u, 1.0], dtype=float)
        if done:
            break

    if not saw_cap_drop:
        return False, f"v_ratio_cap never dropped (<0.95); min_cap={min_cap:.3f}"
    return True, "ok"


def _quick_eval(cfg: Dict, *, seed: int, episodes: int, kp_lat: float) -> Tuple[bool, str]:
    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
    amplitude = float(s_cfg.get("amplitude", scale / 2.0))
    periods = float(s_cfg.get("periods", 2.0))

    paths = {
        "line": build_line(length=scale, num_points=num_points, angle=float(path_cfg.get("line", {}).get("angle", 0.0)) if isinstance(path_cfg.get("line", {}), dict) else 0.0),
        "square": build_open_square(side=scale, num_points=num_points),
        "s_shape": build_s_shape(scale=scale, num_points=num_points, amplitude=amplitude, periods=periods),
    }

    all_ok = True
    for name, pm in paths.items():
        env = _build_env(cfg, pm, max_steps=int(env_cfg.get("max_steps", 200)))
        results: List[EpisodeMetrics] = []
        for ep in range(int(episodes)):
            results.append(_run_episode(env, seed=int(seed) + ep, kp_lat=float(kp_lat), v_policy=None))

        success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))
        oob_rate = float(np.mean([1.0 if r.oob else 0.0 for r in results]))
        steps_mean = float(np.mean([r.steps for r in results])) if results else 0.0
        v_mean = float(np.mean([r.v_mean for r in results])) if results else 0.0
        rmse_mean = float(np.mean([r.rmse for r in results])) if results else 0.0

        print(
            f"[EVAL] path={name} episodes={episodes} success_rate={success_rate:.3f} oob_rate={oob_rate:.3f} rmse_mean={rmse_mean:.4f} steps_mean={steps_mean:.1f} v_mean={v_mean:.3f}"
        )

        if name == "line" and success_rate < 0.95:
            all_ok = False
            print("[FAIL] quick_eval: line success_rate < 0.95")
        if name != "line":
            # 只做“无灾难性退化”检查
            if oob_rate >= 0.99:
                all_ok = False
                print(f"[FAIL] quick_eval: {name} oob_rate ~= 1.0 (catastrophic)")
            if v_mean < 1e-6:
                all_ok = False
                print(f"[FAIL] quick_eval: {name} v_mean ~= 0 (stuck)")

    return (all_ok, "ok" if all_ok else "failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.0 动作量纲统一自检（ratio→phys）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-a", type=int, default=300)
    parser.add_argument("--kp-lat", type=float, default=1.0)
    parser.add_argument("--quick-eval-episodes", type=int, default=5)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)

    ok_a, reason_a = _scenario_a(cfg, steps=int(args.steps_a), seed=int(args.seed))
    if not ok_a:
        print(f"[FAIL] Scenario A: {reason_a}")
        raise SystemExit(2)
    print("[PASS] Scenario A (line constant action) passed.")

    ok_b, reason_b = _scenario_b(cfg, seed=int(args.seed), kp_lat=float(args.kp_lat))
    if not ok_b:
        print(f"[FAIL] Scenario B: {reason_b}")
        raise SystemExit(2)
    print("[PASS] Scenario B (cap chain) passed.")

    ok_eval, reason_eval = _quick_eval(cfg, seed=int(args.seed), episodes=int(args.quick_eval_episodes), kp_lat=float(args.kp_lat))
    if not ok_eval:
        print(f"[FAIL] quick_eval: {reason_eval}")
        raise SystemExit(2)
    print("[PASS] quick_eval passed.")

    print("[PASS] P5.0 unit sanity passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
