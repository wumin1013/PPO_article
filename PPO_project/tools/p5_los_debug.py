"""P5.1 LOS(turn) 自检：alpha/L/kappa_los 对内切(e_n)敏感 + corner/exit 事件对齐回归。

按 09_P5.1 文档要求：
- 在同一条 square 路径上，用短暂不同的 theta_u 造出两条不同内切偏移轨迹；
- 记录拐角段序列：e_n, alpha, kappa_los, v_ratio_cap，并验证它们随偏移变化；
- 验证硬约束：v_ratio_exec <= v_ratio_cap 恒成立；
- quick_eval 回归：line/square/s_shape 各 E=5，避免灾难性退化。
"""

from __future__ import annotations

import argparse
import math
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


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, p4_debug: bool) -> Env:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    reward_weights = dict(cfg.get("reward_weights", {}) or {})
    p4_cfg = dict(reward_weights.get("p4", {}) or {}) if isinstance(reward_weights.get("p4", {}), dict) else {}
    p4_cfg["debug"] = bool(p4_debug)
    reward_weights["p4"] = p4_cfg

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
        max_steps=int(env_cfg.get("max_steps", 200)),
        lookahead_points=int(env_cfg.get("lookahead_points", 5)),
        reward_weights=reward_weights,
    )


def _p_controller_theta(env: Env, corridor_status: Dict[str, object], *, kp: float) -> float:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    return float(np.clip(kp * (0.0 - e_n) / max(half, 1e-6), -1.0, 1.0))


@dataclass
class LosTrace:
    e_n: List[float]
    alpha: List[float]
    kappa_los: List[float]
    v_ratio_cap: List[float]
    v_ratio_exec: List[float]
    sample_info: Dict[str, object]


def _run_offset_case(
    env: Env,
    *,
    seed: int,
    name: str,
    warmup_theta: float,
    warmup_steps: int,
    kp_lat: float,
    max_steps: int,
    print_samples: int,
) -> LosTrace:
    _set_seed(seed)
    env.reset()

    e_corner: List[float] = []
    alpha_corner: List[float] = []
    kappa_corner: List[float] = []
    vcap_corner: List[float] = []
    vexec_corner: List[float] = []

    sample_info: Dict[str, object] = {}
    printed = 0

    action = np.array([float(warmup_theta), 1.0], dtype=float)
    done = False
    info: Dict[str, object] = {}

    for step in range(int(max_steps)):
        _, _, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}

        cs = info.get("corridor_status", {})
        if not isinstance(cs, dict):
            cs = {}
        p4 = info.get("p4_status", {})
        if not isinstance(p4, dict):
            p4 = {}

        v_ratio_cap = float(p4.get("v_ratio_cap", 1.0))
        v_ratio_exec = float(p4.get("v_ratio_exec", 0.0))
        if v_ratio_exec > v_ratio_cap + 1e-6:
            raise RuntimeError(f"[FAIL] {name}: v_ratio_exec exceeded v_ratio_cap (step={step})")

        in_corner = bool(cs.get("corner_phase", False))
        if in_corner:
            e_n = float(cs.get("e_n", 0.0))
            alpha = float(p4.get("alpha", cs.get("alpha", 0.0)))
            kappa_los = float(p4.get("kappa_los", p4.get("kappa", cs.get("kappa_los", 0.0))))

            e_corner.append(e_n)
            alpha_corner.append(alpha)
            kappa_corner.append(kappa_los)
            vcap_corner.append(v_ratio_cap)
            vexec_corner.append(v_ratio_exec)

            if not sample_info:
                sample_info = {
                    "alpha": float(alpha),
                    "L": float(p4.get("L", cs.get("L", float("nan")))),
                    "kappa_los": float(kappa_los),
                    "v_ratio_cap": float(v_ratio_cap),
                    "corner_phase": bool(in_corner),
                    "exit_boost_remaining": float(p4.get("exit_boost_remaining", float("nan"))),
                }

            if printed < int(print_samples):
                printed += 1
                print(
                    f"[LOS] case={name} step={int(step)} e_n={e_n:+.4f} alpha={float(alpha):+.4f} "
                    f"kappa_los={float(kappa_los):.4f} v_ratio_cap={v_ratio_cap:.3f} v_ratio_exec={v_ratio_exec:.3f}"
                )

        if done:
            break

        if step + 1 < int(warmup_steps):
            action = np.array([float(warmup_theta), 1.0], dtype=float)
        else:
            theta_u = _p_controller_theta(env, cs, kp=float(kp_lat))
            action = np.array([theta_u, 1.0], dtype=float)

    return LosTrace(
        e_n=e_corner,
        alpha=alpha_corner,
        kappa_los=kappa_corner,
        v_ratio_cap=vcap_corner,
        v_ratio_exec=vexec_corner,
        sample_info=sample_info,
    )


@dataclass
class EpisodeMetrics:
    success: bool
    oob: bool
    steps: int
    v_mean: float
    rmse: float


def _run_episode(env: Env, *, seed: int, kp_lat: float) -> EpisodeMetrics:
    _set_seed(seed)
    env.reset()

    done = False
    info: Dict[str, object] = {}
    v_samples: List[float] = []
    e_samples: List[float] = []

    p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
    action = np.array([0.0, float(p4_next.get("speed_target", 0.8))], dtype=float)

    while not done:
        _, _, done, info = env.step(action)
        v_samples.append(float(getattr(env, "velocity", 0.0)))
        e_samples.append(float(info.get("contour_error", 0.0)) if isinstance(info, dict) else float("inf"))

        cs = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(cs, dict):
            cs = {}
        theta_u = _p_controller_theta(env, cs, kp=float(kp_lat))

        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_u = float(p4_next.get("speed_target", 0.8))
        action = np.array([theta_u, float(np.clip(v_u, 0.0, 1.0))], dtype=float)

    contour_error = float(info.get("contour_error", 0.0)) if isinstance(info, dict) else float("inf")
    oob = bool(contour_error > float(getattr(env, "half_epsilon", 1.0)))
    success = bool(getattr(env, "reached_target", False))
    steps = int(getattr(env, "current_step", 0))
    v_mean = float(np.mean(v_samples)) if v_samples else 0.0
    rmse = float(np.sqrt(np.mean(np.square(e_samples)))) if e_samples else 0.0
    return EpisodeMetrics(success=success, oob=oob, steps=steps, v_mean=v_mean, rmse=rmse)


def _quick_eval(cfg: Dict, *, seed: int, episodes: int, kp_lat: float) -> Tuple[bool, str]:
    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
    amplitude = float(s_cfg.get("amplitude", scale / 2.0))
    periods = float(s_cfg.get("periods", 2.0))

    paths = {
        "line": build_line(
            length=scale,
            num_points=num_points,
            angle=float(path_cfg.get("line", {}).get("angle", 0.0)) if isinstance(path_cfg.get("line", {}), dict) else 0.0,
        ),
        "square": build_open_square(side=scale, num_points=num_points),
        "s_shape": build_s_shape(scale=scale, num_points=num_points, amplitude=amplitude, periods=periods),
    }

    all_ok = True
    for name, pm in paths.items():
        env = _build_env(cfg, pm, p4_debug=False)
        results: List[EpisodeMetrics] = []
        for ep in range(int(episodes)):
            results.append(_run_episode(env, seed=int(seed) + ep, kp_lat=float(kp_lat)))

        success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))
        oob_rate = float(np.mean([1.0 if r.oob else 0.0 for r in results]))
        steps_mean = float(np.mean([r.steps for r in results])) if results else 0.0
        v_mean = float(np.mean([r.v_mean for r in results])) if results else 0.0
        rmse_mean = float(np.mean([r.rmse for r in results])) if results else 0.0

        print(
            f"[EVAL] path={name} episodes={episodes} success_rate={success_rate:.3f} oob_rate={oob_rate:.3f} "
            f"rmse_mean={rmse_mean:.4f} steps_mean={steps_mean:.1f} v_mean={v_mean:.3f}"
        )

        if name == "line" and success_rate < 0.95:
            all_ok = False
            print("[FAIL] quick_eval: line success_rate < 0.95")
        if name != "line":
            if oob_rate >= 0.99:
                all_ok = False
                print(f"[FAIL] quick_eval: {name} oob_rate ~= 1.0 (catastrophic)")
            if v_mean < 1e-6:
                all_ok = False
                print(f"[FAIL] quick_eval: {name} v_mean ~= 0 (stuck)")

    return (all_ok, "ok" if all_ok else "failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.1 LOS(turn) 自检（alpha/L/kappa_los + corner/exit 对齐）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kp-lat", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=18)
    parser.add_argument("--warmup-theta", type=float, default=0.6, help="用于造出内切偏移的短暂固定 theta_u（正负各跑一次）。")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--print-samples", type=int, default=10)
    parser.add_argument("--quick-eval-episodes", type=int, default=5)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    pm = build_open_square(side=scale, num_points=num_points)
    env = _build_env(cfg, pm, p4_debug=True)

    left = _run_offset_case(
        env,
        seed=int(args.seed),
        name="warmup_left",
        warmup_theta=abs(float(args.warmup_theta)),
        warmup_steps=int(args.warmup_steps),
        kp_lat=float(args.kp_lat),
        max_steps=int(args.max_steps),
        print_samples=int(args.print_samples),
    )
    right = _run_offset_case(
        env,
        seed=int(args.seed),
        name="warmup_right",
        warmup_theta=-abs(float(args.warmup_theta)),
        warmup_steps=int(args.warmup_steps),
        kp_lat=float(args.kp_lat),
        max_steps=int(args.max_steps),
        print_samples=int(args.print_samples),
    )

    if not left.e_n or not right.e_n:
        print("[FAIL] corner samples not found (corner_phase may never enter).")
        raise SystemExit(2)

    e_left = float(np.mean(left.e_n))
    e_right = float(np.mean(right.e_n))
    alpha_left = float(np.mean(np.abs(left.alpha)))
    alpha_right = float(np.mean(np.abs(right.alpha)))
    kappa_left = float(np.mean(left.kappa_los))
    kappa_right = float(np.mean(right.kappa_los))
    cap_left = float(np.mean(left.v_ratio_cap))
    cap_right = float(np.mean(right.v_ratio_cap))

    print(
        f"[SUMMARY] left:  mean_e_n={e_left:+.4f} mean|alpha|={alpha_left:.4f} mean_kappa={kappa_left:.4f} mean_cap={cap_left:.3f}"
    )
    print(
        f"[SUMMARY] right: mean_e_n={e_right:+.4f} mean|alpha|={alpha_right:.4f} mean_kappa={kappa_right:.4f} mean_cap={cap_right:.3f}"
    )

    e_diff = abs(e_left - e_right)
    alpha_diff = abs(alpha_left - alpha_right)
    kappa_diff = abs(kappa_left - kappa_right)
    cap_diff = abs(cap_left - cap_right)

    half = float(getattr(env, "half_epsilon", 1.0))
    e_ok = bool(e_diff > max(0.02, 0.20 * half))
    los_ok = bool(alpha_diff > math.radians(0.5) or kappa_diff > 0.05 or cap_diff > 0.02)

    if not (e_ok and los_ok):
        print(
            f"[FAIL] LOS sensitivity weak: e_diff={e_diff:.4f} (half={half:.4f}) "
            f"alpha_diff={alpha_diff:.4f} kappa_diff={kappa_diff:.4f} cap_diff={cap_diff:.4f}"
        )
        raise SystemExit(2)

    print("[PASS] LOS indicators vary with lateral offset (inner-cut sensitive).")

    print("[SAMPLE] info fields (alpha/L/kappa_los/v_ratio_cap/corner_phase/exit_boost_remaining):", left.sample_info or right.sample_info)

    ok_eval, reason = _quick_eval(cfg, seed=int(args.seed), episodes=int(args.quick_eval_episodes), kp_lat=float(args.kp_lat))
    if not ok_eval:
        print(f"[FAIL] quick_eval: {reason}")
        raise SystemExit(2)

    print("[PASS] quick_eval passed.")
    print("[PASS] P5.1 LOS debug passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()

