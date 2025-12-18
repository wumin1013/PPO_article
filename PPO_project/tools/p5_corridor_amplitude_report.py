"""P5.2 幅度自由报告：走廊奖励不再固定 e_target 后，拐角段 e_n 分布应不再收敛成单点。

按 10_P5_2 文档要求：
- 在 square 路径上跑 E=20 episode，corner_phase 内收集 e_n 样本并输出分位数/直方统计；
- 输出 e_n(或 inner=turn_sign*e_n) 与 P5.1 的 LOS 指标（kappa_los / v_ratio_cap）的相关性；
- 附带 quick_eval（line/square/s_shape，各 E=20）做回归，避免灾难性退化。

说明：本脚本不依赖训练模型；默认使用“每次入弯随机采样一个期望内切幅度”的安全控制器，
用于证明环境侧不再强制固定幅度，并且幅度变化会通过 LOS→cap 链路影响速度上限。
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


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, corridor_enabled: bool) -> Env:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    reward_weights = dict(cfg.get("reward_weights", {}) or {})

    corridor_cfg = (
        dict(reward_weights.get("corridor", {}) or {})
        if isinstance(reward_weights.get("corridor", {}), dict)
        else {}
    )
    corridor_cfg["enabled"] = bool(corridor_enabled)
    reward_weights["corridor"] = corridor_cfg

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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _theta_p_controller(env: Env, *, desired_e_n: float, e_n: float, kp: float) -> float:
    half = float(getattr(env, "half_epsilon", 1.0))
    return _clip(float(kp) * (float(desired_e_n) - float(e_n)) / max(half, 1e-6), -1.0, 1.0)


@dataclass
class Sample:
    e_n: float
    inner: float
    kappa_los: float
    v_ratio_cap: float
    v_ratio_exec: float
    margin_to_edge: float
    barrier_penalty: float


def _run_amplitude_episodes(
    env: Env,
    *,
    episodes: int,
    seed: int,
    kp: float,
    inner_min_frac: float,
    inner_max_frac: float,
    warmup_theta: float,
    warmup_steps_max: int,
    max_steps: int,
) -> List[Sample]:
    samples: List[Sample] = []

    for ep in range(int(episodes)):
        _set_seed(int(seed) + ep)
        env.reset()

        p4_pre = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        alpha0 = float(p4_pre.get("alpha", 0.0))
        warmup_sign = 1 if alpha0 >= 0.0 else -1

        frac = float(np.random.uniform(float(inner_min_frac), float(inner_max_frac)))
        warmup_steps = int(max(0, round(frac * float(max(int(warmup_steps_max), 0)))))

        action = np.array([float(warmup_sign) * abs(float(warmup_theta)), 1.0], dtype=float)
        done = False

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

            corner = bool(cs.get("corner_phase", False))
            turn_sign = int(cs.get("turn_sign", 0))
            in_corridor = bool(cs.get("in_corridor", False))
            e_n = float(cs.get("e_n", 0.0))

            if corner and turn_sign != 0 and in_corridor:
                v_ratio_exec = float(p4.get("v_ratio_exec", 0.0))
                v_ratio_cap = float(p4.get("v_ratio_cap", 1.0))
                if v_ratio_exec > v_ratio_cap + 1e-6:
                    raise RuntimeError(
                        f"[FAIL] v_ratio_exec exceeded v_ratio_cap (ep={ep}, step={step}) "
                        f"exec={v_ratio_exec:.6f} cap={v_ratio_cap:.6f}"
                    )

                samples.append(
                    Sample(
                        e_n=float(e_n),
                        inner=float(turn_sign) * float(e_n),
                        kappa_los=float(p4.get("kappa_los", p4.get("kappa", 0.0))),
                        v_ratio_cap=float(v_ratio_cap),
                        v_ratio_exec=float(v_ratio_exec),
                        margin_to_edge=float(cs.get("margin_to_edge", float("nan"))),
                        barrier_penalty=float(cs.get("barrier_penalty", 0.0)),
                    )
                )

            # 下一步动作：warmup 时固定转向；其余时间回到中心（desired_e_n=0）
            if step + 1 < int(warmup_steps):
                theta_u = float(warmup_sign) * abs(float(warmup_theta))
            else:
                theta_u = _theta_p_controller(env, desired_e_n=0.0, e_n=float(e_n), kp=float(kp))
            p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
            v_u = float(p4_next.get("speed_target", 0.8))
            action = np.array([_clip(theta_u, -1.0, 1.0), _clip(v_u, 0.0, 1.0)], dtype=float)

            if done:
                break

    return samples


@dataclass
class EpisodeMetrics:
    success: bool
    oob: bool
    steps: int
    v_mean: float
    rmse: float


def _run_eval_episode(env: Env, *, seed: int, kp_lat: float) -> EpisodeMetrics:
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
        theta_u = _theta_p_controller(env, desired_e_n=0.0, e_n=float(cs.get("e_n", 0.0)), kp=float(kp_lat))

        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_u = float(p4_next.get("speed_target", 0.8))
        action = np.array([theta_u, _clip(v_u, 0.0, 1.0)], dtype=float)

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
        env = _build_env(cfg, pm, corridor_enabled=True)
        results: List[EpisodeMetrics] = []
        for ep in range(int(episodes)):
            results.append(_run_eval_episode(env, seed=int(seed) + ep, kp_lat=float(kp_lat)))

        success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))
        oob_rate = float(np.mean([1.0 if r.oob else 0.0 for r in results]))
        steps_mean = float(np.mean([r.steps for r in results])) if results else 0.0
        v_mean = float(np.mean([r.v_mean for r in results])) if results else 0.0
        rmse_mean = float(np.mean([r.rmse for r in results])) if results else 0.0

        print(
            f"[EVAL] path={name} episodes={episodes} success_rate={success_rate:.3f} oob_rate={oob_rate:.3f} "
            f"rmse_mean={rmse_mean:.4f} steps_mean={steps_mean:.1f} v_mean={v_mean:.3f}"
        )

        if success_rate < 0.95:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} success_rate < 0.95")
        if oob_rate > 0.10:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} oob_rate > 0.10")
        if v_mean < 1e-6:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} v_mean ~= 0 (stuck)")

    return (all_ok, "ok" if all_ok else "failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="P5.2 走廊幅度自由报告（e_target 取消后的分布/相关性）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20, help="square 幅度统计 episode 数（默认 20）。")
    parser.add_argument("--kp", type=float, default=1.0, help="横向 P 控制系数（用于跟踪随机内切幅度）。")
    parser.add_argument("--inner-min-frac", type=float, default=0.05)
    parser.add_argument("--inner-max-frac", type=float, default=0.95)
    parser.add_argument("--warmup-theta", type=float, default=0.6, help="入弯短暂固定转向幅度（用于快速形成不同内切）。")
    parser.add_argument("--warmup-steps-max", type=int, default=18, help="入弯 warmup 的最大步数（按 inner_frac 线性缩放）。")
    parser.add_argument("--max-steps", type=int, default=220, help="每个 episode 的最大步数（默认 220）。")
    parser.add_argument("--quick-eval-episodes", type=int, default=20)
    parser.add_argument("--kp-lat", type=float, default=1.0, help="quick_eval 横向误差 P 控制系数。")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    pm = build_open_square(side=scale, num_points=num_points)
    env = _build_env(cfg, pm, corridor_enabled=True)

    samples = _run_amplitude_episodes(
        env,
        episodes=int(args.episodes),
        seed=int(args.seed),
        kp=float(args.kp),
        inner_min_frac=float(args.inner_min_frac),
        inner_max_frac=float(args.inner_max_frac),
        warmup_theta=float(args.warmup_theta),
        warmup_steps_max=int(args.warmup_steps_max),
        max_steps=int(args.max_steps),
    )

    if not samples:
        print("[FAIL] no corner samples collected (corner_phase may never enter).")
        raise SystemExit(2)

    e_n = np.asarray([s.e_n for s in samples], dtype=float)
    inner = np.asarray([s.inner for s in samples], dtype=float)
    kappa = np.asarray([s.kappa_los for s in samples], dtype=float)
    cap = np.asarray([s.v_ratio_cap for s in samples], dtype=float)
    margin = np.asarray([s.margin_to_edge for s in samples], dtype=float)
    barrier = np.asarray([s.barrier_penalty for s in samples], dtype=float)

    p10, p50, p90 = (float(np.nanpercentile(e_n, 10)), float(np.nanpercentile(e_n, 50)), float(np.nanpercentile(e_n, 90)))
    i10, i50, i90 = (
        float(np.nanpercentile(inner, 10)),
        float(np.nanpercentile(inner, 50)),
        float(np.nanpercentile(inner, 90)),
    )
    span = float(i90 - i10)
    half = float(getattr(env, "half_epsilon", 1.0))

    print(f"[AMP] corner samples={int(e_n.size)} half_epsilon={half:.4f}")
    print(f"[AMP] e_n percentiles: p10={p10:+.4f} p50={p50:+.4f} p90={p90:+.4f}")
    print(f"[AMP] inner(turn_sign*e_n) percentiles: p10={i10:.4f} p50={i50:.4f} p90={i90:.4f} span={span:.4f}")

    # 幅度自由（硬指标）：p10-p90 有明显跨度
    min_span = max(0.05 * half, 0.02)
    if not (span > min_span):
        print(f"[FAIL] amplitude span too small: span={span:.4f} (need > {min_span:.4f})")
        raise SystemExit(2)

    # 相关性：更内切(inner 更大)应统计上 kappa 更小 / cap 更大（方向以 inner 为准）
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        ok = np.isfinite(a) & np.isfinite(b)
        if int(ok.sum()) < 5:
            return float("nan")
        c = np.corrcoef(a[ok], b[ok])[0, 1]
        return float(c)

    corr_inner_kappa = _corr(inner, kappa)
    corr_inner_cap = _corr(inner, cap)
    print(f"[CORR] corr(inner, kappa_los)={corr_inner_kappa:.4f}  corr(inner, v_ratio_cap)={corr_inner_cap:.4f}")

    # 分桶均值（更直观）
    qs = np.nanpercentile(inner, [0, 25, 50, 75, 100]).tolist()
    print(f"[BINS] inner quantiles={', '.join([f'{q:.4f}' for q in qs])}")
    for lo, hi in zip(qs[:-1], qs[1:]):
        mask = np.isfinite(inner) & (inner >= float(lo)) & (inner <= float(hi)) & np.isfinite(kappa) & np.isfinite(cap)
        if int(mask.sum()) < 10:
            continue
        print(
            f"[BIN] inner∈[{float(lo):.4f},{float(hi):.4f}] n={int(mask.sum())} "
            f"mean_kappa={float(np.mean(kappa[mask])):.4f} mean_cap={float(np.mean(cap[mask])):.3f}"
        )

    # 附带打印：安全裕度/势垒惩罚分布（用于确认不再依赖 e_target）
    if np.isfinite(margin).any():
        print(
            f"[SAFE] margin_to_edge p10={float(np.nanpercentile(margin, 10)):.4f} p50={float(np.nanpercentile(margin, 50)):.4f} "
            f"p90={float(np.nanpercentile(margin, 90)):.4f}"
        )
    if np.isfinite(barrier).any():
        print(
            f"[SAFE] barrier_penalty p10={float(np.nanpercentile(barrier, 10)):.4f} p50={float(np.nanpercentile(barrier, 50)):.4f} "
            f"p90={float(np.nanpercentile(barrier, 90)):.4f}"
        )

    print("[PASS] amplitude freedom check passed.")

    ok_eval, reason = _quick_eval(cfg, seed=int(args.seed), episodes=int(args.quick_eval_episodes), kp_lat=float(args.kp_lat))
    if not ok_eval:
        print(f"[FAIL] quick_eval: {reason}")
        raise SystemExit(2)
    print("[PASS] quick_eval passed.")

    print("[PASS] P5.2 corridor amplitude report passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
