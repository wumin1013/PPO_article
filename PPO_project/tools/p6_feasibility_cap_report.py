"""P6.0 自动化验收：多预瞄 minimax + ω/ω̇/ω̈ 可达性 → 速度上限（v_ratio_cap）。

按 11_P6.0 文档要求，本脚本提供两类报告并做硬指标检查：
1) 时序报告（square 单回合）：输出 step/progress/v_ratio_exec/v_ratio_cap 以及三条边界的最小值，
   验证在急弯前 v_ratio_cap 会提前、平滑地下压（不是到拐点才突然掉）。
2) A/B 对比（square E=20）：
   - A：仅角速度边界（speed_cap_use_wdot=False, speed_cap_use_wddot=False）
   - B：开启 ω̇ + ω̈ 可达性（speed_cap_use_wdot=True, speed_cap_use_wddot=True）
   输出 success/oob/steps/v_mean 以及 angular_jerk_mean、kcm_intervention_mean。

同时提供 quick_eval（line/square/s_shape 各 E=20）做回归，避免灾难性退化。
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt  # noqa: E402

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

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


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


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


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, p4_override: Optional[dict]) -> Env:
    env_cfg = cfg.get("environment", {}) or {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) or {}
    reward_weights = dict(cfg.get("reward_weights", {}) or {})

    p4_cfg = dict(reward_weights.get("p4", {}) or {}) if isinstance(reward_weights.get("p4", {}), dict) else {}
    if p4_override:
        p4_cfg.update(dict(p4_override))
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


def _theta_p_controller(env: Env, corridor_status: Dict[str, object], *, kp: float) -> float:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    return _clip(float(kp) * (0.0 - float(e_n)) / max(half, 1e-6), -1.0, 1.0)


@dataclass
class EpisodeMetrics:
    success: bool
    oob: bool
    steps: int
    v_mean: float
    rmse: float
    angular_jerk_mean: float
    kcm_intervention_mean: float


@dataclass
class Metrics:
    episodes: int
    success_rate: float
    oob_rate: float
    steps_mean: float
    v_mean: float
    rmse_mean: float
    angular_jerk_mean: float
    kcm_intervention_mean: float


def _run_episode(env: Env, *, seed: int, kp_lat: float, max_steps: int) -> EpisodeMetrics:
    _set_seed(seed)
    env.reset()

    done = False
    info: Dict[str, object] = {}
    v_samples: List[float] = []
    e_samples: List[float] = []
    ang_jerk_samples: List[float] = []
    kcm_samples: List[float] = []

    p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
    action = np.array([0.0, float(p4_next.get("speed_target", 0.8))], dtype=float)

    for _ in range(int(max_steps)):
        _, _, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}

        v_samples.append(float(getattr(env, "velocity", 0.0)))
        e_samples.append(float(info.get("contour_error", 0.0)))
        ang_jerk_samples.append(float(getattr(env, "angular_jerk", 0.0)))
        kcm_samples.append(float(info.get("kcm_intervention", 0.0)))

        if done:
            break

        cs = info.get("corridor_status", {})
        if not isinstance(cs, dict):
            cs = {}
        theta_u = _theta_p_controller(env, cs, kp=float(kp_lat))

        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_u = float(p4_next.get("speed_target", 0.8))
        action = np.array([theta_u, _clip(v_u, 0.0, 1.0)], dtype=float)

    contour_error = float(info.get("contour_error", float("inf"))) if isinstance(info, dict) else float("inf")
    oob = bool(contour_error > float(getattr(env, "half_epsilon", 1.0)))
    success = bool(getattr(env, "reached_target", False))
    steps = int(getattr(env, "current_step", 0))

    v_mean = float(np.mean(v_samples)) if v_samples else 0.0
    rmse = float(np.sqrt(np.mean(np.square(e_samples)))) if e_samples else 0.0
    angular_jerk_mean = float(np.mean(np.abs(ang_jerk_samples))) if ang_jerk_samples else 0.0
    kcm_intervention_mean = float(np.mean(kcm_samples)) if kcm_samples else 0.0
    return EpisodeMetrics(
        success=bool(success),
        oob=bool(oob),
        steps=int(steps),
        v_mean=float(v_mean),
        rmse=float(rmse),
        angular_jerk_mean=float(angular_jerk_mean),
        kcm_intervention_mean=float(kcm_intervention_mean),
    )


def _run_eval(env: Env, *, episodes: int, seed: int, kp_lat: float, max_steps: int) -> Metrics:
    results: List[EpisodeMetrics] = []
    for ep in range(int(episodes)):
        results.append(_run_episode(env, seed=int(seed) + ep, kp_lat=float(kp_lat), max_steps=int(max_steps)))

    success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results])) if results else 0.0
    oob_rate = float(np.mean([1.0 if r.oob else 0.0 for r in results])) if results else 0.0
    steps_mean = float(np.mean([r.steps for r in results])) if results else 0.0
    v_mean = float(np.mean([r.v_mean for r in results])) if results else 0.0
    rmse_mean = float(np.mean([r.rmse for r in results])) if results else 0.0
    angular_jerk_mean = float(np.mean([r.angular_jerk_mean for r in results])) if results else 0.0
    kcm_intervention_mean = float(np.mean([r.kcm_intervention_mean for r in results])) if results else 0.0

    return Metrics(
        episodes=int(episodes),
        success_rate=float(success_rate),
        oob_rate=float(oob_rate),
        steps_mean=float(steps_mean),
        v_mean=float(v_mean),
        rmse_mean=float(rmse_mean),
        angular_jerk_mean=float(angular_jerk_mean),
        kcm_intervention_mean=float(kcm_intervention_mean),
    )


def _run_trace(env: Env, *, seed: int, kp_lat: float, max_steps: int) -> Dict[str, List[float]]:
    _set_seed(seed)
    env.reset()

    trace: Dict[str, List[float]] = {
        "step": [],
        "segment_idx": [],
        "progress": [],
        "dist_to_turn": [],
        "v_ratio_exec": [],
        "v_ratio_cap": [],
        "v_cap_w_min": [],
        "v_cap_wdot_min": [],
        "v_cap_wddot_min": [],
        "alpha_max_ahead": [],
        "kappa_max_ahead": [],
    }

    p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
    action = np.array([0.0, float(p4_next.get("speed_target", 0.8))], dtype=float)
    done = False

    for _ in range(int(max_steps)):
        _, _, done, info = env.step(action)
        if not isinstance(info, dict):
            info = {}
        p4 = info.get("p4_status", {})
        if not isinstance(p4, dict):
            p4 = {}

        trace["step"].append(float(info.get("step", len(trace["step"]))))
        trace["segment_idx"].append(float(info.get("segment_idx", float("nan"))))
        trace["progress"].append(float(info.get("progress", float("nan"))))
        trace["dist_to_turn"].append(float(p4.get("dist_to_turn", float("nan"))))
        trace["v_ratio_exec"].append(float(p4.get("v_ratio_exec", float("nan"))))
        trace["v_ratio_cap"].append(float(p4.get("v_ratio_cap", float("nan"))))
        trace["v_cap_w_min"].append(float(p4.get("v_cap_w_min", float("nan"))))
        trace["v_cap_wdot_min"].append(float(p4.get("v_cap_wdot_min", float("nan"))))
        trace["v_cap_wddot_min"].append(float(p4.get("v_cap_wddot_min", float("nan"))))
        trace["alpha_max_ahead"].append(float(p4.get("alpha_max_ahead", float("nan"))))
        trace["kappa_max_ahead"].append(float(p4.get("kappa_max_ahead", float("nan"))))

        if done:
            break

        cs = info.get("corridor_status", {})
        if not isinstance(cs, dict):
            cs = {}
        theta_u = _theta_p_controller(env, cs, kp=float(kp_lat))
        p4_next = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        v_u = float(p4_next.get("speed_target", 0.8))
        action = np.array([theta_u, _clip(v_u, 0.0, 1.0)], dtype=float)

    return trace


def _first_segment_slice(trace: Dict[str, List[float]]) -> slice:
    seg = trace.get("segment_idx", []) or []
    if not seg:
        return slice(0, 0)
    first = seg[0]
    if not math.isfinite(float(first)):
        return slice(0, len(seg))
    first_idx = int(first)
    for i, v in enumerate(seg):
        if not math.isfinite(float(v)):
            return slice(0, i)
        if int(v) != first_idx:
            return slice(0, i)
    return slice(0, len(seg))


def _check_cap_early_and_smooth(trace: Dict[str, List[float]]) -> Tuple[bool, str]:
    v = np.asarray(trace.get("v_ratio_cap", []), dtype=float)
    if v.size < 12 or not np.isfinite(v).any():
        return False, "trace too short"

    # 基于 v_ratio_cap 的“首次下压事件”做时序验收（不依赖 segment_idx / dist_to_turn 的语义）
    finite_head = v[: min(10, v.size)]
    finite_head = finite_head[np.isfinite(finite_head)]
    if finite_head.size == 0:
        return False, "trace head invalid"
    baseline = float(np.median(finite_head))
    thr_active = float(0.99 * baseline)

    start = next((i for i, val in enumerate(v.tolist()) if math.isfinite(float(val)) and float(val) < thr_active), None)
    if start is None:
        return False, "cap never drops below baseline (no event detected)"

    end = int(v.size)
    for j in range(int(start) + 1, int(v.size)):
        tail = v[j : j + 3]
        if tail.size >= 3 and np.all(np.isfinite(tail)) and bool(np.all(tail >= thr_active)):
            end = int(j)
            break

    window = np.asarray(v[int(start) : int(end)], dtype=float)
    window = window[np.isfinite(window)]
    if window.size < 6:
        return False, "cap drop window too short"

    pre = np.asarray(v[max(0, int(start) - 5) : int(start)], dtype=float)
    pre = pre[np.isfinite(pre)]
    v0 = float(np.median(pre)) if pre.size else float(baseline)

    vmin = float(np.min(window))
    total_drop = float(v0 - vmin)
    if total_drop < 0.03:
        return False, f"cap drop too small: drop={total_drop:.4f}"

    idx_vmin = int(np.argmin(window))
    thr = float(v0 - 0.30 * total_drop)  # 30% drop 作为“提前下降”阈值（相对指标）
    idx_first = int(np.where(window <= thr)[0][0]) if np.any(window <= thr) else idx_vmin
    lead_steps = int(idx_vmin - idx_first)
    if lead_steps < 3:
        return False, f"cap not early enough: lead_steps={lead_steps} (need >=3)"

    dv = np.diff(window)
    max_drop = float(np.max(np.clip(-dv, 0.0, None))) if dv.size else 0.0
    if max_drop > 0.70 * total_drop and max_drop > 0.12:
        return False, f"cap too spiky: max_drop={max_drop:.4f}, total_drop={total_drop:.4f}"

    return True, f"ok (start={int(start)}, end={int(end)}, lead_steps={lead_steps}, total_drop={total_drop:.4f}, max_drop={max_drop:.4f})"


def _write_csv(trace: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(trace.keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*[trace[k] for k in keys]):
            w.writerow(list(row))


def _plot_trace(trace: Dict[str, List[float]], out_path: Path, *, title: str) -> None:
    if not _HAS_MPL:
        return

    x = np.asarray(trace.get("dist_to_turn", []), dtype=float)
    v_cap = np.asarray(trace.get("v_ratio_cap", []), dtype=float)
    v_exec = np.asarray(trace.get("v_ratio_exec", []), dtype=float)

    ok_cap = np.isfinite(x) & np.isfinite(v_cap)
    ok_exec = np.isfinite(x) & np.isfinite(v_exec)

    fig, ax = plt.subplots(figsize=(8, 4))
    if ok_exec.any():
        ax.scatter(x[ok_exec], v_exec[ok_exec], s=10, alpha=0.60, label="v_ratio_exec")
    if ok_cap.any():
        ax.scatter(x[ok_cap], v_cap[ok_cap], s=10, alpha=0.45, label="v_ratio_cap")
    ax.set_xlabel("dist_to_turn (state)")
    ax.set_ylabel("velocity ratio")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _print_run_header(cfg: Dict, *, seed: int, episodes: int, kp_lat: float) -> None:
    env_cfg = cfg.get("environment", {}) or {}
    ppo_cfg = cfg.get("ppo", {}) or {}
    dt = float(env_cfg.get("interpolation_period", 0.1))
    gamma = float(ppo_cfg.get("gamma", 0.99))
    horizon_steps = float("inf") if gamma >= 1.0 else 1.0 / max(1e-12, 1.0 - gamma)
    horizon_time = float("inf") if gamma >= 1.0 else dt / max(1e-12, 1.0 - gamma)
    print(
        f"[RUN] seed={int(seed)} dt={dt} gamma={gamma:.6f} H_steps≈{horizon_steps:.1f} H_time≈{horizon_time:.3f} "
        f"episodes={int(episodes)} kp_lat={float(kp_lat)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P6.0 可控性速度上限报告/验收（多预瞄 + ω/ω̇/ω̈）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20, help="A/B 对比与 quick_eval 的 episode 数（默认 20）。")
    parser.add_argument("--kp-lat", type=float, default=1.0, help="横向误差 P 控制系数（默认 1.0）。")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 episode 的最大步数（默认用 YAML）。")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true", help="不生成 PNG（即使 matplotlib 可用）。")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    s_cfg = path_cfg.get("s_shape", {}) if isinstance(path_cfg.get("s_shape", {}), dict) else {}
    amplitude = float(s_cfg.get("amplitude", scale / 2.0))
    periods = float(s_cfg.get("periods", 2.0))

    max_steps = int(args.max_steps) if args.max_steps is not None else int(env_cfg.get("max_steps", 200))
    _print_run_header(cfg, seed=int(args.seed), episodes=int(args.episodes), kp_lat=float(args.kp_lat))

    paths = {
        "line": build_line(
            length=scale,
            num_points=num_points,
            angle=float(path_cfg.get("line", {}).get("angle", 0.0))
            if isinstance(path_cfg.get("line", {}), dict)
            else 0.0,
        ),
        "square": build_open_square(side=scale, num_points=num_points),
        "s_shape": build_s_shape(scale=scale, num_points=num_points, amplitude=amplitude, periods=periods),
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or (REPO_ROOT / "logs" / "p6_0_feasibility" / stamp)
    outdir.mkdir(parents=True, exist_ok=True)

    p4_a = {"speed_cap_use_wdot": False, "speed_cap_use_wddot": False}
    p4_b = {"speed_cap_use_wdot": True, "speed_cap_use_wddot": True}

    # 1) 时序报告（square, B）
    env_trace = _build_env(cfg, paths["square"], p4_override=p4_b)
    trace = _run_trace(env_trace, seed=int(args.seed), kp_lat=float(args.kp_lat), max_steps=int(max_steps))
    csv_path = outdir / "square_trace_modeB.csv"
    _write_csv(trace, csv_path)
    print(f"[OUT] {csv_path}")

    ok_cap, reason = _check_cap_early_and_smooth(trace)
    if ok_cap:
        print(f"[PASS] cap early/smooth check: {reason}")
    else:
        print(f"[FAIL] cap early/smooth check: {reason}")

    if not bool(args.no_plot):
        png_path = outdir / "square_v_ratio_vs_dist.png"
        _plot_trace(trace, png_path, title="P6.0 cap trace (square, mode=B)")
        if png_path.exists():
            print(f"[OUT] {png_path}")

    # 2) A/B 对比（square）
    env_a = _build_env(cfg, paths["square"], p4_override=p4_a)
    env_b = _build_env(cfg, paths["square"], p4_override=p4_b)
    m_a = _run_eval(env_a, episodes=int(args.episodes), seed=int(args.seed), kp_lat=float(args.kp_lat), max_steps=int(max_steps))
    m_b = _run_eval(env_b, episodes=int(args.episodes), seed=int(args.seed), kp_lat=float(args.kp_lat), max_steps=int(max_steps))

    print(
        "[EVAL] square mode=A(w)  "
        f"success_rate={m_a.success_rate:.3f} oob_rate={m_a.oob_rate:.3f} rmse_mean={m_a.rmse_mean:.4f} "
        f"steps_mean={m_a.steps_mean:.1f} v_mean={m_a.v_mean:.3f} "
        f"angular_jerk_mean={m_a.angular_jerk_mean:.4f} kcm_intervention_mean={m_a.kcm_intervention_mean:.4f}"
    )
    print(
        "[EVAL] square mode=B(w+wdot+wddot) "
        f"success_rate={m_b.success_rate:.3f} oob_rate={m_b.oob_rate:.3f} rmse_mean={m_b.rmse_mean:.4f} "
        f"steps_mean={m_b.steps_mean:.1f} v_mean={m_b.v_mean:.3f} "
        f"angular_jerk_mean={m_b.angular_jerk_mean:.4f} kcm_intervention_mean={m_b.kcm_intervention_mean:.4f}"
    )

    all_ok = True
    if not ok_cap:
        all_ok = False

    # 硬指标：稳定性不下降（B 相对 A）
    if m_b.success_rate + 1e-12 < m_a.success_rate:
        all_ok = False
        print("[FAIL] stability: B success_rate decreased vs A")
    if m_b.oob_rate - 1e-12 > m_a.oob_rate:
        all_ok = False
        print("[FAIL] stability: B oob_rate increased vs A")

    # 3) quick_eval（B）
    for name, pm in paths.items():
        env_q = _build_env(cfg, pm, p4_override=p4_b)
        m = _run_eval(env_q, episodes=int(args.episodes), seed=int(args.seed), kp_lat=float(args.kp_lat), max_steps=int(max_steps))
        print(
            f"[EVAL] path={name} mode=B episodes={int(args.episodes)} success_rate={m.success_rate:.3f} "
            f"oob_rate={m.oob_rate:.3f} rmse_mean={m.rmse_mean:.4f} steps_mean={m.steps_mean:.1f} v_mean={m.v_mean:.3f} "
            f"angular_jerk_mean={m.angular_jerk_mean:.4f} kcm_intervention_mean={m.kcm_intervention_mean:.4f}"
        )

        if m.success_rate < 0.95:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} success_rate < 0.95")
        if m.oob_rate > 0.10:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} oob_rate > 0.10")
        if m.v_mean < 1e-6:
            all_ok = False
            print(f"[FAIL] quick_eval: {name} v_mean ~= 0 (stuck)")

    if all_ok:
        print("[PASS] P6.0 feasibility cap acceptance passed.")
        raise SystemExit(0)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
