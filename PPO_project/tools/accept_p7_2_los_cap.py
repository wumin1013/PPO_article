"""P7.2 自动化验收：LOS 可控性边界（v_cap）+ 自适应预瞄。

验收依据：
- `P7_优化指令包/02_P7_2_LOS可控性边界_v4.md`

设计原则（KISS）：
- 不训练；用可复现的常量/启发式控制覆盖 4 项硬验收；
- 所有检查失败返回非 0 退出码，并输出 summary.json + 关键曲线图。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # PPO_project
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt  # noqa: E402

    from src.environment import Env  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python.cmd -m pip install -r PPO_project/requirements.txt")
    raise


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_outdir(outdir: Optional[Path]) -> Path:
    if outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ROOT / "out" / "p7_2" / f"{stamp}_accept"
    outdir = Path(outdir)
    return outdir if outdir.is_absolute() else (ROOT / outdir)


def build_line(length: float, num_points: int, angle: float = 0.0) -> List[np.ndarray]:
    ts = np.linspace(0.0, 1.0, max(2, int(num_points)))
    dx = math.cos(float(angle))
    dy = math.sin(float(angle))
    return [np.array([float(length) * t * dx, float(length) * t * dy], dtype=float) for t in ts]


def build_open_square(side: float, num_points: int) -> List[np.ndarray]:
    """Open square: only 3 edges to avoid near-closed finish-line issues."""
    if num_points < 10:
        raise ValueError("num_points too small for open_square")
    L = float(side)
    vertices = [
        np.array([0.0, 0.0], dtype=float),
        np.array([L, 0.0], dtype=float),
        np.array([L, L], dtype=float),
        np.array([0.0, L], dtype=float),
    ]
    per_edge = max(2, int(num_points) // 3)

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


def _build_env(cfg: Mapping, pm: Sequence[np.ndarray]) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    return Env(
        device="cpu",
        epsilon=float(env_cfg["epsilon"]),
        interpolation_period=float(env_cfg["interpolation_period"]),
        MAX_VEL=float(kcm_cfg["MAX_VEL"]),
        MAX_ACC=float(kcm_cfg["MAX_ACC"]),
        MAX_JERK=float(kcm_cfg["MAX_JERK"]),
        MAX_ANG_VEL=float(kcm_cfg["MAX_ANG_VEL"]),
        MAX_ANG_ACC=float(kcm_cfg["MAX_ANG_ACC"]),
        MAX_ANG_JERK=float(kcm_cfg["MAX_ANG_JERK"]),
        Pm=list(pm),
        max_steps=int(env_cfg["max_steps"]),
        lookahead_points=int(env_cfg.get("lookahead_points", 5)),
        reward_weights=cfg.get("reward_weights", {}),
    )


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_trace_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _plot_series(
    *,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float]]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for name, ys in series:
        ax.plot(xs[: len(ys)], ys, lw=1.2, label=name)
    ax.grid(True, ls=":", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _p50_last_half(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    half = max(1, len(values) // 2)
    tail = _finite_array(values[-half:])
    if tail.size == 0:
        return float("nan")
    return float(np.nanpercentile(tail, 50))


def _first_drop_step(values: Sequence[float], *, threshold: float) -> Optional[int]:
    for i, v in enumerate(values, start=1):
        if math.isfinite(v) and v < float(threshold):
            return i
    return None


def _mean_in_progress_window(
    progress: Sequence[float],
    values: Sequence[float],
    *,
    start: float,
    end: float,
) -> float:
    if not progress or not values:
        return float("nan")
    prog = np.asarray(list(progress), dtype=float)
    vals = np.asarray(list(values), dtype=float)
    n = int(min(prog.size, vals.size))
    if n <= 0:
        return float("nan")
    prog = prog[:n]
    vals = vals[:n]
    mask = np.isfinite(prog) & np.isfinite(vals) & (prog >= float(start)) & (prog <= float(end))
    if not bool(np.any(mask)):
        return float("nan")
    return float(np.nanmean(vals[mask]))


def _corner_progress_open_square(pm: Sequence[np.ndarray], *, side: float) -> float:
    """第一拐角（L,0）在路径弧长上的 progress 比例。"""
    pts = [np.asarray(p, dtype=float) for p in pm]
    if len(pts) < 2:
        return 0.0
    corner = np.array([float(side), 0.0], dtype=float)
    idx = int(np.argmin([float(np.linalg.norm(p - corner)) for p in pts]))
    seg_lens = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
    total = float(sum(seg_lens)) if seg_lens else 1.0
    s_corner = float(sum(seg_lens[:idx])) if idx > 0 else 0.0
    return float(np.clip(s_corner / max(total, 1e-6), 0.0, 1.0))


@dataclass(frozen=True)
class EpisodeTrace:
    steps: int
    nan_count: int
    cap_violation_count: int
    omega_violation_count: int
    progress: List[float]
    v_ratio_cap: List[float]
    v_ratio_exec: List[float]
    alpha: List[float]
    d_target: List[float]
    d_chosen: List[float]
    kappa: List[float]


PolicyFn = Callable[[Env, Dict[str, float], int], np.ndarray]


def _run_episode(
    *,
    env: Env,
    policy: PolicyFn,
    seed: int,
    trace_rows: List[MutableMapping[str, object]],
    tag: str,
) -> EpisodeTrace:
    _set_seed(seed)
    env.reset()

    max_ang_vel = float(getattr(env, "MAX_ANG_VEL", 0.0))

    progress: List[float] = []
    v_ratio_cap: List[float] = []
    v_ratio_exec: List[float] = []
    alpha: List[float] = []
    d_target: List[float] = []
    d_chosen: List[float] = []
    kappa: List[float] = []

    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0

    done = False
    step_idx = 0
    while not done:
        step_idx += 1
        # 用 env 内部计算的 p4 状态驱动 policy（无需 RL）
        p4_pre = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        action = policy(env, p4_pre, step_idx)

        obs, reward, done, info = env.step(action)
        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}

        v_exec = float(p4.get("v_exec", float(getattr(env, "velocity", 0.0))))
        omega_exec = float(p4.get("omega_exec", float(getattr(env, "angular_vel", 0.0))))
        max_vel_cap_phys = float(p4.get("max_vel_cap_phys", float(getattr(env, "MAX_VEL", 0.0))))

        v_ratio_cap_i = float(p4.get("v_ratio_cap", float("nan")))
        v_ratio_exec_i = float(p4.get("v_ratio_exec", float("nan")))
        alpha_i = float(p4.get("alpha", float("nan")))
        d_target_i = float(p4.get("d_target", float("nan")))
        d_chosen_i = float(p4.get("d_chosen", float("nan")))
        kappa_i = float(p4.get("kappa", float("nan")))
        prog_i = float(getattr(env, "state", np.array([0.0, 0.0, 0.0, 0.0, 0.0]))[4])

        progress.append(prog_i)
        v_ratio_cap.append(v_ratio_cap_i)
        v_ratio_exec.append(v_ratio_exec_i)
        alpha.append(alpha_i)
        d_target.append(d_target_i)
        d_chosen.append(d_chosen_i)
        kappa.append(kappa_i)

        if v_exec > max_vel_cap_phys + 1e-6:
            cap_violation_count += 1
        if abs(omega_exec) > max_ang_vel + 1e-6:
            omega_violation_count += 1

        if not isinstance(obs, np.ndarray):
            nan_count += 1
        else:
            nan_count += int(obs.size - int(np.count_nonzero(np.isfinite(obs))))
        if not math.isfinite(float(reward)):
            nan_count += 1

        trace_rows.append(
            {
                "tag": tag,
                "seed": int(seed),
                "step": int(step_idx),
                "progress": float(prog_i),
                "v_ratio_cap": float(v_ratio_cap_i),
                "v_ratio_exec": float(v_ratio_exec_i),
                "alpha": float(alpha_i),
                "d_target": float(d_target_i),
                "d_chosen": float(d_chosen_i),
                "kappa": float(kappa_i),
                "v_exec": float(v_exec),
                "omega_exec": float(omega_exec),
                "max_vel_cap_phys": float(max_vel_cap_phys),
                "contour_error": float(info.get("contour_error", float("nan"))) if isinstance(info, dict) else float("nan"),
                "reward": float(reward),
            }
        )

        if step_idx > int(getattr(env, "max_steps", 2000)) + 50:
            break

    return EpisodeTrace(
        steps=int(step_idx),
        nan_count=int(nan_count),
        cap_violation_count=int(cap_violation_count),
        omega_violation_count=int(omega_violation_count),
        progress=progress,
        v_ratio_cap=v_ratio_cap,
        v_ratio_exec=v_ratio_exec,
        alpha=alpha,
        d_target=d_target,
        d_chosen=d_chosen,
        kappa=kappa,
    )


def _policy_constant(theta: float, v_ratio: float) -> PolicyFn:
    def fn(_env: Env, _p4: Dict[str, float], _step: int) -> np.ndarray:
        return np.array([float(theta), float(v_ratio)], dtype=float)

    return fn


def _policy_turn_track_los(*, v_ratio: float, progress_gate: Optional[float], alpha_gate: Optional[float]) -> PolicyFn:
    def fn(env: Env, p4: Dict[str, float], _step: int) -> np.ndarray:
        theta_u = 0.0
        prog = 0.0
        if getattr(env, "state", None) is not None and len(env.state) > 4:
            prog = float(env.state[4])
        if progress_gate is not None and prog < float(progress_gate):
            theta_u = 0.0
        else:
            alpha = float(p4.get("alpha", 0.0))
            if alpha_gate is not None and abs(alpha) < float(alpha_gate):
                theta_u = 0.0
            else:
                dt = float(getattr(env, "interpolation_period", 0.1))
                max_ang_vel = float(getattr(env, "MAX_ANG_VEL", 1.0))
                theta_u = float(np.clip(alpha / max(max_ang_vel * dt, 1e-6), -1.0, 1.0))
        return np.array([float(theta_u), float(v_ratio)], dtype=float)

    return fn


def main() -> None:
    parser = argparse.ArgumentParser(description="P7.2 自动化验收（LOS cap + 自适应预瞄）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "p7_2_accept.yaml",
        help="用于构建环境/约束默认值的 YAML（默认 configs/original_configs/p7_2_accept.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env_cfg = cfg.get("environment", {}) or {}
    path_cfg = cfg.get("path", {}) or {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    dt = float(env_cfg.get("interpolation_period", 0.1))

    episodes = max(1, int(args.episodes))
    seed0 = int(args.seed)

    trace_rows: List[MutableMapping[str, object]] = []

    # === Test 1: open_line cap 放开 ===
    line = build_line(length=scale, num_points=num_points, angle=float(path_cfg.get("line", {}).get("angle", 0.0)) if isinstance(path_cfg.get("line", {}), dict) else 0.0)
    env_line = _build_env(cfg, line)
    line_p50_caps: List[float] = []
    line_nan = 0
    line_cap_viol = 0
    line_omega_viol = 0
    for ep in range(episodes):
        tr = _run_episode(
            env=env_line,
            policy=_policy_constant(theta=0.0, v_ratio=1.0),
            seed=seed0 + ep,
            trace_rows=trace_rows if ep == 0 else [],
            tag="t1_line",
        )
        line_p50_caps.append(_p50_last_half(tr.v_ratio_cap))
        line_nan += tr.nan_count
        line_cap_viol += tr.cap_violation_count
        line_omega_viol += tr.omega_violation_count
    line_cap_p50 = float(np.nanmedian(np.asarray(line_p50_caps, dtype=float)))
    t1_ok = bool(math.isfinite(line_cap_p50) and line_cap_p50 >= 0.95)

    # === Test 2: open_square 弯前提前下压 ===
    square = build_open_square(side=scale, num_points=num_points)
    env_square = _build_env(cfg, square)
    corner_prog = _corner_progress_open_square(square, side=scale)
    tr2 = _run_episode(
        env=env_square,
        # 用直行逼近拐角：更稳定地产生“弯前提前下压”（避免小 epsilon 下过早越界终止）
        policy=_policy_constant(theta=0.0, v_ratio=0.9),
        seed=seed0 + 1000,
        trace_rows=trace_rows,
        tag="t2_square",
    )
    drop_step2 = _first_drop_step(tr2.v_ratio_cap, threshold=0.98)
    drop_prog2 = tr2.progress[drop_step2 - 1] if drop_step2 is not None and drop_step2 - 1 < len(tr2.progress) else float("nan")
    t2_ok = bool(math.isfinite(drop_prog2) and drop_prog2 < float(corner_prog))

    # === Test 3: 自适应预瞄一致性（0.4 vs 0.9） ===
    env_sq_low = _build_env(cfg, square)
    env_sq_high = _build_env(cfg, square)
    tr3_low = _run_episode(
        env=env_sq_low,
        policy=_policy_constant(theta=0.0, v_ratio=0.4),
        seed=seed0 + 2000,
        trace_rows=trace_rows,
        tag="t3_low",
    )
    tr3_high = _run_episode(
        env=env_sq_high,
        policy=_policy_constant(theta=0.0, v_ratio=0.9),
        seed=seed0 + 2001,
        trace_rows=trace_rows,
        tag="t3_high",
    )
    d_med_low = float(np.nanmedian(_finite_array(tr3_low.d_target))) if tr3_low.d_target else float("nan")
    d_med_high = float(np.nanmedian(_finite_array(tr3_high.d_target))) if tr3_high.d_target else float("nan")
    d_target_ok = bool(math.isfinite(d_med_low) and math.isfinite(d_med_high) and d_med_high > d_med_low + 1e-3)

    drop_step_low = _first_drop_step(tr3_low.v_ratio_cap, threshold=0.98) or 10**9
    drop_step_high = _first_drop_step(tr3_high.v_ratio_cap, threshold=0.98) or 10**9
    cap_earlier_ok = bool(drop_step_high < drop_step_low)
    t3_ok = bool(d_target_ok and cap_earlier_ok)

    # === Test 4: 内切贡献（早转向 vs 晚转向） ===
    env_early = _build_env(cfg, square)
    env_late = _build_env(cfg, square)
    # 晚转向：尽量贴近角点再转，增强与 early 的差异
    late_gate_progress = float(max(0.0, float(corner_prog) - 0.01))
    tr4_early = _run_episode(
        env=env_early,
        policy=_policy_turn_track_los(v_ratio=0.9, progress_gate=None, alpha_gate=None),
        seed=seed0 + 3000,
        trace_rows=trace_rows,
        tag="t4_early",
    )
    tr4_late = _run_episode(
        env=env_late,
        policy=_policy_turn_track_los(v_ratio=0.9, progress_gate=late_gate_progress, alpha_gate=None),
        seed=seed0 + 3001,
        trace_rows=trace_rows,
        tag="t4_late",
    )
    # cap 的“内切贡献”主要出现在拐角附近：用 progress 窗口统计更稳定
    t4_window_start = float(max(0.0, float(corner_prog) - 0.12))
    t4_window_end = float(min(1.0, float(corner_prog) + 0.02))
    mean_cap_early = _mean_in_progress_window(
        tr4_early.progress,
        tr4_early.v_ratio_cap,
        start=t4_window_start,
        end=t4_window_end,
    )
    mean_cap_late = _mean_in_progress_window(
        tr4_late.progress,
        tr4_late.v_ratio_cap,
        start=t4_window_start,
        end=t4_window_end,
    )
    if not (math.isfinite(mean_cap_early) and math.isfinite(mean_cap_late)):
        n = int(min(len(tr4_early.v_ratio_cap), len(tr4_late.v_ratio_cap), 50))
        mean_cap_early = float(np.nanmean(_finite_array(tr4_early.v_ratio_cap[:n]))) if n > 0 else float("nan")
        mean_cap_late = float(np.nanmean(_finite_array(tr4_late.v_ratio_cap[:n]))) if n > 0 else float("nan")
    cap_gain = float(mean_cap_early - mean_cap_late) if math.isfinite(mean_cap_early) and math.isfinite(mean_cap_late) else float("nan")
    t4_ok = bool(math.isfinite(cap_gain) and cap_gain >= 0.1)

    nan_total = int(line_nan + tr2.nan_count + tr3_low.nan_count + tr3_high.nan_count + tr4_early.nan_count + tr4_late.nan_count)
    cap_viol_total = int(line_cap_viol + tr2.cap_violation_count + tr3_low.cap_violation_count + tr3_high.cap_violation_count + tr4_early.cap_violation_count + tr4_late.cap_violation_count)
    omega_viol_total = int(line_omega_viol + tr2.omega_violation_count + tr3_low.omega_violation_count + tr3_high.omega_violation_count + tr4_early.omega_violation_count + tr4_late.omega_violation_count)

    summary = {
        "config": str(args.config),
        "episodes": int(episodes),
        "seed": int(seed0),
        "dt": float(dt),
        "tests": {
            "t1_line_cap_open": bool(t1_ok),
            "t2_square_cap_early_drop": bool(t2_ok),
            "t3_adaptive_preview": bool(t3_ok),
            "t4_inner_cut_coupling": bool(t4_ok),
        },
        "metrics": {
            "t1_line_v_ratio_cap_p50_last_half_median": float(line_cap_p50),
            "t2_corner_progress": float(corner_prog),
            "t2_drop_step": int(drop_step2 or -1),
            "t2_drop_progress": float(drop_prog2),
            "t3_d_target_median_low": float(d_med_low),
            "t3_d_target_median_high": float(d_med_high),
            "t3_drop_step_low": int(drop_step_low if drop_step_low != 10**9 else -1),
            "t3_drop_step_high": int(drop_step_high if drop_step_high != 10**9 else -1),
            "t4_mean_cap_early": float(mean_cap_early),
            "t4_mean_cap_late": float(mean_cap_late),
            "t4_cap_gain": float(cap_gain),
            "t4_corner_progress": float(corner_prog),
            "t4_late_progress_gate": float(late_gate_progress),
            "t4_window_start": float(t4_window_start) if "t4_window_start" in locals() else float("nan"),
            "t4_window_end": float(t4_window_end) if "t4_window_end" in locals() else float("nan"),
        },
        "redlines": {
            "nan_count": int(nan_total),
            "cap_violation_count": int(cap_viol_total),
            "omega_violation_count": int(omega_viol_total),
        },
    }
    _write_json(outdir / "summary.json", summary)
    _write_trace_csv(outdir / "trace.csv", trace_rows)

    # 输出要求：cap_vs_step / delta_vs_step / d_target_vs_step（用 t2_square 的代表回合）
    xs = list(range(1, tr2.steps + 1))
    _plot_series(
        xs=xs,
        series=[("v_ratio_cap", tr2.v_ratio_cap), ("v_ratio_exec", tr2.v_ratio_exec)],
        out_path=outdir / "cap_vs_step.png",
        title="open_square cap vs exec",
        ylabel="ratio",
    )
    _plot_series(
        xs=xs,
        series=[("alpha(delta)", tr2.alpha)],
        out_path=outdir / "delta_vs_step.png",
        title="open_square LOS delta (alpha)",
        ylabel="rad",
    )
    _plot_series(
        xs=xs,
        series=[("d_target", tr2.d_target), ("d_chosen", tr2.d_chosen)],
        out_path=outdir / "d_target_vs_step.png",
        title="open_square adaptive lookahead",
        ylabel="m",
    )

    ok = bool(t1_ok and t2_ok and t3_ok and t4_ok and nan_total == 0 and cap_viol_total == 0 and omega_viol_total == 0)
    if ok:
        print("[PASS] P7.2 acceptance passed.")
        print(f"[OUT] {outdir}")
        raise SystemExit(0)

    print("[FAIL] P7.2 acceptance failed.")
    print(f"[OUT] {outdir}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
