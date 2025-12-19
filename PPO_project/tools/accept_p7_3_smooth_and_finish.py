"""P7.3 自动化验收：平滑与终点可靠性（kappa 奇异点保护 + stall/cap/phase 联动 + open finish 可靠）。

验收依据：
- `P7_优化指令包/04_P7_3_平滑与终点可靠性_v4.md`

设计原则（KISS）：
- 不训练；用可复现的启发式控制覆盖 smooth/finish/numerical safety；
- 同脚本内对比 baseline（关闭 kappa_smoothing）与 smooth（开启）；
- 输出 summary.json + trace.csv + 曲线图，便于审计/回归；
- 任一硬性门槛失败返回非 0。
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
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
        return ROOT / "out" / "p7_3" / f"{stamp}_accept"
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
    field_set = set()
    for row in rows:
        field_set.update(row.keys())
    fieldnames = sorted(field_set)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


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


def _plot_path(*, pm: Sequence[np.ndarray], traj: Sequence[np.ndarray], pl: Sequence, pr: Sequence, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    pm_arr = np.asarray([np.asarray(p, dtype=float) for p in pm], dtype=float)
    ax.plot(pm_arr[:, 0], pm_arr[:, 1], "k--", lw=1.0, label="Pm")
    if pl:
        pl_arr = np.asarray([np.asarray(p, dtype=float) for p in pl if p is not None], dtype=float)
        if pl_arr.size:
            ax.plot(pl_arr[:, 0], pl_arr[:, 1], "g-", lw=0.8, alpha=0.8, label="Pl")
    if pr:
        pr_arr = np.asarray([np.asarray(p, dtype=float) for p in pr if p is not None], dtype=float)
        if pr_arr.size:
            ax.plot(pr_arr[:, 0], pr_arr[:, 1], "g-", lw=0.8, alpha=0.8, label="Pr")
    if traj:
        tr = np.asarray([np.asarray(p, dtype=float) for p in traj], dtype=float)
        ax.plot(tr[:, 0], tr[:, 1], "r-", lw=1.4, label="traj")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", lw=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _kappa_from_exec(*, omega_exec: float, v_exec: float, max_vel: float) -> float:
    v_eps = float(max(1e-6 * float(max_vel), 1e-12))
    denom = float(max(0.0, float(v_exec)) + v_eps)
    if denom <= 0.0:
        return 0.0
    return float(abs(float(omega_exec)) / denom)


def _p95(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.nanpercentile(np.asarray(vals, dtype=float), 95))


@dataclass(frozen=True)
class EpisodeTrace:
    case: str
    steps: int
    done_reason: str
    nan_count: int
    cap_violation_count: int
    omega_violation_count: int
    progress: List[float]
    v_ratio_exec: List[float]
    v_ratio_cap: List[float]
    alpha: List[float]
    kappa_exec: List[float]
    dkappa_exec: List[float]


def _policy_line(_env: Env, _p4: Dict[str, float]) -> np.ndarray:
    return np.array([0.0, 1.0], dtype=float)


def _policy_square(env: Env, p4: Dict[str, float]) -> np.ndarray:
    # LOS 跟随：alpha->theta_u；|alpha| 越大适当降速
    alpha = float(p4.get("alpha", 0.0))
    alpha_scale = float(math.radians(70.0))
    theta_u = float(np.clip(alpha / max(alpha_scale, 1e-6), -1.0, 1.0))

    a_norm = float(min(abs(alpha) / max(float(math.radians(90.0)), 1e-6), 1.0))
    v_ratio = float(0.88 - 0.33 * a_norm)
    v_ratio = float(np.clip(v_ratio, 0.35, 1.0))

    # 边界保护：接近越界时优先回正（避免 OOB 影响 smooth 指标）
    half_eps = float(getattr(env, "half_epsilon", 0.0))
    if half_eps > 1e-6:
        proj, _, _s, _t_hat, n_hat = env._project_onto_progress_path(env.current_position)  # type: ignore[attr-defined]
        e_n = float(np.dot(env.current_position - proj, n_hat))
        if abs(e_n) > 0.90 * half_eps:
            theta_u = -1.0 if e_n > 0.0 else 1.0
            v_ratio = float(min(v_ratio, 0.30))

    return np.array([float(theta_u), float(v_ratio)], dtype=float)


def _run_episode(
    *,
    env: Env,
    seed: int,
    case: str,
    tag: str,
    trace_rows: List[MutableMapping[str, object]],
    policy,
) -> EpisodeTrace:
    _set_seed(seed)
    env.reset()
    done = False
    step_idx = 0

    max_ang_vel = float(getattr(env, "MAX_ANG_VEL", 0.0))
    max_vel = float(getattr(env, "MAX_VEL", 0.0))

    progress: List[float] = []
    v_ratio_exec: List[float] = []
    v_ratio_cap: List[float] = []
    alpha: List[float] = []
    kappa_exec: List[float] = []
    dkappa_exec: List[float] = []

    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0

    while not done:
        step_idx += 1
        p4_pre = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        action = policy(env, p4_pre)
        obs, reward, done, info = env.step(action)

        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}

        v_exec = float(p4.get("v_exec", float(getattr(env, "velocity", 0.0))))
        omega_exec = float(p4.get("omega_exec", float(getattr(env, "angular_vel", 0.0))))
        max_vel_cap_phys = float(p4.get("max_vel_cap_phys", float(getattr(env, "MAX_VEL", 0.0))))

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

        prog = float(getattr(env, "state", np.array([0.0, 0.0, 0.0, 0.0, 0.0]))[4])
        v_ratio_exec_i = float(p4.get("v_ratio_exec", float("nan")))
        v_ratio_cap_i = float(p4.get("v_ratio_cap", float("nan")))
        alpha_i = float(p4.get("alpha", float("nan")))

        kappa_i = p4.get("kappa_exec", None)
        if kappa_i is None:
            kappa_i = _kappa_from_exec(omega_exec=omega_exec, v_exec=v_exec, max_vel=max_vel)
        kappa_i = float(kappa_i)

        dkappa_i = p4.get("dkappa_exec", None)
        if dkappa_i is None:
            if kappa_exec:
                dkappa_i = abs(kappa_i - kappa_exec[-1])
            else:
                dkappa_i = 0.0
        dkappa_i = float(dkappa_i)

        progress.append(float(prog))
        v_ratio_exec.append(float(v_ratio_exec_i))
        v_ratio_cap.append(float(v_ratio_cap_i))
        alpha.append(float(alpha_i))
        kappa_exec.append(float(kappa_i))
        dkappa_exec.append(float(dkappa_i))

        trace_rows.append(
            {
                "tag": tag,
                "case": case,
                "seed": int(seed),
                "step": int(step_idx),
                "progress": float(prog),
                "v_ratio_exec": float(v_ratio_exec_i),
                "v_ratio_cap": float(v_ratio_cap_i),
                "alpha": float(alpha_i),
                "v_exec": float(v_exec),
                "omega_exec": float(omega_exec),
                "kappa_exec": float(kappa_i),
                "dkappa_exec": float(dkappa_i),
                "contour_error": float(info.get("contour_error", float("nan"))) if isinstance(info, dict) else float("nan"),
                "reward": float(reward),
                "done": int(bool(done)),
            }
        )

        if step_idx > int(getattr(env, "max_steps", 2000)) + 50:
            break

    # done reason
    done_reason = "unknown"
    success = bool(getattr(env, "reached_target", False))
    stall = bool(getattr(env, "_p4_stall_triggered", False))
    if isinstance(info, dict):
        contour_error = float(info.get("contour_error", float(env.get_contour_error(env.current_position))))
    else:
        contour_error = float(env.get_contour_error(env.current_position))
    if success:
        done_reason = "success"
    elif stall:
        done_reason = "stall"
    elif math.isfinite(contour_error) and contour_error > float(getattr(env, "half_epsilon", 0.0)):
        done_reason = "oob"
    elif int(getattr(env, "current_step", 0)) >= int(getattr(env, "max_steps", 0)):
        done_reason = "max_steps"

    return EpisodeTrace(
        case=str(case),
        steps=int(step_idx),
        done_reason=str(done_reason),
        nan_count=int(nan_count),
        cap_violation_count=int(cap_violation_count),
        omega_violation_count=int(omega_violation_count),
        progress=progress,
        v_ratio_exec=v_ratio_exec,
        v_ratio_cap=v_ratio_cap,
        alpha=alpha,
        kappa_exec=kappa_exec,
        dkappa_exec=dkappa_exec,
    )


def _override_p7_3_smoothing(cfg: Mapping, *, enabled: bool) -> Dict:
    cfg2 = json.loads(json.dumps(cfg))  # deep copy (YAML-friendly primitives)
    rw = cfg2.get("reward_weights", {})
    if not isinstance(rw, dict):
        rw = {}
    p7_3 = rw.get("p7_3", {})
    if not isinstance(p7_3, dict):
        p7_3 = {}
    p7_3["kappa_smoothing_enabled"] = bool(enabled)
    rw["p7_3"] = p7_3
    cfg2["reward_weights"] = rw
    return cfg2


def main() -> None:
    parser = argparse.ArgumentParser(description="P7.3 自动化验收（平滑 + 终点可靠性）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "p7_3_accept.yaml",
        help="用于构建环境/约束默认值的 YAML（默认 configs/original_configs/p7_3_accept.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--dkappa_reduction_ratio", type=float, default=0.50, help="相对 baseline 的 dkappa_p95 降幅门槛（默认 50%%）。")
    parser.add_argument("--dkappa_abs_threshold", type=float, default=0.18, help="绝对 dkappa_p95 门槛（baseline 不可用时兜底）。")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(Path(args.config))
    episodes = int(max(1, int(args.episodes)))
    seed0 = int(args.seed)

    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    path_cfg = cfg.get("path", {}) if isinstance(cfg.get("path", {}), dict) else {}
    scale = float(path_cfg.get("scale", 3.0))
    num_points = int(path_cfg.get("num_points", 80))

    pm_line = build_line(length=scale, num_points=num_points)
    pm_sq = build_open_square(side=scale, num_points=num_points)

    trace_rows: List[MutableMapping[str, object]] = []

    # --- Baseline: smoothing disabled (P7.2-like) ---
    cfg_base = _override_p7_3_smoothing(cfg, enabled=False)
    base_sq_dkappa: List[float] = []
    base_sq_dkappa_corner: List[float] = []

    base_nan = 0
    base_cap_viol = 0
    base_omega_viol = 0

    for ep in range(episodes):
        env = _build_env(cfg_base, pm_sq)
        tr = _run_episode(env=env, seed=seed0 + 1000 + ep, case="open_square", tag="baseline", trace_rows=trace_rows, policy=_policy_square)
        base_nan += tr.nan_count
        base_cap_viol += tr.cap_violation_count
        base_omega_viol += tr.omega_violation_count
        base_sq_dkappa.extend([float(v) for v in tr.dkappa_exec if math.isfinite(float(v))])
        for v, a in zip(tr.dkappa_exec, tr.alpha):
            if math.isfinite(float(v)) and math.isfinite(float(a)) and abs(float(a)) >= math.radians(15.0):
                base_sq_dkappa_corner.append(float(v))

    base_dkappa_p95 = float(_p95(base_sq_dkappa))
    base_dkappa_p95_corner = float(_p95(base_sq_dkappa_corner))

    # --- Smooth: smoothing enabled ---
    cfg_smooth = _override_p7_3_smoothing(cfg, enabled=True)
    smooth_nan = 0
    smooth_cap_viol = 0
    smooth_omega_viol = 0

    # Finish metrics
    line_reasons: List[str] = []
    sq_reasons: List[str] = []

    smooth_sq_kappa: List[float] = []
    smooth_sq_dkappa: List[float] = []
    smooth_sq_dkappa_corner: List[float] = []

    rep_env: Optional[Env] = None
    rep_trace: Optional[EpisodeTrace] = None

    for ep in range(episodes):
        env_line = _build_env(cfg_smooth, pm_line)
        tr_line = _run_episode(env=env_line, seed=seed0 + ep, case="open_line", tag="smooth", trace_rows=trace_rows, policy=_policy_line)
        line_reasons.append(tr_line.done_reason)
        smooth_nan += tr_line.nan_count
        smooth_cap_viol += tr_line.cap_violation_count
        smooth_omega_viol += tr_line.omega_violation_count

        env_sq = _build_env(cfg_smooth, pm_sq)
        tr_sq = _run_episode(env=env_sq, seed=seed0 + 1000 + ep, case="open_square", tag="smooth", trace_rows=trace_rows, policy=_policy_square)
        sq_reasons.append(tr_sq.done_reason)
        smooth_nan += tr_sq.nan_count
        smooth_cap_viol += tr_sq.cap_violation_count
        smooth_omega_viol += tr_sq.omega_violation_count

        smooth_sq_kappa.extend([float(v) for v in tr_sq.kappa_exec if math.isfinite(float(v))])
        smooth_sq_dkappa.extend([float(v) for v in tr_sq.dkappa_exec if math.isfinite(float(v))])
        for v, a in zip(tr_sq.dkappa_exec, tr_sq.alpha):
            if math.isfinite(float(v)) and math.isfinite(float(a)) and abs(float(a)) >= math.radians(15.0):
                smooth_sq_dkappa_corner.append(float(v))

        if rep_env is None:
            rep_env = env_sq
            rep_trace = tr_sq

    def _rate(reasons: Sequence[str], target: str) -> float:
        if not reasons:
            return float("nan")
        return float(sum(1 for r in reasons if r == target) / max(1, len(reasons)))

    line_success_rate = float(_rate(line_reasons, "success"))
    line_stall_rate = float(_rate(line_reasons, "stall"))
    line_oob_rate = float(_rate(line_reasons, "oob"))

    sq_success_rate = float(_rate(sq_reasons, "success"))
    sq_stall_rate = float(_rate(sq_reasons, "stall"))
    sq_oob_rate = float(_rate(sq_reasons, "oob"))

    smooth_kappa_max = float(np.nanmax(np.asarray(smooth_sq_kappa, dtype=float))) if smooth_sq_kappa else float("nan")
    smooth_dkappa_max = float(np.nanmax(np.asarray(smooth_sq_dkappa, dtype=float))) if smooth_sq_dkappa else float("nan")
    smooth_dkappa_p95 = float(_p95(smooth_sq_dkappa))
    smooth_dkappa_p95_corner = float(_p95(smooth_sq_dkappa_corner))

    reduction_ok = False
    reduction_ratio = float("nan")
    if math.isfinite(base_dkappa_p95) and base_dkappa_p95 > 1e-9 and math.isfinite(smooth_dkappa_p95):
        reduction_ratio = float((base_dkappa_p95 - smooth_dkappa_p95) / base_dkappa_p95)
        reduction_ok = bool(reduction_ratio >= float(args.dkappa_reduction_ratio))
    else:
        reduction_ok = bool(math.isfinite(smooth_dkappa_p95) and smooth_dkappa_p95 <= float(args.dkappa_abs_threshold))

    t1_smooth_ok = bool(reduction_ok and math.isfinite(smooth_kappa_max) and math.isfinite(smooth_dkappa_max))
    t2_finish_ok = bool(line_success_rate >= 0.95 and sq_success_rate >= 0.80 and (line_stall_rate + sq_stall_rate) <= 0.05)
    t3_safe_ok = bool(smooth_nan == 0 and smooth_cap_viol == 0 and smooth_omega_viol == 0)

    summary = {
        "config": str(args.config),
        "episodes": int(episodes),
        "seed": int(seed0),
        "tests": {
            "t1_smoothness_dkappa_reduced": bool(t1_smooth_ok),
            "t2_finish_reliability": bool(t2_finish_ok),
            "t3_numerical_safety": bool(t3_safe_ok),
        },
        "metrics": {
            "baseline_dkappa_p95": float(base_dkappa_p95),
            "baseline_dkappa_p95_corner": float(base_dkappa_p95_corner),
            "smooth_kappa_max": float(smooth_kappa_max),
            "smooth_dkappa_max": float(smooth_dkappa_max),
            "smooth_dkappa_p95": float(smooth_dkappa_p95),
            "smooth_dkappa_p95_corner": float(smooth_dkappa_p95_corner),
            "dkappa_reduction_ratio": float(reduction_ratio),
            "line_success_rate": float(line_success_rate),
            "square_success_rate": float(sq_success_rate),
            "line_stall_rate": float(line_stall_rate),
            "square_stall_rate": float(sq_stall_rate),
            "line_oob_rate": float(line_oob_rate),
            "square_oob_rate": float(sq_oob_rate),
        },
        "redlines": {
            "nan_count": int(smooth_nan),
            "cap_violation_count": int(smooth_cap_viol),
            "omega_violation_count": int(smooth_omega_viol),
        },
    }

    _write_json(outdir / "summary.json", summary)
    _write_trace_csv(outdir / "trace.csv", trace_rows)

    # 图：使用代表回合（smooth + open_square）
    if rep_env is not None and rep_trace is not None:
        xs = list(range(1, rep_trace.steps + 1))
        _plot_series(
            xs=xs,
            series=[("kappa_exec", rep_trace.kappa_exec)],
            out_path=outdir / "kappa_vs_step.png",
            title="open_square kappa_exec (smooth)",
            ylabel="1/m",
        )
        _plot_series(
            xs=xs,
            series=[("dkappa_exec", rep_trace.dkappa_exec)],
            out_path=outdir / "dkappa_vs_step.png",
            title="open_square dkappa_exec (abs diff)",
            ylabel="1/m",
        )
        _plot_series(
            xs=xs,
            series=[("v_ratio_exec", rep_trace.v_ratio_exec), ("v_ratio_cap", rep_trace.v_ratio_cap)],
            out_path=outdir / "v_ratio_exec_vs_step.png",
            title="open_square v_ratio (exec vs cap)",
            ylabel="ratio",
        )
        _plot_path(
            pm=rep_env.Pm,
            traj=getattr(rep_env, "trajectory", []),
            pl=getattr(rep_env, "Pl", rep_env.cache.get("Pl", [])),
            pr=getattr(rep_env, "Pr", rep_env.cache.get("Pr", [])),
            out_path=outdir / "path.png",
        )

    ok = bool(t1_smooth_ok and t2_finish_ok and t3_safe_ok)
    if ok:
        print("[PASS] P7.3 acceptance passed.")
        print(f"[OUT] {outdir}")
        raise SystemExit(0)

    print("[FAIL] P7.3 acceptance failed.")
    print(f"[OUT] {outdir}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
