"""P7.0 自动化验收：动作尺度（ratio->物理量）与航向积分（heading integration）。

验收依据：
- `P7_优化指令包/00_README_P7_发送顺序与验收SOP_v4.md`
- `P7_优化指令包/01_P7_0_动作尺度与航向积分_v4.md`

设计原则（KISS）：
- 不做训练，仅用可复现的常量动作覆盖 line/square 两个场景；
- 产出 summary.json/trace.csv/path.png 与关键曲线图，便于审计与回归；
- 任何硬性门槛失败返回非 0 退出码。
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # PPO_project
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt  # noqa: E402

    from src.environment import Env  # noqa: E402
    from src.utils.geometry import count_polyline_self_intersections  # noqa: E402
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
        return ROOT / "out" / "p7_0" / f"{stamp}_accept"
    outdir = Path(outdir)
    return outdir if outdir.is_absolute() else (ROOT / outdir)


def _pm_for_case(case: str, cfg: Mapping) -> List[np.ndarray]:
    path_cfg = cfg.get("path", {}) if isinstance(cfg.get("path", {}), dict) else {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    def build_line(length: float, num_points: int, angle: float = 0.0) -> List[np.ndarray]:
        ts = np.linspace(0.0, 1.0, max(2, int(num_points)))
        dx = math.cos(float(angle))
        dy = math.sin(float(angle))
        return [np.array([float(length) * t * dx, float(length) * t * dy], dtype=float) for t in ts]

    def build_open_square(side: float, num_points: int) -> List[np.ndarray]:
        # open：仅 3 条边（避免“几乎闭合”导致 open_finish_line 贴近起点、回合 1 步结束）
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

    if case == "open_line":
        line_cfg = path_cfg.get("line", {}) if isinstance(path_cfg.get("line", {}), dict) else {}
        angle = float(line_cfg.get("angle", 0.0))
        return build_line(length=scale, num_points=num_points, angle=angle)

    if case == "open_square":
        return build_open_square(side=scale, num_points=num_points)

    raise ValueError(f"unknown case: {case}")


def _as_xy_points(points: object) -> List[List[float]]:
    if not isinstance(points, list):
        return []
    out: List[List[float]] = []
    for p in points:
        if p is None:
            continue
        arr = np.asarray(p, dtype=float).reshape(-1)
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            out.append([float(arr[0]), float(arr[1])])
    return out


def _boundary_stats(env: Env) -> Dict[str, object]:
    cache = getattr(env, "cache", {}) or {}
    pl_pts = _as_xy_points(cache.get("Pl", None))
    pr_pts = _as_xy_points(cache.get("Pr", None))
    closed = bool(getattr(env, "closed", False))
    pl_closed = bool(pl_pts) and bool(np.allclose(pl_pts[0], pl_pts[-1], atol=1e-6))
    pr_closed = bool(pr_pts) and bool(np.allclose(pr_pts[0], pr_pts[-1], atol=1e-6))
    return {
        "env_closed": closed,
        "pl_closed": pl_closed,
        "pr_closed": pr_closed,
        "pl_self_intersections": int(count_polyline_self_intersections(pl_pts, closed=closed)),
        "pr_self_intersections": int(count_polyline_self_intersections(pr_pts, closed=closed)),
    }


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


def _is_finite_scalar(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _count_non_finite(*values: object) -> int:
    return sum(0 if _is_finite_scalar(v) else 1 for v in values)


@dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    stall_triggered: bool
    steps: int
    mean_v_ratio_exec_last20: float
    heading_jump_max: float
    nan_count: int
    cap_violation_count: int
    omega_violation_count: int


def _last20_mean(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    n = max(1, int(math.ceil(0.2 * len(samples))))
    tail = samples[-n:]
    return float(np.mean(np.asarray(tail, dtype=float)))


def _write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_path(
    *,
    env: Env,
    trajectory: Sequence[Sequence[float]],
    out_path: Path,
    title: str,
) -> None:
    pm = np.asarray(env.Pm, dtype=float)
    traj = np.asarray(trajectory, dtype=float) if trajectory else np.zeros((0, 2), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot(pm[:, 0], pm[:, 1], "k--", lw=1.2, label="Pm(ref)")

    pl = getattr(env, "cache", {}).get("Pl", None)
    pr = getattr(env, "cache", {}).get("Pr", None)
    if isinstance(pl, list) and isinstance(pr, list):
        pl_pts = np.asarray([p for p in pl if p is not None], dtype=float) if any(p is not None for p in pl) else None
        pr_pts = np.asarray([p for p in pr if p is not None], dtype=float) if any(p is not None for p in pr) else None
        if pl_pts is not None and pl_pts.size > 0:
            ax.plot(pl_pts[:, 0], pl_pts[:, 1], color="tab:blue", lw=0.8, alpha=0.85, label="Pl")
        if pr_pts is not None and pr_pts.size > 0:
            ax.plot(pr_pts[:, 0], pr_pts[:, 1], color="tab:orange", lw=0.8, alpha=0.85, label="Pr")

    if traj.size > 0:
        ax.plot(traj[:, 0], traj[:, 1], color="tab:red", lw=1.4, label="trajectory")
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="tab:red", s=18, marker="o")
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="tab:red", s=18, marker="x")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", lw=0.6)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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


def _run_episode(
    *,
    env: Env,
    action: np.ndarray,
    seed: int,
    dt: float,
    max_ang_vel: float,
    capture_trace: bool,
    trace_rows: List[MutableMapping[str, object]],
    case: str,
) -> EpisodeMetrics:
    _set_seed(seed)
    env.reset()

    headings: List[float] = []
    v_ratio_exec_series: List[float] = []

    done = False
    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0
    heading_jump_max = 0.0

    heading_prev = float(getattr(env, "heading", float(getattr(env, "_current_direction_angle", 0.0))))

    while not done:
        obs, reward, done, info = env.step(action)

        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}
        corridor = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(corridor, dict):
            corridor = {}
        e_n = float(corridor.get("e_n", float("nan")))

        v_exec = float(p4.get("v_exec", float(getattr(env, "velocity", 0.0))))
        omega_exec = float(p4.get("omega_exec", float(getattr(env, "angular_vel", 0.0))))
        max_vel_cap_phys = float(p4.get("max_vel_cap_phys", float(getattr(env, "MAX_VEL", 0.0))))
        v_ratio_exec = float(p4.get("v_ratio_exec", 0.0))
        v_ratio_cap = float(p4.get("v_ratio_cap", float("nan")))
        v_intent = float(p4.get("v_intent", float("nan")))
        omega_intent = float(p4.get("omega_intent", float("nan")))
        kappa = float(p4.get("kappa_max_ahead", p4.get("kappa_los", p4.get("kappa", float("nan")))))

        heading_now = float(getattr(env, "heading", float(getattr(env, "_current_direction_angle", 0.0))))
        d_heading = abs(heading_now - heading_prev)
        heading_jump_max = float(max(heading_jump_max, d_heading))
        heading_prev = heading_now

        headings.append(heading_now)
        v_ratio_exec_series.append(v_ratio_exec)

        if v_exec > max_vel_cap_phys + 1e-6:
            cap_violation_count += 1

        if abs(omega_exec) > float(max_ang_vel) + 1e-6:
            omega_violation_count += 1

        if not _is_finite_scalar(reward):
            nan_count += 1
        if isinstance(obs, np.ndarray):
            nan_count += int(obs.size - int(np.count_nonzero(np.isfinite(obs))))
        else:
            nan_count += 1
        nan_count += _count_non_finite(
            info.get("contour_error", 0.0) if isinstance(info, dict) else 0.0,
            p4.get("v_exec", 0.0),
            p4.get("omega_exec", 0.0),
            p4.get("v_ratio_cap", 0.0),
            p4.get("max_vel_cap_phys", 0.0),
            v_intent,
            omega_intent,
            e_n,
        )

        if capture_trace:
            trace_rows.append(
                {
                    "case": case,
                    "seed": int(seed),
                    "step": int(info.get("step", getattr(env, "current_step", 0))),
                    "x": float(getattr(env, "current_position", np.array([float("nan"), float("nan")]))[0]),
                    "y": float(getattr(env, "current_position", np.array([float("nan"), float("nan")]))[1]),
                    "heading": float(heading_now),
                    "d_heading": float(d_heading),
                    "theta_u_policy": float(action[0]),
                    "v_ratio_policy": float(action[1]),
                    "omega_intent": float(omega_intent),
                    "v_intent": float(v_intent),
                    "omega_exec": float(omega_exec),
                    "v_exec": float(v_exec),
                    "v_ratio_exec": float(v_ratio_exec),
                    "v_ratio_cap": float(v_ratio_cap),
                    "max_vel_cap_phys": float(max_vel_cap_phys),
                    "kappa": float(kappa),
                    "e_n": float(e_n),
                    "contour_error": float(info.get("contour_error", float("nan"))),
                    "reward": float(reward),
                }
            )

    success = bool(getattr(env, "reached_target", False))
    stall_triggered = bool(getattr(env, "_p4_stall_triggered", False))

    return EpisodeMetrics(
        success=success,
        stall_triggered=stall_triggered,
        steps=int(getattr(env, "current_step", 0)),
        mean_v_ratio_exec_last20=float(_last20_mean(v_ratio_exec_series)),
        heading_jump_max=float(heading_jump_max),
        nan_count=int(nan_count),
        cap_violation_count=int(cap_violation_count),
        omega_violation_count=int(omega_violation_count),
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="P7.0 自动化验收（动作尺度 + 航向积分）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于构建环境/约束/路径参数的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[ERROR] missing config: {args.config}")
        raise SystemExit(2)

    cfg = _load_yaml(args.config)
    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    episodes = max(1, int(args.episodes))
    seed0 = int(args.seed)

    pm_line = _pm_for_case("open_line", cfg)
    env_line = _build_env(cfg, pm_line)
    dt = float(getattr(env_line, "interpolation_period", 0.0))
    max_ang_vel = float(getattr(env_line, "MAX_ANG_VEL", 0.0))

    # 固定动作：line 直线极速；square 用恒定转向观察“尖角是否消失”
    action_line = np.array([0.0, 1.0], dtype=float)
    action_square = np.array([0.3, 0.6], dtype=float)

    trace_rows: List[MutableMapping[str, object]] = []

    # === open_line ===
    line_eps: List[EpisodeMetrics] = []
    heading_jump_max = 0.0
    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0
    for ep in range(episodes):
        m = _run_episode(
            env=env_line,
            action=action_line,
            seed=seed0 + ep,
            dt=dt,
            max_ang_vel=max_ang_vel,
            capture_trace=(ep == 0),
            trace_rows=trace_rows,
            case="open_line",
        )
        line_eps.append(m)
        heading_jump_max = max(heading_jump_max, float(m.heading_jump_max))
        nan_count += int(m.nan_count)
        cap_violation_count += int(m.cap_violation_count)
        omega_violation_count += int(m.omega_violation_count)

    success_rate_line = float(np.mean([1.0 if m.success else 0.0 for m in line_eps]))
    stall_rate_line = float(np.mean([1.0 if m.stall_triggered else 0.0 for m in line_eps]))
    mean_v_ratio_exec_last20_line = float(np.mean([m.mean_v_ratio_exec_last20 for m in line_eps]))

    # === open_square ===
    pm_square = _pm_for_case("open_square", cfg)
    env_square = _build_env(cfg, pm_square)
    square_boundary_stats = _boundary_stats(env_square)
    square_eps: List[EpisodeMetrics] = []
    square_traj: List[np.ndarray] = []
    for ep in range(episodes):
        capture = ep == 0
        m = _run_episode(
            env=env_square,
            action=action_square,
            seed=seed0 + 1000 + ep,
            dt=dt,
            max_ang_vel=max_ang_vel,
            capture_trace=capture,
            trace_rows=trace_rows,
            case="open_square",
        )
        square_eps.append(m)
        heading_jump_max = max(heading_jump_max, float(m.heading_jump_max))
        nan_count += int(m.nan_count)
        cap_violation_count += int(m.cap_violation_count)
        omega_violation_count += int(m.omega_violation_count)
        if capture:
            square_traj = [np.array(p, dtype=float) for p in getattr(env_square, "trajectory", [])]

    heading_jump_threshold = float(1.2 * max_ang_vel * dt)

    summary = {
        "config": str(args.config),
        "episodes": int(episodes),
        "seed": int(seed0),
        "dt": float(dt),
        "MAX_VEL": float(getattr(env_line, "MAX_VEL", float("nan"))),
        "MAX_ANG_VEL": float(max_ang_vel),
        "success_rate_line": float(success_rate_line),
        "stall_rate_line": float(stall_rate_line),
        "mean_v_ratio_exec_last20_line": float(mean_v_ratio_exec_last20_line),
        "heading_jump_max": float(heading_jump_max),
        "heading_jump_threshold": float(heading_jump_threshold),
        "nan_count": int(nan_count),
        "cap_violation_count": int(cap_violation_count),
        "omega_violation_count": int(omega_violation_count),
        "square_env_closed": bool(square_boundary_stats.get("env_closed", False)),
        "square_pl_closed": bool(square_boundary_stats.get("pl_closed", False)),
        "square_pr_closed": bool(square_boundary_stats.get("pr_closed", False)),
        "square_pl_self_intersections": int(square_boundary_stats.get("pl_self_intersections", 0)),
        "square_pr_self_intersections": int(square_boundary_stats.get("pr_self_intersections", 0)),
    }

    _write_json(outdir / "summary.json", summary)
    _write_trace_csv(outdir / "trace.csv", trace_rows)

    # 代表性可视化：square 的轨迹（尖角/圆弧）
    if square_traj:
        _plot_path(
            env=env_square,
            trajectory=square_traj,
            out_path=outdir / "path.png",
            title="open_square trajectory (P7.0 heading integration)",
        )

    # 曲线图：从 trace.csv 中提取 open_square 的代表回合
    square_trace = [r for r in trace_rows if str(r.get("case")) == "open_square"]
    if square_trace:
        xs = [float(r.get("step", i)) for i, r in enumerate(square_trace)]
        v_ratio_exec = [float(r.get("v_ratio_exec", float("nan"))) for r in square_trace]
        v_ratio_cap = [float(r.get("v_ratio_cap", float("nan"))) for r in square_trace]
        kappa = [float(r.get("kappa", float("nan"))) for r in square_trace]
        dkappa = [abs(kappa[i] - kappa[i - 1]) if i > 0 else 0.0 for i in range(len(kappa))]
        d_heading = [float(r.get("d_heading", 0.0)) for r in square_trace]
        e_n = [float(r.get("e_n", float("nan"))) for r in square_trace]

        _plot_series(
            xs=xs,
            series=[("v_ratio_exec", v_ratio_exec), ("v_ratio_cap", v_ratio_cap)],
            out_path=outdir / "v_ratio_cap.png",
            title="open_square v_ratio (exec vs cap)",
            ylabel="ratio",
        )
        _plot_series(
            xs=xs,
            series=[("kappa", kappa)],
            out_path=outdir / "kappa.png",
            title="open_square kappa (max_ahead/los)",
            ylabel="1/m",
        )
        _plot_series(
            xs=xs,
            series=[("dkappa", dkappa)],
            out_path=outdir / "dkappa.png",
            title="open_square dkappa (abs diff)",
            ylabel="1/m",
        )
        _plot_series(
            xs=xs,
            series=[("e_n", e_n)],
            out_path=outdir / "e_n.png",
            title="open_square lateral error (e_n)",
            ylabel="m",
        )
        _plot_series(
            xs=xs,
            series=[("abs(d_heading)", [abs(v) for v in d_heading]), ("threshold", [heading_jump_threshold] * len(xs))],
            out_path=outdir / "d_heading.png",
            title="open_square |Δheading| per step",
            ylabel="rad",
        )

    lines = [
        "[P7.0 ACCEPT] 动作尺度与航向积分",
        f"outdir: {outdir}",
        f"episodes={episodes} seed={seed0} dt={dt} MAX_ANG_VEL={max_ang_vel}",
        f"success_rate_line={success_rate_line:.3f} (>=0.95)",
        f"mean_v_ratio_exec_last20_line={mean_v_ratio_exec_last20_line:.3f} (>=0.85)",
        f"stall_rate_line={stall_rate_line:.3f} (=0.0)",
        f"heading_jump_max={heading_jump_max:.6f} (<= {heading_jump_threshold:.6f})",
        f"nan_count={nan_count} (=0)",
        f"cap_violation_count={cap_violation_count} (=0)",
        f"omega_violation_count={omega_violation_count} (=0)",
        f"square: env_closed={bool(square_boundary_stats.get('env_closed', False))} "
        f"pl_closed={bool(square_boundary_stats.get('pl_closed', False))} pr_closed={bool(square_boundary_stats.get('pr_closed', False))} "
        f"pl_self_intersections={int(square_boundary_stats.get('pl_self_intersections', 0))} "
        f"pr_self_intersections={int(square_boundary_stats.get('pr_self_intersections', 0))}",
        "manual: 请目视检查 path.png（open_square 角点附近应为圆弧/连续转弯，而非折线尖角）。",
    ]
    _write_text(outdir / "summary.txt", lines)

    ok = True
    if success_rate_line < 0.95:
        ok = False
    if mean_v_ratio_exec_last20_line < 0.85:
        ok = False
    if stall_rate_line > 0.0:
        ok = False
    if nan_count != 0:
        ok = False
    if cap_violation_count != 0:
        ok = False
    if omega_violation_count != 0:
        ok = False
    if heading_jump_max > heading_jump_threshold:
        ok = False
    if int(square_boundary_stats.get("pl_self_intersections", 0)) != 0:
        ok = False
    if int(square_boundary_stats.get("pr_self_intersections", 0)) != 0:
        ok = False

    if ok:
        print("[PASS] P7.0 acceptance passed.")
        print(f"[OUT] {outdir}")
        raise SystemExit(0)

    print("[FAIL] P7.0 acceptance failed.")
    print(f"[OUT] {outdir}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
