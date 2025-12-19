"""P7.1 自动化验收：走廊奖励重构（自由幅度 + ramp + hysteresis）。

验收依据：
- `P7_优化指令包/03_P7_1_走廊奖励重构_v4.md`

设计原则（KISS）：
- 不训练；用可复现的启发式控制覆盖 4 项硬验收；
- 输出 summary.json + trace.csv + 关键曲线图，便于审计/回归；
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
        return ROOT / "out" / "p7_1" / f"{stamp}_accept"
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


def _first_true_segment(mask: Sequence[bool]) -> Optional[Tuple[int, int]]:
    """Return [start,end] inclusive indices for the first contiguous True segment."""
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if start is not None and (not v):
            return start, i - 1
    if start is not None:
        return start, len(mask) - 1
    return None


def _count_zero_crossings(values: Sequence[float], eps: float) -> int:
    last = 0
    count = 0
    for v in values:
        s = 0
        if math.isfinite(v) and abs(float(v)) > float(eps):
            s = 1 if v > 0 else -1
        if last != 0 and s != 0 and s != last:
            count += 1
        if s != 0:
            last = s
    return int(count)


@dataclass(frozen=True)
class EpisodeTrace:
    steps: int
    nan_count: int
    cap_violation_count: int
    omega_violation_count: int
    progress: List[float]
    e_n: List[float]
    w_center: List[float]
    corner_phase: List[bool]
    turn_sign: List[int]
    toggle_count: List[int]
    v_ratio_exec: List[float]


def _run_episode_open_line(*, env: Env, seed: int, trace_rows: List[MutableMapping[str, object]], tag: str) -> EpisodeTrace:
    _set_seed(seed)
    env.reset()
    done = False
    step_idx = 0

    max_ang_vel = float(getattr(env, "MAX_ANG_VEL", 0.0))
    progress: List[float] = []
    e_n: List[float] = []
    w_center: List[float] = []
    corner_phase: List[bool] = []
    turn_sign: List[int] = []
    toggle_count: List[int] = []
    v_ratio_exec: List[float] = []
    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0

    while not done:
        step_idx += 1
        action = np.array([0.0, 1.0], dtype=float)
        obs, reward, done, info = env.step(action)
        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        cs = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}
        if not isinstance(cs, dict):
            cs = {}

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
        progress.append(prog)
        e_n.append(float(cs.get("e_n", float("nan"))))
        w_center.append(float(cs.get("w_center", float("nan"))))
        corner_phase.append(bool(cs.get("corner_phase", False)))
        turn_sign.append(int(cs.get("turn_sign", 0)))
        toggle_count.append(int(cs.get("toggle_count", 0)))
        v_ratio_exec.append(float(p4.get("v_ratio_exec", float("nan"))))

        trace_rows.append(
            {
                "tag": tag,
                "seed": int(seed),
                "step": int(step_idx),
                "progress": float(prog),
                "v_ratio_exec": float(p4.get("v_ratio_exec", float("nan"))),
                "e_n": float(cs.get("e_n", float("nan"))),
                "w_center": float(cs.get("w_center", float("nan"))),
                "corner_phase": int(bool(cs.get("corner_phase", False))),
                "turn_sign": int(cs.get("turn_sign", 0)),
                "toggle_count": int(cs.get("toggle_count", 0)),
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
        e_n=e_n,
        w_center=w_center,
        corner_phase=corner_phase,
        turn_sign=turn_sign,
        toggle_count=toggle_count,
        v_ratio_exec=v_ratio_exec,
    )


def _policy_open_square(env: Env, p4_pre: Dict[str, float], *, center_gain: float) -> np.ndarray:
    # KISS：用 LOS 角误差做基础转向，但避免 1-step 纠偏导致饱和/甩尾
    alpha = float(p4_pre.get("alpha", 0.0))
    alpha_scale = float(math.radians(70.0))  # alpha≈70deg → theta_u≈±1
    theta_u = float(np.clip(alpha / max(alpha_scale, 1e-6), -1.0, 1.0))

    proj, _, _s, t_hat, n_hat = env._project_onto_progress_path(env.current_position)  # type: ignore[attr-defined]
    e_n = float(np.dot(env.current_position - proj, n_hat))
    corridor_half = max(float(getattr(env, "half_epsilon", 0.0)) - float(getattr(env, "_corridor_margin", 0.0)), 1e-6)

    cs = getattr(env, "last_corridor_status", {}) or {}
    in_corner = bool(cs.get("corner_phase", False)) if isinstance(cs, dict) else False
    exit_timer = int(cs.get("exit_timer", -1)) if isinstance(cs, dict) else -1
    exit_steps = int(cs.get("exit_ramp_steps", 0)) if isinstance(cs, dict) else 0
    in_exit_ramp = bool(exit_timer >= 0 and exit_steps > 0 and exit_timer <= exit_steps)

    # exit ramp 期间：不再“看远处拐角”，只做回中，避免沿偏移线跑
    if in_exit_ramp:
        # 纯几何：瞄准“投影点前方一点”，既回中也保证沿路径前进，避免过零后继续甩到另一侧
        e_norm = float(min(abs(float(e_n)) / float(corridor_half), 1.0))
        forward = float(corridor_half) * float(0.35 + 0.45 * (1.0 - e_norm))
        forward = float(max(forward, 0.25))
        target = np.asarray(proj, dtype=float) + float(forward) * np.asarray(t_hat, dtype=float)
        vec = np.asarray(target, dtype=float) - np.asarray(env.current_position, dtype=float)
        theta_target = float(math.atan2(float(vec[1]), float(vec[0])))
        heading = float(getattr(env, "_current_direction_angle", getattr(env, "heading", 0.0)))
        err = float((theta_target - heading + math.pi) % (2.0 * math.pi) - math.pi)
        err_scale_deg = float(35.0 + 25.0 * (1.0 - e_norm))  # 近中心更柔和，但保留足够纠偏能力
        err_scale = float(math.radians(err_scale_deg))
        theta_u = float(np.clip(err / max(err_scale, 1e-6), -1.0, 1.0))
        return np.array([float(theta_u), 0.55], dtype=float)

    half_eps = float(getattr(env, "half_epsilon", 0.0))
    # 贴边提前保护：宁可先减速回到安全区，避免 corner_phase 内越界提前 done → T2 NaN
    if half_eps > 1e-6 and abs(float(e_n)) > 0.90 * half_eps:
        theta_u = -1.0 if float(e_n) > 0.0 else 1.0
        return np.array([float(theta_u), 0.25], dtype=float)

    dist_to_turn = float(cs.get("dist_to_turn", float("inf"))) if isinstance(cs, dict) else float("inf")

    turn_sign = int(cs.get("turn_sign", 0)) if isinstance(cs, dict) else 0

    # 速度：远处更快，入弯/拐角更慢；exit_ramp 已在上面单独处理
    v_ratio = 0.82
    if math.isfinite(dist_to_turn):
        if dist_to_turn <= 0.9:
            v_ratio = 0.55
        elif dist_to_turn <= 1.8:
            v_ratio = 0.68
    if in_corner:
        v_ratio = float(min(v_ratio, 0.55))

    # 走廊内自由幅度：用“随 dist_to_turn 线性拉升的期望内切量”制造足够 std，但不钉死单一幅度
    if turn_sign != 0 and math.isfinite(dist_to_turn):
        approach_span = 2.2
        if dist_to_turn <= approach_span:
            x = float(np.clip(1.0 - float(dist_to_turn) / float(approach_span), 0.0, 1.0))
            desired = float(corridor_half) * float(0.10 + 0.45 * x)  # 0.10~0.55 corridor_half
            signed_e = float(turn_sign) * float(e_n)
            err = float(desired - signed_e)
            side_gain = 1.2
            theta_u = float(np.clip(theta_u + float(side_gain) * float(np.clip(err / corridor_half, -0.8, 0.8)) * float(turn_sign), -1.0, 1.0))

    # 远离拐角/非 corner_phase：轻微回中，避免“越走越歪”
    if not in_corner:
        theta_u = float(np.clip(theta_u - float(center_gain) * (e_n / corridor_half), -1.0, 1.0))

    return np.array([float(theta_u), float(v_ratio)], dtype=float)


def _run_episode_open_square(*, env: Env, seed: int, trace_rows: List[MutableMapping[str, object]], tag: str) -> EpisodeTrace:
    _set_seed(seed)
    env.reset()
    done = False
    step_idx = 0

    max_ang_vel = float(getattr(env, "MAX_ANG_VEL", 0.0))
    progress: List[float] = []
    e_n: List[float] = []
    w_center: List[float] = []
    corner_phase: List[bool] = []
    turn_sign: List[int] = []
    toggle_count: List[int] = []
    v_ratio_exec: List[float] = []
    nan_count = 0
    cap_violation_count = 0
    omega_violation_count = 0

    center_gain = 1.5

    while not done:
        step_idx += 1
        p4_pre = env._compute_p4_pre_step_status()  # type: ignore[attr-defined]
        action = _policy_open_square(env, p4_pre, center_gain=center_gain)
        obs, reward, done, info = env.step(action)
        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        cs = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}
        if not isinstance(cs, dict):
            cs = {}

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
        progress.append(prog)
        e_n.append(float(cs.get("e_n", float("nan"))))
        w_center.append(float(cs.get("w_center", float("nan"))))
        corner_phase.append(bool(cs.get("corner_phase", False)))
        turn_sign.append(int(cs.get("turn_sign", 0)))
        toggle_count.append(int(cs.get("toggle_count", 0)))
        v_ratio_exec.append(float(p4.get("v_ratio_exec", float("nan"))))

        trace_rows.append(
            {
                "tag": tag,
                "seed": int(seed),
                "step": int(step_idx),
                "progress": float(prog),
                "v_ratio_exec": float(p4.get("v_ratio_exec", float("nan"))),
                "e_n": float(cs.get("e_n", float("nan"))),
                "w_center": float(cs.get("w_center", float("nan"))),
                "corner_phase": int(bool(cs.get("corner_phase", False))),
                "turn_sign": int(cs.get("turn_sign", 0)),
                "toggle_count": int(cs.get("toggle_count", 0)),
                "alpha": float(cs.get("alpha", float("nan"))),
                "L": float(cs.get("L", float("nan"))),
                "dist_to_turn": float(cs.get("dist_to_turn", float("nan"))),
                "exit_timer": int(cs.get("exit_timer", -1)),
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
        e_n=e_n,
        w_center=w_center,
        corner_phase=corner_phase,
        turn_sign=turn_sign,
        toggle_count=toggle_count,
        v_ratio_exec=v_ratio_exec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="P7.1 自动化验收（走廊奖励重构：自由幅度 + ramp + hysteresis）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "p7_1_accept.yaml",
        help="用于构建环境/约束默认值的 YAML（默认 configs/original_configs/p7_1_accept.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    episodes = int(max(1, int(args.episodes)))
    seed0 = int(args.seed)

    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    path_cfg = cfg.get("path", {}) if isinstance(cfg.get("path", {}), dict) else {}
    scale = float(path_cfg.get("scale", 2.0))
    num_points = int(path_cfg.get("num_points", 50))

    trace_rows: List[MutableMapping[str, object]] = []

    # T4: 直线段速度不被“压死”（走廊不应影响 open_line）
    pm_line = build_line(length=scale, num_points=num_points)
    line_caps: List[float] = []
    line_nans = 0
    line_cap_viol = 0
    line_omega_viol = 0
    for i in range(episodes):
        env = _build_env(cfg, pm_line)
        tr = _run_episode_open_line(env=env, seed=seed0 + i, trace_rows=trace_rows, tag=f"line_{i}")
        line_nans += tr.nan_count
        line_cap_viol += tr.cap_violation_count
        line_omega_viol += tr.omega_violation_count
        tail = tr.v_ratio_exec[max(0, int(0.8 * len(tr.v_ratio_exec))) :]
        tail = [v for v in tail if math.isfinite(v)]
        line_caps.append(float(np.mean(tail)) if tail else float("nan"))

    mean_line_tail = float(np.nanmean(np.asarray(line_caps, dtype=float))) if line_caps else float("nan")
    t4_ok = bool(math.isfinite(mean_line_tail) and mean_line_tail >= 0.95)

    # T1~T3: open_square（自由幅度/回中/抖动）
    pm_sq = build_open_square(side=scale, num_points=num_points)
    t1_ok_all: List[bool] = []
    t2_ok_all: List[bool] = []
    t3_ok_all: List[bool] = []

    t1_std_list: List[float] = []
    t1_dir_ratio_list: List[float] = []
    t2_med_abs_list: List[float] = []
    t2_zero_cross_list: List[int] = []
    chatter_list: List[int] = []
    toggle_count_list: List[int] = []

    sq_nans = 0
    sq_cap_viol = 0
    sq_omega_viol = 0

    rep_env: Optional[Env] = None
    rep_trace: Optional[EpisodeTrace] = None

    for i in range(episodes):
        env = _build_env(cfg, pm_sq)
        tr = _run_episode_open_square(env=env, seed=seed0 + 1000 + i, trace_rows=trace_rows, tag=f"square_{i}")
        sq_nans += tr.nan_count
        sq_cap_viol += tr.cap_violation_count
        sq_omega_viol += tr.omega_violation_count

        if rep_env is None:
            rep_env = env
            rep_trace = tr

        # corner_phase segments（只取第一个拐角段做验收，避免 open_square 多拐角计数干扰）
        seg = _first_true_segment(tr.corner_phase)
        corridor_half = max(float(getattr(env, "half_epsilon", 0.0)) - float(getattr(env, "_corridor_margin", 0.0)), 1e-6)
        if seg is None:
            t1_ok_all.append(False)
            t2_ok_all.append(False)
            t3_ok_all.append(False)
            t1_std_list.append(float("nan"))
            t1_dir_ratio_list.append(float("nan"))
            t2_med_abs_list.append(float("nan"))
            t2_zero_cross_list.append(-1)
            chatter_list.append(-1)
            toggle_count_list.append(-1)
            continue

        s0, s1 = seg
        e_corner = [v for v in tr.e_n[s0 : s1 + 1] if math.isfinite(v)]
        ts_corner = tr.turn_sign[s0 : s1 + 1]

        std_e = float(np.nanstd(np.asarray(e_corner, dtype=float))) if e_corner else float("nan")
        dir_ok_count = 0
        for v, ts in zip(tr.e_n[s0 : s1 + 1], ts_corner):
            if not math.isfinite(v) or int(ts) == 0:
                continue
            if float(v) * float(ts) >= 0.0:
                dir_ok_count += 1
        dir_total = int(sum(1 for v, ts in zip(tr.e_n[s0 : s1 + 1], ts_corner) if math.isfinite(v) and int(ts) != 0))
        dir_ratio = float(dir_ok_count / max(dir_total, 1))

        t1_ok = bool(math.isfinite(std_e) and std_e >= 0.1 * corridor_half and dir_ratio >= 0.8)

        # exit window after the first corner segment
        ramp_steps = int(getattr(env, "_corridor_exit_center_ramp_steps", 0))
        w_start = s1 + 1
        w_end = min(len(tr.e_n) - 1, w_start + max(1, ramp_steps) - 1)
        exit_vals = [abs(float(v)) for v in tr.e_n[w_start : w_end + 1] if math.isfinite(v)]
        med_abs = float(np.nanmedian(np.asarray(exit_vals, dtype=float))) if exit_vals else float("nan")
        zero_cross = _count_zero_crossings(tr.e_n[w_start : w_end + 1], eps=1e-6)

        t2_ok = bool(math.isfinite(med_abs) and med_abs <= 0.2 * corridor_half and zero_cross <= 3)

        # hysteresis jitter: compare total toggles vs expected toggles for this episode
        toggles = int(sum(1 for j in range(1, len(tr.corner_phase)) if tr.corner_phase[j] != tr.corner_phase[j - 1]))
        entries = int(sum(1 for j in range(1, len(tr.corner_phase)) if (not tr.corner_phase[j - 1]) and tr.corner_phase[j]))
        expected = int(2 * entries)
        if tr.corner_phase and tr.corner_phase[-1]:
            expected = max(0, expected - 1)
        chatter = int(max(0, toggles - expected))
        t3_ok = bool(chatter <= 1)
        toggle_count_list.append(int(toggles))

        t1_ok_all.append(bool(t1_ok))
        t2_ok_all.append(bool(t2_ok))
        t3_ok_all.append(bool(t3_ok))
        t1_std_list.append(float(std_e))
        t1_dir_ratio_list.append(float(dir_ratio))
        t2_med_abs_list.append(float(med_abs))
        t2_zero_cross_list.append(int(zero_cross))
        chatter_list.append(int(chatter))

    t1_pass = bool(all(t1_ok_all)) if t1_ok_all else False
    t2_pass = bool(all(t2_ok_all)) if t2_ok_all else False
    t3_pass = bool(all(t3_ok_all)) if t3_ok_all else False

    nan_total = int(line_nans + sq_nans)
    cap_viol_total = int(line_cap_viol + sq_cap_viol)
    omega_viol_total = int(line_omega_viol + sq_omega_viol)

    summary = {
        "config": str(args.config),
        "episodes": int(episodes),
        "seed": int(seed0),
        "dt": float(cfg.get("environment", {}).get("interpolation_period", float("nan"))),
        "tests": {
            "t1_free_amplitude_and_direction": bool(t1_pass),
            "t2_exit_recenter": bool(t2_pass),
            "t3_phase_hysteresis_no_chatter": bool(t3_pass),
            "t4_line_speed_not_suppressed": bool(t4_ok),
        },
        "metrics": {
            "t1_std_e_n_min": float(np.nanmin(np.asarray(t1_std_list, dtype=float))) if t1_std_list else float("nan"),
            "t1_dir_ratio_min": float(np.nanmin(np.asarray(t1_dir_ratio_list, dtype=float))) if t1_dir_ratio_list else float("nan"),
            "t2_median_abs_e_n_max": float(np.nanmax(np.asarray(t2_med_abs_list, dtype=float))) if t2_med_abs_list else float("nan"),
            "t2_zero_cross_max": int(max(t2_zero_cross_list)) if t2_zero_cross_list else -1,
            "t3_toggle_count_max": int(max(toggle_count_list)) if toggle_count_list else -1,
            "t3_chatter_max": int(max(chatter_list)) if chatter_list else -1,
            "t4_mean_v_ratio_exec_last20_line": float(mean_line_tail),
            "ramp_steps": int(getattr(rep_env, "_corridor_exit_center_ramp_steps", 0)) if rep_env is not None else 0,
        },
        "redlines": {
            "nan_count": int(nan_total),
            "cap_violation_count": int(cap_viol_total),
            "omega_violation_count": int(omega_viol_total),
        },
    }
    _write_json(outdir / "summary.json", summary)
    _write_trace_csv(outdir / "trace.csv", trace_rows)

    # 图：使用代表回合（square_0）
    if rep_env is not None and rep_trace is not None:
        xs = list(range(1, rep_trace.steps + 1))
        _plot_series(
            xs=xs,
            series=[("e_n", rep_trace.e_n)],
            out_path=outdir / "e_n_vs_step.png",
            title="open_square e_n",
            ylabel="m",
        )
        _plot_series(
            xs=xs,
            series=[("w_center", rep_trace.w_center)],
            out_path=outdir / "w_center_vs_step.png",
            title="open_square w_center ramp",
            ylabel="weight",
        )
        _plot_series(
            xs=xs,
            series=[("corner_phase", [1.0 if v else 0.0 for v in rep_trace.corner_phase])],
            out_path=outdir / "corner_phase_toggle_marks.png",
            title="open_square corner_phase",
            ylabel="0/1",
        )
        _plot_path(
            pm=rep_env.Pm,
            traj=getattr(rep_env, "trajectory", []),
            pl=getattr(rep_env, "Pl", rep_env.cache.get("Pl", [])),
            pr=getattr(rep_env, "Pr", rep_env.cache.get("Pr", [])),
            out_path=outdir / "path.png",
        )

    ok = bool(t1_pass and t2_pass and t3_pass and t4_ok and nan_total == 0 and cap_viol_total == 0 and omega_viol_total == 0)
    if ok:
        print("[PASS] P7.1 acceptance passed.")
        print(f"[OUT] {outdir}")
        raise SystemExit(0)

    print("[FAIL] P7.1 acceptance failed.")
    print(f"[OUT] {outdir}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
