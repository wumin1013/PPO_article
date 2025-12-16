"""P3.1 对比脚本：直角拐角案例，走廊开/关的轨迹与 e_n 可视化。

用途：
- 直观看到 corridor_status 是否进入 corner_phase、turn_sign 是否正确；
- 同一控制策略下，走廊 ON 会将轨迹在拐角阶段引导到 e_target（内侧偏置），OFF 更趋近中心线；
- 输出 PNG 结果，便于验收/粘贴。
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import matplotlib.pyplot as plt  # noqa: E402
    import yaml  # noqa: E402

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


def build_right_angle_path(
    *,
    side: float,
    num_points: int,
    turn: str,
) -> List[np.ndarray]:
    """两段折线直角：起点(0,0)→(L,0)→(L,±L)。turn=left|right."""
    if num_points < 20:
        raise ValueError("num_points too small")
    L = float(side)
    if turn not in {"left", "right"}:
        raise ValueError("turn must be left or right")
    p0 = np.array([0.0, 0.0], dtype=float)
    p1 = np.array([L, 0.0], dtype=float)
    p2 = np.array([L, L if turn == "left" else -L], dtype=float)

    per_edge = max(10, num_points // 2)
    pts: List[np.ndarray] = []
    pts.extend(_linspace_points(p0, p1, per_edge, include_start=True))
    pts.extend(_linspace_points(p1, p2, num_points - len(pts), include_start=False))
    return [np.array(p, dtype=float) for p in pts]


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_env(cfg: Dict, pm: Sequence[np.ndarray], *, corridor_cfg: dict) -> Env:
    env_cfg = cfg["environment"]
    kcm_cfg = cfg["kinematic_constraints"]
    reward_weights = dict(cfg.get("reward_weights", {}) or {})
    reward_weights["corridor"] = dict(corridor_cfg)
    return Env(
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


def _controller_action(env: Env, corridor_status: Dict[str, object], *, vel: float, kp: float) -> np.ndarray:
    half = float(getattr(env, "half_epsilon", 1.0))
    e_n = float(corridor_status.get("e_n", 0.0))
    desired = 0.0
    if bool(corridor_status.get("enabled", False)) and bool(corridor_status.get("corner_phase", False)):
        desired = float(corridor_status.get("e_target", 0.0))
    theta = float(np.clip(kp * (desired - e_n) / max(half, 1e-6), -1.0, 1.0))
    return np.array([theta, float(vel)], dtype=float)


@dataclass
class EpisodeTrace:
    x: List[float]
    y: List[float]
    e_n: List[float]
    e_target: List[float]
    lower: List[float]
    upper: List[float]
    corner_phase: List[int]
    in_corridor: List[int]
    reward: List[float]
    reached_target: bool
    oob: bool
    steps: int


@dataclass
class Summary:
    episodes: int
    success_rate: float
    oob_rate: float
    steps_mean: float
    mean_e_n_corner: float
    in_corridor_ratio: float


def _run_episodes(
    env: Env,
    *,
    episodes: int,
    seed: int,
    vel: float,
    kp: float,
    keep_first_trace: bool,
) -> Tuple[Summary, Optional[EpisodeTrace]]:
    successes = 0
    oobs = 0
    steps: List[int] = []
    e_corner: List[float] = []
    in_corridor_hits = 0
    corner_hits = 0
    first_trace: Optional[EpisodeTrace] = None

    for ep in range(episodes):
        _set_seed(seed + ep)
        env.reset()
        action = np.array([0.0, float(vel)], dtype=float)
        done = False
        info: Dict[str, object] = {}

        trace = EpisodeTrace(
            x=[],
            y=[],
            e_n=[],
            e_target=[],
            lower=[],
            upper=[],
            corner_phase=[],
            in_corridor=[],
            reward=[],
            reached_target=False,
            oob=False,
            steps=0,
        )

        while not done:
            _, r, done, info = env.step(action)
            cs = info.get("corridor_status", {}) if isinstance(info, dict) else {}
            if not isinstance(cs, dict):
                cs = {}
            action = _controller_action(env, cs, vel=float(vel), kp=float(kp))

            trace.x.append(float(getattr(env, "current_position", np.array([0.0, 0.0]))[0]))
            trace.y.append(float(getattr(env, "current_position", np.array([0.0, 0.0]))[1]))
            trace.reward.append(float(r))
            trace.e_n.append(float(cs.get("e_n", 0.0)))
            trace.e_target.append(float(cs.get("e_target", 0.0)))
            trace.lower.append(float(cs.get("lower", 0.0)))
            trace.upper.append(float(cs.get("upper", 0.0)))
            corner = 1 if bool(cs.get("corner_phase", False)) else 0
            inside = 1 if bool(cs.get("in_corridor", False)) else 0
            trace.corner_phase.append(corner)
            trace.in_corridor.append(inside)

            if corner:
                corner_hits += 1
                e_corner.append(float(cs.get("e_n", 0.0)))
                if inside:
                    in_corridor_hits += 1

        trace.steps = (
            int(info.get("step", getattr(env, "current_step", 0))) if isinstance(info, dict) else int(getattr(env, "current_step", 0))
        )
        reached = bool(getattr(env, "reached_target", False))
        if isinstance(info, dict):
            reached = reached or bool(info.get("reached_target", False))
        trace.reached_target = reached

        oob = bool(getattr(env, "is_oob", False))
        if isinstance(info, dict):
            oob = oob or bool(info.get("oob", False))
        trace.oob = oob

        if trace.reached_target:
            successes += 1
        if trace.oob:
            oobs += 1
        steps.append(int(trace.steps))

        if keep_first_trace and first_trace is None:
            first_trace = trace

    summary = Summary(
        episodes=int(episodes),
        success_rate=float(successes) / max(1, int(episodes)),
        oob_rate=float(oobs) / max(1, int(episodes)),
        steps_mean=float(np.mean(steps)) if steps else 0.0,
        mean_e_n_corner=float(np.mean(e_corner)) if e_corner else 0.0,
        in_corridor_ratio=float(in_corridor_hits) / max(1, corner_hits),
    )
    return summary, first_trace


def _extract_band(env: Env) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    pl = [(float(p[0]), float(p[1])) for p in getattr(env, "cache", {}).get("Pl", [])]
    pr = [(float(p[0]), float(p[1])) for p in getattr(env, "cache", {}).get("Pr", [])]
    return pl, pr


def _plot_trajectory(
    *,
    out_path: Path,
    pm: Sequence[np.ndarray],
    pl: Sequence[Tuple[float, float]],
    pr: Sequence[Tuple[float, float]],
    trace_off: EpisodeTrace,
    trace_on: EpisodeTrace,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    if pl and pr:
        band_x = [p[0] for p in pl] + [p[0] for p in pr][::-1]
        band_y = [p[1] for p in pl] + [p[1] for p in pr][::-1]
        ax.fill(band_x, band_y, color="#228be6", alpha=0.12, label="Tolerance Band")

    ax.plot([float(p[0]) for p in pm], [float(p[1]) for p in pm], "--", color="#1f77b4", linewidth=1.5, label="Reference Path")
    ax.plot(trace_off.x, trace_off.y, color="#e03131", linewidth=2.0, label="Corridor OFF")
    ax.plot(trace_on.x, trace_on.y, color="#2f9e44", linewidth=2.0, label="Corridor ON")

    xs = [float(p[0]) for p in pm] + trace_off.x + trace_on.x
    ys = [float(p[1]) for p in pm] + trace_off.y + trace_on.y
    ax.set_xlim([min(xs) - 0.5, max(xs) + 0.5])
    ax.set_ylim([min(ys) - 0.5, max(ys) + 0.5])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_lateral_error(
    *,
    out_path: Path,
    trace_off: EpisodeTrace,
    trace_on: EpisodeTrace,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))

    t_off = np.arange(len(trace_off.e_n))
    t_on = np.arange(len(trace_on.e_n))
    ax.plot(t_off, trace_off.e_n, color="#e03131", linewidth=1.6, label="e_n OFF")
    ax.plot(t_on, trace_on.e_n, color="#2f9e44", linewidth=1.6, label="e_n ON")

    # 走廊 ON 的边界/目标（只画 ON，避免图过于拥挤）
    ax.plot(t_on, trace_on.e_target, color="#2f9e44", linestyle="--", linewidth=1.2, label="e_target (ON)")
    ax.plot(t_on, trace_on.lower, color="#495057", linestyle=":", linewidth=1.1, label="lower/upper (ON)")
    ax.plot(t_on, trace_on.upper, color="#495057", linestyle=":", linewidth=1.1)

    # corner_phase 作为背景色
    if any(trace_on.corner_phase):
        on = np.array(trace_on.corner_phase, dtype=int)
        idx = np.where(on > 0)[0]
        if idx.size > 0:
            ax.axvspan(int(idx.min()), int(idx.max()), color="#fab005", alpha=0.12, label="corner_phase (ON)")

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("e_n (left + / right -)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3.1 直角案例 corridor ON/OFF 对比可视化")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "original_configs" / "smoke.yaml",
        help="用于读取环境/约束默认值的 YAML（默认 original_configs/smoke.yaml）。",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vel", type=float, default=0.8)
    parser.add_argument("--kp", type=float, default=1.0)
    parser.add_argument("--turn", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument(
        "--side",
        type=float,
        default=0.0,
        help="直角两段的边长；<=0 时按 dt/max_steps/vel 自动估算可跑完的尺度。",
    )
    parser.add_argument("--theta-enter-deg", type=float, default=15.0)
    parser.add_argument("--theta-exit-deg", type=float, default=8.0)
    parser.add_argument("--dist-enter", type=float, default=0.0, help="<=0 时使用 3*lookahead_spacing")
    parser.add_argument("--dist-exit", type=float, default=0.0, help="<=0 时使用 1.5*dist_enter")
    parser.add_argument("--margin-ratio", type=float, default=0.1)
    parser.add_argument("--heading-weight", type=float, default=2.0)
    parser.add_argument("--outside-penalty-weight", type=float, default=20.0)
    parser.add_argument("--outdir", type=Path, default=None)
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
        est_travel = float(args.vel) * dt * max_steps * 0.85
        side = float(np.clip(est_travel / 2.0 * 0.9, 1.0, 10.0))
        print(f"[AUTO] side={side:.2f} (dt={dt}, max_steps={max_steps}, vel={args.vel})")

    outdir = args.outdir
    if outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = REPO_ROOT / "logs" / "p3_1_corridor_compare" / f"right_angle_{args.turn}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    pm = build_right_angle_path(side=side, num_points=int(args.num_points), turn=str(args.turn))

    corridor_cfg_on = {
        "enabled": True,
        "theta_enter_deg": float(args.theta_enter_deg),
        "theta_exit_deg": float(args.theta_exit_deg),
        "dist_enter": None if float(args.dist_enter) <= 0 else float(args.dist_enter),
        "dist_exit": None if float(args.dist_exit) <= 0 else float(args.dist_exit),
        "margin_ratio": float(args.margin_ratio),
        "heading_weight": float(args.heading_weight),
        "outside_penalty_weight": float(args.outside_penalty_weight),
    }
    corridor_cfg_off = dict(corridor_cfg_on)
    corridor_cfg_off["enabled"] = False

    env_off = _build_env(cfg, pm, corridor_cfg=corridor_cfg_off)
    env_on = _build_env(cfg, pm, corridor_cfg=corridor_cfg_on)
    pl, pr = _extract_band(env_on)

    print(f"[RUN] turn={args.turn} dt={dt} max_steps={max_steps} side={side:.2f} episodes={args.episodes} seed={args.seed}")
    print(f"[RUN] corridor(theta_enter={args.theta_enter_deg:.1f}°, theta_exit={args.theta_exit_deg:.1f}°, margin_ratio={args.margin_ratio:.3f})")

    s_off, trace_off = _run_episodes(
        env_off,
        episodes=int(args.episodes),
        seed=int(args.seed),
        vel=float(args.vel),
        kp=float(args.kp),
        keep_first_trace=True,
    )
    s_on, trace_on = _run_episodes(
        env_on,
        episodes=int(args.episodes),
        seed=int(args.seed),
        vel=float(args.vel),
        kp=float(args.kp),
        keep_first_trace=True,
    )

    print(
        f"[EVAL] corridor=OFF episodes={s_off.episodes} success_rate={s_off.success_rate:.3f} "
        f"oob_rate={s_off.oob_rate:.3f} steps_mean={s_off.steps_mean:.1f} "
        f"mean_e_n_corner={s_off.mean_e_n_corner:.4f} in_corridor_ratio={s_off.in_corridor_ratio:.3f}"
    )
    print(
        f"[EVAL] corridor=ON  episodes={s_on.episodes} success_rate={s_on.success_rate:.3f} "
        f"oob_rate={s_on.oob_rate:.3f} steps_mean={s_on.steps_mean:.1f} "
        f"mean_e_n_corner={s_on.mean_e_n_corner:.4f} in_corridor_ratio={s_on.in_corridor_ratio:.3f}"
    )

    if trace_off is None or trace_on is None:
        print("[FAIL] missing traces.")
        raise SystemExit(2)

    traj_path = outdir / "trajectory_compare.png"
    _plot_trajectory(
        out_path=traj_path,
        pm=pm,
        pl=pl,
        pr=pr,
        trace_off=trace_off,
        trace_on=trace_on,
        title=f"Right-angle ({args.turn}) · Corridor ON/OFF",
    )

    en_path = outdir / "e_n_compare.png"
    _plot_lateral_error(
        out_path=en_path,
        trace_off=trace_off,
        trace_on=trace_on,
        title="Signed lateral error e_n (corridor_status) with corridor bounds/target",
    )

    # 最小有效性检查：走廊 ON 的 corner_phase 内应出现与 turn 同号的偏置趋势
    if args.turn == "left" and s_on.mean_e_n_corner <= 1e-3:
        print("[FAIL] corridor ON mean_e_n_corner not positive for left-turn.")
        raise SystemExit(2)
    if args.turn == "right" and s_on.mean_e_n_corner >= -1e-3:
        print("[FAIL] corridor ON mean_e_n_corner not negative for right-turn.")
        raise SystemExit(2)
    if s_on.oob_rate > s_off.oob_rate + 0.10:
        print("[FAIL] oob_rate worsened too much with corridor enabled.")
        raise SystemExit(2)

    print(f"[OUT] {traj_path}")
    print(f"[OUT] {en_path}")
    print("[PASS] P3.1 corridor right-angle compare passed.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
