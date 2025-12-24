"""P8.1 自动化验收（Phase A / Step 1-3）。

目标：
1) 观测语义一致性：state[3]=scan.dist_to_turn，state[5]=scan.turn_angle（严格断言）
2) corner_phase 隔离：enable_corridor 关闭时不得影响全局状态；开启时仅由 scan 决定进入/退出
3) dist_to_turn 必须为弧长（Arc Length），并与欧式距离可区分
4) Square 专家策略（PD + speed_target）闭环验证并产出图表/trace/summary

默认产出目录（符合 Runbook DoD）：
- artifacts/phaseA/square_v_ratio_exec.png
- artifacts/phaseA/square_v_ratio_cap.png
- artifacts/phaseA/square_v_cap_breakdown.png
- artifacts/phaseA/square_speed_util.png
- artifacts/phaseA/square_e_n.png
- artifacts/phaseA/summary.json
- artifacts/phaseA/trace.csv
- artifacts/phaseA/path.png

也支持应力测试：
python tools/accept_p8_1_observation_and_corner_phase.py --config configs/stress_tiny_eps_high_vel.yaml --out artifacts/phaseA_stress
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # PPO_project
sys.path.insert(0, str(ROOT))

V_RATIO_CMD_K = 0.9
V_RATIO_CMD_MIN = 0.02
V_RATIO_CMD_SMOOTHING = 0.2
RECENTER_STEPS = 12
STRAIGHT_OMEGA_MAX = 0.6  # 恢复到合理值
CORNER_OMEGA_SCALE = 0.55  # Patch P8.1 v2: 直接作为 omega_ratio，~0.55 可在 ~30 步完成 90° 转弯
RECOVERY_E_ON_RATIO = 0.40  # 适中的阈值
RECOVERY_E_OFF_RATIO = 0.20
RECOVERY_OMEGA_RATIO_BOOST = 1.5
RECOVERY_OMEGA_RATIO_MAX = 0.85
V_RECOVERY = 0.06
V_RECOVERY_CAP_RATIO = 0.9

# Patch P8.1 (ExitDriftFix): 低速时限制 omega，防止漂移
LOW_VCAP_OMEGA_LIMIT_ENABLE = True  # 启用低速 omega 限制
LOW_VCAP_THRESHOLD = 0.10           # v_cap < 10% 时触发限制
LOW_VCAP_OMEGA_MAX = 0.45           # 低速时最大 omega ratio

STRAIGHT_K_PSI = 0.9
STRAIGHT_K_I = 0.0

GATE_V_RATIO_EXEC_NUNIQUE_MIN = 10
GATE_SPEED_UTIL_IN_BAND_MIN = 0.7
GATE_V_RATIO_CAP_STD_MIN = 0.01
GATE_CORR_V_EXEC_V_CAP_MIN = 0.3
GATE_MAX_ABS_E_N_RATIO = 0.98
GATE_CAP_ANG_ACTIVE_RATIO_MIN = 0.05
# Patch-3: ang cap 在直线段的允许偏差（相对于 corner_mode_ratio）
GATE_CAP_ANG_EXCESS_OVER_CORNER_MAX = 0.2
DIST_STRAIGHT_THRESHOLD = 2.0
DIST_TURN_THRESHOLD = 0.5
NUNIQUE_DECIMALS = 4
SPEED_UTIL_EPS = 1e-9
TURN_ANGLE_EPS = 1e-6


def _parse_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return bool(default)


# Feature flags（便于快速回滚对比）
EXPERT_STRAIGHT_PD_ENABLE = _parse_env_bool("EXPERT_STRAIGHT_PD_ENABLE", True)
EXPERT_RECOVERY_MODE_ENABLE = _parse_env_bool("EXPERT_RECOVERY_MODE_ENABLE", True)
# Patch-2: turn completion 按 turn_angle 截断
TURN_COMPLETION_CLAMP_ENABLE = _parse_env_bool("TURN_COMPLETION_CLAMP_ENABLE", True)
# Patch-3: 诊断 ang cap 在直线段的滞留（默认开启检测）
CAP_ANG_STUCK_CHECK_ENABLE = _parse_env_bool("CAP_ANG_STUCK_CHECK_ENABLE", True)

try:
    import matplotlib.pyplot as plt  # noqa: E402

    from src.environment import Env  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python.cmd -m pip install -r PPO_project/requirements.txt")
    raise


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_trace_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    headers: List[str] = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _plot_series(
    *,
    xs: Sequence[float],
    series: Sequence[Tuple[str, Sequence[float]]],
    out_path: Path,
    title: str,
    ylabel: str,
    hlines: Optional[Sequence[Tuple[float, str]]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    for name, ys in series:
        ax.plot(xs[: len(ys)], ys, lw=1.2, label=name)
    if hlines:
        for y, label in hlines:
            ax.axhline(float(y), ls="--", lw=1.0, color="#868e96", label=label)
    ax.grid(True, ls=":", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


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


def _plot_path(
    *,
    env: Env,
    trajectory: Sequence[Sequence[float]],
    out_path: Path,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    mark_start_end: bool = False,
) -> None:
    pm = _as_xy_points(getattr(env, "Pm", []))
    cache = getattr(env, "cache", {}) or {}
    pl = _as_xy_points(cache.get("Pl"))
    pr = _as_xy_points(cache.get("Pr"))

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    if pm:
        xs, ys = zip(*pm)
        ax.plot(xs, ys, lw=1.0, color="#1971c2", label="Pm")
    if pl:
        xs, ys = zip(*pl)
        ax.plot(xs, ys, lw=0.9, color="#12b886", alpha=0.8, label="Pl")
    if pr:
        xs, ys = zip(*pr)
        ax.plot(xs, ys, lw=0.9, color="#12b886", alpha=0.8, label="Pr")
    if trajectory:
        xs, ys = zip(*trajectory)
        ax.plot(xs, ys, lw=1.2, color="#f76707", label="traj")
        if mark_start_end:
            ax.scatter([xs[0]], [ys[0]], s=30, color="#2f9e44", zorder=5, label="start")
            ax.scatter([xs[-1]], [ys[-1]], s=30, color="#e03131", zorder=5, label="end")

    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, ls=":", lw=0.6)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _compute_zoom_limits(
    trajectory: Sequence[Sequence[float]],
    env: Env,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if not trajectory:
        return None, None
    xs, ys = zip(*[(float(p[0]), float(p[1])) for p in trajectory if p is not None and len(p) >= 2])
    if not xs or not ys:
        return None, None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = float(max(max_x - min_x, max_y - min_y, 1e-6))
    half_eps = float(getattr(env, "half_epsilon", 0.0))
    pad = float(max(0.5 * span, 4.0 * half_eps, 0.2))
    return (min_x - pad, max_x + pad), (min_y - pad, max_y + pad)


def build_line(length: float, num_points: int, angle: float = 0.0) -> List[np.ndarray]:
    ts = np.linspace(0.0, 1.0, max(2, int(num_points)))
    dx = math.cos(float(angle))
    dy = math.sin(float(angle))
    return [np.array([float(length) * t * dx, float(length) * t * dy], dtype=float) for t in ts]


def build_open_square(side: float, num_points: int) -> List[np.ndarray]:
    if num_points < 12:
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
        return_normalized_obs=True,
    )


def _set_pose_at_s(env: Env, *, s: float, lateral_offset: float = 0.0) -> float:
    """把 env 放到弧长 s 的位置，并沿法向偏移 lateral_offset。返回实际投影的 s_now。"""
    p_on_path = np.array(env._interpolate_progress_point_at_s(float(s)), dtype=float)
    theta = float(env._tangent_angle_at_s(float(s)))
    n_hat = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

    env.current_position = p_on_path + float(lateral_offset) * n_hat
    env.heading = float(theta)
    env._current_direction_angle = float(theta)

    seg_idx = int(env._find_containing_segment(env.current_position))
    if seg_idx >= 0:
        env.current_segment_idx = int(seg_idx)

    _proj, _seg_idx_progress, s_now, _t_hat, _n_hat = env._project_onto_progress_path(env.current_position)
    scan = env._scan_for_next_turn(float(s_now))
    dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
    if not math.isfinite(dist_to_turn):
        dist_to_turn = float(getattr(env, "_progress_total_length", 0.0))
    turn_angle = float(scan.get("turn_angle", 0.0))
    tau_next = float(env.calculate_direction_deviation(env.current_position))
    overall_progress = (
        env._calculate_closed_path_progress(env.current_position)
        if bool(getattr(env, "closed", False))
        else env._calculate_path_progress(env.current_position)
    )
    lookahead_features = env._compute_lookahead_features()
    base_state = np.array(
        [
            0.0,  # theta_prime
            0.0,  # length_prime
            tau_next,
            dist_to_turn,
            overall_progress,
            turn_angle,
            float(getattr(env, "velocity", 0.0)),
            float(getattr(env, "acceleration", 0.0)),
            float(getattr(env, "jerk", 0.0)),
            float(getattr(env, "angular_vel", 0.0)),
            float(getattr(env, "angular_acc", 0.0)),
            float(getattr(env, "angular_jerk", 0.0)),
        ],
        dtype=float,
    )
    env.state = np.concatenate([base_state, lookahead_features])
    return float(s_now)


def _assert_close(a: float, b: float, tol: float, msg: str) -> None:
    if not (abs(float(a) - float(b)) <= float(tol)):
        raise AssertionError(f"{msg}: a={a} b={b} tol={tol}")


def _assert_state_semantics(env: Env, *, tol: float = 1e-6) -> None:
    """Step 1：state[3]/state[5] 与 scan 输出绑定一致。"""
    _proj, _seg_idx, s_now, _t_hat, _n_hat = env._project_onto_progress_path(env.current_position)
    scan = env._scan_for_next_turn(float(s_now))
    dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
    if not math.isfinite(dist_to_turn):
        dist_to_turn = float(getattr(env, "_progress_total_length", 0.0))
    turn_angle = float(scan.get("turn_angle", 0.0))

    if getattr(env, "state", None) is None or len(env.state) < 6:
        raise AssertionError("env.state not initialized or too short")
    _assert_close(float(env.state[3]), dist_to_turn, tol, "state[3] != scan.dist_to_turn")
    _assert_close(float(env.state[5]), turn_angle, tol, "state[5] != scan.turn_angle")


def _expert_policy(
    *,
    env: Env,
    k_pursuit: float = 3.5,  # 提高pure pursuit增益
    k_e: float = 3.0,  # 进一步提高k_e增益
    k_psi: float = STRAIGHT_K_PSI,
    k_i: float = STRAIGHT_K_I,
    v_ratio_cmd_prev: Optional[float] = None,
    recenter_state: Optional[MutableMapping[str, object]] = None,
) -> Tuple[np.ndarray, float, float, bool]:
    """专家策略：Pure Pursuit（Phase A 闭环健康验证用）。

    关键点：
    - 目标点取弧长 `s_now + L` 的中心线点，提前进入转弯
    - 横向误差项用于回中，避免出弯外飘
    - 速度跟随 v_ratio_cap，确保速度剖面被真实激活
    - 进入 Corner Mode 时改用可行圆弧角速度
    """
    proj, _seg_idx, s_now, _t_hat, n_hat = env._project_onto_progress_path(env.current_position)
    pos = np.asarray(env.current_position, dtype=float)

    half_eps = float(max(float(getattr(env, "half_epsilon", 1.0)), 1e-6))
    heading = float(getattr(env, "heading", getattr(env, "_current_direction_angle", 0.0)))
    scan = env._scan_for_next_turn(float(s_now))
    dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
    if not math.isfinite(dist_to_turn):
        dist_to_turn = float(getattr(env, "_progress_total_length", 0.0))

    # 预瞄距离：直线段更远，近拐角更短以避免切角
    L_min = float(max(1.5 * half_eps, 0.8))
    L_max = float(max(4.0 * half_eps, 2.5))
    L = float(np.clip(dist_to_turn + L_min, L_min, L_max))
    p_target = np.asarray(env._interpolate_progress_point_at_s(float(s_now) + L), dtype=float)
    theta_des = float(math.atan2(float(p_target[1] - pos[1]), float(p_target[0] - pos[0])))
    heading_err = float(env._wrap_angle(theta_des - heading))
    turn_angle = float(scan.get("turn_angle", 0.0))
    preturn_dist = float(3.0 * half_eps)
    if preturn_dist > 1e-6 and dist_to_turn < preturn_dist:
        blend = float(np.clip(1.0 - dist_to_turn / preturn_dist, 0.0, 1.0))
        heading_err = float(env._wrap_angle(heading_err + 0.6 * turn_angle * blend))
    if math.isfinite(dist_to_turn) and dist_to_turn > preturn_dist:
        theta_path = float(env._tangent_angle_at_s(float(s_now)))
        heading_err = float(env._wrap_angle(theta_path - heading))
    e_n = float(np.dot(pos - proj, n_hat))
    turn_boost = 1.0
    if dist_to_turn < 2.0 * half_eps:
        turn_boost = 1.5
    elif dist_to_turn < 4.0 * half_eps:
        turn_boost = 1.2
    # e_n: left + / right -；omega_ratio: CCW(左转) +
    # 横向误差纠偏符号必须是 (-e_n)，否则直线段会“慢漂移 -> 末端越界”。
    omega_ratio = float(np.clip(turn_boost * k_pursuit * heading_err + k_e * (0.0 - e_n) / half_eps, -1.0, 1.0))

    if recenter_state is None:
        recenter_state = {}

    # Patch-2: turn completion 追踪（防止 theta_prog 跑到 2π）
    theta_prog = float(recenter_state.get("theta_prog", 0.0))
    turn_done = bool(recenter_state.get("turn_done", False))
    prev_turn_angle_abs = float(recenter_state.get("prev_turn_angle_abs", 0.0))
    dt = float(max(float(getattr(env, "interpolation_period", 0.01)), 1e-6))

    corner_mode = False
    r_allow = float("nan")
    if math.isfinite(turn_angle) and abs(turn_angle) > TURN_ANGLE_EPS and half_eps > 1e-9:
        sin_min = float(getattr(env, "_p8_ang_cap_sin_min", 0.2))
        sin_min = float(np.clip(sin_min, 1e-6, 1.0))
        sin_half = float(math.sin(min(abs(turn_angle) * 0.5, 0.5 * math.pi)))
        sin_half = float(max(sin_half, sin_min))
        r_allow = float(half_eps / max(sin_half, 1e-9))
        fillet_scale = float(getattr(env, "_p8_corner_fillet_scale", 1.0))
        fillet_scale = float(max(fillet_scale, 0.0))
        d_fillet = float(r_allow * math.tan(abs(turn_angle) * 0.5) * max(fillet_scale, 1e-6))
        if math.isfinite(dist_to_turn) and math.isfinite(d_fillet) and dist_to_turn <= d_fillet:
            # Patch-2: 若 turn_done 且仍在同一个弯，强制退出 corner_mode
            if TURN_COMPLETION_CLAMP_ENABLE and turn_done and abs(abs(turn_angle) - prev_turn_angle_abs) < 0.1:
                corner_mode = False
            else:
                corner_mode = True
                # 进入新的转弯时复位 turn progress
                if abs(abs(turn_angle) - prev_turn_angle_abs) >= 0.1:
                    theta_prog = 0.0
                    turn_done = False
                    prev_turn_angle_abs = float(abs(turn_angle))

    in_recovery_now = False
    integ_e = 0.0
    if corner_mode and math.isfinite(r_allow) and r_allow > 0.0:
        # Patch P8.1 (ExitDriftFix v2): 使用基于 r_allow 的目标 omega
        # 核心问题：v_cap_geom 被预瞄曲率压得很低（0.01-0.04），导致 omega 太小无法完成转弯
        # 解决方案：使用能在 r_allow 半径下安全转弯的目标速度（而非被压低的 v_cap）
        
        # 计算在 r_allow 半径下能达到的最大角速度
        # omega_max_at_r = MAX_ANG_VEL（直接使用最大角速度）
        # 对应的线速度 v = omega * r_allow
        r_min = float(max(half_eps, 1e-6))
        r_eff = float(max(r_allow, r_min))
        
        # 目标角速度：基于转弯半径，使用适中的 omega_ratio（不要太大导致漂移）
        # CORNER_OMEGA_SCALE 控制转弯强度
        omega_target_ratio = float(CORNER_OMEGA_SCALE)  # 直接使用 scale 作为 omega_ratio
        omega_target = float(omega_target_ratio * float(env.MAX_ANG_VEL))
        omega_target = float(math.copysign(omega_target, turn_angle))
        omega_ratio = float(np.clip(omega_target / float(env.MAX_ANG_VEL), -1.0, 1.0))
        
        # Patch P8.1 (ExitDriftFix): 低速时限制 omega 上限
        # 不在 corner_mode 内限制，因为转弯需要大 omega

        # Patch-2: 累计转弯进度，按 turn_angle 截断
        if TURN_COMPLETION_CLAMP_ENABLE:
            omega_exec = float(getattr(env, "angular_vel", 0.0))
            theta_prog += float(abs(omega_exec) * dt)
            theta_prog = float(np.clip(theta_prog, 0.0, abs(turn_angle)))
            if theta_prog >= abs(turn_angle) - 1e-6:
                turn_done = True
                # 转弯完成后强制进入直线段闭环（下一步生效）
    else:
        # Patch-2: 离开 corner_mode 时复位 turn progress
        if TURN_COMPLETION_CLAMP_ENABLE and turn_done:
            theta_prog = 0.0
            # turn_done 保持 True 直到进入新的弯

        straight_mode = bool(math.isfinite(dist_to_turn) and dist_to_turn > preturn_dist)
        if (not EXPERT_STRAIGHT_PD_ENABLE) and straight_mode:
            # 回滚：保留旧版直线段逻辑
            omega_ratio = float(np.clip(k_e * (e_n / half_eps), -STRAIGHT_OMEGA_MAX, STRAIGHT_OMEGA_MAX))
        elif EXPERT_STRAIGHT_PD_ENABLE and straight_mode:
            # Patch-1：直线段稳定闭环（去死区 + 软限幅）
            e_n_norm = float(e_n / half_eps)
            psi_err = float(heading_err)  # dist_to_turn>preturn_dist 时已绑定为切向误差

            in_recovery = bool(recenter_state.get("in_recovery", False))
            if EXPERT_RECOVERY_MODE_ENABLE:
                aen = abs(e_n_norm)
                if in_recovery:
                    if aen <= float(RECOVERY_E_OFF_RATIO):
                        in_recovery = False
                else:
                    if aen >= float(RECOVERY_E_ON_RATIO):
                        in_recovery = True
            else:
                in_recovery = False

            omega_ratio_max = float(STRAIGHT_OMEGA_MAX)
            if in_recovery:
                omega_ratio_max = float(
                    min(float(RECOVERY_OMEGA_RATIO_MAX), float(STRAIGHT_OMEGA_MAX) * float(RECOVERY_OMEGA_RATIO_BOOST))
                )

            # Patch P8.1 (ExitDriftFix): 低速时限制 omega，防止漂移
            # 使用上一步的 v_ratio_cmd 作为当前 v_cap 的近似
            if LOW_VCAP_OMEGA_LIMIT_ENABLE and v_ratio_cmd_prev is not None:
                v_cap_approx = float(v_ratio_cmd_prev)
                if v_cap_approx < float(LOW_VCAP_THRESHOLD):
                    omega_ratio_max = float(min(omega_ratio_max, float(LOW_VCAP_OMEGA_MAX)))

            if float(k_i) != 0.0:
                integ_e = float(recenter_state.get("integ_e", 0.0))
                dt = float(max(float(getattr(env, "interpolation_period", 0.0)), 0.0))
                integ_e = float(np.clip(integ_e + e_n_norm * dt, -5.0, 5.0))

            omega_raw = float((0.0 - float(k_e)) * e_n_norm + float(k_psi) * psi_err + float(k_i) * integ_e)
            omega_ratio = float(np.clip(omega_raw, -omega_ratio_max, omega_ratio_max))
            in_recovery_now = bool(in_recovery)

    if not EXPERT_STRAIGHT_PD_ENABLE:
        # 旧版：出弯“硬归零窗口”（仅回滚时保留，默认不启用）
        prev_corner_mode = bool(recenter_state.get("prev_corner_mode", False))
        recenter_left = int(recenter_state.get("recenter_left", 0))
        if prev_corner_mode and not corner_mode:
            recenter_left = int(RECENTER_STEPS)
        if recenter_left > 0:
            omega_ratio = 0.0
            recenter_left -= 1
        recenter_state["prev_corner_mode"] = bool(corner_mode)
        recenter_state["recenter_left"] = int(recenter_left)

    recenter_state["in_recovery"] = bool(in_recovery_now)
    recenter_state["integ_e"] = float(integ_e)
    # Patch-2: 保存 turn progress 状态
    recenter_state["theta_prog"] = float(theta_prog)
    recenter_state["turn_done"] = bool(turn_done)
    recenter_state["prev_turn_angle_abs"] = float(prev_turn_angle_abs)

    p4 = env._compute_p4_pre_step_status()
    v_ratio_cap = float(p4.get("v_ratio_cap", float("nan")))
    if not math.isfinite(v_ratio_cap):
        v_ratio_cap = float(p4.get("speed_target", 0.0))
    if not math.isfinite(v_ratio_cap):
        v_ratio_cap = 0.0

    v_ratio_cmd = float(np.clip(V_RATIO_CMD_K * v_ratio_cap, V_RATIO_CMD_MIN, 1.0))
    if v_ratio_cmd_prev is not None and math.isfinite(v_ratio_cmd_prev):
        v_ratio_cmd = float((1.0 - V_RATIO_CMD_SMOOTHING) * v_ratio_cmd_prev + V_RATIO_CMD_SMOOTHING * v_ratio_cmd)

    recenter_state["omega_ratio_prev"] = float(omega_ratio)

    if EXPERT_STRAIGHT_PD_ENABLE and EXPERT_RECOVERY_MODE_ENABLE and in_recovery_now:
        v_ratio_cmd = float(min(float(v_ratio_cmd), float(V_RECOVERY)))
        if math.isfinite(v_ratio_cap):
            v_ratio_cmd = float(min(float(v_ratio_cmd), float(V_RECOVERY_CAP_RATIO) * float(v_ratio_cap)))

    return np.array([omega_ratio, v_ratio_cmd], dtype=float), float(v_ratio_cmd), float(v_ratio_cap), bool(corner_mode)


@dataclass
class EpisodeResult:
    reached_target: bool
    stall_triggered: bool
    max_abs_e_n: float
    rmse_e_n: float
    mean_v_ratio_exec: float
    trace_rows: List[MutableMapping[str, object]]
    trajectory: List[Tuple[float, float]]


def _run_square_expert_episode(env: Env) -> EpisodeResult:
    obs = env.reset()
    if not isinstance(obs, np.ndarray):
        raise AssertionError("Env.reset() did not return ndarray obs")

    trace_rows: List[MutableMapping[str, object]] = []
    trajectory: List[Tuple[float, float]] = []
    pos0 = getattr(env, "current_position", None)
    if pos0 is not None and len(pos0) >= 2:
        trajectory.append((float(pos0[0]), float(pos0[1])))

    e_n_series: List[float] = []
    v_ratio_exec_series: List[float] = []
    lookahead_s_mid_seen = False
    kappa_near_corner_nonzero = False

    done = False
    last_dkappa_exec = 0.0
    v_ratio_cmd_prev: Optional[float] = None
    recenter_state: Dict[str, object] = {}
    while not done:
        action, v_ratio_cmd, v_ratio_cap_pre, corner_mode = _expert_policy(
            env=env,
            v_ratio_cmd_prev=v_ratio_cmd_prev,
            recenter_state=recenter_state,
        )
        v_ratio_cmd_prev = v_ratio_cmd
        obs, _reward, done, info = env.step(action)

        if not isinstance(obs, np.ndarray):
            raise AssertionError("Env.step() did not return ndarray obs")
        if not np.all(np.isfinite(obs)):
            raise AssertionError("obs contains NaN/Inf")

        _assert_state_semantics(env)

        proj, _seg_idx, s_now, _t_hat, n_hat = env._project_onto_progress_path(env.current_position)
        e_n = float(np.dot(env.current_position - proj, n_hat))
        scan = env._scan_for_next_turn(float(s_now))
        dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
        if not math.isfinite(dist_to_turn):
            dist_to_turn = float(getattr(env, "_progress_total_length", 0.0))
        turn_angle = float(scan.get("turn_angle", 0.0))

        p4 = info.get("p4_status", {}) if isinstance(info, dict) else {}
        if not isinstance(p4, dict):
            p4 = {}
        v_ratio_cap = float(p4.get("v_ratio_cap", float("nan")))
        v_ratio_cap_brake = float(p4.get("v_ratio_cap_brake", float("nan")))
        v_ratio_cap_ang = float(p4.get("v_ratio_cap_ang", float("nan")))
        v_ratio_exec = float(p4.get("v_ratio_exec", float("nan")))
        omega_exec = float(p4.get("omega_exec", float(getattr(env, "angular_vel", 0.0))))
        dkappa_exec = float(p4.get("dkappa_exec", float("nan")))
        stall_triggered = bool(float(p4.get("stall_triggered", 0.0)) > 0.5)

        if not math.isfinite(v_ratio_cap):
            v_ratio_cap = v_ratio_cap_pre
        if not math.isfinite(v_ratio_cap):
            v_ratio_cap = float("nan")

        speed_util = float("nan")
        if math.isfinite(v_ratio_exec) and math.isfinite(v_ratio_cap):
            speed_util = float(v_ratio_exec / (v_ratio_cap + SPEED_UTIL_EPS))

        corridor = info.get("corridor_status", {}) if isinstance(info, dict) else {}
        if not isinstance(corridor, dict):
            corridor = {}
        corner_phase = bool(corridor.get("corner_phase", False))

        if not math.isfinite(dkappa_exec):
            dkappa_exec = last_dkappa_exec
        last_dkappa_exec = dkappa_exec

        e_n_series.append(e_n)
        v_ratio_exec_series.append(v_ratio_exec)

        # Step 2：lookahead 不全饱和 + 弯道附近 kappa_rate 非零（用 raw state 检查）
        base_len = int(len(getattr(env, "base_state_keys", [])))
        lookahead_raw = np.asarray(env.state[base_len:], dtype=float).reshape(-1, 3) if len(env.state) > base_len else None
        if lookahead_raw is not None and lookahead_raw.size > 0:
            kappa_rate_raw = lookahead_raw[:, 2]
            # 只要在整个回合中出现过一次非零即可（square 的非零必然发生在拐角附近）
            if np.any(np.abs(kappa_rate_raw) > 1e-10):
                kappa_near_corner_nonzero = True

        lookahead_norm = np.asarray(obs[base_len:], dtype=float).reshape(-1, 3) if obs.size > base_len else None
        if lookahead_norm is not None and lookahead_norm.size > 0:
            s_norm = lookahead_norm[:, 0]
            if np.any((s_norm > 0.05) & (s_norm < 0.95)):
                lookahead_s_mid_seen = True

        pos = getattr(env, "current_position", None)
        if pos is not None and len(pos) >= 2:
            trajectory.append((float(pos[0]), float(pos[1])))

        trace_rows.append(
            {
                "step": int(info.get("step", getattr(env, "current_step", 0))) if isinstance(info, dict) else int(getattr(env, "current_step", 0)),
                "s_now": float(s_now),
                "dist_to_turn": float(dist_to_turn),
                "turn_angle": float(turn_angle),
                "corner_phase": int(1 if corner_phase else 0),
                "corner_mode": int(1 if corner_mode else 0),
                "omega_ratio_cmd": float(action[0]) if action is not None and len(action) > 0 else float("nan"),
                "v_ratio_cmd": float(v_ratio_cmd),
                "v_ratio_cap_brake": float(v_ratio_cap_brake),
                "v_ratio_cap_ang": float(v_ratio_cap_ang),
                "v_ratio_cap": float(v_ratio_cap),
                "v_ratio_exec": float(v_ratio_exec),
                "speed_util": float(speed_util),
                "e_n": float(e_n),
                "omega_exec": float(omega_exec),
                "dkappa_exec": float(dkappa_exec),
                "stall_triggered": int(1 if stall_triggered else 0),
                "progress": float(getattr(env, "state", [0, 0, 0, 0, 0])[4]) if getattr(env, "state", None) is not None and len(env.state) > 4 else 0.0,
            }
        )

    if not e_n_series:
        raise AssertionError("no samples collected")

    e_arr = np.asarray(e_n_series, dtype=float)
    v_arr = np.asarray(v_ratio_exec_series, dtype=float)
    max_abs_e_n = float(np.max(np.abs(e_arr)))
    rmse_e_n = float(math.sqrt(float(np.mean(e_arr * e_arr))))
    mean_v_ratio_exec = float(np.mean(v_arr))
    reached_target = bool(getattr(env, "reached_target", False))
    stall_triggered = bool(getattr(env, "_p4_stall_triggered", False))

    if not lookahead_s_mid_seen:
        raise AssertionError("lookahead s appears fully saturated (no mid-range samples)")
    if not kappa_near_corner_nonzero:
        raise AssertionError("kappa_rate appears zero near corner (lookahead raw kappa_rate all ~0)")

    return EpisodeResult(
        reached_target=reached_target,
        stall_triggered=stall_triggered,
        max_abs_e_n=max_abs_e_n,
        rmse_e_n=rmse_e_n,
        mean_v_ratio_exec=mean_v_ratio_exec,
        trace_rows=trace_rows,
        trajectory=trajectory,
    )


def _test_corner_phase_isolation(cfg: Mapping) -> None:
    """Step 3：直线+横向偏置时 corner_phase 不得误触发；v_cap 不受 alpha 变化影响。"""
    env_cfg = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) if isinstance(cfg.get("kinematic_constraints", {}), dict) else {}

    # 构造足够长直线，避免 dist_to_end/刹车包络干扰 v_cap
    tmp_env = _build_env(cfg, build_line(length=10.0, num_points=50))
    a_abs = float(abs(float(tmp_env.MAX_ACC)))
    v_req = 0.95 * float(tmp_env.MAX_VEL)
    d_req = float(max((v_req * v_req) / max(2.0 * a_abs, 1e-6), 1.0))
    line_len = float(max(200.0, 1.6 * d_req))
    env_line = _build_env(cfg, build_line(length=line_len, num_points=200))

    total = float(getattr(env_line, "_progress_total_length", 0.0))
    s_test = total / 3.0
    y_error = 0.5 * float(env_line.half_epsilon)

    # corridor enabled: corner_phase 不能因 alpha 误触发
    env_line.enable_corridor = True
    env_line.in_corner_phase = False
    _set_pose_at_s(env_line, s=s_test, lateral_offset=y_error)
    cs = env_line._compute_corridor_status()
    if bool(cs.get("corner_phase", False)):
        alpha_dbg = float(cs.get("alpha", float("nan")))
        raise AssertionError(f"corner_phase mis-triggered on straight (alpha={alpha_dbg})")

    # v_cap 不随横向偏移变化
    _set_pose_at_s(env_line, s=s_test, lateral_offset=0.0)
    v0 = float(env_line._compute_p4_pre_step_status().get("v_ratio_cap", float("nan")))
    _set_pose_at_s(env_line, s=s_test, lateral_offset=y_error)
    v1 = float(env_line._compute_p4_pre_step_status().get("v_ratio_cap", float("nan")))
    _assert_close(v0, v1, 1e-3, "v_ratio_cap depends on lateral offset on straight")

    # corridor disabled: 不得更新 self.in_corner_phase，且强制 corner_phase=False
    env_line.enable_corridor = False
    env_line.in_corner_phase = True
    cs2 = env_line._compute_corridor_status()
    if bool(cs2.get("corner_phase", False)):
        raise AssertionError("corner_phase should be forced False when enable_corridor=False")
    if not bool(env_line.in_corner_phase):
        raise AssertionError("enable_corridor=False should not update env.in_corner_phase")

    _ = env_cfg
    _ = kcm_cfg


def _test_dist_to_turn_arc_length(cfg: Mapping) -> None:
    """Step 3.1：Arc vs Euclid 分离测试。"""
    path_cfg = cfg.get("path", {}) if isinstance(cfg.get("path", {}), dict) else {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    env = _build_env(cfg, build_open_square(side=scale, num_points=num_points))
    env.reset()
    scan0 = env._scan_for_next_turn(0.0)
    turn_s = float(scan0.get("turn_s", float("nan")))
    if not math.isfinite(turn_s):
        raise AssertionError("cannot locate turn_s on square path")

    # 选取“离拐角足够近 + 足够横向偏移”的姿态，确保 Arc 与 Euclid 至少相差 5%
    half_eps = float(getattr(env, "half_epsilon", 0.0))
    dist_arc_target = float(max(2.0 * half_eps, 1e-3))
    s_pose = float(max(turn_s - dist_arc_target, 0.0))
    s_now = _set_pose_at_s(env, s=s_pose, lateral_offset=half_eps)
    scan = env._scan_for_next_turn(float(s_now))
    dist_arc = float(scan.get("dist_to_turn", float("inf")))
    if not math.isfinite(dist_arc):
        raise AssertionError("dist_to_turn is not finite on square near turn")

    turn_s2 = float(scan.get("turn_s", float("nan")))
    if not math.isfinite(turn_s2):
        raise AssertionError("turn_s missing in scan")

    corner_xy = np.asarray(env._interpolate_progress_point_at_s(float(turn_s2)), dtype=float)
    dist_euclid = float(np.linalg.norm(corner_xy - np.asarray(env.current_position, dtype=float)))
    rel = float(abs(dist_euclid - dist_arc) / max(dist_arc, 1e-9))
    if rel < 0.05:
        raise AssertionError(f"Arc vs Euclid separation too small: rel_diff={rel:.6f} arc={dist_arc} euclid={dist_euclid}")

    # open path: dist_arc 应等于 max(turn_s - s_now, 0)
    dist_indep = float(max(turn_s2 - s_now, 0.0))
    if abs(dist_indep - dist_arc) > 1e-3:
        raise AssertionError(f"dist_arc mismatch independent arc-length: arc={dist_arc} indep={dist_indep}")

    _assert_state_semantics(env, tol=1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "train_square.yaml"),
        help="环境/动力学配置 YAML（用于读取 epsilon/约束/reward_weights）。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path("artifacts") / "phaseA"),
        help="输出目录（相对路径将基于 PPO_project 解析）。",
    )
    args = parser.parse_args()

    cfg = _load_yaml(_resolve_path(args.config))
    outdir = _resolve_path(args.out)
    _ensure_dir(outdir)

    # ===== 先跑硬性逻辑断言（失败即退出）=====
    _test_corner_phase_isolation(cfg)
    _test_dist_to_turn_arc_length(cfg)

    # ===== 专家策略闭环（Square / open）=====
    path_cfg = cfg.get("path", {}) if isinstance(cfg.get("path", {}), dict) else {}
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))

    env = _build_env(cfg, build_open_square(side=scale, num_points=num_points))
    result = _run_square_expert_episode(env)

    env_cfg = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
    kcm_cfg = cfg.get("kinematic_constraints", {}) if isinstance(cfg.get("kinematic_constraints", {}), dict) else {}
    epsilon = float(env_cfg.get("epsilon", float(getattr(env, "epsilon", float("nan")))))
    max_vel = float(kcm_cfg.get("MAX_VEL", float(getattr(env, "MAX_VEL", float("nan")))))
    is_stress = bool(float(max_vel) >= 10000.0)
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    if is_stress:
        xlim, ylim = _compute_zoom_limits(result.trajectory, env)

    _write_trace_csv(outdir / "trace.csv", result.trace_rows)
    _plot_path(
        env=env,
        trajectory=result.trajectory,
        out_path=outdir / "path.png",
        title="open_square expert policy",
        xlim=xlim,
        ylim=ylim,
        mark_start_end=is_stress,
    )

    xs = [float(r.get("step", i)) for i, r in enumerate(result.trace_rows)]
    v_ratio_exec = [float(r.get("v_ratio_exec", float("nan"))) for r in result.trace_rows]
    v_ratio_cap = [float(r.get("v_ratio_cap", float("nan"))) for r in result.trace_rows]
    v_ratio_cap_brake = [float(r.get("v_ratio_cap_brake", float("nan"))) for r in result.trace_rows]
    v_ratio_cap_ang = [float(r.get("v_ratio_cap_ang", float("nan"))) for r in result.trace_rows]
    speed_util = [float(r.get("speed_util", float("nan"))) for r in result.trace_rows]
    dist_to_turn = [float(r.get("dist_to_turn", float("nan"))) for r in result.trace_rows]
    e_n = [float(r.get("e_n", float("nan"))) for r in result.trace_rows]
    turn_angle = [float(r.get("turn_angle", float("nan"))) for r in result.trace_rows]
    corner_mode = [int(r.get("corner_mode", 0)) for r in result.trace_rows]

    half_eps = float(getattr(env, "half_epsilon", 0.0))
    v_exec_arr = np.asarray(v_ratio_exec, dtype=float)
    v_cap_arr = np.asarray(v_ratio_cap, dtype=float)
    v_cap_brake_arr = np.asarray(v_ratio_cap_brake, dtype=float)
    v_cap_ang_arr = np.asarray(v_ratio_cap_ang, dtype=float)
    e_arr = np.asarray(e_n, dtype=float)
    speed_util_arr = np.asarray(speed_util, dtype=float)
    dist_arr = np.asarray(dist_to_turn, dtype=float)
    turn_angle_arr = np.asarray(turn_angle, dtype=float)
    corner_mode_arr = np.asarray(corner_mode, dtype=float)

    v_exec_clean = v_exec_arr[np.isfinite(v_exec_arr)]
    if v_exec_clean.size > 0:
        v_ratio_exec_nunique = int(np.unique(np.round(v_exec_clean, NUNIQUE_DECIMALS)).size)
    else:
        v_ratio_exec_nunique = 0

    band_mask = np.isfinite(speed_util_arr) & np.isfinite(e_arr) & (np.abs(e_arr) <= half_eps)
    mean_speed_util_in_band = float(np.mean(speed_util_arr[band_mask])) if np.any(band_mask) else float("nan")

    straight_mask = np.isfinite(v_exec_arr) & np.isfinite(dist_arr) & (dist_arr > DIST_STRAIGHT_THRESHOLD)
    mean_v_ratio_exec_straight = float(np.mean(v_exec_arr[straight_mask])) if np.any(straight_mask) else float("nan")

    turn_mask = np.isfinite(v_exec_arr) & np.isfinite(dist_arr) & (dist_arr < DIST_TURN_THRESHOLD)
    mean_v_ratio_exec_near_turn = float(np.mean(v_exec_arr[turn_mask])) if np.any(turn_mask) else float("nan")

    corr_v_exec_vs_v_cap = float("nan")
    v_ratio_cap_std = float("nan")
    corr_mask = np.isfinite(v_exec_arr) & np.isfinite(v_cap_arr)
    if int(np.sum(corr_mask)) >= 2:
        v_exec_corr = v_exec_arr[corr_mask]
        v_cap_corr = v_cap_arr[corr_mask]
        v_ratio_cap_std = float(np.std(v_cap_corr))
        if float(np.std(v_exec_corr)) > 1e-9 and float(np.std(v_cap_corr)) > 1e-9:
            corr_v_exec_vs_v_cap = float(np.corrcoef(v_exec_corr, v_cap_corr)[0, 1])

    cap_ang_active_ratio = float("nan")
    cap_ang_mask = np.isfinite(v_cap_brake_arr) & np.isfinite(v_cap_ang_arr)
    if np.any(cap_ang_mask):
        cap_ang_active_ratio = float(np.mean(v_cap_ang_arr[cap_ang_mask] < v_cap_brake_arr[cap_ang_mask]))

    corner_mode_ratio = float("nan")
    if corner_mode_arr.size > 0:
        corner_mode_ratio = float(np.mean(corner_mode_arr > 0.5))

    _plot_series(
        xs=xs,
        series=[("v_ratio_exec", v_ratio_exec)],
        out_path=outdir / "square_v_ratio_exec.png",
        title="square v_ratio_exec(t)",
        ylabel="ratio",
    )
    _plot_series(
        xs=xs,
        series=[("v_ratio_cap", v_ratio_cap)],
        out_path=outdir / "square_v_ratio_cap.png",
        title="square v_ratio_cap(t)",
        ylabel="ratio",
    )
    _plot_series(
        xs=xs,
        series=[
            ("v_ratio_cap_brake", v_ratio_cap_brake),
            ("v_ratio_cap_ang", v_ratio_cap_ang),
            ("v_ratio_cap_final", v_ratio_cap),
        ],
        out_path=outdir / "square_v_cap_breakdown.png",
        title="square v_ratio_cap breakdown",
        ylabel="ratio",
    )
    _plot_series(
        xs=xs,
        series=[("speed_util", speed_util)],
        out_path=outdir / "square_speed_util.png",
        title="square speed_util(t)",
        ylabel="util",
        hlines=[(1.0, "util=1")],
    )
    _plot_series(
        xs=xs,
        series=[("e_n", e_n)],
        out_path=outdir / "square_e_n.png",
        title="square e_n(t)",
        ylabel="e_n",
        hlines=[(+half_eps, "+half_epsilon"), (-half_eps, "-half_epsilon")],
    )

    summary = {
        "epsilon": float(epsilon),
        "MAX_VEL": float(max_vel),
        "max_abs_e_n": float(result.max_abs_e_n),
        "rmse_e_n": float(result.rmse_e_n),
        "mean_v_ratio_exec": float(result.mean_v_ratio_exec),
        "v_ratio_exec_nunique": int(v_ratio_exec_nunique),
        "mean_speed_util_in_band": float(mean_speed_util_in_band),
        "mean_v_ratio_exec_straight": float(mean_v_ratio_exec_straight),
        "mean_v_ratio_exec_near_turn": float(mean_v_ratio_exec_near_turn),
        "corr_v_exec_vs_v_cap": float(corr_v_exec_vs_v_cap),
        "cap_ang_active_ratio": float(cap_ang_active_ratio),
        "corner_mode_ratio": float(corner_mode_ratio),
        "mean_v_ratio_cap_brake": float(np.nanmean(v_cap_brake_arr)) if v_cap_brake_arr.size > 0 else float("nan"),
        "mean_v_ratio_cap_ang": float(np.nanmean(v_cap_ang_arr)) if v_cap_ang_arr.size > 0 else float("nan"),
        "mean_v_ratio_cap_final": float(np.nanmean(v_cap_arr)) if v_cap_arr.size > 0 else float("nan"),
        "reached_target": bool(result.reached_target),
        "stall_triggered": bool(result.stall_triggered),
        "steps": int(len(result.trace_rows)),
        "outdir": str(outdir),
    }
    _write_json(outdir / "summary.json", summary)

    def _cap_stats(arr: np.ndarray) -> Tuple[float, float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.min(finite)), float(np.mean(finite)), float(np.max(finite))

    def _gate_debug() -> str:
        abs_turn = np.abs(turn_angle_arr[np.isfinite(turn_angle_arr)])
        if abs_turn.size == 0:
            turn_mean = float("nan")
            turn_max = float("nan")
            turn_nonzero = float("nan")
        else:
            turn_mean = float(np.mean(abs_turn))
            turn_max = float(np.max(abs_turn))
            turn_nonzero = float(np.mean(abs_turn > TURN_ANGLE_EPS))
        cap_brake_stats = _cap_stats(v_cap_brake_arr)
        cap_ang_stats = _cap_stats(v_cap_ang_arr)
        cap_final_stats = _cap_stats(v_cap_arr)
        return (
            "gate_debug:"
            f" turn_angle_abs_mean={turn_mean:.6f}"
            f" turn_angle_abs_max={turn_max:.6f}"
            f" turn_angle_nonzero_ratio={turn_nonzero:.3f}"
            f" v_ratio_cap_brake(min/mean/max)={cap_brake_stats[0]:.4f}/{cap_brake_stats[1]:.4f}/{cap_brake_stats[2]:.4f}"
            f" v_ratio_cap_ang(min/mean/max)={cap_ang_stats[0]:.4f}/{cap_ang_stats[1]:.4f}/{cap_ang_stats[2]:.4f}"
            f" v_ratio_cap_final(min/mean/max)={cap_final_stats[0]:.4f}/{cap_final_stats[1]:.4f}/{cap_final_stats[2]:.4f}"
            f" corner_mode_ratio={corner_mode_ratio:.3f}"
        )

    if not is_stress:
        if not summary["reached_target"]:
            raise AssertionError(f"reached_target=False. {_gate_debug()}")
        if summary["stall_triggered"]:
            raise AssertionError(f"stall_triggered=True. {_gate_debug()}")
        if not (math.isfinite(result.max_abs_e_n) and result.max_abs_e_n <= GATE_MAX_ABS_E_N_RATIO * half_eps):
            raise AssertionError(
                f"max_abs_e_n too large: max_abs_e_n={result.max_abs_e_n:.6f} half_eps={half_eps:.6f}. {_gate_debug()}"
            )
        if v_ratio_exec_nunique < GATE_V_RATIO_EXEC_NUNIQUE_MIN:
            raise AssertionError(
                f"Degenerate speed: v_ratio_exec is (almost) constant. Check expert speed command or action mapping. {_gate_debug()}"
            )
        if not (math.isfinite(mean_speed_util_in_band) and mean_speed_util_in_band >= GATE_SPEED_UTIL_IN_BAND_MIN):
            raise AssertionError(
                f"Speed not utilized: expert not tracking v_cap/speed_target, or v_cap is too small everywhere. {_gate_debug()}"
            )
        if not (math.isfinite(v_ratio_cap_std) and v_ratio_cap_std >= GATE_V_RATIO_CAP_STD_MIN):
            raise AssertionError(
                f"v_ratio_cap not varying; braking envelope may be broken or dist_to_turn/turn_angle not wired. {_gate_debug()}"
            )
        if not (math.isfinite(corr_v_exec_vs_v_cap) and corr_v_exec_vs_v_cap >= GATE_CORR_V_EXEC_V_CAP_MIN):
            raise AssertionError(
                f"v_exec does not track v_cap: check expert v_ratio_cmd computation or execution-layer filtering/clipping. {_gate_debug()}"
            )
        if not (math.isfinite(cap_ang_active_ratio) and cap_ang_active_ratio >= GATE_CAP_ANG_ACTIVE_RATIO_MIN):
            raise AssertionError(
                f"cap_ang_active_ratio too low: {cap_ang_active_ratio:.4f}. Check v_ratio_cap_ang wiring/sin_min. {_gate_debug()}"
            )
        # Patch-3: ang cap 滞留检测（直线段 ang cap 占比不应远大于 corner_mode 占比）
        if CAP_ANG_STUCK_CHECK_ENABLE:
            if math.isfinite(cap_ang_active_ratio) and math.isfinite(corner_mode_ratio):
                excess = float(cap_ang_active_ratio - corner_mode_ratio)
                if excess > GATE_CAP_ANG_EXCESS_OVER_CORNER_MAX:
                    # 计算直线段 ang cap 的均值供诊断
                    straight_ang_mask = np.isfinite(dist_arr) & (dist_arr > DIST_STRAIGHT_THRESHOLD) & np.isfinite(v_cap_ang_arr)
                    mean_ang_on_straight = float(np.mean(v_cap_ang_arr[straight_ang_mask])) if np.any(straight_ang_mask) else float("nan")
                    raise AssertionError(
                        f"ang cap likely stuck on straight: cap_ang_active_ratio={cap_ang_active_ratio:.4f} - corner_mode_ratio={corner_mode_ratio:.4f} = {excess:.4f} > {GATE_CAP_ANG_EXCESS_OVER_CORNER_MAX}. "
                        f"mean(v_ratio_cap_ang on straight)={mean_ang_on_straight:.4f}. {_gate_debug()}"
                    )

    # ===== 通过标准：常规/应力二选一 =====
    ok = True
    if not is_stress:
        if summary["stall_triggered"]:
            ok = False
    else:
        if not (summary["stall_triggered"] or float(summary["mean_v_ratio_exec"]) < 0.05):
            ok = False

    print(
        "[P8.1 ACCEPT] "
        f"out={outdir} epsilon={epsilon} MAX_VEL={max_vel} "
        f"mean_v_ratio_exec={summary['mean_v_ratio_exec']:.6f} max_abs_e_n={summary['max_abs_e_n']:.6f} "
        f"stall_triggered={bool(summary['stall_triggered'])} reached_target={bool(summary['reached_target'])}"
    )

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
