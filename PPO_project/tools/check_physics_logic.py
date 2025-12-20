"""P8.0 自验证：真实几何 v_cap + 刹车包络（不训练）。

断言（按 P8_0_最终优化指令.md）：
1) 直线不因横向误差降低 v_cap
2) 拐角刹车包络生效
3) 终点刹车包络生效（Open Path）
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]  # PPO_project
sys.path.insert(0, str(ROOT))

try:
    from src.environment import Env  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装依赖，例如: python.cmd -m pip install -r PPO_project/requirements.txt")
    raise


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
    )


def _set_pose_at_s(env: Env, *, s: float, lateral_offset: float = 0.0) -> None:
    p_on_path = np.array(env._interpolate_progress_point_at_s(float(s)), dtype=float)
    theta = float(env._tangent_angle_at_s(float(s)))
    n_hat = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

    env.current_position = p_on_path + float(lateral_offset) * n_hat
    env.heading = float(theta)
    env._current_direction_angle = float(theta)

    seg_idx = int(env._find_containing_segment(env.current_position))
    if seg_idx >= 0:
        env.current_segment_idx = int(seg_idx)

    tau_next = float(env.calculate_direction_deviation(env.current_position))
    overall_progress = (
        env._calculate_closed_path_progress(env.current_position)
        if bool(getattr(env, "closed", False))
        else env._calculate_path_progress(env.current_position)
    )
    scan = env._scan_for_next_turn(float(s))
    dist_to_turn_state = float(scan.get("dist_to_turn", float("inf")))
    if not math.isfinite(dist_to_turn_state):
        dist_to_turn_state = float(getattr(env, "_progress_total_length", 0.0))
    turn_angle = float(scan.get("turn_angle", 0.0))

    lookahead_features = env._compute_lookahead_features()
    base_state = np.array(
        [
            0.0,  # theta_prime
            0.0,  # length_prime
            tau_next,
            dist_to_turn_state,
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


def _print_status(tag: str, status: Mapping[str, float]) -> None:
    dist_to_turn = status.get("dist_to_turn", float("nan"))
    turn_angle = status.get("turn_angle", float("nan"))
    v_ratio_cap = status.get("v_ratio_cap", float("nan"))
    v_brake_turn_ratio = status.get("v_brake_turn_ratio", float("nan"))
    v_brake_end_ratio = status.get("v_brake_end_ratio", float("nan"))
    speed_target_ratio = status.get("speed_target", float("nan"))
    print(
        f"[{tag}] dist_to_turn={dist_to_turn:.6g} turn_angle={turn_angle:.6g} "
        f"v_ratio_cap={v_ratio_cap:.6g} v_brake_turn_ratio={v_brake_turn_ratio:.6g} "
        f"v_brake_end_ratio={v_brake_end_ratio:.6g} speed_target_ratio={speed_target_ratio:.6g}"
    )


def _assert_close(a: float, b: float, tol: float, msg: str) -> None:
    if not (abs(float(a) - float(b)) <= float(tol)):
        raise AssertionError(f"{msg}: a={a} b={b} tol={tol}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "train_square.yaml"),
        help="训练/环境配置 YAML（用于读取动力学与 reward_weights）。",
    )
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    eps = 1e-6

    # ===== Assert 1: 直线不因横向误差降 v_cap =====
    env_tmp = _build_env(cfg, build_line(length=10.0, num_points=50))
    a_abs = float(abs(float(env_tmp.MAX_ACC)))
    v_req = 0.95 * float(env_tmp.MAX_VEL)
    d_req = float(max((v_req * v_req) / (2.0 * a_abs + eps), 1.0))
    line_len = float(max(200.0, 1.6 * d_req))
    line_pm = build_line(length=line_len, num_points=200)
    env_line = _build_env(cfg, line_pm)

    total = float(getattr(env_line, "_progress_total_length", 0.0))
    s_test = total / 3.0
    y_error = 0.5 * float(env_line.half_epsilon)

    _set_pose_at_s(env_line, s=s_test, lateral_offset=0.0)
    s0 = env_line._compute_p4_pre_step_status()
    _print_status("line_y0", s0)

    _set_pose_at_s(env_line, s=s_test, lateral_offset=y_error)
    s1 = env_line._compute_p4_pre_step_status()
    _print_status("line_yerr", s1)

    v_ratio_0 = float(s0.get("v_ratio_cap", 0.0))
    v_ratio_1 = float(s1.get("v_ratio_cap", 0.0))
    if v_ratio_1 < 0.95:
        raise AssertionError(f"Assert1 failed: v_ratio_cap too low on straight: {v_ratio_1}")
    _assert_close(v_ratio_0, v_ratio_1, 1e-3, "Assert1 failed: v_ratio_cap depends on lateral error")

    # ===== Assert 2: 拐角刹车包络生效 =====
    square_pm = build_open_square(side=10.0, num_points=200)
    env_square = _build_env(cfg, square_pm)
    scan0 = env_square._scan_for_next_turn(0.0)
    turn_s = float(scan0.get("turn_s", float("nan")))
    if not math.isfinite(turn_s):
        raise AssertionError("Assert2 failed: cannot locate turn_s on square path")

    d_test = 1.5 * float(getattr(env_square, "lookahead_spacing", 1.0))
    s_before = float(max(turn_s - d_test, 0.0))
    _set_pose_at_s(env_square, s=s_before, lateral_offset=0.0)
    st2 = env_square._compute_p4_pre_step_status()
    _print_status("square_before_turn", st2)

    v_brake_turn_ratio = float(st2.get("v_brake_turn_ratio", 1.0))
    speed_target_ratio = float(st2.get("speed_target", 1.0))
    if not (speed_target_ratio <= v_brake_turn_ratio + 1e-6):
        raise AssertionError(
            f"Assert2 failed: speed_target_ratio not limited by v_brake_turn_ratio: "
            f"speed_target_ratio={speed_target_ratio} v_brake_turn_ratio={v_brake_turn_ratio}"
        )

    # ===== Assert 3: 终点刹车包络生效（Open Path）=====
    d_end_test = 1.5 * float(getattr(env_line, "lookahead_spacing", 1.0))
    s_end = float(max(total - d_end_test, 0.0))
    _set_pose_at_s(env_line, s=s_end, lateral_offset=0.0)
    st3 = env_line._compute_p4_pre_step_status()
    _print_status("line_near_end", st3)

    v_brake_end = float(math.sqrt(2.0 * a_abs * float(d_end_test) + eps))
    speed_target_phys = float(st3.get("speed_target_phys", float("nan")))
    if not (speed_target_phys <= v_brake_end + 1e-6):
        raise AssertionError(
            f"Assert3 failed: speed_target_phys not limited by v_brake_end: "
            f"speed_target_phys={speed_target_phys} v_brake_end={v_brake_end}"
        )

    print("[OK] All P8.0 physics logic asserts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

