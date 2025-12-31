from __future__ import annotations

from typing import Dict, Tuple


class RewardCalculator:
    """P0 reward: progress-dominant with tracking/heading/time penalties."""

    def __init__(
        self,
        weights: Dict[str, float],
        max_vel: float,
        half_epsilon: float,
        max_jerk: float,
        max_ang_jerk: float,
        max_ang_acc: float | None = None,
        safe_ratio: float | None = None,  # 兼容旧参数，逻辑中不再使用
    ):
        self.weights = weights or {}
        self.max_vel = max_vel
        self.max_ang_acc = max_ang_acc
        self.max_jerk = max_jerk
        self.max_ang_jerk = max_ang_jerk
        self.half_epsilon = max(half_epsilon, 1e-6)
        self.last_progress = 0.0

    def reset(self) -> None:
        self.last_progress = 0.0

    def calculate_reward(
        self,
        contour_error: float,
        progress: float,
        velocity: float,
        heading_error: float,
        kcm_intervention: float,
        end_distance: float,
        jerk: float,
        angular_jerk: float,
        # P6.1: 动作变化率惩罚（policy 层）
        du_theta_u: float = 0.0,
        du_v_u: float = 0.0,
        du_enabled: bool = False,
        du_weight: float = 0.0,
        du_mode: str = "l1",
        lap_completed: bool = False,
        is_closed: bool = False,
        # P4.0: turn-aware speed target / time penalty / exit boost / stall
        v_ratio_exec: float | None = None,
        speed_target: float | None = None,
        speed_weight: float = 6.0,
        time_penalty: float = 0.0,
        progress_multiplier: float = 1.0,
        stall_triggered: bool = False,
        stall_penalty: float = 0.0,
        corridor_enabled: bool = False,
        corridor_active: bool = False,
        corridor_in_corridor: bool = False,
        corridor_target_error: float = 0.0,  # P5.2 起不再用于 shaping（保留兼容字段）
        corridor_outside_distance: float = 0.0,
        corridor_e_n: float = 0.0,
        corridor_margin_to_edge: float = float("nan"),
        corridor_safe_margin: float = 0.0,
        corridor_barrier_scale: float = 0.0,
        corridor_barrier_weight: float = 0.0,
        corridor_center_weight: float = 0.0,
        corridor_center_power: float = 2.0,
        corridor_heading_cos: float = 0.0,
        corridor_heading_weight: float = 2.0,
        corridor_outside_penalty_weight: float = 20.0,
        # P7.1：方向性偏好（tanh 弱引导）
        corridor_corner_phase: bool = False,
        corridor_turn_sign: int = 0,
        corridor_dir_pref_weight: float = 0.0,
        corridor_dir_pref_beta: float = 2.0,
        **_: object,
    ) -> Tuple[float, Dict[str, float]]:
        """P0: progress-dominant reward with pure penalties."""
        w_s = abs(float(self.weights.get("w_s", 20.0)))
        w_e = abs(float(self.weights.get("w_e", 5.0)))
        w_tau = abs(float(self.weights.get("w_tau", 2.0)))
        w_t = abs(float(self.weights.get("w_t", 1.0)))
        w_smooth = abs(float(self.weights.get("w_smooth", 0.0)))

        progress_now = float(progress)
        progress_diff = max(0.0, progress_now - float(self.last_progress))

        error_ratio = abs(float(contour_error)) / max(float(self.half_epsilon), 1e-6)
        tau = abs(float(heading_error))

        r_progress = w_s * progress_diff
        r_track = -w_e * (error_ratio**2)
        r_dir = -w_tau * (tau**2)
        r_time = -w_t

        r_smooth = 0.0
        if w_smooth > 0.0:
            jerk_ratio = abs(float(jerk)) / max(float(self.max_jerk), 1e-6)
            ang_jerk_ratio = abs(float(angular_jerk)) / max(float(self.max_ang_jerk), 1e-6)
            r_smooth = -w_smooth * (jerk_ratio**2 + ang_jerk_ratio**2)

        total = float(r_progress + r_track + r_dir + r_time + r_smooth)

        self.last_progress = progress_now

        components = {
            "progress_diff": float(progress_diff),
            "r_progress": float(r_progress),
            "r_track": float(r_track),
            "r_dir": float(r_dir),
            "r_time": float(r_time),
            "r_smooth": float(r_smooth),
            "total": float(total),
        }
        return total, components
