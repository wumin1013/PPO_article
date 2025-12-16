from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class RewardCalculator:
    """奖励计算：与单文件版本 `PPO最终版_改进.py` 对齐，不再使用额外权重项。"""

    def __init__(
        self,
        weights: Dict[str, float],
        max_vel: float,
        half_epsilon: float,
        max_jerk: float,
        max_ang_jerk: float,
        safe_ratio: float | None = None,  # 兼容旧参数，逻辑中不再使用
    ):
        self.weights = weights or {}
        self.max_vel = max_vel
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
        corridor_target_error: float = 0.0,
        corridor_outside_distance: float = 0.0,
        corridor_heading_cos: float = 0.0,
        corridor_heading_weight: float = 2.0,
        corridor_outside_penalty_weight: float = 20.0,
        **_: object,
    ) -> Tuple[float, Dict[str, float]]:
        """精确复现单文件脚本中的奖励逻辑。"""
        use_corridor = bool(corridor_enabled and corridor_active)
        tracking_error = float(corridor_target_error if use_corridor else contour_error)
        distance_ratio = float(np.clip(tracking_error / self.half_epsilon, 0.0, 2.0))
        tracking_reward = 10.0 * np.exp(-3.0 * distance_ratio) - 5.0

        progress_diff = max(0.0, progress - self.last_progress)
        progress_reward = float(progress_multiplier) * (20.0 * progress_diff)

        tau = abs(heading_error)
        direction_reward = 5.0 * np.exp(-2.0 * tau) - 2.5

        corridor_outside_penalty = 0.0
        corridor_heading_reward = 0.0
        if use_corridor:
            outside_ratio = float(np.clip(float(corridor_outside_distance) / self.half_epsilon, 0.0, 2.0))
            corridor_outside_penalty = -float(corridor_outside_penalty_weight) * (outside_ratio**2)
            corridor_heading_reward = float(corridor_heading_weight) * float(np.clip(corridor_heading_cos, -1.0, 1.0))

        speed_target_reward = 0.0
        if speed_target is not None and v_ratio_exec is not None:
            err = float(v_ratio_exec) - float(speed_target)
            speed_target_reward = -abs(float(speed_weight)) * (err**2)

        velocity_ratio = velocity / max(self.max_vel, 1e-6)
        velocity_reward = 2.0 * (1.0 - abs(velocity_ratio - 0.7)) if speed_target is None else 0.0

        jerk_penalty = -4.0 * np.clip(abs(jerk) / max(self.max_jerk, 1e-6), 0.0, 1.0)
        ang_jerk_penalty = -4.0 * np.clip(abs(angular_jerk) / max(self.max_ang_jerk, 1e-6), 0.0, 1.0)
        smoothness_reward = jerk_penalty + ang_jerk_penalty

        constraint_penalty = -2.0 * kcm_intervention

        completion_reward = 0.0
        end_distance_ratio = end_distance / self.half_epsilon
        # P4.0：终点接近奖励更依赖 progress（兼容 open-path 过冲 success）
        if progress > 0.8:
            completion_reward += 40.0 * (progress - 0.8) ** 2
            proximity_bonus = 50.0 * (progress - 0.8) * np.exp(-5.0 * end_distance_ratio)
            completion_reward += proximity_bonus
            if end_distance < self.half_epsilon * 0.6:
                completion_reward += 100.0 * np.exp(-10.0 * end_distance_ratio)
                if end_distance < self.half_epsilon * 0.2:
                    completion_reward += 200.0

        if is_closed and lap_completed:
            completion_reward += 150.0

        survival_reward = 0.1
        step_time_penalty = float(time_penalty)
        stall_term_penalty = float(stall_penalty) if bool(stall_triggered) else 0.0

        total = (
            tracking_reward
            + progress_reward
            + direction_reward
            + velocity_reward
            + speed_target_reward
            + smoothness_reward
            + constraint_penalty
            + corridor_outside_penalty
            + corridor_heading_reward
            + completion_reward
            + survival_reward
            + step_time_penalty
            + stall_term_penalty
        )
        total = float(np.clip(total, -20.0, 100.0))

        self.last_progress = progress

        components = {
            "tracking_reward": float(tracking_reward),
            "progress_reward": float(progress_reward),
            "progress_multiplier": float(progress_multiplier),
            "direction_reward": float(direction_reward),
            "velocity_reward": float(velocity_reward),
            "speed_target": float(speed_target) if speed_target is not None else float("nan"),
            "v_ratio_exec": float(v_ratio_exec) if v_ratio_exec is not None else float("nan"),
            "speed_target_reward": float(speed_target_reward),
            "smoothness_reward": float(smoothness_reward),
            "constraint_penalty": float(constraint_penalty),
            "corridor_active": float(1.0 if use_corridor else 0.0),
            "corridor_in_corridor": float(1.0 if (use_corridor and corridor_in_corridor) else 0.0),
            "corridor_outside_penalty": float(corridor_outside_penalty),
            "corridor_heading_reward": float(corridor_heading_reward),
            "completion_reward": float(completion_reward),
            "survival_reward": float(survival_reward),
            "time_penalty": float(step_time_penalty),
            "stall_penalty": float(stall_term_penalty),
            "total": float(total),
        }
        return total, components
