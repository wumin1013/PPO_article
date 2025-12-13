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
        **_: object,
    ) -> Tuple[float, Dict[str, float]]:
        """精确复现单文件脚本中的奖励逻辑。"""
        distance_ratio = float(np.clip(contour_error / self.half_epsilon, 0.0, 2.0))
        tracking_reward = 10.0 * np.exp(-3.0 * distance_ratio) - 5.0

        progress_diff = max(0.0, progress - self.last_progress)
        progress_reward = 20.0 * progress_diff

        tau = abs(heading_error)
        direction_reward = 5.0 * np.exp(-2.0 * tau) - 2.5

        velocity_ratio = velocity / max(self.max_vel, 1e-6)
        velocity_reward = 2.0 * (1.0 - abs(velocity_ratio - 0.7))

        jerk_penalty = -4.0 * np.clip(abs(jerk) / max(self.max_jerk, 1e-6), 0.0, 1.0)
        ang_jerk_penalty = -4.0 * np.clip(abs(angular_jerk) / max(self.max_ang_jerk, 1e-6), 0.0, 1.0)
        smoothness_reward = jerk_penalty + ang_jerk_penalty

        constraint_penalty = -2.0 * kcm_intervention

        completion_reward = 0.0
        end_distance_ratio = end_distance / self.half_epsilon
        if progress > 0.8:
            proximity_bonus = 50.0 * (progress - 0.8) * np.exp(-5.0 * end_distance_ratio)
            completion_reward += proximity_bonus
            if end_distance < self.half_epsilon * 0.6:
                completion_reward += 100.0 * np.exp(-10.0 * end_distance_ratio)
                if end_distance < self.half_epsilon * 0.2:
                    completion_reward += 200.0

        if is_closed and lap_completed:
            completion_reward += 150.0

        survival_reward = 0.1

        total = (
            tracking_reward
            + progress_reward
            + direction_reward
            + velocity_reward
            + smoothness_reward
            + constraint_penalty
            + completion_reward
            + survival_reward
        )
        total = float(np.clip(total, -20.0, 100.0))

        self.last_progress = progress

        components = {
            "tracking_reward": float(tracking_reward),
            "progress_reward": float(progress_reward),
            "direction_reward": float(direction_reward),
            "velocity_reward": float(velocity_reward),
            "smoothness_reward": float(smoothness_reward),
            "constraint_penalty": float(constraint_penalty),
            "completion_reward": float(completion_reward),
            "survival_reward": float(survival_reward),
            "total": float(total),
        }
        return total, components
