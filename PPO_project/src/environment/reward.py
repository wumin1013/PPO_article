from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class RewardCalculator:
    """Corridor-based reward calculator focused on speed, smoothness, and KCM cooperation."""

    SAFE_RATIO = 0.8
    BUFFER_RATIO = 1.0

    def __init__(
        self,
        weights: Dict[str, float],
        max_vel: float,
        half_epsilon: float,
        safe_ratio: float | None = None,
    ):
        self.weights = weights or {}
        self.max_vel = max_vel
        self.half_epsilon = max(half_epsilon, 1e-6)
        self.safe_ratio = safe_ratio if safe_ratio is not None else self.SAFE_RATIO
        self.buffer_ratio = self.BUFFER_RATIO

        self.last_progress = 0.0
        self.prev_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.last_progress = 0.0
        self.prev_action = None

    def calculate_reward(
        self,
        contour_error: float,
        progress: float,
        velocity: float,
        action: Sequence[float],
        kcm_intervention: float,
        end_distance: float,
        lap_completed: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        error_ratio = float(np.clip(contour_error / self.half_epsilon, 0.0, np.inf))
        speed_reward = self.weights.get("w_velocity", 1.0) * self._normalized_velocity(velocity)
        corridor_reward, zone, buffer_penalty = self._corridor_term(error_ratio, speed_reward)

        progress_reward = self.weights.get("w_progress", 1.0) * max(0.0, progress - self.last_progress)
        smooth_penalty = self._action_smooth_penalty(action)
        kcm_penalty = -self.weights.get("w_kcm_penalty", 0.0) * float(kcm_intervention)
        completion_bonus = self._completion_bonus(end_distance, progress, lap_completed)
        survival_reward = self.weights.get("w_survival", 0.05)

        total = corridor_reward + progress_reward + smooth_penalty + kcm_penalty + completion_bonus + survival_reward
        total = float(np.clip(total, -30.0, 120.0))

        self.prev_action = np.array(action, dtype=float)
        self.last_progress = progress

        components = {
            "zone": zone,
            "speed": float(speed_reward),
            "corridor": float(corridor_reward),
            "buffer_penalty": float(buffer_penalty),
            "progress": float(progress_reward),
            "action_smooth": float(smooth_penalty),
            "kcm_penalty": float(kcm_penalty),
            "completion": float(completion_bonus),
            "survival": float(survival_reward),
            "total": float(total),
        }
        return total, components

    def _normalized_velocity(self, velocity: float) -> float:
        return float(np.clip(velocity / max(self.max_vel, 1e-6), 0.0, 1.2))

    def _corridor_term(self, error_ratio: float, speed_reward: float) -> Tuple[float, str, float]:
        if error_ratio < self.safe_ratio:
            return speed_reward, "safe", 0.0

        if error_ratio <= self.buffer_ratio:
            penalty_strength = (error_ratio - self.safe_ratio) / max(self.buffer_ratio - self.safe_ratio, 1e-6)
            buffer_penalty = self.weights.get("w_e", 1.0) * (penalty_strength**2)
            scaled_speed = speed_reward * (1.0 - 0.5 * penalty_strength)
            return scaled_speed - buffer_penalty, "buffer", buffer_penalty

        violation_penalty = self.weights.get("w_violation", 8.0) * (1.0 + (error_ratio - self.buffer_ratio) * 2.0)
        return -violation_penalty, "violation", violation_penalty

    def _action_smooth_penalty(self, action: Sequence[float]) -> float:
        if self.prev_action is None:
            return 0.0
        delta = np.abs(np.array(action, dtype=float) - self.prev_action)
        return -self.weights.get("w_action_smooth", 0.0) * float(delta.sum())

    def _completion_bonus(self, end_distance: float, progress: float, lap_completed: bool) -> float:
        bonus = 0.0
        if lap_completed:
            bonus += 30.0
        if progress > 0.8:
            proximity = 1.0 - np.clip(end_distance / (self.half_epsilon + 1e-6), 0.0, 1.0)
            bonus += 20.0 * proximity
        return bonus
