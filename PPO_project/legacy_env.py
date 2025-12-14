"""Legacy Env extracted from single-file PPO script for parity checks.

精简版，仅保留 reset/step/reward/done 逻辑，用于与新版 Env 做行为一致性对比。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import math
import numpy as np

from src.environment.kinematics import apply_kinematic_constraints
from src.environment.reward import RewardCalculator
from src.utils.geometry import generate_offset_paths, point_to_line_distance, project_point_to_segment


class LegacyEnv:
    def __init__(
        self,
        device,
        epsilon: float,
        interpolation_period: float,
        MAX_VEL: float,
        MAX_ACC: float,
        MAX_JERK: float,
        MAX_ANG_VEL: float,
        MAX_ANG_ACC: float,
        MAX_ANG_JERK: float,
        Pm: Iterable,
        max_steps: int,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.observation_dim = 12
        self.action_space_dim = 2
        self.epsilon = epsilon
        self.half_epsilon = epsilon / 2
        self.rmax = 3 * epsilon
        self.device = device
        self.max_steps = max_steps
        self.interpolation_period = float(interpolation_period)
        self.MAX_VEL = float(MAX_VEL)
        self.MAX_ACC = float(MAX_ACC)
        self.MAX_JERK = float(MAX_JERK)
        self.MAX_ANG_VEL = float(MAX_ANG_VEL)
        self.MAX_ANG_ACC = float(MAX_ANG_ACC)
        self.MAX_ANG_JERK = float(MAX_ANG_JERK)
        self.current_step = 0
        self.trajectory = []
        self.trajectory_states = []

        self.Pm = [np.array(p, dtype=float) for p in Pm]
        self.closed = len(self.Pm) > 2 and np.allclose(self.Pm[0], self.Pm[-1], atol=1e-6)
        self.cache = {}
        self._precompute_geometry()
        self.reward_calculator = RewardCalculator(
            weights=reward_weights or {},
            max_vel=self.MAX_VEL,
            half_epsilon=self.half_epsilon,
            max_jerk=self.MAX_JERK,
            max_ang_jerk=self.MAX_ANG_JERK,
        )
        self.normalization_params = {
            "theta_prime": self.MAX_ANG_VEL,
            "length_prime": self.MAX_VEL,
            "tau_next": math.pi,
            "distance_to_next_turn": self.cache["total_path_length"] or 10.0,
            "overall_progress": 1.0,
            "next_angle": math.pi,
            "velocity": self.MAX_VEL,
            "acceleration": self.MAX_ACC,
            "jerk": self.MAX_JERK,
            "angular_vel": self.MAX_ANG_VEL,
            "angular_acc": self.MAX_ANG_ACC,
            "angular_jerk": self.MAX_ANG_JERK,
        }
        self.reset()

    # ------------------------------------------------------------------
    # 初始化与几何工具
    # ------------------------------------------------------------------
    def _precompute_geometry(self) -> None:
        segment_lengths = []
        angles = []
        segment_directions = []

        for i in range(len(self.Pm) - 1):
            p1 = self.Pm[i]
            p2 = self.Pm[i + 1]
            seg_vec = p2 - p1
            length = float(np.linalg.norm(seg_vec))
            segment_lengths.append(length)
            direction = math.atan2(seg_vec[1], seg_vec[0]) if length > 1e-8 else 0.0
            segment_directions.append(direction)

        for i in range(1, len(self.Pm) - 1):
            v1 = self.Pm[i] - self.Pm[i - 1]
            v2 = self.Pm[i + 1] - self.Pm[i]
            ang = math.atan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2) + 1e-8)
            angles.append(ang)

        if self.closed and len(self.Pm) > 2:
            v1 = self.Pm[0] - self.Pm[-2]
            v2 = self.Pm[1] - self.Pm[0]
            ang = math.atan2(v1[0] * v2[1] - v1[1] * v2[0], np.dot(v1, v2) + 1e-8)
            angles.append(ang)

        pl, pr = generate_offset_paths(self.Pm, self.half_epsilon, closed=self.closed)
        polygons = []
        for i in range(len(pl) - 1):
            if pl[i] is None or pr[i] is None or pl[i + 1] is None or pr[i + 1] is None:
                polygons.append(None)
                continue
            polygons.append([pl[i], pl[i + 1], pr[i + 1], pr[i]])

        self.cache = {
            "segment_lengths": segment_lengths,
            "segment_directions": segment_directions,
            "angles": angles,
            "Pl": pl,
            "Pr": pr,
            "polygons": polygons,
            "total_path_length": sum(segment_lengths) if segment_lengths else 0.0,
        }

    def _get_path_direction(self, pt: np.ndarray) -> float:
        idx = self._find_containing_segment(pt)
        if idx >= 0 and idx < len(self.cache["segment_directions"]):
            return self.cache["segment_directions"][idx]
        return 0.0

    def _find_containing_segment(self, pt: np.ndarray) -> int:
        min_dist = float("inf")
        nearest_idx = -1
        for i in range(len(self.Pm) - 1):
            p1 = self.Pm[i]
            p2 = self.Pm[i + 1]
            projection = project_point_to_segment(pt, p1, p2)
            dist = np.linalg.norm(pt - projection)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        return nearest_idx

    def _get_next_angle(self, segment_idx: int) -> float:
        if segment_idx < len(self.cache["angles"]):
            return self.cache["angles"][segment_idx]
        return 0.0

    def _calculate_path_progress(self, pt: np.ndarray) -> float:
        total_length = self.cache["total_path_length"] or 1.0
        idx = self._find_containing_segment(pt)
        if idx < 0:
            return 0.0
        current_dist = sum(self.cache["segment_lengths"][:idx])
        p1 = self.Pm[idx]
        p2 = self.Pm[idx + 1]
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len > 1e-8:
            t = np.clip(np.dot(pt - p1, seg_vec) / (seg_len**2), 0.0, 1.0)
            current_dist += t * seg_len
        return current_dist / total_length

    def get_contour_error(self, pt: np.ndarray) -> float:
        idx = self._find_containing_segment(pt)
        if idx < 0:
            return 0.0
        p1 = self.Pm[idx]
        p2 = self.Pm[idx + 1]
        return point_to_line_distance(pt, p1, p2)

    # ------------------------------------------------------------------
    # 交互接口
    # ------------------------------------------------------------------
    def reset(self, random_start: bool = False):
        self.current_step = 0
        self.current_position = np.array(self.Pm[0], dtype=float)
        self.trajectory = [self.current_position.copy()]
        self.velocity = self.acceleration = self.jerk = 0.0
        self.angular_vel = self.angular_acc = self.angular_jerk = 0.0
        self.kcm_intervention = 0.0
        self._current_direction_angle = self._get_path_direction(self.current_position)
        distance_to_next_turn = self.cache["segment_lengths"][0] if self.cache["segment_lengths"] else 0.0
        overall_progress = 0.0
        tau_initial = 0.0
        next_angle = self._get_next_angle(0)
        self.state = np.array(
            [
                0.0,
                0.0,
                tau_initial,
                distance_to_next_turn,
                overall_progress,
                next_angle,
                self.velocity,
                self.acceleration,
                self.jerk,
                self.angular_vel,
                self.angular_acc,
                self.angular_jerk,
            ]
        )
        return self.normalize_state(self.state)

    def step(self, action: Tuple[float, float]):
        self.current_step += 1
        prev_vel, prev_acc = self.velocity, self.acceleration
        prev_ang_vel, prev_ang_acc = self.angular_vel, self.angular_acc

        arr = np.array(action, dtype=float).flatten()
        policy_theta = float(arr[0]) if arr.size > 0 else 0.0
        policy_len = float(arr[1]) if arr.size > 1 else 0.0
        policy_theta = np.clip(policy_theta, -1.0, 1.0)
        policy_len = np.clip(policy_len, 0.0, 1.0)
        action_policy = np.array([policy_theta, policy_len], dtype=float)

        (self.velocity, self.acceleration, self.jerk, self.angular_vel, self.angular_acc, self.angular_jerk) = (
            apply_kinematic_constraints(
                prev_vel,
                prev_acc,
                prev_ang_vel,
                prev_ang_acc,
                policy_len,
                policy_theta,
                self.interpolation_period,
                self.MAX_VEL,
                self.MAX_ACC,
                self.MAX_JERK,
                self.MAX_ANG_VEL,
                self.MAX_ANG_ACC,
                self.MAX_ANG_JERK,
            )
        )

        velocity_diff = abs(self.velocity - policy_len)
        angular_vel_diff = abs(self.angular_vel - policy_theta)
        self.kcm_intervention = velocity_diff + angular_vel_diff

        safe_action = (self.angular_vel, self.velocity)
        next_state = self.apply_action(safe_action)
        self.state = next_state

        contour_error = self.get_contour_error(self.current_position)
        heading_error = abs(self.state[2])
        progress = self.state[4]
        end_point = np.array(self.Pm[-1])
        end_distance = float(np.linalg.norm(self.current_position - end_point))

        reward, components = self.reward_calculator.calculate_reward(
            contour_error=contour_error,
            progress=progress,
            velocity=self.velocity,
            heading_error=heading_error,
            kcm_intervention=self.kcm_intervention,
            end_distance=end_distance,
            jerk=self.jerk,
            angular_jerk=self.angular_jerk,
            lap_completed=False,
            is_closed=self.closed,
        )
        self.last_reward_components = components
        done = self.is_done()

        info = {
            "position": self.current_position.copy(),
            "step": self.current_step,
            "contour_error": contour_error,
            "segment_idx": self._find_containing_segment(self.current_position),
            "progress": progress,
            "action_policy": action_policy,
            "action_exec": np.array(safe_action, dtype=float),
            "action_gap_abs": np.abs(np.array(safe_action, dtype=float) - action_policy),
        }
        return self.normalize_state(self.state), reward, done, info

    def apply_action(self, action: Tuple[float, float]) -> np.ndarray:
        theta_prime, length_prime = action
        path_angle = self._get_path_direction(self.current_position)
        effective_angle = path_angle + theta_prime * self.interpolation_period
        self._current_direction_angle = effective_angle
        displacement = length_prime * self.interpolation_period
        dx = displacement * math.cos(effective_angle)
        dy = displacement * math.sin(effective_angle)
        self.current_position = self.current_position + np.array([dx, dy])
        self.trajectory.append(self.current_position.copy())

        tau_next = self.calculate_direction_deviation(self.current_position)
        distance_to_next_turn = 0.0
        segment_idx = self._find_containing_segment(self.current_position)
        if segment_idx >= 0 and segment_idx < len(self.cache["segment_lengths"]):
            next_turn_point = np.array(self.Pm[segment_idx + 1])
            distance_to_next_turn = float(np.linalg.norm(next_turn_point - self.current_position))
        overall_progress = self._calculate_path_progress(self.current_position)
        next_angle = self._get_next_angle(segment_idx)

        return np.array(
            [
                theta_prime,
                length_prime,
                tau_next,
                distance_to_next_turn,
                overall_progress,
                next_angle,
                self.velocity,
                self.acceleration,
                self.jerk,
                self.angular_vel,
                self.angular_acc,
                self.angular_jerk,
            ]
        )

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    def calculate_direction_deviation(self, pt: np.ndarray) -> float:
        path_direction = self._get_path_direction(pt)
        tau = self._current_direction_angle - path_direction
        tau = (tau + np.pi) % (2 * np.pi) - np.pi
        return tau

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(state, dtype=float)
        keys = list(self.normalization_params.keys())
        for i, key in enumerate(keys):
            max_val = self.normalization_params[key]
            if key == "distance_to_next_turn":
                normalized[i] = np.clip(np.log1p(state[i]) / np.log1p(max_val), 0, 1)
            elif key == "overall_progress":
                normalized[i] = state[i]
            else:
                normalized[i] = np.clip(state[i] / max_val, -1, 1)
        return normalized

    def is_done(self) -> bool:
        contour_error = self.get_contour_error(self.current_position)
        if contour_error > self.half_epsilon or self.current_step >= self.max_steps:
            return True
        end_point = np.array(self.Pm[-1])
        end_distance = float(np.linalg.norm(self.current_position - end_point))
        progress = self.state[4] if len(self.state) > 4 else 0.0
        if progress > 0.95 and end_distance < self.half_epsilon * 0.6:
            return True
        return False


__all__ = ["LegacyEnv"]
