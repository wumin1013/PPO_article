"""CNC 轨迹环境：解耦遗留实现并引入几何工具函数。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import copy
import json
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
try:
    from rtree import index as rtree_index
except ImportError:  # pragma: no cover
    rtree_index = None

from src.environment.kinematics import apply_kinematic_constraints
from src.environment.reward import RewardCalculator
from src.utils.geometry import (
    generate_offset_paths,
    is_point_in_polygon,
    point_to_line_distance,
    project_point_to_segment,
)

# 固定沿用单文件版本的动力学参数，避免配置漂移导致的震荡
LEGACY_INTERPOLATION_PERIOD = 0.01
LEGACY_KINEMATICS = {
    "MAX_VEL": 1000.0,
    "MAX_ACC": 5000.0,
    "MAX_JERK": 50000.0,
    "MAX_ANG_VEL": math.pi * 2,
    "MAX_ANG_ACC": math.pi * 10,
    "MAX_ANG_JERK": math.pi * 100,
}


class Env:
    def __init__(
        self,
        device,
        epsilon,
        interpolation_period,
        MAX_VEL,
        MAX_ACC,
        MAX_JERK,
        MAX_ANG_VEL,
        MAX_ANG_ACC,
        MAX_ANG_JERK,
        Pm,
        max_steps,
        reward_weights: Optional[Dict[str, float]] = None,
        lookahead_points: int = 5,
        return_normalized_obs: bool = True,
    ):
        self.lookahead_points = max(1, int(lookahead_points))
        self.lookahead_feature_size = 3
        self.base_state_keys = [
            "theta_prime",
            "length_prime",
            "tau_next",
            "distance_to_next_turn",
            "overall_progress",
            "next_angle",
            "velocity",
            "acceleration",
            "jerk",
            "angular_vel",
            "angular_acc",
            "angular_jerk",
        ]
        self.observation_dim = len(self.base_state_keys) + self.lookahead_points * self.lookahead_feature_size
        self.action_space_dim = 2
        self.epsilon = epsilon  # 总带宽（Pl到Pr的距离）
        self.half_epsilon = epsilon / 2  # 单侧偏移距离（Pm到Pl或Pr的距离）
        self.rmax = 3 * epsilon
        self.device = device
        self.max_steps = max_steps
        self.interpolation_period = float(
            interpolation_period if interpolation_period is not None else LEGACY_INTERPOLATION_PERIOD
        )
        self.reward_weights = reward_weights or {}
        # 确保所有约束参数都是浮点数
        self.MAX_VEL = float(MAX_VEL if MAX_VEL is not None else LEGACY_KINEMATICS["MAX_VEL"])
        self.MAX_ACC = float(MAX_ACC if MAX_ACC is not None else LEGACY_KINEMATICS["MAX_ACC"])
        self.MAX_JERK = float(MAX_JERK if MAX_JERK is not None else LEGACY_KINEMATICS["MAX_JERK"])
        self.MAX_ANG_VEL = float(
            MAX_ANG_VEL if MAX_ANG_VEL is not None else LEGACY_KINEMATICS["MAX_ANG_VEL"]
        )
        self.MAX_ANG_ACC = float(
            MAX_ANG_ACC if MAX_ANG_ACC is not None else LEGACY_KINEMATICS["MAX_ANG_ACC"]
        )
        self.MAX_ANG_JERK = float(
            MAX_ANG_JERK if MAX_ANG_JERK is not None else LEGACY_KINEMATICS["MAX_ANG_JERK"]
        )
        self.lookahead_lateral_soft_k = 3.0  # d_i 归一化软因子（越大越不易饱和）
        self.current_step = 0
        # Step 2：归一化链路二选一
        # - return_normalized_obs=True：Env 输出 normalized obs（训练端禁用 StateNormalizer）
        # - return_normalized_obs=False：Env 输出 raw obs（训练端启用 StateNormalizer）
        self.return_normalized_obs = bool(return_normalized_obs)
        self.trajectory = []
        self.trajectory_states = []
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reward_calculator = RewardCalculator(
            weights=self.reward_weights,
            max_vel=self.MAX_VEL,
            half_epsilon=self.half_epsilon,
            max_jerk=self.MAX_JERK,
            max_ang_acc=self.MAX_ANG_ACC,
            max_ang_jerk=self.MAX_ANG_JERK,
        )
        self._log_effective_params()

        # Episode 结束诊断（默认关闭，tools 可开启）
        self.enable_episode_diagnostics = False
        self._episode_summary_printed = False

        # P3.1 VirtualCorridor（走廊奖励/滞回开关）
        self.enable_corridor = False
        self._corridor_theta_enter = math.radians(15.0)
        self._corridor_theta_exit = math.radians(8.0)
        self._corridor_dist_enter = 0.0
        self._corridor_dist_exit = 0.0
        self._corridor_margin = 0.0
        self._corridor_heading_weight = 2.0
        self._corridor_outside_penalty_weight = 20.0
        self.in_corner_phase = False
        self.last_corridor_status = {}
        
        # 完成判据/闭环 lap 跟踪（P3.0）
        self.lap_completed = False
        self.s_travelled = 0.0
        self._progress_s_prev = 0.0
        self._progress_s_max = 0.0
        self._progress_segment_lengths = []
        self._progress_cumulative_lengths = []
        self._progress_total_length = 0.0
        
        # 轨迹误差历史记录（用于过弯后修正
        self.error_history = []         # 最近的误差历史
        self.max_error_history = 20     # 保存最0步的误差
        self.last_corner_segment = -1   # 上一个过弯的线段

        
        # 运动学状态变
        self.velocity = 0.0       # 当前速度
        self.acceleration = 0.0   # 当前加速度
        self.jerk = 0.0           # 当前捷度
        self.angular_vel = 0.0  # 当前角速度
        self.angular_acc = 0.0  # 当前角加速度
        self.angular_jerk = 0.0  # 当前角加加速度

        # 最近一步的奖励分解，便于日志记
        self.last_reward_components = {}

        self.Pm = [np.array(p) for p in Pm]
        # 检查路径是否闭
        self.closed = len(Pm) > 2 and np.allclose(Pm[0], Pm[-1], atol=1e-6)
        
        self.geometric_features = self._compute_geometric_features()
        self.current_position = np.array(self.Pm[0])
        pl, pr = generate_offset_paths(self.Pm, self.half_epsilon, closed=self.closed)

        # 新增缓存字典
        self.cache = {
            'segment_lengths': None,
            'segment_directions': None,  # 新增路径方向缓存
            'angles': None,
            'Pl': pl,
            'Pr': pr,
            'polygons': None,
            'total_path_length': None,
            'segment_info': {},  # 存储每个线段的缓存信
            'cumulative_lengths': None,
        }
        # 预计算并缓存几何特征
        self._precompute_and_cache_geometric_features()
        self.curvature_profile, self.curvature_rate_profile = self._compute_curvature_profile()
        max_segment = max(self.cache['segment_lengths'] or [1.0])
        total_length = self.cache['total_path_length'] or max_segment
        # 前瞻距离按弧长等间距，保证尺度与路径长度相关
        self.lookahead_spacing = max(total_length / (self.lookahead_points + 1), self.half_epsilon * 0.5)
        self.lookahead_longitudinal_scale = max(self.lookahead_spacing * self.lookahead_points, 1.0)
        # lateral scale 用 soft k 缩放，避免远端饱和
        self.lookahead_lateral_scale = max(self.half_epsilon * self.lookahead_lateral_soft_k, 1.0)
        max_curvature_rate = max([abs(v) for v in self.curvature_rate_profile] + [0.0])
        self.curvature_rate_scale = max(max_curvature_rate, 1e-3)
        # 创建三角函数查找表
        self._create_trig_lookup_table()
        
        # 添加新属性用于跟踪线段信
        self.current_segment_idx = 0
        self.segment_count = len(self.Pm) - 1 if not self.closed else len(self.Pm)
        # 创建 R-tree 空间索引（可选依赖；缺失时退化为投影搜索）
        self.rtree_idx = None
        if rtree_index is not None:
            self.rtree_idx = rtree_index.Index()
            for idx, polygon in enumerate(self.cache["polygons"]):
                if polygon:
                    min_x = min(p[0] for p in polygon)
                    min_y = min(p[1] for p in polygon)
                    max_x = max(p[0] for p in polygon)
                    max_y = max(p[1] for p in polygon)
                    self.rtree_idx.insert(idx, (min_x, min_y, max_x, max_y))
        
        # 现在缓存已经填充，可以安全地设置归一化参        
        self.normalization_params = {
            'theta_prime': self.MAX_ANG_VEL,
            'length_prime': self.MAX_VEL,
            'tau_next': math.pi,
            'distance_to_next_turn': self.cache['total_path_length'] or 10.0,  # 使用缓存的总长度
            'overall_progress': 1.0,  # 本身就是[0,1]范围
            'next_angle': math.pi,
            'velocity': self.MAX_VEL,
            'acceleration': self.MAX_ACC,
            'jerk': self.MAX_JERK,
            'angular_vel': self.MAX_ANG_VEL,
            'angular_acc': self.MAX_ANG_ACC,
            'angular_jerk': self.MAX_ANG_JERK,
            'lookahead_longitudinal': self.lookahead_longitudinal_scale,
            'lookahead_lateral': self.lookahead_lateral_scale,
            'lookahead_curvature_rate': self.curvature_rate_scale,
        }
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # 初始化走廊配置（允许通过 reward_weights.corridor 覆盖）
        self._init_corridor_config()
        # 初始化 P1 配置（LOS 前瞻 / deadzone / corner phase）
        self._init_p1_config()
        # 初始化 P4.0 配置（turn-aware speed target / exit boost / stall / speed cap）
        self._init_p4_config()
        # 初始化 P6.1 配置（抑制抖动：Δu 惩罚 + 目标速度平滑器）
        self._init_p6_1_config()
        # 初始化 P7.3 配置（平滑与终点可靠性）
        self._init_p7_3_config()
        # 初始化 P8.0 配置（真实几何 + 刹车包络）
        self._init_p8_config()
        
        self.reset()

    def _log_effective_params(self) -> None:
        """打印最终生效的时间步长与运动学约束，用于验证YAML是否生效。"""
        print("[ENV] Effective parameters (from YAML or defaults):")
        print(f"  dt (interpolation_period): {self.interpolation_period}")
        print(
            f"  MAX_VEL={self.MAX_VEL}, MAX_ACC={self.MAX_ACC}, MAX_JERK={self.MAX_JERK}, "
            f"MAX_ANG_VEL={self.MAX_ANG_VEL}, MAX_ANG_ACC={self.MAX_ANG_ACC}, MAX_ANG_JERK={self.MAX_ANG_JERK}"
        )
    
    def _create_trig_lookup_table(self):
        """Create trig lookup table for fast trig computation"""
        # 创建360个点的查找表-359度）
        self.COS_TABLE = {}
        self.SIN_TABLE = {}
        
        for deg in range(360):
            rad = math.radians(deg)
            self.COS_TABLE[deg] = math.cos(rad)
            self.SIN_TABLE[deg] = math.sin(rad)
        
        # 添加特殊角度
        for rad in [0, math.pi/2, math.pi, 3*math.pi/2]:
            deg = round(math.degrees(rad)) % 360
            self.COS_TABLE[rad] = math.cos(rad)
            self.SIN_TABLE[rad] = math.sin(rad)
    
    def fast_cos(self, rad):
        """Fast cosine lookup"""
        # 尝试直接查找特殊角度
        if rad in self.COS_TABLE:
            return self.COS_TABLE[rad]
        
        # 转换为角度并查找
        deg = round(math.degrees(rad)) % 360
        return self.COS_TABLE.get(deg, math.cos(rad))
    
    def fast_sin(self, rad):
        """Fast sine lookup"""
        # 尝试直接查找特殊角度
        if rad in self.SIN_TABLE:
            return self.SIN_TABLE[rad]
        
        # 转换为角度并查找
        deg = round(math.degrees(rad)) % 360
        return self.SIN_TABLE.get(deg, math.sin(rad))
        
    def _precompute_and_cache_geometric_features(self):
        """预计算并缓存所有几何特征，重用已有函数"""
        # 重用已有的_compute_geometric_features函数
        segment_lengths, angles = self._compute_geometric_features()
        
        # 缓存线段长度和角
        self.cache['segment_lengths'] = segment_lengths
        self.cache['angles'] = angles
        
        # 计算线段方向（新增）
        segment_directions = []
        n = len(self.Pm)
        if self.closed:
            # 闭合路径：包括从最后一个点到第一个点的线
            for i in range(n-1):
                p1 = np.array(self.Pm[i])
                p2 = np.array(self.Pm[(i + 1) % n])
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                direction = math.atan2(dy, dx)
                segment_directions.append(direction)
        else:
            # 非闭合路
            for i in range(n - 1):
                p1 = np.array(self.Pm[i])
                p2 = np.array(self.Pm[i + 1])
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                direction = math.atan2(dy, dx)
                segment_directions.append(direction)
        
        self.cache['segment_directions'] = segment_directions
            
        # 计算总路径长
        total_length = sum(segment_lengths) if segment_lengths else 0.0
        self.cache['total_path_length'] = total_length
        cumulative = [0.0]
        for l in segment_lengths:
            cumulative.append(cumulative[-1] + l)
        self.cache['cumulative_lengths'] = cumulative
        
        # 创建多边形并缓存（重用_create_polygons函数
        self.cache['polygons'] = self._create_polygons()
        
        # 预缓存线段信
        n = len(self.Pm)
        for idx in range(len(segment_lengths)):
            length = segment_lengths[idx]
            
            # 获取当前线段的方
            current_direction = segment_directions[idx] if idx < len(segment_directions) else 0.0
            
            # 获取多边形（如果有）
            polygon = self.cache['polygons'][idx] if idx < len(self.cache['polygons']) else None
            
            # 计算下一拐角角度（重用_get_next_angle逻辑
            next_angle = 0.0
            if self.closed:
                turn_idx = (idx + 1) % len(angles) if angles else 0
                next_angle = angles[turn_idx] if turn_idx < len(angles) else 0.0
            else:
                turn_idx = idx + 1
                next_angle = angles[turn_idx] if turn_idx < len(angles) else 0.0
            
            self.cache['segment_info'][idx] = {
                'length': length,
                'direction': current_direction,
                'polygon': polygon,
                'next_angle': next_angle
            }
    
    def _rebuild_progress_cache(self) -> None:
        """预计算弧长累积，用于稳定的 progress/lap 判定（P3.0）。"""
        n = len(self.Pm)
        if n < 2:
            self._progress_segment_lengths = []
            self._progress_cumulative_lengths = [0.0]
            self._progress_total_length = 0.0
            return

        seg_lengths = []
        cumulative = [0.0]
        total = 0.0
        for i in range(n - 1):
            seg_len = float(np.linalg.norm(self.Pm[i + 1] - self.Pm[i]))
            seg_lengths.append(seg_len)
            total += seg_len
            cumulative.append(total)

        self._progress_segment_lengths = seg_lengths
        self._progress_cumulative_lengths = cumulative
        self._progress_total_length = total

    def reset(self, random_start: bool = False):
        self.current_step = 0
        self.trajectory_states = []
        # 强制从路径起点开始，忽略外部的 random_start 请求

        self.current_segment_idx = 0
        self.current_position = np.array(self.Pm[0])
        self._current_direction_angle, self._current_step_length = self.initialize_starting_conditions()
        # P7.0：显式维护航向积分（禁止每步用 path_direction 重置航向）
        self.heading = float(self._current_direction_angle)
        self._theta_ref_prev = None
        self._theta_ref_last = float(self._current_direction_angle)
        self._theta_ref_delta = 0.0
        self._p1_los_last = {}
        self.trajectory = [self.current_position.copy()]
        self._episode_summary_printed = False
        self.in_corner_phase = False
        # P7.3：kappa 执行量（奇异点保护）与平滑状态（每回合重置）
        self._p7_3_prev_kappa_exec = 0.0
        self._p7_3_kappa_filt = 0.0
        self._p7_3_trace_ring = []
        # P7.1：出弯回中 ramp + corner_phase 抖动计数（每回合重置）
        self._p7_1_exit_timer = -1
        self._p7_1_corner_toggle_count = 0
        self.last_corridor_status = {}
        self._p4_exit_boost_remaining = 0
        self._p4_stall_counter = 0
        self._p4_last_progress_for_stall = 0.0
        self._p4_stall_triggered = False
        self._p4_step_status = {}
        # P8.x: corner exit hysteresis / cap release state
        self._p8_corner_mode = False
        self._p8_corner_exit_hold = 0
        self._p8_vcap_prev = None
        self._p8_recovery_active = False

        # 重置完成判据相关状态（P3.0）
        self._rebuild_progress_cache()
        self.lap_completed = False
        self.s_travelled = 0.0
        self._progress_s_prev = 0.0
        self._progress_s_max = 0.0

        # 重置误差与成功标记
        self.error_history = []
        self.reached_target = False

        # 运动学状态变量
        self.velocity = 0.0
        self.acceleration = 0.0
        self.jerk = 0.0
        self.angular_vel = 0.0
        self.angular_acc = 0.0
        self.angular_jerk = 0.0
        self.kcm_intervention = 0.0
        self.reward_calculator.reset()

        # P6.1：重置抖动抑制相关状态
        self._p6_prev_action_policy = None
        self._p6_v_target_prev = 0.0
        self._p6_a_ref_prev = 0.0
        self._p6_v_target_initialized = False

        # 初始观测对齐单文件脚本：固定起点、零进度、使用首段长度
        _proj, _seg_idx, s_now, _t_hat, _n_hat = self._project_onto_progress_path(self.current_position)
        scan = self._scan_for_next_turn(float(s_now))
        distance_to_next_turn = float(scan.get("dist_to_turn", float("inf")))
        if not math.isfinite(distance_to_next_turn):
            distance_to_next_turn = float(getattr(self, "_progress_total_length", 0.0))
        overall_progress = 0.0
        tau_initial = 0.0
        next_angle = float(scan.get("turn_angle", 0.0))
        lookahead_features = self._compute_lookahead_features()

        self.state = np.array(
            [
                0.0,  # 初始theta_prime
                0.0,  # 初始length_prime
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
        self.state = np.concatenate([self.state, lookahead_features])
        normalized_state = self.normalize_state(self.state)
        return normalized_state if self.return_normalized_obs else self.state.copy()

    def initialize_starting_conditions(self):
        p1 = self.Pm[0]
        p2 = self.Pm[1]
        delta = p2 - p1
        initial_direction_angle = np.arctan2(delta[1], delta[0])
        initial_step_length = self.MAX_VEL  
        
        return initial_direction_angle, initial_step_length
  
    def step(self, action):
        self.current_step += 1
        prev_vel = self.velocity
        prev_acc = self.acceleration
        # 角运动约束处
        prev_ang_vel = self.angular_vel
        prev_ang_acc = self.angular_acc

        policy_action = np.array(action, dtype=float).flatten()
        policy_theta = float(policy_action[0]) if policy_action.size > 0 else 0.0
        policy_length = float(policy_action[1]) if policy_action.size > 1 else 0.0

        # 先按动作空间边界裁剪，保持 PPO 使用的 log_prob 与送入环境的一致
        policy_theta = np.clip(policy_theta, -1.0, 1.0)
        policy_length = np.clip(policy_length, 0.0, 1.0)
        action_policy = np.array([policy_theta, policy_length], dtype=float)

        # P6.1：动作变化率（policy 层，ratio 动作）
        du_theta = 0.0
        du_v = 0.0
        prev_u = getattr(self, "_p6_prev_action_policy", None)
        if isinstance(prev_u, np.ndarray) and prev_u.shape == (2,):
            du_theta = float(action_policy[0] - prev_u[0])
            du_v = float(action_policy[1] - prev_u[1])
        self._p6_prev_action_policy = action_policy.copy()

        corner_phase_before = bool(getattr(self, "in_corner_phase", False))
        exit_boost_before = int(getattr(self, "_p4_exit_boost_remaining", 0))

        # P4.0：预计算 turn/speed 量（基于 s_t），用于速度目标与硬上限裁剪
        p4_status = self._compute_p4_pre_step_status()
        v_u_policy = float(policy_length)
        omega_u_policy = float(policy_theta)
        p4_status["v_ratio_policy"] = float(v_u_policy)
        p4_status["exit_boost_remaining"] = float(exit_boost_before)

        v_ratio_cap = float(p4_status.get("v_ratio_cap", 1.0))
        if not bool(getattr(self, "_p4_speed_cap_enabled", True)):
            # P8.0：即便关闭 turning-feasible cap，也必须保留刹车包络（终点停下/进弯可刹住）
            v_ratio_cap = float(p4_status.get("v_ratio_brake", 1.0))
            p4_status["v_ratio_cap"] = float(v_ratio_cap)

        # === P5.0：比率层裁剪（仍是 ratio）===
        v_u_exec = float(min(v_u_policy, v_ratio_cap))

        # === P5.0：比率→物理量映射（KCM 输入必须是物理量）===
        omega_intent = float(omega_u_policy) * float(self.MAX_ANG_VEL)
        v_intent = float(v_u_exec) * float(self.MAX_VEL)

        # turning-feasible cap 在物理量上的等价 max_vel
        max_vel_cap_phys = float(min(float(self.MAX_VEL), float(v_ratio_cap) * float(self.MAX_VEL)))
        p4_status["omega_u_policy"] = float(omega_u_policy)
        p4_status["omega_intent"] = float(omega_intent)
        p4_status["v_u_policy"] = float(v_u_policy)
        p4_status["v_u_exec"] = float(v_u_exec)
        p4_status["v_intent"] = float(v_intent)  # v_des（已包含 cap）
        p4_status["max_vel_cap_phys"] = float(max_vel_cap_phys)
        p4_status["du_theta_u"] = float(du_theta)
        p4_status["du_v_u"] = float(du_v)
        p4_status["du_l1"] = float(abs(du_theta) + abs(du_v))

        # policy 意图作为物理量输入约束（速度意图先经 turning-feasible cap，再进 KCM）
        raw_angular_vel_intent = float(omega_intent)

        # P6.1：目标速度平滑器（v_des -> v_target）
        v_des = float(v_intent)
        v_target = float(v_des)
        if bool(getattr(self, "_p6_v_target_smoother_enabled", False)):
            dt = float(self.interpolation_period)
            if not bool(getattr(self, "_p6_v_target_initialized", False)):
                v_target = float(v_des)
                self._p6_v_target_prev = float(v_target)
                self._p6_a_ref_prev = 0.0
                self._p6_v_target_initialized = True
            else:
                v_prev = float(getattr(self, "_p6_v_target_prev", 0.0))
                mode = str(getattr(self, "_p6_v_target_mode", "accel")).lower()
                if mode == "lowpass":
                    tau = float(max(float(getattr(self, "_p6_v_target_tau", 0.10)), 1e-6))
                    beta = float(dt / (tau + dt))
                    v_target = float(v_prev + beta * (v_des - v_prev))
                else:
                    a_ref_max = float(max(float(getattr(self, "_p6_a_ref_max", 0.0)), 0.0))
                    a_prev = float(getattr(self, "_p6_a_ref_prev", 0.0))
                    a_des = float((v_des - v_prev) / max(dt, 1e-6))
                    a_ref = float(np.clip(a_des, -a_ref_max, a_ref_max))
                    if bool(getattr(self, "_p6_j_ref_enabled", True)):
                        j_ref_max = float(max(float(getattr(self, "_p6_j_ref_max", 0.0)), 0.0))
                        a_low = float(a_prev - j_ref_max * dt)
                        a_high = float(a_prev + j_ref_max * dt)
                        a_ref = float(np.clip(a_ref, a_low, a_high))
                    v_target = float(v_prev + a_ref * dt)
                    # 避免跨越 v_des（防止振荡）
                    if (v_des - v_prev) * (v_des - v_target) < 0.0:
                        v_target = float(v_des)
                        a_ref = float((v_target - v_prev) / max(dt, 1e-6))
                    self._p6_a_ref_prev = float(a_ref)
                self._p6_v_target_prev = float(v_target)

        # 目标器输出仍需遵守物理上限与 turning-feasible cap
        v_target = float(np.clip(v_target, 0.0, max_vel_cap_phys))
        p4_status["v_des"] = float(v_des)
        p4_status["v_target"] = float(v_target)
        raw_linear_vel_intent = float(v_target)

        # P7.3：曲率平滑（基于执行量定义 kappa=|omega|/(v+eps)），在约束前做温和滤波（默认关闭）
        raw_angular_vel_intent = float(self._p7_3_apply_kappa_smoothing(raw_angular_vel_intent, raw_linear_vel_intent))

        # 使用Numba优化的约束计算
        (self.velocity, self.acceleration, self.jerk,
         self.angular_vel, self.angular_acc, self.angular_jerk) = apply_kinematic_constraints(
            prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
            raw_linear_vel_intent, raw_angular_vel_intent, self.interpolation_period,
            max_vel_cap_phys, self.MAX_ACC, self.MAX_JERK,
            self.MAX_ANG_VEL, self.MAX_ANG_ACC, self.MAX_ANG_JERK
        )
        
        # === 计算KCM干预程度：物理量差值（归一化到约[0,2]量级，避免奖励尺度爆炸）===
        velocity_diff = abs(self.velocity - raw_linear_vel_intent) / max(float(self.MAX_VEL), 1e-6)
        angular_vel_diff = abs(self.angular_vel - raw_angular_vel_intent) / max(float(self.MAX_ANG_VEL), 1e-6)
        self.kcm_intervention = float(velocity_diff + angular_vel_diff)
        
        # === 使用修正后的动作执行状态转===
        # 构建最终安全动
        safe_action = (self.angular_vel, self.velocity)  # final_vel是线速度
        next_state = self.apply_action(safe_action)
        self.state = next_state
        self.trajectory_states.append(next_state)

        # P4.0：更新本步实际执行速度比（考虑 KCM 与 cap 后的最终速度）
        v_ratio_exec = float(self.velocity / max(float(self.MAX_VEL), 1e-6))
        omega_u_exec = float(self.angular_vel / max(float(self.MAX_ANG_VEL), 1e-6))
        p4_status["v_ratio_exec"] = float(v_ratio_exec)
        p4_status["omega_u_exec"] = float(omega_u_exec)
        p4_status["omega_exec"] = float(self.angular_vel)
        p4_status["v_exec"] = float(self.velocity)

        # P7.3：执行曲率（奇异点保护）+ dkappa 诊断（用于验收脚本）
        kappa_exec = float(self._p7_3_compute_kappa_exec(v_exec=float(self.velocity), omega_exec=float(self.angular_vel)))
        prev_kappa_exec = float(getattr(self, "_p7_3_prev_kappa_exec", 0.0))
        dkappa_exec = float(abs(kappa_exec - prev_kappa_exec))
        self._p7_3_prev_kappa_exec = float(kappa_exec)
        p4_status["kappa_exec"] = float(kappa_exec)
        p4_status["dkappa_exec"] = float(dkappa_exec)

        # P7.3：数值安全（kappa/dkappa/state/reward/obs 永不 NaN/Inf；异常立刻 dump）
        self._p7_3_trace_append(p4_status=p4_status)
        self._p7_3_assert_finite(name="kappa_exec", value=kappa_exec, p4_status=p4_status)
        self._p7_3_assert_finite(name="dkappa_exec", value=dkappa_exec, p4_status=p4_status)
        self._p7_3_assert_finite_array(name="state", arr=self.state, p4_status=p4_status)

        # P4.0：停滞检测（progress_diff 很小 + 低速，持续 N 步则终止）
        self._p4_stall_triggered = False
        progress_now = float(self.state[4]) if len(self.state) > 4 else 0.0
        progress_diff = max(0.0, progress_now - float(getattr(self, "_p4_last_progress_for_stall", 0.0)))
        self._p4_last_progress_for_stall = float(progress_now)

        if bool(getattr(self, "_p4_stall_enabled", True)) and not bool(getattr(self, "reached_target", False)):
            # P7.3：stall 与 corner_phase/cap 联动
            if bool(corner_phase_before):
                # corner_phase 内不触发 stall（避免转弯低速被误杀）
                self._p4_stall_counter = 0
            else:
                stall_progress_eps = float(getattr(self, "_p4_stall_progress_eps", 1e-4))
                stall_v_eps = float(getattr(self, "_p4_stall_v_eps", 0.05))
                cap_low = float(getattr(self, "_p7_3_stall_cap_low", 0.25))
                v_ratio_cap_now = float(p4_status.get("v_ratio_cap", 1.0))
                low_cap = bool(math.isfinite(v_ratio_cap_now) and v_ratio_cap_now < cap_low)

                if progress_diff < stall_progress_eps and (low_cap or v_ratio_exec < stall_v_eps):
                    self._p4_stall_counter = int(getattr(self, "_p4_stall_counter", 0)) + 1
                else:
                    self._p4_stall_counter = 0
            if int(getattr(self, "_p4_stall_counter", 0)) >= int(getattr(self, "_p4_stall_steps", 300)):
                self._p4_stall_triggered = True

        p4_status["stall_counter"] = float(int(getattr(self, "_p4_stall_counter", 0)))
        p4_status["stall_triggered"] = float(1.0 if bool(getattr(self, "_p4_stall_triggered", False)) else 0.0)

        # 更新误差历史
        current_error = self.get_contour_error(self.current_position)
        self.error_history.append(current_error)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)  # 保持固定长度

        # 保存本步 P4 状态供 reward/info 使用（reward 基于 s_t 的目标 + s_{t+1} 的结果）
        self._p4_step_status = dict(p4_status)

        reward = self.calculate_reward()
        self._p7_3_assert_finite(name="reward", value=reward, p4_status=p4_status)
        done = self.is_done()

        # P4.0：出弯 boost 更新（基于 corner_phase s_t -> s_{t+1} 的状态转移）
        corner_phase_after = bool(getattr(self, "last_corridor_status", {}).get("corner_phase", False))
        if bool(getattr(self, "_p4_exit_boost_enabled", True)) and corner_phase_before and not corner_phase_after:
            self._p4_exit_boost_remaining = int(getattr(self, "_p4_exit_window_steps", 0))
            if bool(getattr(self, "enable_p4_diagnostics", False)):
                cs = getattr(self, "last_corridor_status", {}) or {}
                if not isinstance(cs, dict):
                    cs = {}
                alpha_dbg = float(cs.get("alpha", float("nan")))
                L_dbg = float(cs.get("L", float("nan")))
                v_ratio_exec_dbg = float(getattr(self, "_p4_step_status", {}).get("v_ratio_exec", float("nan")))
                exit_window = int(getattr(self, "_p4_exit_window_steps", 0))
                print(
                    f"[P4_EXIT] step={int(getattr(self, 'current_step', 0))} alpha={alpha_dbg:.4f} L={L_dbg:.4f} "
                    f"v_ratio_exec={v_ratio_exec_dbg:.4f} exit_window={exit_window}"
                )
        else:
            if int(getattr(self, "_p4_exit_boost_remaining", 0)) > 0:
                self._p4_exit_boost_remaining = int(getattr(self, "_p4_exit_boost_remaining", 0)) - 1

        # 添加info字典作为第四个返回值
        info = {
            "position": self.current_position.copy(),
            "step": self.current_step,
            "contour_error": self.get_contour_error(self.current_position),
            "segment_idx": self.current_segment_idx,
            "progress": self.state[4],  # 添加进度信息
            "jerk": self.jerk,  # 添加当前捷度指标
            "kcm_intervention": self.kcm_intervention,  # 添加运动学约束干预程度
            "corridor_status": copy.deepcopy(getattr(self, "last_corridor_status", {})),
            "p4_status": copy.deepcopy(getattr(self, "_p4_step_status", {})),
            "action_policy": action_policy.copy(),
            "action_exec": np.array(safe_action, dtype=float),
            # P5.0：gap 统一在比率空间比较，便于判断 KCM/cap 干预程度
            "action_gap_abs": np.abs(np.array([omega_u_exec, v_ratio_exec], dtype=float) - action_policy),
        }
        normalized_state = self.normalize_state(self.state)
        obs = normalized_state if self.return_normalized_obs else np.asarray(self.state, dtype=float).copy()
        self._p7_3_assert_finite_array(name="obs", arr=obs, p4_status=p4_status)
        return obs, reward, done, info
    
    def normalize_state(self, state):
        normalized = np.zeros_like(state, dtype=float)
        base_len = len(self.base_state_keys)
        for i, key in enumerate(self.base_state_keys):
            max_val = self.normalization_params[key]
            # 特殊处理距离和进度
            if key == 'distance_to_next_turn':
                scaled = np.log1p(state[i]) / np.log1p(max_val)
                normalized[i] = np.clip(scaled, 0, 1)
            elif key == 'overall_progress':
                normalized[i] = state[i]
            else:
                normalized[i] = np.clip(state[i] / max_val, -1, 1)

        offset = base_len
        for idx in range(self.lookahead_points):
            base = offset + idx * self.lookahead_feature_size
            if base + 2 >= len(state):
                break
            # s_i: 前向距离，归一化到[0,1]
            normalized[base] = np.clip(state[base] / self.lookahead_longitudinal_scale, 0, 1)
            # d_i: 法向距离，采用软缩放+ tanh 防饱和，左正右负
            d_scaled = (state[base + 1] / max(self.half_epsilon, 1e-6)) / self.lookahead_lateral_soft_k
            normalized[base + 1] = np.tanh(d_scaled)
            # 曲率变化率（可选），压到[-1,1]
            if self.curvature_rate_scale > 0:
                normalized[base + 2] = np.clip(state[base + 2] / self.curvature_rate_scale, -1, 1)
            else:
                normalized[base + 2] = state[base + 2]
        return normalized
    
    def apply_action(self, action):
        theta_prime, length_prime = action
        # Path tangent + residual steering.
        new_pos = self.calculate_new_position(theta_prime, length_prime)
        self.current_position = new_pos
        self.trajectory.append(self.current_position.copy())
        tau_next = self.calculate_direction_deviation(self.current_position)
        seg_idx = int(self._find_containing_segment(self.current_position))
        if seg_idx >= 0:
            self.current_segment_idx = int(seg_idx)
        _proj, _seg_idx_progress, s_now, _t_hat, _n_hat = self._project_onto_progress_path(self.current_position)
        scan = self._scan_for_next_turn(float(s_now))
        distance_to_next_turn = float(scan.get("dist_to_turn", float("inf")))
        if not math.isfinite(distance_to_next_turn):
            distance_to_next_turn = float(getattr(self, "_progress_total_length", 0.0))
        overall_progress = (
            self._calculate_closed_path_progress(self.current_position)
            if self.closed
            else self._calculate_path_progress(self.current_position)
        )
        next_angle = float(scan.get("turn_angle", 0.0))

        lookahead_features = self._compute_lookahead_features()
        base_state = np.array(
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
        return np.concatenate([base_state, lookahead_features])

    def _update_segment_info(self):
        """使用缓存的线段信息"""
        pt = self.current_position
        segment_idx = self._find_containing_segment(pt)
        
        if segment_idx >= 0 and segment_idx in self.cache['segment_info']:
            # 获取线段终点
            next_turn_point = np.array(self.Pm[segment_idx + 1])
            # 计算到下一拐点的欧氏距
            distance_to_next_turn = np.linalg.norm(next_turn_point - pt)
            return segment_idx, distance_to_next_turn
        
        return self.current_segment_idx, float('inf')
    
    def _get_next_angle(self, segment_idx):
        """从缓存中获取下一拐角角度"""
        if segment_idx in self.cache['segment_info']:
            return self.cache['segment_info'][segment_idx]['next_angle']
        return 0.0
    
    def _calculate_closed_path_progress(self, pt):
        """Closed path: monotonic arc-length accumulation + start gate (P3.0)."""
        if not self.closed:
            return self._calculate_path_progress(pt)

        if self.lap_completed:
            return 1.0

        total_length = float(getattr(self, "_progress_total_length", 0.0))
        if total_length <= 1e-9:
            return 0.0

        seg_idx = self._find_nearest_segment_for_progress(pt)
        if seg_idx < 0 or seg_idx >= len(self.Pm) - 1:
            return 0.0
        if seg_idx >= len(getattr(self, "_progress_segment_lengths", [])):
            return 0.0

        p1 = np.array(self.Pm[seg_idx])
        p2 = np.array(self.Pm[seg_idx + 1])
        seg_vec = p2 - p1
        denom = float(np.dot(seg_vec, seg_vec))
        t = 0.0
        if denom > 1e-12:
            t = float(np.dot(pt - p1, seg_vec) / denom)
        t = float(np.clip(t, 0.0, 1.0))

        s_current = float(self._progress_cumulative_lengths[seg_idx] + t * self._progress_segment_lengths[seg_idx])

        s_prev = float(getattr(self, "_progress_s_prev", 0.0))
        delta_s = s_current - s_prev
        # 过起点时 s 会从接近 total 跳到接近 0：解包裹为正增量
        if delta_s < -0.5 * total_length:
            delta_s += total_length
        # 保证单调（向后走/最近段抖动不回退）
        if delta_s < 0.0:
            delta_s = 0.0

        self.s_travelled = float(getattr(self, "s_travelled", 0.0) + delta_s)
        self._progress_s_prev = s_current
        self._progress_s_max = max(float(getattr(self, "_progress_s_max", 0.0)), s_current)

        tol = 0.02
        start_tol = 0.6 * self.half_epsilon
        dist_to_start = float(np.linalg.norm(pt - np.array(self.Pm[0])))
        threshold = (1.0 - tol) * total_length

        if self.s_travelled >= threshold and self._progress_s_max >= threshold and dist_to_start < start_tol:
            self.lap_completed = True
            self.reached_target = True
            return 1.0

        progress = self.s_travelled / total_length
        return float(np.clip(progress, 0.0, 0.99))
    
    def calculate_new_position(self, theta_prime_action, length_prime_action):
        # Residual control: follow path tangent with a small angular residual.
        dt = float(self.interpolation_period)
        omega_exec = float(theta_prime_action)
        v_exec = float(length_prime_action)
        theta_ref = float(self._get_path_direction(self.current_position, v_exec=v_exec, record=True))

        effective = float(self._wrap_angle(theta_ref + omega_exec * dt))
        # Keep heading for diagnostics; dynamics follow path tangent + residual.
        self.heading = float(effective)
        self._current_direction_angle = float(effective)

        displacement = v_exec * dt
        x_next = float(self.current_position[0]) + displacement * float(np.cos(effective))
        y_next = float(self.current_position[1]) + displacement * float(np.sin(effective))

        return np.array([x_next, y_next], dtype=float)

    def _compute_geometric_features(self):
        """
        计算并存储几何特征：线段长度、角
        返回一个包含三个元素的元组线段长度列表, 角度列表)
        """
        if self.closed:
            n = len(self.Pm)-1
        else:
            n = len(self.Pm)
        # 计算线段长度
        segment_lengths = []
        for i in range(n - 1):
            p1 = np.array(self.Pm[i])
            p2 = np.array(self.Pm[i + 1])
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
        if self.closed:
            # 闭合路径：最后一个点到第一个点的线段长度为0（因为它们是同一个点
            # 不需要添加额外的线段长度
            pass
        
        
        # 计算角度
        angles = []
        
        if n >= 3:  # 至少需个点才能计算角度
            for i in range(n):
                # 对于闭合路径，所有点都有角度
                # 对于非闭合路径，只有中间点有角度
                if self.closed or (not self.closed and i > 0 and i < n - 1):
                    # 获取三个连续点（考虑闭合路径
                    prev_index = (i - 1) % n
                    next_index = (i + 1) % n
                    
                    p0 = np.array(self.Pm[prev_index])
                    p1 = np.array(self.Pm[i])
                    p2 = np.array(self.Pm[next_index])
                    
                    # 计算向量
                    vec1 = p1 - p0
                    vec2 = p2 - p1
                    
                    # 计算向量长度
                    len1 = np.linalg.norm(vec1)
                    len2 = np.linalg.norm(vec2)
                    
                    # 避免除以
                    if len1 < 1e-6 or len2 < 1e-6:
                        angle = 0.0
                    else:
                        # 计算角度（单位：弧度），确保逆时针为正，顺时针为
                        dot_product = np.dot(vec1, vec2) / (len1 * len2)
                        cross_product = np.cross(vec1, vec2) / (len1 * len2)
                        
                        # 使用atan2确保正确的角度方
                        angle = math.atan2(cross_product, dot_product)
                    
                    angles.append(angle)
        
        # 存储几何特征
        self.segment_lengths = segment_lengths
        self.angles = angles
        
        return segment_lengths, angles

    def _compute_curvature_profile(self):
        """Approximate curvature via three-point method and compute d/ds"""
        points = [np.array(p) for p in self.Pm]
        n = len(points)
        curvatures = [0.0 for _ in range(n)]
        curvature_rate = [0.0 for _ in range(n)]

        if n < 3:
            return curvatures, curvature_rate

        def curvature_at(idx_prev, idx_curr, idx_next):
            p0, p1, p2 = points[idx_prev], points[idx_curr], points[idx_next]
            a = p1 - p0
            b = p2 - p1
            cross = np.cross(a, b)
            denom = (np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(p2 - p0) + 1e-6)
            return 2 * cross / denom

        for i in range(n):
            prev_idx = (i - 1) % n if self.closed else max(i - 1, 0)
            next_idx = (i + 1) % n if self.closed else min(i + 1, n - 1)
            curvatures[i] = curvature_at(prev_idx, i, next_idx)

        for i in range(1, n - 1):
            ds = np.linalg.norm(points[i + 1] - points[i - 1]) + 1e-6
            curvature_rate[i] = (curvatures[i + 1] - curvatures[i - 1]) / ds

        if self.closed and n > 3:
            curvature_rate[0] = curvature_rate[-2]
            curvature_rate[-1] = curvature_rate[1]

        return curvatures, curvature_rate

    def _project_onto_path(self, pt):
        """投影点到路径，返回投影坐标、段索引、弧长、切向/法向单位向量。"""
        seg_idx = self._find_containing_segment(pt)
        if seg_idx < 0:
            seg_idx = max(self.current_segment_idx, 0)
        p1 = np.array(self.Pm[seg_idx])
        p2 = np.array(self.Pm[(seg_idx + 1) % len(self.Pm)])
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            t_hat = np.array([1.0, 0.0])
            proj = p1
            t = 0.0
        else:
            t = np.clip(np.dot(pt - p1, seg_vec) / (seg_len**2), 0.0, 1.0)
            proj = p1 + t * seg_vec
            t_hat = seg_vec / seg_len
        n_hat = np.array([-t_hat[1], t_hat[0]])  # 左侧为正
        s_along = (self.cache['cumulative_lengths'][seg_idx] if self.cache['cumulative_lengths'] else 0.0) + t * seg_len
        return proj, seg_idx, s_along, t_hat, n_hat

    def _interpolate_point_at_s(self, s_value: float) -> tuple[np.ndarray, int, float]:
        """按弧长获取路径点，返回坐标、段索引、局部t。"""
        total = self.cache['total_path_length'] or 1.0
        s_clamped = np.clip(s_value, 0.0, total)
        cumulative = self.cache.get('cumulative_lengths') or []
        if not cumulative or len(cumulative) < 2:
            return np.array(self.Pm[0]), 0, 0.0
        # 找到所在段
        seg_idx = min(np.searchsorted(cumulative, s_clamped, side="right") - 1, len(self.Pm) - 2)
        seg_start_s = cumulative[seg_idx]
        seg_len = self.cache['segment_lengths'][seg_idx] if seg_idx < len(self.cache['segment_lengths']) else 1.0
        if seg_len <= 0:
            t = 0.0
        else:
            t = np.clip((s_clamped - seg_start_s) / seg_len, 0.0, 1.0)
        p1 = np.array(self.Pm[seg_idx])
        p2 = np.array(self.Pm[seg_idx + 1])
        point = p1 + t * (p2 - p1)
        return point, seg_idx, t

    def _get_lookahead_targets(self, s_current: float):
        """按弧长等间距向前采样目标点。"""
        total = self.cache['total_path_length'] or 1.0
        targets = []
        for i in range(self.lookahead_points):
            s_target = min(s_current + self.lookahead_spacing * (i + 1), total)
            pt, seg_idx, _ = self._interpolate_point_at_s(s_target)
            targets.append((pt, seg_idx))
        return targets

    def _compute_lookahead_features(self):
        """使用路径切向坐标系的前瞻特征: [s_i, d_i, curvature_rate]."""
        proj, seg_idx, s_current, t_hat, n_hat = self._project_onto_path(self.current_position)
        targets = self._get_lookahead_targets(s_current)
        features = []
        for pt, target_seg in targets:
            delta = pt - proj
            s_along = float(np.dot(delta, t_hat))
            d_signed = float(np.dot(delta, n_hat))  # 左正右负
            kappa_rate = 0.0
            if target_seg < len(self.curvature_rate_profile):
                kappa_rate = float(self.curvature_rate_profile[target_seg])
            features.extend([s_along, d_signed, kappa_rate])
        return np.array(features, dtype=float)

    def _init_corridor_config(self) -> None:
        """读取/初始化 VirtualCorridor 配置（P3.1，可在 YAML 的 reward_weights.corridor 下设置）。"""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("corridor", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self.enable_corridor = bool(cfg.get("enabled", False))

        theta_enter_deg = float(cfg.get("theta_enter_deg", 15.0))
        theta_exit_deg = float(cfg.get("theta_exit_deg", 8.0))
        theta_enter = float(math.radians(theta_enter_deg))
        theta_exit = float(math.radians(theta_exit_deg))
        if theta_exit > theta_enter:
            theta_exit = 0.5 * theta_enter
        self._corridor_theta_enter = theta_enter
        self._corridor_theta_exit = theta_exit

        dist_enter = cfg.get("dist_enter", None)
        dist_exit = cfg.get("dist_exit", None)
        default_enter = 3.0 * float(getattr(self, "lookahead_spacing", 1.0))
        self._corridor_dist_enter = float(default_enter if dist_enter is None else dist_enter)
        self._corridor_dist_exit = float(1.5 * self._corridor_dist_enter if dist_exit is None else dist_exit)
        if self._corridor_dist_exit <= self._corridor_dist_enter:
            self._corridor_dist_exit = 1.5 * self._corridor_dist_enter

        margin_ratio = float(cfg.get("margin_ratio", 0.1))
        margin = margin_ratio * self.half_epsilon
        self._corridor_margin = float(np.clip(margin, 0.0, 0.4 * self.half_epsilon))

        self._corridor_heading_weight = float(cfg.get("heading_weight", 2.0))
        self._corridor_outside_penalty_weight = float(cfg.get("outside_penalty_weight", 20.0))

        # P5.2：走廊奖励重构（取消固定 e_target）：边界势垒 + 轻微中心偏好（可选）
        safe_margin_ratio = float(cfg.get("safe_margin_ratio", 0.2))
        safe_margin_ratio = float(np.clip(safe_margin_ratio, 0.0, 0.9))
        self._corridor_safe_margin = float(safe_margin_ratio * self.half_epsilon)

        barrier_scale_ratio = float(cfg.get("barrier_scale_ratio", 0.05))
        barrier_scale_ratio = float(np.clip(barrier_scale_ratio, 1e-4, 1.0))
        self._corridor_barrier_scale = float(max(barrier_scale_ratio * self.half_epsilon, 1e-6))
        self._corridor_barrier_weight = float(cfg.get("barrier_weight", 2.0))

        self._corridor_center_weight = float(cfg.get("center_weight", 0.0))
        self._corridor_center_power = float(cfg.get("center_power", 2.0))

        # P7.1：方向性偏好（tanh 弱引导） + 出弯回中权重线性 ramp
        self._corridor_dir_pref_weight = float(cfg.get("dir_pref_weight", 0.0))
        self._corridor_dir_pref_beta = float(cfg.get("dir_pref_beta", 2.0))
        if not math.isfinite(self._corridor_dir_pref_beta) or self._corridor_dir_pref_beta <= 0.0:
            self._corridor_dir_pref_beta = 1.0

        exit_steps = cfg.get("exit_center_ramp_steps", None)
        if exit_steps is None:
            dt = float(max(float(getattr(self, "interpolation_period", 0.1)), 1e-6))
            exit_steps = max(10, int(1.0 / dt))
        self._corridor_exit_center_ramp_steps = int(max(1, int(exit_steps)))

    def _init_p1_config(self) -> None:
        """Initialize P1 config: LOS reference and corner-phase helpers."""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("p1", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        self._p1_use_los = bool(cfg.get("use_los", False))

        base = float(getattr(self, "lookahead_spacing", 1.0))
        L0 = cfg.get("L0", None)
        Lmin = cfg.get("Lmin", None)
        Lmax = cfg.get("Lmax", None)
        kL = float(cfg.get("kL", 0.0))

        L0 = float(base if L0 is None else L0)
        Lmin = float(0.5 * base if Lmin is None else Lmin)
        Lmax = float(3.0 * base if Lmax is None else Lmax)

        if not math.isfinite(L0) or L0 <= 0.0:
            L0 = float(max(base, 1e-6))
        if not math.isfinite(Lmin) or Lmin <= 0.0:
            Lmin = float(max(0.5 * base, 1e-6))
        if not math.isfinite(Lmax) or Lmax <= 0.0:
            Lmax = float(max(3.0 * base, Lmin))
        if Lmax < Lmin:
            Lmax = float(Lmin)

        self._p1_los_L0 = float(L0)
        self._p1_los_kL = float(kL)
        self._p1_los_Lmin = float(Lmin)
        self._p1_los_Lmax = float(Lmax)
        self._p1_los_last = {}

        corner_phase_enabled = bool(cfg.get("corner_phase_enabled", False))
        if not corner_phase_enabled:
            dz_corner = float(cfg.get("deadzone_corner_ratio", 0.0))
            w_ang_acc = float(cfg.get("w_ang_acc", 0.0))
            corner_phase_enabled = bool(dz_corner > 0.0 or w_ang_acc > 0.0)
        self._p1_corner_phase_enabled = bool(corner_phase_enabled)

    def _init_p4_config(self) -> None:
        """读取/初始化 P4.0 配置（可在 YAML 的 reward_weights.p4 下设置）。"""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("p4", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # 1) 每步时间惩罚：逼迫更快完成（负值）
        self._p4_time_penalty = float(cfg.get("time_penalty", -0.01))

        # 2) turn-aware speed_target
        self._p4_v_max = float(cfg.get("v_max", 1.0))
        self._p4_v_min = float(cfg.get("v_min", 0.35))
        theta_max_deg = float(cfg.get("theta_max_deg", 90.0))
        self._p4_theta_max = float(math.radians(theta_max_deg))
        d_scale = cfg.get("d_scale", None)
        default_d_scale = 3.0 * float(getattr(self, "lookahead_spacing", 1.0))
        self._p4_d_scale = float(default_d_scale if d_scale is None else d_scale)
        self._p4_speed_weight = float(cfg.get("speed_weight", 6.0))

        # 3) exit boost
        self._p4_exit_boost_enabled = bool(cfg.get("exit_boost_enabled", True))
        exit_window_sec = float(cfg.get("exit_window_sec", 0.25))
        default_exit_steps = max(5, int(exit_window_sec / max(self.interpolation_period, 1e-6)))
        self._p4_exit_window_steps = int(cfg.get("exit_window_steps", default_exit_steps))
        self._p4_exit_progress_mult = float(cfg.get("exit_progress_mult", 1.35))
        self._p4_exit_speed_target_min = float(cfg.get("exit_speed_target_min", 0.9))

        # 4) stall termination
        self._p4_stall_enabled = bool(cfg.get("stall_enabled", True))
        self._p4_stall_steps = int(cfg.get("stall_steps", 300))
        self._p4_stall_progress_eps = float(cfg.get("stall_progress_eps", 1e-4))
        self._p4_stall_v_eps = float(cfg.get("stall_v_eps", 0.05))
        self._p4_stall_penalty = float(cfg.get("stall_penalty", -8.0))

        # 5) speed cap derived from MAX_ANG_VEL
        self._p4_speed_cap_enabled = bool(cfg.get("speed_cap_enabled", True))
        # eps 仅用于防止除零；默认取更小值，避免在 κ≈0 的直线段因 ω̈ 公式被误限速
        self._p4_speed_cap_eps = float(cfg.get("speed_cap_eps", 1e-12))

        # P7.2：自适应预瞄（d_target = d0 + k*v_exec）
        d0_default = float(getattr(self, "lookahead_spacing", 1.0))
        d_min_default = 0.5 * d0_default
        d_max_default = 4.0 * d0_default
        k_default = d0_default / max(float(self.MAX_VEL), 1e-6)
        self._p7_2_d0 = float(cfg.get("p7_2_d0", d0_default))
        self._p7_2_k = float(cfg.get("p7_2_k", k_default))
        self._p7_2_d_min = float(cfg.get("p7_2_d_min", d_min_default))
        self._p7_2_d_max = float(cfg.get("p7_2_d_max", d_max_default))
        if self._p7_2_d_max < self._p7_2_d_min:
            self._p7_2_d_max = float(self._p7_2_d_min)

        # kappa 估计使用的 lookahead 弧长（默认按 lookahead_spacing*k，并做上限截断，避免过长导致 kappa 过小）
        k = int(cfg.get("speed_cap_k", 4))
        k = int(np.clip(k, 1, max(1, int(getattr(self, "lookahead_points", 1)))))
        base_s = float(getattr(self, "lookahead_spacing", 1.0)) * float(k)
        s_min = float(cfg.get("speed_cap_s_min", 0.5 * self.half_epsilon))
        s_max = float(cfg.get("speed_cap_s_max", 2.0 * self.half_epsilon))
        if s_max < s_min:
            s_max = max(s_min, s_max)
        self._p4_speed_cap_s = float(np.clip(base_s, max(s_min, 1e-6), max(s_max, 1e-6)))

        # 6) P6.0：多预瞄 minimax + ω/ω̇/ω̈ 可达性 → 速度上限
        preview_points = int(cfg.get("speed_cap_preview_points", int(getattr(self, "lookahead_points", 5))))
        self._p6_speed_cap_points = int(np.clip(preview_points, 1, 16))
        preview_spacing = cfg.get("speed_cap_preview_spacing", None)
        if preview_spacing is None:
            preview_spacing = float(getattr(self, "_p4_speed_cap_s", 0.0))
        self._p6_speed_cap_spacing = float(max(float(preview_spacing), 1e-6))
        self._p6_speed_cap_use_wdot = bool(cfg.get("speed_cap_use_wdot", True))
        self._p6_speed_cap_use_wddot = bool(cfg.get("speed_cap_use_wddot", True))

        # 诊断输出开关（默认关闭）
        self.enable_p4_diagnostics = bool(cfg.get("debug", False))

        # 运行时状态（reset 时会重置）
        self._p4_exit_boost_remaining = 0
        self._p4_stall_counter = 0
        self._p4_last_progress_for_stall = 0.0
        self._p4_stall_triggered = False
        self._p4_step_status = {}

    def _init_p6_1_config(self) -> None:
        """读取/初始化 P6.1 配置（可在 YAML 的 reward_weights.p6_1 下设置）。"""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("p6_1", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # 1) 动作变化率惩罚（reward 层面）：Δu 使用 action_policy 的 ratio 动作
        self._p6_du_enabled = bool(cfg.get("du_enabled", True))
        self._p6_du_weight = float(cfg.get("w_du", 0.01))
        self._p6_du_mode = str(cfg.get("du_mode", "l1")).lower()  # l1|l2

        # 2) 目标速度平滑器（env 层面）：policy v_des -> v_target -> KCM
        self._p6_v_target_smoother_enabled = bool(cfg.get("v_target_smoother_enabled", True))
        self._p6_v_target_mode = str(cfg.get("v_target_mode", "accel")).lower()  # lowpass|accel

        # A) 一阶低通
        self._p6_v_target_tau = float(cfg.get("v_target_tau", 0.10))

        # B) a/j 限幅的目标器（推荐）
        a_ratio = float(cfg.get("a_ref_max_ratio", 0.6))
        j_ratio = float(cfg.get("j_ref_max_ratio", 0.6))
        self._p6_a_ref_max = float(max(0.0, min(1.0, a_ratio))) * float(self.MAX_ACC)
        self._p6_j_ref_max = float(max(0.0, min(1.0, j_ratio))) * float(self.MAX_JERK)
        self._p6_j_ref_enabled = bool(cfg.get("j_ref_enabled", True))

        # 运行时状态（reset 时会重置）
        self._p6_prev_action_policy = None
        self._p6_v_target_prev = 0.0
        self._p6_a_ref_prev = 0.0
        self._p6_v_target_initialized = False

    def _init_p7_3_config(self) -> None:
        """读取/初始化 P7.3 配置（平滑与终点可靠性）。"""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("p7_3", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # A) kappa 奇异点保护：v_eps 按 MAX_VEL 缩放（写死默认值）
        self._p7_3_v_eps = float(max(1e-6 * float(getattr(self, "MAX_VEL", 1.0)), 1e-12))

        # B) stall 与 cap 联动阈值（cap 很低时不以低速判 stall）
        stall_cap_low = cfg.get("stall_cap_low", 0.25)
        try:
            stall_cap_low = float(stall_cap_low)
        except Exception:
            stall_cap_low = 0.25
        self._p7_3_stall_cap_low = float(np.clip(stall_cap_low, 0.0, 1.0))

        # C) kappa 平滑：用于削弱 dkappa 尖峰（默认关闭，避免影响 P7.0~P7.2）
        self._p7_3_kappa_smoothing_enabled = bool(cfg.get("kappa_smoothing_enabled", False))
        beta = cfg.get("kappa_smoothing_beta", 0.25)
        try:
            beta = float(beta)
        except Exception:
            beta = 0.25
        self._p7_3_kappa_smoothing_beta = float(np.clip(beta, 0.0, 1.0))
        dkappa_limit = cfg.get("kappa_dkappa_limit", None)
        if dkappa_limit is None:
            self._p7_3_kappa_dkappa_limit = float("inf")
        else:
            try:
                dkappa_limit = float(dkappa_limit)
            except Exception:
                dkappa_limit = float("inf")
            self._p7_3_kappa_dkappa_limit = float(max(dkappa_limit, 0.0))

        # D) trace ring buffer（仅用于 NaN/Inf dump）
        ring = cfg.get("trace_ring_size", 200)
        try:
            ring = int(ring)
        except Exception:
            ring = 200
        self._p7_3_trace_ring_size = int(max(50, min(ring, 2000)))

    def _init_p8_config(self) -> None:
        """读取/初始化 P8.0 配置（turn scan + braking envelope）。"""
        cfg = {}
        if isinstance(self.reward_weights, dict):
            cfg = self.reward_weights.get("p8", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}

        kappa_th = cfg.get("kappa_th", 0.0)
        try:
            kappa_th = float(kappa_th)
        except Exception:
            kappa_th = 0.0
        self._p8_kappa_th = float(max(kappa_th, 0.0))

        refine = cfg.get("turn_refine_steps", 4)
        try:
            refine = int(refine)
        except Exception:
            refine = 4
        self._p8_turn_refine_steps = int(max(1, min(refine, 50)))

        w_mult = cfg.get("turn_angle_window_mult", 3)
        try:
            w_mult = int(w_mult)
        except Exception:
            w_mult = 3
        self._p8_turn_angle_window_mult = int(max(2, min(w_mult, 20)))

        # Corner exit / cap release controls (Phase A patch)
        self._p8_use_corner_exit_hysteresis = bool(cfg.get("use_corner_exit_hysteresis", True))
        self._p8_use_vcap_rate_limit = bool(cfg.get("use_vcap_rate_limit", True))
        self._p8_use_recovery_cap = bool(cfg.get("use_recovery_cap", True))

        corner_on_scale = cfg.get("corner_on_dist_scale", 2.0)
        corner_off_scale = cfg.get("corner_off_dist_scale", 4.0)
        try:
            corner_on_scale = float(corner_on_scale)
        except Exception:
            corner_on_scale = 2.0
        try:
            corner_off_scale = float(corner_off_scale)
        except Exception:
            corner_off_scale = 4.0
        self._p8_corner_on_dist_scale = float(max(corner_on_scale, 0.0))
        self._p8_corner_off_dist_scale = float(max(corner_off_scale, self._p8_corner_on_dist_scale + 1e-6))

        hold_steps = cfg.get("corner_exit_hold_steps", 3)
        try:
            hold_steps = int(hold_steps)
        except Exception:
            hold_steps = 3
        self._p8_corner_exit_hold_steps = int(max(1, min(hold_steps, 20)))

        e_release_ratio = cfg.get("corner_exit_e_release_ratio", 0.6)
        psi_release_deg = cfg.get("corner_exit_psi_release_deg", 8.0)
        try:
            e_release_ratio = float(e_release_ratio)
        except Exception:
            e_release_ratio = 0.6
        try:
            psi_release_deg = float(psi_release_deg)
        except Exception:
            psi_release_deg = 8.0
        self._p8_corner_exit_e_release_ratio = float(np.clip(e_release_ratio, 0.0, 1.0))
        self._p8_corner_exit_psi_release = float(max(math.radians(psi_release_deg), 0.0))

        vcap_rate_up = cfg.get("vcap_rate_up", 0.002)
        vcap_rate_down = cfg.get("vcap_rate_down", 0.02)
        try:
            vcap_rate_up = float(vcap_rate_up)
        except Exception:
            vcap_rate_up = 0.002
        try:
            vcap_rate_down = float(vcap_rate_down)
        except Exception:
            vcap_rate_down = 0.02
        self._p8_vcap_rate_up = float(max(vcap_rate_up, 0.0))
        self._p8_vcap_rate_down = float(max(vcap_rate_down, 0.0))

        e_warn_ratio = cfg.get("recovery_e_warn_ratio", 0.85)
        e_recover_ratio = cfg.get("recovery_e_release_ratio", 0.6)
        v_recovery = cfg.get("recovery_vcap", 0.015)
        try:
            e_warn_ratio = float(e_warn_ratio)
        except Exception:
            e_warn_ratio = 0.85
        try:
            e_recover_ratio = float(e_recover_ratio)
        except Exception:
            e_recover_ratio = 0.6
        try:
            v_recovery = float(v_recovery)
        except Exception:
            v_recovery = 0.015
        self._p8_recovery_e_warn_ratio = float(np.clip(e_warn_ratio, 0.0, 1.0))
        self._p8_recovery_e_release_ratio = float(np.clip(e_recover_ratio, 0.0, 1.0))
        self._p8_recovery_vcap = float(max(v_recovery, 0.0))

        # Patch P8.1: ang_cap_min_ratio - minimum v_ratio during corner_mode (prevent stall)
        ang_cap_min = cfg.get("ang_cap_min_ratio", 0.01)
        try:
            ang_cap_min = float(ang_cap_min)
        except Exception:
            ang_cap_min = 0.01
        self._p8_ang_cap_min_ratio = float(np.clip(ang_cap_min, 0.01, 0.5))

        # Patch P8.1: corner_mode_ignore_geom_cap - ignore v_cap_geom in corner_mode
        self._p8_corner_mode_ignore_geom_cap = bool(cfg.get("corner_mode_ignore_geom_cap", False))
        # Patch v3.6: straight_mode_ignore_geom_cap - ignore v_cap_geom in straight_mode
        self._p8_straight_mode_ignore_geom_cap = bool(cfg.get("straight_mode_ignore_geom_cap", False))

    def _p7_3_compute_kappa_exec(self, *, v_exec: float, omega_exec: float) -> float:
        """P7.3：执行曲率（奇异点保护）。"""
        v = float(v_exec) if math.isfinite(float(v_exec)) else 0.0
        w = float(omega_exec) if math.isfinite(float(omega_exec)) else 0.0
        v = float(max(v, 0.0))
        v_eps = float(getattr(self, "_p7_3_v_eps", 1e-12))
        denom = float(v + v_eps)
        kappa = float(abs(w) / denom) if denom > 0.0 else 0.0
        if not math.isfinite(kappa):
            return 0.0
        return float(kappa)

    def _p7_3_apply_kappa_smoothing(self, omega_intent: float, v_intent: float) -> float:
        """P7.3：对曲率 kappa=|omega|/(v+eps) 做简单一阶滤波，抑制 dkappa 尖峰（KISS）。"""
        if not bool(getattr(self, "_p7_3_kappa_smoothing_enabled", False)):
            return float(omega_intent)

        beta = float(getattr(self, "_p7_3_kappa_smoothing_beta", 0.0))
        if not math.isfinite(beta) or beta <= 0.0:
            return float(omega_intent)

        v_eps = float(getattr(self, "_p7_3_v_eps", 1e-12))
        v = float(v_intent) if math.isfinite(float(v_intent)) else 0.0
        v = float(max(v, 0.0))
        denom = float(v + v_eps)

        w = float(omega_intent) if math.isfinite(float(omega_intent)) else 0.0
        sign = 0.0
        if w > 0.0:
            sign = 1.0
        elif w < 0.0:
            sign = -1.0

        kappa_des = float(abs(w) / denom) if denom > 0.0 else 0.0
        prev = float(getattr(self, "_p7_3_kappa_filt", 0.0))
        if not math.isfinite(prev) or prev < 0.0:
            prev = 0.0

        # 先做指数平滑，再做 dkappa 限幅（对尖峰更有效）
        kappa_raw = float((1.0 - beta) * prev + beta * kappa_des)
        dkappa_limit = float(getattr(self, "_p7_3_kappa_dkappa_limit", float("inf")))
        if math.isfinite(dkappa_limit) and dkappa_limit > 0.0:
            delta = float(np.clip(kappa_raw - prev, -dkappa_limit, dkappa_limit))
            kappa_filt = float(prev + delta)
        else:
            kappa_filt = float(kappa_raw)
        if not math.isfinite(kappa_filt) or kappa_filt < 0.0:
            kappa_filt = 0.0
        self._p7_3_kappa_filt = float(kappa_filt)

        w_smooth = float(sign * kappa_filt * denom)
        w_max = float(getattr(self, "MAX_ANG_VEL", 0.0))
        if math.isfinite(w_max) and w_max > 0.0:
            w_smooth = float(np.clip(w_smooth, -w_max, w_max))
        return float(w_smooth)

    def _p7_3_trace_append(self, *, p4_status: Dict[str, float]) -> None:
        ring = getattr(self, "_p7_3_trace_ring", None)
        if not isinstance(ring, list):
            ring = []
            self._p7_3_trace_ring = ring

        try:
            ring.append(
                {
                    "step": int(getattr(self, "current_step", 0)),
                    "pos": [float(self.current_position[0]), float(self.current_position[1])],
                    "progress": float(self.state[4]) if getattr(self, "state", None) is not None and len(self.state) > 4 else float("nan"),
                    "contour_error": float(self.get_contour_error(self.current_position)),
                    "v_exec": float(p4_status.get("v_exec", float("nan"))),
                    "omega_exec": float(p4_status.get("omega_exec", float("nan"))),
                    "v_ratio_exec": float(p4_status.get("v_ratio_exec", float("nan"))),
                    "v_ratio_cap": float(p4_status.get("v_ratio_cap", float("nan"))),
                    "kappa_exec": float(p4_status.get("kappa_exec", float("nan"))),
                    "dkappa_exec": float(p4_status.get("dkappa_exec", float("nan"))),
                    "alpha": float(p4_status.get("alpha", float("nan"))),
                }
            )
        except Exception:
            return

        max_len = int(getattr(self, "_p7_3_trace_ring_size", 200))
        if len(ring) > max_len:
            del ring[: max(0, len(ring) - max_len)]

    def _p7_3_dump_trace(self, *, reason: str, p4_status: Dict[str, float], extra: Optional[Dict[str, object]] = None) -> None:
        """P7.3：发现 NaN/Inf 时 dump trace 到 PPO_project/out。"""
        try:
            ppo_root = Path(__file__).resolve().parents[2]
            out_dir = ppo_root / "out" / "p7_3_nan_dumps"
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time() * 1000.0)
            path = out_dir / f"dump_{stamp}_{reason}.json"

            payload = {
                "reason": str(reason),
                "step": int(getattr(self, "current_step", 0)),
                "p4_status": {k: float(v) for k, v in (p4_status or {}).items() if isinstance(v, (int, float, np.floating))},
                "trace_tail": list(getattr(self, "_p7_3_trace_ring", []) or []),
            }
            if extra:
                payload["extra"] = extra
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _p7_3_assert_finite(self, *, name: str, value: float, p4_status: Dict[str, float]) -> None:
        v = float(value)
        if math.isfinite(v):
            return
        self._p7_3_dump_trace(reason=f"non_finite_{name}", p4_status=p4_status, extra={"value": str(value)})
        raise AssertionError(f"[P7.3] non-finite {name}: {value}")

    def _p7_3_assert_finite_array(self, *, name: str, arr: np.ndarray, p4_status: Dict[str, float]) -> None:
        try:
            a = np.asarray(arr, dtype=float)
        except Exception:
            self._p7_3_dump_trace(reason=f"non_numeric_{name}", p4_status=p4_status)
            raise AssertionError(f"[P7.3] non-numeric {name}")

        finite = np.isfinite(a)
        if bool(np.all(finite)):
            return

        nan_count = int(a.size - int(np.count_nonzero(finite)))
        self._p7_3_dump_trace(
            reason=f"non_finite_{name}",
            p4_status=p4_status,
            extra={"shape": list(a.shape), "nan_count": int(nan_count)},
        )
        raise AssertionError(f"[P7.3] non-finite {name}: nan_count={nan_count}")

    def _wrap_angle(self, angle: float) -> float:
        return (float(angle) + math.pi) % (2 * math.pi) - math.pi

    def _compute_p4_pre_step_status(self) -> Dict[str, float]:
        """计算 P4.0 的 turn/speed 相关量（基于当前状态 s_t，用于裁剪与奖励）。"""
        EPS = 1e-6
        status: Dict[str, float] = {
            "kappa": 0.0,
            "kappa_los": 0.0,
            "alpha": 0.0,
            "L": float("inf"),
            "theta_los": 0.0,
            "v_cap": float(self.MAX_VEL),
            # P6.0 debug: 三条边界的最小值 + 最终 cap（用于诊断哪条边界主导）
            "v_cap_w_min": float("inf"),
            "v_cap_wdot_min": float("inf"),
            "v_cap_wddot_min": float("inf"),
            "v_cap_final": float(self.MAX_VEL),
            "alpha_max_ahead": 0.0,
            "kappa_max_ahead": 0.0,
            "v_ratio_cap": 1.0,
            "v_ratio_cap_geom": 1.0,
            "v_ratio_cap_brake": 1.0,
            "v_ratio_cap_ang": 1.0,
            "v_ratio_brake": 1.0,
            "v_ratio_policy": 0.0,
            "v_ratio_exec": 0.0,
            "speed_target": float(getattr(self, "_p4_v_max", 1.0)),
            "speed_target_raw": float(getattr(self, "_p4_v_max", 1.0)),
            "speed_target_brake": float(getattr(self, "_p4_v_max", 1.0)),
            "speed_target_phys": float(getattr(self, "_p4_v_max", 1.0)) * float(self.MAX_VEL),
            "turn_severity": 0.0,
            "dist_to_turn": float("inf"),
            "dist_to_end": float("inf"),
            "turn_angle": 0.0,
            "turn_sign": 0.0,
            "turn_s": float("nan"),
            "is_isolated_corner": 0.0,
            "s_lookahead": float(getattr(self, "_p4_speed_cap_s", 0.0)),
            # P7.2: 自适应预瞄诊断字段
            "d_target": 0.0,
            "d_chosen": 0.0,
            "kappa_chosen": 0.0,
            "progress_multiplier": 1.0,
            "time_penalty": float(getattr(self, "_p4_time_penalty", 0.0)),
            # P8.0: 刹车包络（physical + ratio）
            "v_brake_turn": float(self.MAX_VEL),
            "v_brake_end": float(self.MAX_VEL),
            "v_brake_turn_ratio": 1.0,
            "v_brake_end_ratio": 1.0,
        }

        total = float(getattr(self, "_progress_total_length", 0.0))
        if not math.isfinite(total) or total <= 1e-9:
            return status

        proj, _seg_idx, s_now, _t_hat, n_hat = self._project_onto_progress_path(self.current_position)
        _ = proj  # 保持结构一致
        s_now = float(s_now)
        e_n = float(np.dot(self.current_position - proj, n_hat))
        heading_err = float(abs(self.calculate_direction_deviation(self.current_position)))

        closed = bool(getattr(self, "closed", False))
        remaining = float(max(0.0, total - s_now)) if not closed else float("inf")
        status["dist_to_end"] = float(remaining if not closed else float("inf"))

        # P8.0：从 s 域扫描下一处拐角事件（dist_to_turn/turn_angle 等）
        scan = self._scan_for_next_turn(float(s_now))
        dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
        turn_angle = float(scan.get("turn_angle", 0.0))
        turn_sign = float(scan.get("turn_sign", 0.0))
        status["dist_to_turn"] = float(dist_to_turn)
        status["turn_angle"] = float(turn_angle)
        status["turn_sign"] = float(turn_sign)
        status["turn_s"] = float(scan.get("turn_s", float("nan")))
        status["is_isolated_corner"] = 1.0 if bool(scan.get("is_isolated_corner", False)) else 0.0

        # LOS 指标仅用于 debug/reward（不得参与 v_cap）
        s_lookahead = float(getattr(self, "_p4_speed_cap_s", 0.0))
        if not closed:
            s_lookahead = float(min(s_lookahead, max(0.0, total - s_now)))
        los = self._compute_los_metrics(s_now=float(s_now), s_lookahead=float(s_lookahead))
        status["alpha"] = float(los.get("alpha", 0.0))
        status["L"] = float(los.get("L", float("inf")))
        status["kappa_los"] = float(los.get("kappa_los", 0.0))
        status["theta_los"] = float(los.get("theta_los", float(getattr(self, "_current_direction_angle", 0.0))))

        # P7.2：自适应预瞄距离（保留诊断字段；不再用于 v_cap 选择）
        v_exec_now = float(getattr(self, "velocity", 0.0))
        d0 = float(getattr(self, "_p7_2_d0", float(getattr(self, "lookahead_spacing", 1.0))))
        k_adapt = float(getattr(self, "_p7_2_k", 1.0))
        d_min = float(getattr(self, "_p7_2_d_min", 0.5 * d0))
        d_max = float(getattr(self, "_p7_2_d_max", 3.0 * d0))
        if d_max < d_min:
            d_max = d_min
        d_target = float(d0 + k_adapt * max(v_exec_now, 0.0))
        d_target = float(np.clip(d_target, max(d_min, EPS), max(d_max, EPS)))
        status["d_target"] = float(d_target)

        # P8.0：turning-feasible v_cap 只能由路径几何 + 动力学决定
        preview_points = int(max(1, int(getattr(self, "_p6_speed_cap_points", getattr(self, "lookahead_points", 5)))))
        preview_spacing = float(max(float(getattr(self, "lookahead_spacing", 1.0)), EPS))

        use_wdot = bool(getattr(self, "_p6_speed_cap_use_wdot", True))
        use_wddot = bool(getattr(self, "_p6_speed_cap_use_wddot", True))

        half_eps = float(getattr(self, "half_epsilon", 0.0))
        kappa_tol = 1.0 / (half_eps + EPS)

        theta0 = float(self._tangent_angle_at_s(float(s_now)))
        v_cap_geom = float(self.MAX_VEL)
        v_cap_w_min = float("inf")
        v_cap_wdot_min = float("inf")
        v_cap_wddot_min = float("inf")
        kappa_max = 0.0
        delta_at_kappa_max = 0.0

        d_chosen = 0.0
        kappa_chosen = 0.0
        for i in range(int(preview_points)):
            s_i = float(preview_spacing) * float(i + 1)
            if not closed:
                s_i = float(min(s_i, remaining))
            if s_i <= 1e-9:
                continue

            theta1 = float(self._tangent_angle_at_s(float(s_now) + float(s_i)))
            delta = float(self._wrap_angle(theta1 - theta0))
            kappa_i = abs(delta) / (s_i + EPS)
            if kappa_i > kappa_max:
                kappa_max = float(kappa_i)
                delta_at_kappa_max = float(delta)

            kappa_eff = float(min(kappa_i, kappa_tol))
            denom = float(kappa_eff + EPS)

            v_cap_w = float(self.MAX_ANG_VEL) / denom
            v_cap_w_min = float(min(v_cap_w_min, v_cap_w))

            v_cap_wdot = float("inf")
            if use_wdot:
                v_cap_wdot = float(math.sqrt(float(self.MAX_ANG_ACC) * float(s_i) / denom + EPS))
                v_cap_wdot_min = float(min(v_cap_wdot_min, v_cap_wdot))

            v_cap_wddot = float("inf")
            if use_wddot:
                v_cap_wddot = float((0.5 * float(self.MAX_ANG_JERK) * float(s_i) * float(s_i) / denom + EPS) ** (1.0 / 3.0))
                v_cap_wddot_min = float(min(v_cap_wddot_min, v_cap_wddot))

            v_cap_i = float(min(float(self.MAX_VEL), v_cap_w, v_cap_wdot, v_cap_wddot))
            if v_cap_i < v_cap_geom:
                v_cap_geom = float(v_cap_i)
                d_chosen = float(s_i)
                kappa_chosen = float(kappa_eff)

        v_ratio_cap_geom = float(np.clip(v_cap_geom / max(float(self.MAX_VEL), EPS), 0.0, 1.0))
        status["v_ratio_cap_geom"] = float(v_ratio_cap_geom)
        status["d_chosen"] = float(d_chosen)
        status["kappa_chosen"] = float(kappa_chosen)

        # P8.0：刹车包络（拐角 + 终点）
        a_abs = float(abs(float(self.MAX_ACC)))

        v_brake_turn_ratio = 1.0
        v_brake_turn = float(self.MAX_VEL)
        if math.isfinite(dist_to_turn):
            v_limit_at_turn = float(min(float(self.MAX_VEL), float(self.MAX_ANG_VEL) / (kappa_tol + EPS)))
            v_brake_turn = float(math.sqrt(v_limit_at_turn * v_limit_at_turn + 2.0 * a_abs * max(dist_to_turn, 0.0) + EPS))
            v_brake_turn_ratio = float(np.clip(v_brake_turn / max(float(self.MAX_VEL), EPS), 0.0, 1.0))

        v_brake_end_ratio = 1.0
        v_brake_end = float(self.MAX_VEL)
        if not closed:
            dist_to_end = float(max(total - s_now, 0.0))
            status["dist_to_end"] = float(dist_to_end)
            v_brake_end = float(math.sqrt(2.0 * a_abs * dist_to_end + EPS))
            v_brake_end_ratio = float(np.clip(v_brake_end / max(float(self.MAX_VEL), EPS), 0.0, 1.0))

        v_ratio_brake = float(min(v_brake_turn_ratio, v_brake_end_ratio))

        # P8.x: corner exit hysteresis + ang cap gating
        corner_mode = bool(getattr(self, "_p8_corner_mode", False))
        corner_exit_hold = int(getattr(self, "_p8_corner_exit_hold", 0))
        r_allow = float("nan")
        d_fillet = float("inf")
        turn_active = bool(math.isfinite(turn_angle) and abs(turn_angle) > 1e-6 and half_eps > EPS)
        if turn_active:
            sin_min = float(getattr(self, "_p8_ang_cap_sin_min", 0.2))
            sin_min = float(np.clip(sin_min, EPS, 1.0))
            sin_half = float(math.sin(min(abs(turn_angle) * 0.5, 0.5 * math.pi)))
            sin_half = float(max(sin_half, sin_min))
            r_allow = float(half_eps / max(sin_half, EPS))
            fillet_scale = float(getattr(self, "_p8_corner_fillet_scale", 1.0))
            fillet_scale = float(max(fillet_scale, 0.0))
            d_fillet = float(r_allow * math.tan(abs(turn_angle) * 0.5) * max(fillet_scale, EPS))
            d_on = float(max(d_fillet, float(getattr(self, "_p8_corner_on_dist_scale", 0.0)) * half_eps))
            d_off = float(max(d_on + EPS, float(getattr(self, "_p8_corner_off_dist_scale", 0.0)) * half_eps))
            if bool(getattr(self, "_p8_use_corner_exit_hysteresis", True)):
                if not corner_mode:
                    if math.isfinite(dist_to_turn) and dist_to_turn <= d_on:
                        corner_mode = True
                        corner_exit_hold = 0
                else:
                    e_release = float(getattr(self, "_p8_corner_exit_e_release_ratio", 0.6)) * half_eps
                    psi_release = float(getattr(self, "_p8_corner_exit_psi_release", 0.0))
                    exit_ready = bool(
                        math.isfinite(dist_to_turn)
                        and dist_to_turn >= d_off
                        and abs(e_n) <= e_release
                        and heading_err <= psi_release
                    )
                    if exit_ready:
                        corner_exit_hold += 1
                    else:
                        corner_exit_hold = 0
                    if corner_exit_hold >= int(getattr(self, "_p8_corner_exit_hold_steps", 1)):
                        corner_mode = False
                        corner_exit_hold = 0
            else:
                corner_mode = bool(math.isfinite(dist_to_turn) and dist_to_turn <= d_on)
                corner_exit_hold = 0
        else:
            corner_mode = False
            corner_exit_hold = 0

        self._p8_corner_mode = bool(corner_mode)
        self._p8_corner_exit_hold = int(corner_exit_hold)
        status["corner_mode"] = 1.0 if corner_mode else 0.0

        v_cap_ang = float(self.MAX_VEL)
        v_ratio_cap_ang = 1.0
        if turn_active and math.isfinite(r_allow):
            apply_ang_cap = False
            if bool(getattr(self, "_p8_use_corner_exit_hysteresis", True)):
                apply_ang_cap = bool(corner_mode)
            else:
                apply_ang_cap = bool(math.isfinite(dist_to_turn) and dist_to_turn <= d_fillet)
            if apply_ang_cap:
                v_cap_ang = float(self.MAX_ANG_VEL) * float(r_allow)
                v_min_ratio = float(getattr(self, "_p8_ang_cap_min_ratio", 0.01))
                v_min_ratio = float(max(v_min_ratio, 0.0))
                v_min = float(v_min_ratio) * float(self.MAX_VEL)
                if math.isfinite(v_cap_ang):
                    v_cap_ang = float(max(v_cap_ang, v_min))
                else:
                    v_cap_ang = float(self.MAX_VEL)
        if not math.isfinite(v_cap_ang):
            v_cap_ang = float(self.MAX_VEL)
        v_ratio_cap_ang = float(np.clip(v_cap_ang / max(float(self.MAX_VEL), EPS), 0.0, 1.0))

        v_ratio_cap_brake = float(v_ratio_brake)
        v_cap_brake = float(v_ratio_cap_brake) * float(self.MAX_VEL)
        
        # Patch P8.1 (ExitDriftFix): 在 corner_mode 时可选忽略 v_cap_geom
        # Patch v3.6: 新增 straight_mode_ignore_geom_cap 支持
        use_geom_cap = True
        if corner_mode and bool(getattr(self, "_p8_corner_mode_ignore_geom_cap", False)):
            use_geom_cap = False
        if (not corner_mode) and bool(getattr(self, "_p8_straight_mode_ignore_geom_cap", False)):
            use_geom_cap = False
        
        if use_geom_cap:
            v_ratio_cap_raw = float(min(v_ratio_cap_geom, v_ratio_cap_brake, v_ratio_cap_ang))
            v_cap_final = float(min(v_cap_geom, v_cap_brake, v_cap_ang))
        else:
            v_ratio_cap_raw = float(min(v_ratio_cap_brake, v_ratio_cap_ang))
            v_cap_final = float(min(v_cap_brake, v_cap_ang))

        # P8.x: recovery cap near boundary
        v_ratio_cap = float(v_ratio_cap_raw)
        recovery_active = bool(getattr(self, "_p8_recovery_active", False))
        if bool(getattr(self, "_p8_use_recovery_cap", True)) and half_eps > EPS:
            e_abs = float(abs(e_n))
            e_warn = float(getattr(self, "_p8_recovery_e_warn_ratio", 0.85)) * half_eps
            e_release = float(getattr(self, "_p8_recovery_e_release_ratio", 0.6)) * half_eps
            if recovery_active:
                if e_abs < e_release:
                    recovery_active = False
            else:
                if e_abs > e_warn:
                    recovery_active = True
            if recovery_active:
                v_ratio_cap = float(min(v_ratio_cap, float(getattr(self, "_p8_recovery_vcap", 0.0))))
        self._p8_recovery_active = bool(recovery_active)
        status["recovery_cap_active"] = 1.0 if recovery_active else 0.0

        # P8.x: rate limiter for cap release
        if bool(getattr(self, "_p8_use_vcap_rate_limit", True)):
            prev = getattr(self, "_p8_vcap_prev", None)
            if prev is None or not math.isfinite(float(prev)):
                prev = float(v_ratio_cap)
            v_up = float(getattr(self, "_p8_vcap_rate_up", 0.0))
            v_down = float(getattr(self, "_p8_vcap_rate_down", 0.0))
            if v_ratio_cap > prev + v_up:
                v_ratio_cap = float(prev + v_up)
            elif v_ratio_cap < prev - v_down:
                v_ratio_cap = float(prev - v_down)
            self._p8_vcap_prev = float(v_ratio_cap)
        else:
            self._p8_vcap_prev = float(v_ratio_cap)

        v_ratio_cap = float(min(v_ratio_cap, v_ratio_cap_raw))
        v_ratio_cap = float(np.clip(v_ratio_cap, 0.0, 1.0))
        v_cap_final = float(min(v_cap_final, v_ratio_cap * float(self.MAX_VEL)))

        status.update(
            {
                "kappa": float(kappa_chosen),
                "v_cap_final": float(v_cap_final),
                "v_cap_w_min": float(v_cap_w_min),
                "v_cap_wdot_min": float(v_cap_wdot_min),
                "v_cap_wddot_min": float(v_cap_wddot_min),
                "alpha_max_ahead": float(delta_at_kappa_max),
                "kappa_max_ahead": float(kappa_max),
                "v_brake_turn": float(v_brake_turn),
                "v_brake_end": float(v_brake_end),
                "v_brake_turn_ratio": float(v_brake_turn_ratio),
                "v_brake_end_ratio": float(v_brake_end_ratio),
                "v_ratio_brake": float(v_ratio_brake),
                "v_ratio_cap_brake": float(v_ratio_cap_brake),
                "v_ratio_cap_ang": float(v_ratio_cap_ang),
                "v_ratio_cap": float(v_ratio_cap),
                "v_cap": float(v_ratio_cap) * float(self.MAX_VEL),
            }
        )

        # speed_target（先按 turn_severity/near 计算，再应用 cap）
        v_min = float(getattr(self, "_p4_v_min", 0.35))
        v_max = float(getattr(self, "_p4_v_max", 1.0))
        d_scale = max(float(getattr(self, "_p4_d_scale", 1.0)), EPS)
        near = float(math.exp(-dist_to_turn / d_scale)) if math.isfinite(dist_to_turn) else 0.0
        theta_max = max(float(getattr(self, "_p4_theta_max", math.pi / 2)), EPS)
        turn_severity = float(np.clip(abs(turn_angle) / theta_max, 0.0, 1.0))
        status["turn_severity"] = float(turn_severity)

        speed_target_raw = v_min + (v_max - v_min) * (1.0 - turn_severity * near)

        # 出弯 boost：窗口内强制目标更快回升（仍受 cap 限制）
        if int(getattr(self, "_p4_exit_boost_remaining", 0)) > 0:
            speed_target_raw = max(speed_target_raw, float(getattr(self, "_p4_exit_speed_target_min", 0.9)))
            status["progress_multiplier"] = float(getattr(self, "_p4_exit_progress_mult", 1.35))

        speed_target_raw = float(np.clip(speed_target_raw, 0.0, v_max))
        speed_target_brake = float(min(speed_target_raw, float(status["v_brake_turn_ratio"]), float(status["v_brake_end_ratio"])))
        speed_target = float(min(speed_target_brake, float(status["v_ratio_cap"])))
        status["speed_target_raw"] = float(speed_target_raw)
        status["speed_target_brake"] = float(speed_target_brake)
        status["speed_target"] = float(speed_target)
        status["speed_target_phys"] = float(speed_target) * float(self.MAX_VEL)

        return status

    def _project_onto_progress_path(self, pt: np.ndarray) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
        """基于 progress cache 的投影（用于走廊/turn 检测，避免依赖 pnpoly/rtree 的抖动）。"""
        if not getattr(self, "_progress_segment_lengths", None):
            self._rebuild_progress_cache()
        seg_idx = self._find_nearest_segment_for_progress(pt)
        if seg_idx < 0:
            seg_idx = 0

        p1 = np.array(self.Pm[seg_idx])
        p2 = np.array(self.Pm[seg_idx + 1])
        seg_vec = p2 - p1
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-9:
            t = 0.0
            proj = p1
            t_hat = np.array([1.0, 0.0], dtype=float)
        else:
            t = float(np.clip(np.dot(pt - p1, seg_vec) / (seg_len**2), 0.0, 1.0))
            proj = p1 + t * seg_vec
            t_hat = seg_vec / seg_len
        n_hat = np.array([-t_hat[1], t_hat[0]], dtype=float)
        s_base = float(self._progress_cumulative_lengths[seg_idx]) if getattr(self, "_progress_cumulative_lengths", None) else 0.0
        s_along = s_base + t * seg_len
        return proj, seg_idx, s_along, t_hat, n_hat

    def _tangent_angle_at_s(self, s_value: float) -> float:
        if not getattr(self, "_progress_segment_lengths", None):
            self._rebuild_progress_cache()
        total = float(getattr(self, "_progress_total_length", 0.0))
        if total <= 1e-9:
            return float(getattr(self, "_current_direction_angle", 0.0))

        s = float(s_value)
        if self.closed:
            s = s % total
        else:
            s = float(np.clip(s, 0.0, total))

        cumulative = self._progress_cumulative_lengths
        seg_idx = int(np.searchsorted(cumulative, s, side="right") - 1)
        seg_idx = int(np.clip(seg_idx, 0, len(self._progress_segment_lengths) - 1))
        p1 = np.array(self.Pm[seg_idx])
        p2 = np.array(self.Pm[seg_idx + 1])
        v = p2 - p1
        if float(np.linalg.norm(v)) < 1e-9:
            return float(getattr(self, "_current_direction_angle", 0.0))
        return float(math.atan2(float(v[1]), float(v[0])))

    def _scan_for_next_turn(self, s_now: float) -> Dict[str, object]:
        """在弧长 s 域扫描下一处显著转向事件（P8.0）。

        返回字段：dist_to_turn/turn_angle/turn_sign/turn_s/is_isolated_corner。
        - dist_to_turn 必须是弧长（Arc Length），且在接近拐角时应趋近于 0。
        - 使用“顶点域（segment 方向差）”检测转向事件，避免窗口中心漂移导致 dist_to_turn 固定为 ds/2。
        """
        EPS = 1e-6
        if not getattr(self, "_progress_segment_lengths", None):
            self._rebuild_progress_cache()

        total = float(getattr(self, "_progress_total_length", 0.0))
        if not math.isfinite(total) or total <= 1e-9:
            return {
                "dist_to_turn": float("inf"),
                "turn_angle": 0.0,
                "turn_sign": 0,
                "turn_s": float("nan"),
                "is_isolated_corner": False,
            }

        seg_lengths = list(getattr(self, "_progress_segment_lengths", []) or [])
        cumulative = list(getattr(self, "_progress_cumulative_lengths", []) or [])
        seg_count = int(len(seg_lengths))
        if seg_count < 2 or len(cumulative) != seg_count + 1:
            return {
                "dist_to_turn": float("inf"),
                "turn_angle": 0.0,
                "turn_sign": 0,
                "turn_s": float("nan"),
                "is_isolated_corner": False,
            }

        seg_dirs = self.cache.get("segment_directions", None) if isinstance(getattr(self, "cache", None), dict) else None
        if not isinstance(seg_dirs, list) or len(seg_dirs) != seg_count:
            seg_dirs = []
            for i in range(seg_count):
                p1 = np.asarray(self.Pm[i], dtype=float)
                p2 = np.asarray(self.Pm[i + 1], dtype=float)
                seg_dirs.append(float(math.atan2(float(p2[1] - p1[1]), float(p2[0] - p1[0]))))

        closed = bool(getattr(self, "closed", False))
        s0 = float(s_now)
        if closed:
            s0 = s0 % total
        else:
            s0 = float(np.clip(s0, 0.0, total))

        # 阈值必须与 half_eps 关联（P8.0）
        kappa_cfg = float(getattr(self, "_p8_kappa_th", 0.0))
        half_eps = float(getattr(self, "half_epsilon", 0.0))
        kappa_th = float(max(kappa_cfg, 0.25 / (half_eps + EPS)))

        def _vertex_kappa(v_idx: int) -> Tuple[float, float]:
            """v_idx: 顶点索引（等价于 point index），对应 segment v_idx 的起点；返回(kappa_proxy, delta_theta)。"""
            if closed:
                prev_seg = (v_idx - 1) % seg_count
                next_seg = v_idx % seg_count
            else:
                prev_seg = v_idx - 1
                next_seg = v_idx
            theta_prev = float(seg_dirs[prev_seg])
            theta_next = float(seg_dirs[next_seg])
            delta = float(self._wrap_angle(theta_next - theta_prev))
            len_prev = float(seg_lengths[prev_seg])
            len_next = float(seg_lengths[next_seg]) if 0 <= next_seg < seg_count else float(len_prev)
            ds_local = float(max(0.5 * (len_prev + len_next), EPS))
            return float(abs(delta) / (ds_local + EPS)), float(delta)

        # 当前所在 segment（用于确定“下一拐角”从哪开始找）
        seg_idx0 = int(np.searchsorted(np.asarray(cumulative, dtype=float), s0, side="right") - 1)
        seg_idx0 = int(np.clip(seg_idx0, 0, seg_count - 1))

        # 在“顶点域”前向扫描：找第一个 kappa_proxy 超阈值的事件段
        start_vertex = None
        if closed:
            for step in range(1, seg_count):
                v = int((seg_idx0 + step) % seg_count)
                kappa_p, _ = _vertex_kappa(v)
                if kappa_p > kappa_th:
                    start_vertex = int(v)
                    break
        else:
            for v in range(max(1, seg_idx0 + 1), seg_count):
                kappa_p, _ = _vertex_kappa(int(v))
                if kappa_p > kappa_th:
                    start_vertex = int(v)
                    break

        if start_vertex is None:
            return {
                "dist_to_turn": float("inf"),
                "turn_angle": 0.0,
                "turn_sign": 0,
                "turn_s": float("nan"),
                "is_isolated_corner": False,
            }

        # 收集“第一个事件段”内的连续顶点，并在其中取最大 kappa 作为 turn_s（避免跳到后续拐角）
        event_vertices: List[int] = []
        if closed:
            v = int(start_vertex)
            for _ in range(seg_count):
                kappa_p, _ = _vertex_kappa(v)
                if kappa_p <= kappa_th:
                    break
                event_vertices.append(int(v))
                v = int((v + 1) % seg_count)
        else:
            for v in range(int(start_vertex), seg_count):
                kappa_p, _ = _vertex_kappa(int(v))
                if kappa_p <= kappa_th:
                    break
                event_vertices.append(int(v))

        if not event_vertices:
            return {
                "dist_to_turn": float("inf"),
                "turn_angle": 0.0,
                "turn_sign": 0,
                "turn_s": float("nan"),
                "is_isolated_corner": False,
            }

        best_v = int(event_vertices[0])
        best_kappa = -1.0
        turn_angle_sum = 0.0
        for v in event_vertices:
            kappa_p, delta = _vertex_kappa(int(v))
            if kappa_p > best_kappa:
                best_kappa = float(kappa_p)
                best_v = int(v)
            turn_angle_sum += float(delta)

        turn_s = float(cumulative[best_v])
        dist_to_turn = float((turn_s - s0) % total) if closed else float(max(turn_s - s0, 0.0))

        turn_angle = float(self._wrap_angle(float(turn_angle_sum)))
        if not math.isfinite(turn_angle):
            turn_angle = 0.0

        turn_sign = 0
        if abs(turn_angle) > 1e-6:
            turn_sign = 1 if turn_angle > 0.0 else -1

        # S 型 vs 孤立拐角：窗口内同时存在显著正/负曲率则视为复合弯
        ds_nominal = float(max(float(getattr(self, "lookahead_spacing", 1.0)), EPS))
        w_mult = int(max(2, int(getattr(self, "_p8_turn_angle_window_mult", 3))))
        w_max = float(w_mult) * ds_nominal

        pos = False
        neg = False
        if closed:
            for v in range(seg_count):
                dv = float((cumulative[v] - turn_s) % total)
                dv = float(min(dv, total - dv))
                if dv > w_max + 1e-12:
                    continue
                kappa_p, delta = _vertex_kappa(int(v))
                if kappa_p <= kappa_th:
                    continue
                if delta > 0.0:
                    pos = True
                elif delta < 0.0:
                    neg = True
                if pos and neg:
                    break
        else:
            s_lo = float(max(turn_s - w_max, 0.0))
            s_hi = float(min(turn_s + w_max, total))
            v_lo = int(np.searchsorted(np.asarray(cumulative, dtype=float), s_lo, side="left"))
            v_hi = int(np.searchsorted(np.asarray(cumulative, dtype=float), s_hi, side="right"))
            for v in range(max(1, v_lo), min(seg_count, v_hi)):
                kappa_p, delta = _vertex_kappa(int(v))
                if kappa_p <= kappa_th:
                    continue
                if delta > 0.0:
                    pos = True
                elif delta < 0.0:
                    neg = True
                if pos and neg:
                    break

        is_isolated_corner = bool(not (pos and neg))
        return {
            "dist_to_turn": float(dist_to_turn),
            "turn_angle": float(turn_angle),
            "turn_sign": int(turn_sign),
            "turn_s": float(turn_s),
            "is_isolated_corner": bool(is_isolated_corner),
        }

    def _interpolate_progress_point_at_s(self, s_value: float) -> np.ndarray:
        """按 progress cache 的弧长插值路径点（用于 LOS 指标）。"""
        if not getattr(self, "_progress_segment_lengths", None):
            self._rebuild_progress_cache()
        total = float(getattr(self, "_progress_total_length", 0.0))
        if total <= 1e-9:
            return np.array(self.Pm[0], dtype=float)

        s = float(s_value)
        if self.closed:
            s = s % total
        else:
            s = float(np.clip(s, 0.0, total))

        cumulative = self._progress_cumulative_lengths
        seg_idx = int(np.searchsorted(cumulative, s, side="right") - 1)
        seg_idx = int(np.clip(seg_idx, 0, max(0, len(self._progress_segment_lengths) - 1)))
        seg_len = float(self._progress_segment_lengths[seg_idx]) if seg_idx < len(self._progress_segment_lengths) else 0.0
        seg_start = float(cumulative[seg_idx]) if seg_idx < len(cumulative) else 0.0
        t = float((s - seg_start) / seg_len) if seg_len > 1e-9 else 0.0
        t = float(np.clip(t, 0.0, 1.0))

        p1 = np.array(self.Pm[seg_idx], dtype=float)
        p2 = np.array(self.Pm[seg_idx + 1], dtype=float)
        return p1 + t * (p2 - p1)

    def _compute_los_metrics(self, *, s_now: float, s_lookahead: float) -> Dict[str, float]:
        """计算 LOS 几何指标：alpha/L/kappa_los，用于 turn/cap/phase。"""
        heading = float(getattr(self, "_current_direction_angle", 0.0))
        lookahead = float(max(float(s_lookahead), 0.0))
        eps = float(getattr(self, "_p4_speed_cap_eps", 1e-6))
        if lookahead <= 1e-9:
            return {"alpha": 0.0, "L": float("inf"), "kappa_los": 0.0, "theta_los": heading}

        p_far = self._interpolate_progress_point_at_s(float(s_now) + lookahead)
        r = np.array(p_far, dtype=float) - np.array(self.current_position, dtype=float)
        L = float(np.linalg.norm(r))
        if not math.isfinite(L) or L <= 1e-9:
            return {"alpha": 0.0, "L": float(max(L, eps)), "kappa_los": 0.0, "theta_los": heading}

        theta_los = float(math.atan2(float(r[1]), float(r[0])))
        alpha = float(self._wrap_angle(theta_los - heading))
        kappa_los = float(abs(alpha) / (L + eps))
        return {"alpha": alpha, "L": L, "kappa_los": kappa_los, "theta_los": theta_los}

    def _compute_corridor_status(self) -> Dict[str, object]:
        """计算并更新走廊状态（P3.1），并可用于 reward/info。"""
        status: Dict[str, object] = {
            "enabled": bool(getattr(self, "enable_corridor", False)),
            "corner_phase": False,
            # P7.1: 抖动/回中诊断字段
            "toggle_count": 0,
            "exit_timer": -1,
            "exit_ramp_steps": int(getattr(self, "_corridor_exit_center_ramp_steps", 0)),
            "exit_ratio": 0.0,
            "w_center": 0.0,
            "turn_sign": 0,
            "e_n": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "e_target": 0.0,
            "in_corridor": False,
            "dist_to_interval": 0.0,
            # P5.2: 幅度自由诊断字段
            "margin_to_edge": float("nan"),
            "barrier_penalty": 0.0,
            # P5.1: LOS 几何 turn 指标
            "alpha": 0.0,
            "L": float("inf"),
            "kappa_los": 0.0,
            "theta_los": 0.0,
            # 兼容旧字段：用 LOS 语义填充
            "delta_theta": 0.0,
            "theta_far": 0.0,
            "heading_cos": 0.0,
            "dist_to_turn": float("inf"),
            "dist_to_turn_enter": float("inf"),
            "dist_to_turn_exit": float("inf"),
        }

        proj, seg_idx, s_now, _t_hat, n_hat = self._project_onto_progress_path(self.current_position)
        e_n = float(np.dot(self.current_position - proj, n_hat))  # 左正右负
        corridor_enabled = bool(getattr(self, "enable_corridor", False))
        # P5.1: LOS 角误差 alpha（内切敏感）仅用于 debug/reward（不得驱动 corner_phase/v_cap）
        s_lookahead = float(getattr(self, "_p4_speed_cap_s", 0.0))
        los = self._compute_los_metrics(s_now=float(s_now), s_lookahead=float(s_lookahead))
        alpha = float(los.get("alpha", 0.0))
        L = float(los.get("L", float("inf")))
        kappa_los = float(los.get("kappa_los", 0.0))
        theta_los = float(los.get("theta_los", float(getattr(self, "_current_direction_angle", 0.0))))

        # P8.0：turn_sign/dist_to_turn 必须来自 s 域扫描（禁止基于索引/点距）
        scan = self._scan_for_next_turn(float(s_now))
        dist_to_turn = float(scan.get("dist_to_turn", float("inf")))
        turn_angle = float(scan.get("turn_angle", 0.0))
        turn_sign_scan = int(scan.get("turn_sign", 0))
        is_isolated_corner = bool(scan.get("is_isolated_corner", False))
        # S 型：关闭内切偏置与方向性偏好，但保留走廊/中线奖励
        turn_sign_effective = int(turn_sign_scan) if (is_isolated_corner and turn_sign_scan != 0) else 0

        corner_before = bool(self.in_corner_phase)
        corner_after = corner_before
        if corridor_enabled:
            # Step 3：corner_phase 进入/退出只使用 scan 的 dist_to_turn/turn_angle
            abs_turn_angle = abs(float(turn_angle))
            if not corner_before:
                if abs_turn_angle >= float(getattr(self, "_corridor_theta_enter", 0.0)) and dist_to_turn <= float(
                    getattr(self, "_corridor_dist_enter", float("inf"))
                ):
                    self.in_corner_phase = True
            else:
                if abs_turn_angle <= float(getattr(self, "_corridor_theta_exit", 0.0)) or dist_to_turn >= float(
                    getattr(self, "_corridor_dist_exit", float("inf"))
                ):
                    self.in_corner_phase = False
            corner_after = bool(self.in_corner_phase)

            # P7.1：corner_phase toggle 计数 + exit_timer
            if corner_after != corner_before:
                self._p7_1_corner_toggle_count = int(getattr(self, "_p7_1_corner_toggle_count", 0)) + 1
            if corner_after:
                self._p7_1_exit_timer = -1
            else:
                if corner_before and not corner_after:
                    self._p7_1_exit_timer = 0
                else:
                    prev_exit_timer = int(getattr(self, "_p7_1_exit_timer", -1))
                    if prev_exit_timer >= 0:
                        self._p7_1_exit_timer = prev_exit_timer + 1

        lower = 0.0
        upper = 0.0
        e_target = 0.0
        in_corridor = False
        dist_to_interval = 0.0
        margin_to_edge = float("nan")

        corridor_half = max(self.half_epsilon - float(self._corridor_margin), 0.0)
        if corridor_enabled and self.in_corner_phase and corridor_half > 0.0:
            if is_isolated_corner and turn_sign_scan != 0:
                if turn_sign_scan > 0:
                    lower, upper = 0.0, corridor_half
                else:
                    lower, upper = -corridor_half, 0.0

                ramp_den = max(float(getattr(self, "_corridor_dist_enter", 1.0)), 1e-6)
                ramp = float(np.clip(1.0 - dist_to_turn / ramp_den, 0.0, 1.0)) if math.isfinite(dist_to_turn) else 0.0
                e_target = float(turn_sign_scan) * float(self.half_epsilon) * float(ramp)
            else:
                lower, upper = -corridor_half, corridor_half
                e_target = 0.0

            in_corridor = bool(lower <= e_n <= upper)
            if e_n < lower:
                dist_to_interval = float(lower - e_n)
            elif e_n > upper:
                dist_to_interval = float(e_n - upper)
            margin_to_edge = float(min(e_n - lower, upper - e_n))

        heading_cos = float(math.cos(float(self._wrap_angle(float(getattr(self, "_current_direction_angle", 0.0)) - theta_los))))

        # P7.1：回中 ramp 权重（严格线性）
        exit_timer = int(getattr(self, "_p7_1_exit_timer", -1))
        ramp_steps = int(getattr(self, "_corridor_exit_center_ramp_steps", 0))
        exit_ratio = 0.0
        w_center = 0.0
        if exit_timer >= 0 and ramp_steps > 0:
            exit_ratio = float(np.clip(float(exit_timer) / max(float(ramp_steps), 1.0), 0.0, 1.0))
            w_center = abs(float(getattr(self, "_corridor_center_weight", 0.0))) * float(exit_ratio)

        status.update(
            {
                # Step 3：enable_corridor==False 时强制 corner_phase=False，且不更新 self.in_corner_phase
                "corner_phase": bool(self.in_corner_phase) if corridor_enabled else False,
                "toggle_count": int(getattr(self, "_p7_1_corner_toggle_count", 0)),
                "exit_timer": int(exit_timer),
                "exit_ramp_steps": int(ramp_steps),
                "exit_ratio": float(exit_ratio),
                "w_center": float(w_center),
                "turn_sign": int(turn_sign_effective),
                "is_isolated_corner": bool(is_isolated_corner),
                "e_n": float(e_n),
                "lower": float(lower),
                "upper": float(upper),
                "e_target": float(e_target),
                "in_corridor": bool(in_corridor),
                "dist_to_interval": float(dist_to_interval),
                "margin_to_edge": float(margin_to_edge),
                "alpha": float(alpha),
                "L": float(L),
                "kappa_los": float(kappa_los),
                "theta_los": float(theta_los),
                "delta_theta": float(alpha),
                "theta_far": float(theta_los),
                "heading_cos": float(heading_cos),
                "dist_to_turn": float(dist_to_turn),
                "dist_to_turn_enter": float(dist_to_turn),
                "dist_to_turn_exit": float(dist_to_turn),
                "turn_angle": float(turn_angle),
            }
        )

        self.last_corridor_status = status
        return status

    def _get_path_direction(self, pt):
        """Use cached segment tangent; allow a small lookahead near segment end for sharp corners."""
        seg_dirs = self.cache.get("segment_directions") if isinstance(getattr(self, "cache", None), dict) else None
        if not isinstance(seg_dirs, list) or not seg_dirs:
            return self._current_direction_angle

        segment_index = int(self._find_containing_segment(pt))
        if segment_index < 0 or segment_index >= len(seg_dirs):
            return self._current_direction_angle

        # If the point is very close to the end of the current segment, switch to the next segment's tangent.
        # This mitigates sharp-corner overshoot under "path tangent + residual" dynamics (P0).
        try:
            p1 = np.array(self.Pm[segment_index], dtype=float)
            p2 = np.array(self.Pm[segment_index + 1], dtype=float)
            seg_vec = p2 - p1
            denom = float(np.dot(seg_vec, seg_vec))
            if denom > 1e-12:
                t = float(np.dot(np.asarray(pt, dtype=float) - p1, seg_vec) / denom)
                if t > 0.98:
                    next_idx = segment_index + 1
                    if next_idx >= len(seg_dirs):
                        next_idx = 0 if self.closed else segment_index
                    if 0 <= next_idx < len(seg_dirs):
                        return float(seg_dirs[next_idx])
        except Exception:
            pass

        return float(seg_dirs[segment_index])
    
    def calculate_direction_deviation(self, pt):
        path_direction = self._get_path_direction(pt, v_exec=float(getattr(self, "velocity", 0.0)))
        current_direction = self._current_direction_angle
        
        tau = current_direction - path_direction
        tau = (tau + np.pi) % (2 * np.pi) - np.pi
        return tau
    
    def calculate_reward(self):
        contour_error = self.get_contour_error(self.current_position)
        progress = float(self.state[4]) if len(self.state) > 4 else 0.0
        heading_error = abs(self.state[2]) if len(self.state) > 2 else abs(self.calculate_direction_deviation(self.current_position))
        end_point = np.array(self.Pm[-1])
        end_distance = float(np.linalg.norm(self.current_position - end_point))
        lap_done = self.lap_completed or getattr(self, "reached_target", False)
        corridor_status = self._compute_corridor_status()
        corridor_corner_phase = bool(corridor_status.get("corner_phase"))
        turn_sign = int(corridor_status.get("turn_sign", 0))
        exit_timer = int(corridor_status.get("exit_timer", -1))
        exit_steps = int(corridor_status.get("exit_ramp_steps", 0))
        exit_active = bool(exit_timer >= 0 and exit_steps > 0 and exit_timer <= exit_steps)
        corridor_active = bool(corridor_status.get("enabled")) and bool(corridor_corner_phase or exit_active)
        corridor_e_n = float(corridor_status.get("e_n", 0.0))
        corridor_target_error = abs(corridor_e_n - float(corridor_status.get("e_target", 0.0)))
        corridor_heading_cos = float(corridor_status.get("heading_cos", 0.0))
        corridor_margin_to_edge = corridor_status.get("margin_to_edge", float("nan"))
        corridor_margin_to_edge = float(corridor_margin_to_edge) if corridor_margin_to_edge is not None else float("nan")
        w_center = float(corridor_status.get("w_center", 0.0))

        p4_status = getattr(self, "_p4_step_status", {}) or {}
        if not isinstance(p4_status, dict):
            p4_status = {}
        speed_target = p4_status.get("speed_target", None)
        speed_target = float(speed_target) if speed_target is not None else None
        v_ratio_exec = p4_status.get("v_ratio_exec", None)
        v_ratio_exec = float(v_ratio_exec) if v_ratio_exec is not None else None
        progress_multiplier = float(p4_status.get("progress_multiplier", 1.0))
        time_penalty = float(p4_status.get("time_penalty", getattr(self, "_p4_time_penalty", 0.0)))
        du_theta_u = float(p4_status.get("du_theta_u", 0.0))
        du_v_u = float(p4_status.get("du_v_u", 0.0))

        reward, components = self.reward_calculator.calculate_reward(
            contour_error=contour_error,
            progress=progress,
            velocity=self.velocity,
            heading_error=heading_error,
            kcm_intervention=self.kcm_intervention,
            end_distance=end_distance,
            jerk=self.jerk,
            angular_jerk=self.angular_jerk,
            du_theta_u=float(du_theta_u),
            du_v_u=float(du_v_u),
            du_enabled=bool(getattr(self, "_p6_du_enabled", False)),
            du_weight=float(getattr(self, "_p6_du_weight", 0.0)),
            du_mode=str(getattr(self, "_p6_du_mode", "l1")),
            lap_completed=lap_done,
            is_closed=self.closed,
            v_ratio_exec=v_ratio_exec,
            speed_target=speed_target,
            speed_weight=float(getattr(self, "_p4_speed_weight", 6.0)),
            time_penalty=time_penalty,
            progress_multiplier=progress_multiplier,
            stall_triggered=bool(getattr(self, "_p4_stall_triggered", False)),
            stall_penalty=float(getattr(self, "_p4_stall_penalty", 0.0)),
            corridor_enabled=bool(corridor_status.get("enabled", False)),
            corridor_active=corridor_active,
            corridor_in_corridor=bool(corridor_status.get("in_corridor", False)),
            corridor_target_error=float(corridor_target_error),
            corridor_outside_distance=float(corridor_status.get("dist_to_interval", 0.0)),
            corridor_e_n=float(corridor_e_n),
            corridor_margin_to_edge=float(corridor_margin_to_edge),
            corridor_safe_margin=float(getattr(self, "_corridor_safe_margin", 0.0)),
            corridor_barrier_scale=float(getattr(self, "_corridor_barrier_scale", 0.0)),
            corridor_barrier_weight=float(getattr(self, "_corridor_barrier_weight", 0.0)),
            corridor_center_weight=float(w_center),
            corridor_center_power=float(getattr(self, "_corridor_center_power", 2.0)),
            corridor_heading_cos=float(corridor_heading_cos),
            corridor_heading_weight=float(self._corridor_heading_weight),
            corridor_outside_penalty_weight=float(self._corridor_outside_penalty_weight),
            corridor_corner_phase=bool(corridor_corner_phase),
            corridor_turn_sign=int(turn_sign),
            corridor_dir_pref_weight=float(getattr(self, "_corridor_dir_pref_weight", 0.0)),
            corridor_dir_pref_beta=float(getattr(self, "_corridor_dir_pref_beta", 2.0)),
        )

        if isinstance(corridor_status, dict):
            corridor_status["barrier_penalty"] = float(components.get("corridor_barrier_penalty", 0.0))

        self.last_reward_components = components
        return reward

    def _calculate_segment_adaptive_reward(self):
        """根据路径段类型计算自适应奖励，加强直线段跟踪精度"""
        current_segment = self.current_segment_idx
        distance_error = self.get_contour_error(self.current_position)
        # 修正：使half_epsilon 作为归一化基
        error_ratio = distance_error / self.half_epsilon
        
        # 获取当前和下一个拐角的角度
        current_angle = self._get_current_segment_angle()
        next_angle = abs(self.state[5])  # 下一拐角角度
        distance_to_turn = self.state[3]  # 到下一拐点距离
        
        # 判断路径段类
        turn_threshold = 0.8
        is_near_turn = distance_to_turn < turn_threshold
        is_sharp_turn = next_angle > np.pi / 8  # 22.5度以上为急转
        is_moderate_turn = np.pi / 16 < next_angle <= np.pi / 8  # 11.25-22.5度为中等转弯
        is_straight = next_angle <= np.pi / 16  # 11.25度以下为直线
        
        # 判断是否刚过转弯
        just_passed_turn = False
        if current_segment > 0:
            prev_angle = abs(self.cache['angles'][current_segment - 1]) if current_segment - 1 < len(self.cache['angles']) else 0
            just_passed_turn = prev_angle > np.pi / 8 and distance_to_turn > 2.0
        
        if is_near_turn and is_sharp_turn:
            # 接近急转弯处：优先控制速度，适当放宽精度要求
            speed_penalty = -12.0 * (self.velocity / self.MAX_VEL) ** 2
            precision_bonus = 15.0 * np.exp(-8 * error_ratio)  # 相对宽松的精度要
            return speed_penalty + precision_bonus
            
        elif is_near_turn and is_moderate_turn:
            # 接近中等转弯：平衡速度和精
            speed_penalty = -6.0 * (self.velocity / self.MAX_VEL) ** 2
            precision_bonus = 20.0 * np.exp(-12 * error_ratio)
            return speed_penalty + precision_bonus
            
        elif just_passed_turn:
            # 刚过转弯：加强误差修正奖励，并鼓励出弯弹射
            error_correction = -30.0 * (error_ratio) ** 2
            direction_bonus = 20.0 * np.exp(-12 * abs(self.state[2]))  # 方向对齐奖励

            # 出弯弹射奖励：正向加速度且尚未恢复到巡航速度时奖励越大
            exit_burst_bonus = 0.0
            if self.acceleration > 0 and self.velocity < self.MAX_VEL * 0.95:
                exit_burst_bonus = 10.0 * (self.acceleration / self.MAX_ACC)

            return error_correction + direction_bonus + exit_burst_bonus
            
        elif is_straight:
            # 直线段：极度重视跟踪精度，要求紧贴PM
            # 大幅加强精度惩罚
            if error_ratio > 0.3:  # 误差超过30%容差时严厉惩
                precision_penalty = -80.0 * (error_ratio) ** 3
            else:
                precision_penalty = -40.0 * (error_ratio) ** 2
            
            # 精度奖励：越接近PM越好
            precision_bonus = 50.0 * np.exp(-20 * error_ratio)  # 极高的精度要
            
            # 速度奖励：在保证精度的前提下鼓励合理速度
            if error_ratio < 0.2:  # 误差小于20%时才给速度奖励
                speed_bonus = 12.0 * (self.velocity / self.MAX_VEL) * np.exp(-10 * error_ratio)
            else:
                speed_bonus = 0.0  # 误差大时不给速度奖励
            
            # 方向对齐奖励：直线段要求严格的方向对
            direction_bonus = 25.0 * np.exp(-15 * abs(self.state[2]))
            
            return precision_penalty + precision_bonus + speed_bonus + direction_bonus
            
        else:
            # 其他情况：平衡处
            precision_penalty = -20.0 * (error_ratio) ** 2
            precision_bonus = 15.0 * np.exp(-10 * error_ratio)
            speed_bonus = 8.0 * (self.velocity / self.MAX_VEL)
            return precision_penalty + precision_bonus + speed_bonus

    def _calculate_velocity_reward(self):
        """Speed reward: high on straights, low on turns"""
        distance_to_turn = self.state[3]
        next_angle = abs(self.state[5])
        
        # 计算期望速度
        if distance_to_turn < 0.3 and next_angle > np.pi / 8:  # 接近急转
            expected_speed_ratio = 0.3  # 期望30%最大速度
        elif distance_to_turn < 0.6 and next_angle > np.pi / 12:  # 接近中等转弯
            expected_speed_ratio = 0.6  # 期望60%最大速度
        else:  # 直线
            expected_speed_ratio = 0.9  # 期望90%最大速度
        
        current_speed_ratio = self.velocity / self.MAX_VEL
        speed_error = abs(current_speed_ratio - expected_speed_ratio)
        
        return -10.0 * speed_error

    def _calculate_smoothness_reward(self):
        """Smoothness reward"""
        jerk_penalty = -2.0 * (abs(self.jerk) / self.MAX_JERK) ** 2
        ang_jerk_penalty = -2.0 * (abs(self.angular_jerk) / self.MAX_ANG_JERK) ** 2
        return jerk_penalty + ang_jerk_penalty

    def _get_current_segment_angle(self):
        """Get angle info for the current segment"""
        if self.current_segment_idx < len(self.cache['angles']):
            return abs(self.cache['angles'][self.current_segment_idx])
        return 0.0

    def _calculate_corner_correction_reward(self):
        """Turn-exit correction reward using error history trends"""
        distance_from_path = self.get_contour_error(self.current_position)
        current_segment = self.current_segment_idx
        
        # 判断是否刚过转弯
        if current_segment > 0:
            prev_angle = abs(self.cache['angles'][current_segment - 1]) if current_segment - 1 < len(self.cache['angles']) else 0
            
            # 如果上一个转弯角度较大，说明刚经过急转
            if prev_angle > np.pi / 8:  # 降低阈值到22.5
                # 在转弯后的一段距离内加强修正奖励
                distance_to_turn = self.state[3]
                if distance_to_turn > 0.5 and distance_to_turn < 2.0:  # 转弯后适当距离
                    
                    # 基础修正奖励：距离路径越近越
                    # 修正：使half_epsilon 作为归一化基
                    base_correction = 20.0 * np.exp(-8 * distance_from_path / self.half_epsilon)
                    
                    # 趋势奖励：如果误差在减少，给予额外奖
                    trend_bonus = 0.0
                    if len(self.error_history) >= 3:
                        recent_errors = self.error_history[-3:]
                        if recent_errors[-1] < recent_errors[0]:  # 误差在减
                            error_reduction = (recent_errors[0] - recent_errors[-1]) / self.half_epsilon
                            trend_bonus = 10.0 * error_reduction
                    
                    return base_correction + trend_bonus
        
        return 0.0
    
    def _calculate_start_stability_reward(self):
        """计算初始稳定奖励，防止出发时偏离路径"""
        # 在前50步内加强跟踪精度要求
        if self.current_step < 50:
            distance_error = self.get_contour_error(self.current_position)
            distance_ratio = distance_error / self.epsilon
            
            # 前期步骤越小，要求精度越
            step_factor = (50 - self.current_step) / 50.0  # 步数越少，因子越
            
            # 距离路径越近，奖励越
            precision_reward = 30.0 * step_factor * np.exp(-10 * distance_ratio)
            
            # 额外的方向对齐奖
            tau = abs(self.state[2])
            direction_reward = 20.0 * step_factor * np.exp(-15 * tau)
            
            return precision_reward + direction_reward
        
        return 0.0
        
    def _calculate_path_progress(self, pt):
        """Fixed path progress calculation"""
        n = len(self.Pm)
        total_length = self.cache['total_path_length'] or 1.0
        
        # 找到当前所在线
        segment_index = self._find_containing_segment(pt)
        if segment_index >= 0:
            current_dist = 0.0
            
            # 累加之前所有线段长
            for i in range(segment_index):
                current_dist += self.cache['segment_lengths'][i]
            
            # 计算在当前线段上的进
            if self.closed and segment_index == len(self.Pm) - 1:
                # 闭合路径的最后一段（连接到起点）
                p1 = np.array(self.Pm[-1])
                p2 = np.array(self.Pm[0])
            else:
                p1 = np.array(self.Pm[segment_index])
                p2 = np.array(self.Pm[segment_index + 1])
            
            # 投影到线段上
            seg_vec = p2 - p1
            pt_vec = pt - p1
            seg_length = np.linalg.norm(seg_vec)
            
            if seg_length > 1e-6:
                t = np.clip(np.dot(pt_vec, seg_vec) / (seg_length ** 2), 0, 1)
                segment_progress = t * seg_length
                current_dist += segment_progress
            
            progress = current_dist / total_length
            
            # 闭合路径特殊处理：确保进度不超过1
            if self.closed:
                progress = min(progress, 1.0)
            
            return progress
        
        return 0.0
    
    def _create_polygons(self):
        polygons: List[List[np.ndarray]] = []
        n = len(self.Pm)
        if n < 2:
            return polygons

        closed = self.closed
        has_duplicate_last = closed and np.allclose(self.Pm[0], self.Pm[-1], atol=1e-6)
        m = (n - 1) if has_duplicate_last else n

        if not closed:
            for i in range(n - 1):
                if (
                    self.cache["Pl"][i] is not None
                    and self.cache["Pl"][i + 1] is not None
                    and self.cache["Pr"][i] is not None
                    and self.cache["Pr"][i + 1] is not None
                ):
                    polygons.append([self.cache["Pl"][i], self.cache["Pl"][i + 1], self.cache["Pr"][i + 1], self.cache["Pr"][i]])
            return polygons

        for i in range(m):
            j = (i + 1) % m
            if self.cache["Pl"][i] is None or self.cache["Pl"][j] is None or self.cache["Pr"][i] is None or self.cache["Pr"][j] is None:
                continue
            polygons.append([self.cache["Pl"][i], self.cache["Pl"][j], self.cache["Pr"][j], self.cache["Pr"][i]])
        return polygons
    
    def get_contour_error(self, pt):
        """使用缓存的多边形信息"""
        segment_idx = self._find_containing_segment(pt)
        if segment_idx >= 0:
            p1 = np.array(self.Pm[segment_idx])
            p2 = np.array(self.Pm[segment_idx + 1])
            return point_to_line_distance(pt, p1, p2)
        return self._traditional_shortest_distance(pt)
 
    def _find_containing_segment(self, pt):
        """Use R-tree for faster queries"""
        if self.rtree_idx is None:
            return self._find_nearest_segment_by_projection(pt)

        x, y = pt
        candidate_idxs = list(self.rtree_idx.intersection((x, y, x, y)))
        
        # 先检查当前线
        if self.current_segment_idx in candidate_idxs:
            polygon = self.cache['polygons'][self.current_segment_idx]
            if polygon and is_point_in_polygon((x,y), polygon):
                return self.current_segment_idx
        
        # 检查候选线
        for idx in candidate_idxs:
            polygon = self.cache['polygons'][idx]
            if polygon and is_point_in_polygon((x,y), polygon):
                return idx
        
        # 后备方案：投影法
        return self._find_nearest_segment_by_projection(pt)
    
    def _find_nearest_segment_by_projection(self, pt):
        """传统方法：通过投影找到最近的线段"""
        min_dist = float('inf')
        nearest_segment_index = -1
        n = len(self.Pm)
        closed = self.closed
        segments = n if closed else n-1
        
        for i in range(segments):
            j = (i+1) % n
            p1 = np.array(self.Pm[i])
            p2 = np.array(self.Pm[j])
            
            # 计算点到线段的投
            seg_vec = p2 - p1
            pt_vec = pt - p1
            seg_length_sq = np.dot(seg_vec, seg_vec)
            
            # 避免除以
            if seg_length_sq < 1e-6:
                continue
                
            # 计算投影参数 t
            t = np.dot(pt_vec, seg_vec) / seg_length_sq
            t = np.clip(t, 0, 1)
            projection = p1 + t * seg_vec
            
            # 计算距离
            dist = np.linalg.norm(pt - projection)
            
            # 更新最近线
            if dist < min_dist:
                min_dist = dist
                nearest_segment_index = i
        
        return nearest_segment_index
    
    def _find_nearest_segment_for_progress(self, pt):
        """专门为进度计算找最近线段，避免套圈问题"""
        min_dist = float('inf')
        nearest_segment_index = -1
        n = len(self.Pm)
        
        # 只考虑实际的线段数量（闭合路径排除最后一个重复点
        actual_segments = n - 1 if self.closed else n - 1
        
        for i in range(actual_segments):
            p1 = np.array(self.Pm[i])
            p2 = np.array(self.Pm[i + 1])
            
            # 计算点到线段的距
            seg_vec = p2 - p1
            pt_vec = pt - p1
            seg_length_sq = np.dot(seg_vec, seg_vec)
            
            if seg_length_sq < 1e-6:
                continue
                
            # 计算投影参数 t
            t = np.dot(pt_vec, seg_vec) / seg_length_sq
            t = np.clip(t, 0, 1)
            projection = p1 + t * seg_vec
            
            # 计算距离
            dist = np.linalg.norm(pt - projection)
            
            # 更新最近线
            if dist < min_dist:
                min_dist = dist
                nearest_segment_index = i
        
        return nearest_segment_index

    def _traditional_path_progress(self, pt, total_length, closed):
        """Fallback path progress when pnpoly fails"""
        current_dist = 0.0
        n = len(self.Pm)
        found = False
        
        for i in range(n-1):
            p1, p2 = self.Pm[i], self.Pm[i+1]
            segment_length = np.linalg.norm(p2-p1)
            
            if segment_length < 1e-6:
                continue
                
            projection = project_point_to_segment(pt, p1, p2)
            dist_to_segment = np.linalg.norm(pt - projection)
            
            # 修正：使half_epsilon 作为容差范围
            if dist_to_segment < self.half_epsilon:
                current_dist += np.linalg.norm(projection - p1)
                found = True
                break  # 找到最近的线段后退出循
            # 不再累加不相关的线段长度
        
        # 检查闭合路径的最后一个线
        if not found and closed:
            p1, p2 = self.Pm[-1], self.Pm[0]
            segment_length = np.linalg.norm(p2-p1)
            if segment_length > 1e-6:
                projection = project_point_to_segment(pt, p1, p2)
                dist_to_segment = np.linalg.norm(pt - projection)
                
                # 修正：使half_epsilon 作为容差范围
                if dist_to_segment < self.half_epsilon:
                    current_dist += np.linalg.norm(projection - p1)
                    found = True
        
        # 关键修改：未找到时返进度
        return current_dist / total_length if found and total_length > 0 else 0.0
    
    def _traditional_shortest_distance(self, pt):
        min_dist = float('inf')
        n = len(self.Pm)
        closed = np.allclose(self.Pm[0], self.Pm[-1], atol=1e-6)
        segments = n if closed else n-1
        
        for i in range(segments):
            j = (i+1) % n
            p1 = np.array(self.Pm[i])
            p2 = np.array(self.Pm[j])
            seg_vec = p2 - p1
            pt_vec = np.array(pt) - p1
            t = np.dot(pt_vec, seg_vec) / (np.linalg.norm(seg_vec)**2 + 1e-6)
            t = np.clip(t, 0, 1)
            projection = p1 + t * seg_vec
            dist = np.linalg.norm(np.array(pt) - projection)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _open_reached_target(self, pt: np.ndarray) -> bool:
        """Open path: allow crossing the finish line on last segment (P3.0)."""
        if self.closed or len(self.Pm) < 2:
            return False

        p1 = np.array(self.Pm[-2])
        p2 = np.array(self.Pm[-1])
        seg_vec = p2 - p1
        denom = float(np.dot(seg_vec, seg_vec))
        if denom < 1e-12:
            return False

        t = float(np.dot(pt - p1, seg_vec) / denom)  # do NOT clip upper bound
        dist_to_finish_line = float(point_to_line_distance(pt, p1, p2))
        # Patch v3.31: 使用放宽的 OOB 边界作为终点判定容差
        finish_tol = float(getattr(self, "_oob_half_epsilon", self.half_epsilon))
        return t >= 1.0 and dist_to_finish_line <= finish_tol

    def _maybe_print_episode_summary(self, *, contour_error: float) -> None:
        if not getattr(self, "enable_episode_diagnostics", False):
            return
        if getattr(self, "_episode_summary_printed", False):
            return

        progress = float(self.state[4]) if getattr(self, "state", None) is not None and len(self.state) > 4 else 0.0
        end_point = np.array(self.Pm[-1])
        start_point = np.array(self.Pm[0])
        end_distance = float(np.linalg.norm(self.current_position - end_point))
        dist_to_start = float(np.linalg.norm(self.current_position - start_point))

        mean_velocity = float(self.velocity)
        if getattr(self, "trajectory_states", None):
            try:
                v_samples = [float(s[6]) for s in self.trajectory_states if len(s) > 6]
                if v_samples:
                    mean_velocity = float(np.mean(v_samples))
            except Exception:
                mean_velocity = float(self.velocity)

        p4 = getattr(self, "_p4_step_status", {}) or {}
        if not isinstance(p4, dict):
            p4 = {}
        v_intent = float(p4.get("v_intent", float("nan")))
        v_target = float(p4.get("v_target", float("nan")))
        v_exec = float(p4.get("v_exec", float(getattr(self, "velocity", 0.0))))
        v_ratio_exec = float(p4.get("v_ratio_exec", float("nan")))
        v_ratio_cap = float(p4.get("v_ratio_cap", float("nan")))
        omega_intent = float(p4.get("omega_intent", float("nan")))
        omega_exec = float(p4.get("omega_exec", float(getattr(self, "angular_vel", 0.0))))

        print(
            "[EPISODE_END] "
            f"closed={bool(self.closed)} lap_completed={bool(self.lap_completed)} reached_target={bool(getattr(self, 'reached_target', False))} "
            f"final_progress={progress:.4f} final_end_distance={end_distance:.4f} final_dist_to_start={dist_to_start:.4f} "
            f"final_contour_error={float(contour_error):.4f} mean_velocity={mean_velocity:.4f} steps={int(self.current_step)} dt={float(self.interpolation_period)} "
            f"v_intent={v_intent:.4f} v_target={v_target:.4f} v_exec={v_exec:.4f} v_ratio_exec={v_ratio_exec:.4f} v_ratio_cap={v_ratio_cap:.4f} "
            f"omega_intent={omega_intent:.4f} omega_exec={omega_exec:.4f}"
        )
        self._episode_summary_printed = True
    
    def is_done(self):
        contour_error = float(self.get_contour_error(self.current_position))
        
        # 1) OOB：误差超限立即终止
        # 修正：使用 half_epsilon，因为边界在 ±epsilon/2
        # Patch v3.28: 允许通过 _disable_oob_check 禁用 OOB 检查
        # Patch v3.29: 支持通过 _oob_half_epsilon 设置独立的 OOB 边界
        oob_boundary = float(getattr(self, "_oob_half_epsilon", self.half_epsilon))
        if not bool(getattr(self, "_disable_oob_check", False)) and contour_error > oob_boundary:
            self._maybe_print_episode_summary(contour_error=contour_error)
            return True

        # 2) success（优先于 stall/max_steps，避免“到终点但被误判失败”）
        if self.closed and self.lap_completed:
            self.reached_target = True
            self._maybe_print_episode_summary(contour_error=contour_error)
            return True

        if not self.closed:
            progress = float(self.state[4]) if getattr(self, "state", None) is not None and len(self.state) > 4 else 0.0
            end_point = np.array(self.Pm[-1])
            end_distance = float(np.linalg.norm(self.current_position - end_point))

            # 保留跨终点线，同时加备用判据（P7.3）：progress 足够大且终点距离足够小
            if self._open_reached_target(self.current_position) or (
                progress > 0.995 and end_distance < self.half_epsilon
            ):
                self.reached_target = True
                self._maybe_print_episode_summary(contour_error=contour_error)
                return True

        # 3) stall：避免无意义拖回合
        if bool(getattr(self, "_p4_stall_triggered", False)):
            self._maybe_print_episode_summary(contour_error=contour_error)
            return True

        # 4) max_steps：最后兜底
        if self.current_step >= self.max_steps:
            self._maybe_print_episode_summary(contour_error=contour_error)
            return True

        return False


def create_environment_from_config(config: Dict, path_points: Iterable, device=None) -> 'Env':
    """根据配置字典构建 Env 实例。"""
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    Pm = [np.array(pt) for pt in path_points]
    reward_weights = config.get("reward_weights", {})

    return Env(
        device=device or env_cfg.get("device"),
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=Pm,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=reward_weights,
    )


__all__ = ["Env", "create_environment_from_config", "apply_kinematic_constraints"]
