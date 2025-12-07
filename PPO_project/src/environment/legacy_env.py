import os
# 解决 OpenMP 多副本冲突问- 必须在所有其他导入之前设
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import copy
import numpy as np
import gymnasium as gym
from torch import nn
import torch
import math
from numpy.linalg import norm
from torch.distributions import Normal
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec,font_manager
import pandas as pd
import matplotlib as mpl
import torch.nn.functional as F
from tqdm import tqdm
import csv
from math import degrees,acos,sqrt
from typing import Dict, List, Tuple, Optional
import time
from rtree import index
from src.utils import rl_utils
from src.environment.kinematics import apply_kinematic_constraints
from src.environment.reward import RewardCalculator
from src.utils.metrics import PaperMetrics
from src.utils.plotter import configure_chinese_font, visualize_final_path

configure_chinese_font()

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
        self.interpolation_period = interpolation_period
        self.reward_weights = reward_weights or {}
        # 确保所有约束参数都是浮点数
        self.MAX_VEL = float(MAX_VEL)
        self.MAX_ACC = float(MAX_ACC)
        self.MAX_JERK = float(MAX_JERK)
        self.MAX_ANG_VEL = float(MAX_ANG_VEL)
        self.MAX_ANG_ACC = float(MAX_ANG_ACC)
        self.MAX_ANG_JERK = float(MAX_ANG_JERK)
        self.current_step = 0
        self.trajectory = []
        self.trajectory_states = []
        self.action_space = gym.spaces.Box(
            low=np.array([-self.MAX_ANG_VEL, 0.0], dtype=np.float32),
            high=np.array([self.MAX_ANG_VEL, self.MAX_VEL], dtype=np.float32),
            dtype=np.float32,
        )

        self.reward_calculator = RewardCalculator(
            weights=self.reward_weights,
            max_vel=self.MAX_VEL,
            half_epsilon=self.half_epsilon,
        )
        
        # 闭合路径追踪相关
        self.accumulated_distance = 0.0  # 累计距离
        self.previous_segment_idx = -1   # 上一个线段索
        self.reached_final_segment = False  # 是否到达最后一
        self.lap_completed = False       # 是否完成一
        self.segment_transition_history = []  # 线段转换历史
        
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
        pl, pr = self.generate_offset_paths()

        # 新增缓存字典
        self.cache = {
            'segment_lengths': None,
            'segment_directions': None,  # 新增路径方向缓存
            'angles': None,
            'Pl': pl,
            'Pr': pr,
            'polygons': None,
            'total_path_length': None,
            'segment_info': {}  # 存储每个线段的缓存信
        }
        # 预计算并缓存几何特征
        self._precompute_and_cache_geometric_features()
        self.curvature_profile, self.curvature_rate_profile = self._compute_curvature_profile()
        max_segment = max(self.cache['segment_lengths'] or [1.0])
        self.lookahead_longitudinal_scale = max(max_segment * self.lookahead_points, 1.0)
        self.lookahead_lateral_scale = max(self.half_epsilon, 1.0)
        max_curvature_rate = max([abs(v) for v in self.curvature_rate_profile] + [0.0])
        self.curvature_rate_scale = max(max_curvature_rate, 1e-3)
        # 创建三角函数查找表
        self._create_trig_lookup_table()
        
        # 添加新属性用于跟踪线段信
        self.current_segment_idx = 0
        self.segment_count = len(self.Pm) - 1 if not self.closed else len(self.Pm)
        # 创建 R-tree 空间索引
        self.rtree_idx = index.Index()
        for idx, polygon in enumerate(self.cache['polygons']):
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
        
        self.reset()
    
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
        self.cache['total_path_length'] = sum(segment_lengths) if segment_lengths else 0.0
        
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
    
    def reset(self):
        self.current_step = 0
        self.current_position = np.array(self.Pm[0])  # 确保是数组形
        self.trajectory = [self.current_position.copy()]
        self.trajectory_states = [ ]
        self._current_direction_angle, self._current_step_length = self.initialize_starting_conditions()

        # 初始线段索引和进
        self.current_segment_idx = 0
        distance_to_next_turn = self.cache['segment_lengths'][0] if self.cache['segment_lengths'] else 0.0
        overall_progress = 0.0
        
        # 重置闭合路径追踪状
        self.accumulated_distance = 0.0
        self.previous_segment_idx = -1
        self.reached_final_segment = False
        self.lap_completed = False
        self.segment_transition_history = []
        
        # 重置误差历史
        self.error_history = []
        
        # 重置成功标志
        self.reached_target = False
        
        # 计算初始方向偏差（应该是0或很小）
        tau_initial = 0.0  # 初始时应该完全对
        
        # 计算下一个转折点夹角
        next_angle = self._get_next_angle(self.current_segment_idx)
        
        # 运动学状态变
        self.velocity = 0.0       # 当前速度
        self.acceleration = 0.0   # 当前加速度
        self.jerk = 0.0           # 当前捷度
        self.angular_vel = 0.0  # 当前角速度
        self.angular_acc = 0.0  # 当前角加速度
        self.angular_jerk = 0.0  # 当前角加加速度
        self.kcm_intervention = 0.0  # 运动学约束干预程        
        self.reward_calculator.reset()
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
        return normalized_state

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
        #解包动作
        theta_prime, length_prime = action
        
        # === 保存网络输出的原始意图（raw intent===
        raw_angular_vel_intent = theta_prime  # 网络输出的原始角速度意图
        raw_linear_vel_intent = length_prime  # 网络输出的原始线速度意图
        
        # 使用Numba优化的约束计
        (self.velocity, self.acceleration, self.jerk,
         self.angular_vel, self.angular_acc, self.angular_jerk) = apply_kinematic_constraints(
            prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
            length_prime, theta_prime, self.interpolation_period,
            self.MAX_VEL, self.MAX_ACC, self.MAX_JERK,
            self.MAX_ANG_VEL, self.MAX_ANG_ACC, self.MAX_ANG_JERK
        )
        
        # === 计算KCM干预程度：原始意图与实际执行动作的差===
        velocity_diff = abs(self.velocity - raw_linear_vel_intent)
        angular_vel_diff = abs(self.angular_vel - raw_angular_vel_intent)
        self.kcm_intervention = velocity_diff + angular_vel_diff
        
        # === 使用修正后的动作执行状态转===
        # 构建最终安全动
        safe_action = (self.angular_vel, self.velocity)  # final_vel是线速度
        next_state = self.apply_action(safe_action)
        self.trajectory_states.append(next_state)
        
        # 更新误差历史
        current_error = self.get_contour_error(self.current_position)
        self.error_history.append(current_error)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)  # 保持固定长度
        
        reward = self.calculate_reward()
        done = self.is_done()

        # 添加info字典作为第四个返回值
        info = {
            "position": self.current_position.copy(),
            "step": self.current_step,
            "contour_error": self.get_contour_error(self.current_position),
            "segment_idx": self.current_segment_idx,
            "progress": next_state[4],  # 添加进度信息
            "jerk": self.jerk,  # 添加当前捷度指标
            "kcm_intervention": self.kcm_intervention,  # 添加运动学约束干预程度
        }
        normalized_state = self.normalize_state(next_state)
        self.state = next_state
        return normalized_state, reward, done, info
    
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
            normalized[base] = np.clip(state[base] / self.lookahead_longitudinal_scale, -1, 1)
            normalized[base + 1] = np.clip(state[base + 1] / self.lookahead_lateral_scale, -1, 1)
            if self.curvature_rate_scale > 0:
                normalized[base + 2] = np.clip(state[base + 2] / self.curvature_rate_scale, -1, 1)
            else:
                normalized[base + 2] = state[base + 2]
        return normalized
    
    def apply_action(self, action):
        theta_prime, length_prime = action
        path_angle = self._get_path_direction(self.current_position)
        effective_angle = path_angle + theta_prime * self.interpolation_period
        self._current_direction_angle = effective_angle

        displacement = length_prime * self.interpolation_period
        cos_angle = self.fast_cos(effective_angle)
        sin_angle = self.fast_sin(effective_angle)
        x_next = self.current_position[0] + displacement * cos_angle
        y_next = self.current_position[1] + displacement * sin_angle

        self.current_position = np.array([x_next, y_next])
        self.trajectory.append(self.current_position.copy())
        tau_next = self.calculate_direction_deviation(self.current_position)
        self.current_segment_idx, distance_to_next_turn = self._update_segment_info()
        overall_progress = self._calculate_closed_path_progress(self.current_position) if self.closed else self._calculate_path_progress(self.current_position)
        next_angle = self._get_next_angle(self.current_segment_idx)

        lookahead_features = self._compute_lookahead_features()
        base_state = np.array([theta_prime, length_prime, tau_next, distance_to_next_turn, overall_progress,
                               next_angle, self.velocity, self.acceleration, self.jerk,
                               self.angular_vel, self.angular_acc, self.angular_jerk])
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
        """Closed-path progress using segment jump detection"""
        if not self.closed:
            return self._calculate_path_progress(pt)
        
        # 如果已完成一圈，直接返回1.0
        if self.lap_completed:
            return 1.0
        
        # 找到当前最近的线段
        current_segment_idx = self._find_nearest_segment_for_progress(pt)
        if current_segment_idx < 0:
            return 0.0
        
        # 检测线段跳转，判断是否完成一
        self._detect_lap_completion_by_segment_jump(current_segment_idx)
        
        if self.lap_completed:
            return 1.0
        
        # 正常进度计算（排除重复的最后一个点
        total_segments = len(self.Pm) - 1  # 闭合路径实际只有n-1个线 
        current_dist = 0.0
        
        # 累加到当前线段之前的所有长
        for i in range(current_segment_idx):
            if i < len(self.cache['segment_lengths']):
                current_dist += self.cache['segment_lengths'][i]
        
        # 计算在当前线段上的投影位
        if current_segment_idx < total_segments:
            p1 = np.array(self.Pm[current_segment_idx])
            p2 = np.array(self.Pm[current_segment_idx + 1])
        else:
            return 0.95  # 接近完成但还未完
        
        # 投影计算
        seg_vec = p2 - p1
        pt_vec = pt - p1
        seg_length = np.linalg.norm(seg_vec)
        
        if seg_length > 1e-6:
            t = np.clip(np.dot(pt_vec, seg_vec) / (seg_length ** 2), 0, 1)
            segment_progress = t * seg_length
            current_dist += segment_progress
        
        # 计算真正的总路径长度（不包括回到起点的虚拟线段
        actual_total_length = sum(self.cache['segment_lengths'][:total_segments])
        raw_progress = current_dist / actual_total_length if actual_total_length > 0 else 0.0
        
        # 确保进度在合理范围内，但不会重置
        return np.clip(raw_progress, 0.0, 0.99)  # 最.99，完成时才返.0
    
    def _detect_lap_completion(self, current_segment):
        """Simplified lap detection to prevent infinite loops"""
        # 只在非常明确的情况下才认为完成了一
        if current_segment != self.previous_segment:
            # 检测到回到起点附近且已经有很高的进
            if (current_segment == 0 and self.last_raw_progress > 0.9 
                and self.previous_segment == len(self.Pm) - 2):
                
                self.completed_laps += 1
                
                # 防止过度套圈 - 立即停止
                if self.completed_laps >= 1:  # 只允许完成一
                    self.last_raw_progress = 1.0  # 设置为完成状
            
            self.previous_segment = current_segment
    
    def _detect_lap_completion_by_segment_jump(self, current_segment_idx):
        """Detect lap completion via segment jumps"""
        if self.lap_completed:
            return
        
        # 记录线段转换历史
        if current_segment_idx != self.previous_segment_idx:
            self.segment_transition_history.append({
                'from': self.previous_segment_idx,
                'to': current_segment_idx,
                'step': self.current_step
            })
            
            # 只保留最近的转换历史（避免内存过度使用）
            if len(self.segment_transition_history) > 20:
                self.segment_transition_history.pop(0)
            
            # 检测关键的线段跳转模式
            total_segments = len(self.Pm) - 1  # 闭合路径的实际线段数
            
            # 模式1：从最后一个线段跳转到第一个线
            if (self.previous_segment_idx == total_segments - 1 and 
                current_segment_idx == 0):
                
                # 额外验证：确保已经访问过大部分线
                visited_segments = set()
                for transition in self.segment_transition_history:
                    if transition['from'] >= 0:
                        visited_segments.add(transition['from'])
                    if transition['to'] >= 0:
                        visited_segments.add(transition['to'])
                
                # 如果访问了超0%的线段，认为完成了一
                coverage_ratio = len(visited_segments) / total_segments
                if coverage_ratio > 0.7:
                    self.lap_completed = True
                    print(f"检测到完成一圈：从线段{self.previous_segment_idx}跳转到线段{current_segment_idx}，覆盖率{coverage_ratio:.2f}")
            
            # 模式2：连续访问模式检
            elif len(self.segment_transition_history) >= 3:
                # 检查是否有连续的线段访问模式表明完成了循环
                recent_segments = [t['to'] for t in self.segment_transition_history[-5:] if t['to'] >= 0]
                
                # 如果最近访问的线段包含了从高索引到低索引的跳转
                if len(recent_segments) >= 3:
                    max_recent = max(recent_segments)
                    min_recent = min(recent_segments)
                    
                    # 从接近末尾的线段跳转到接近开头的线段
                    if (max_recent >= total_segments * 0.8 and 
                        min_recent <= total_segments * 0.2 and 
                        current_segment_idx <= 2):  # 当前在前几个线段
                        
                        # 再次检查覆盖率
                        all_visited = set(recent_segments)
                        if len(all_visited) >= total_segments * 0.6:
                            self.lap_completed = True
                            print(f"检测到完成一圈：连续访问模式，从线段{max_recent}区域跳转到线段{current_segment_idx}")
            
            self.previous_segment_idx = current_segment_idx
    
    def generate_offset_paths(self):
        """生成偏移路径，左边界(Pl)和右边界(Pr)"""
        pm=self.Pm
        # half_epsilon: 单侧偏移距离（Pm到Pl或Pr的距离）
        half_epsilon = self.half_epsilon
        
        def normalize(v):
            length = math.sqrt(v[0]**2 + v[1]**2)
            if length == 0: return (0, 0)  # 防止除以零的情况
            return (v[0] / length, v[1] / length)

        def get_parallel_lines(p1, p2, offset_distance):
            """
            计算平行
            offset_distance: 偏移距离（单侧）
            """
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 计算单位法向
            normal_vector = (-dy, dx)  # 左侧法线
            unit_normal_vector = normalize(normal_vector)
            
            A, B = unit_normal_vector  # 法线向量即为直线方程中的系数
            
            def line_equation(distance, point=p1):
                # 计算常数C
                C = -(A * point[0] + B * point[1]) + distance
                return A, B, C
            
            return line_equation(offset_distance), line_equation(-offset_distance)

        def find_intersection(line1, line2):
            A1, B1, C1 = line1
            A2, B2, C2 = line2
            
            determinant = A1 * B2 - A2 * B1
            
            if determinant == 0:
                return None  # 平行线或重合线，没有唯一交点
                
            x = (B1 * C2 - B2 * C1) / determinant  # 注意这里的顺
            y = (C1 * A2 - C2 * A1) / determinant   # 确保 y 的符号正
            
            return (x, y)

        def calculate_intersections(pm, offset_distance):
            """
            计算三个点的偏移路径交点
            offset_distance: 单侧偏移距离
            """
            p1, p2, p3 = pm
            # 获取P1P2的平行线（左右各偏移offset_distance
            l1, r1 = get_parallel_lines(p1, p2, offset_distance)
            # 获取P2P3的平行线（左右各偏移offset_distance
            l2, r2 = get_parallel_lines(p2, p3, offset_distance)
            
            # 寻找交点，允许延长线
            def extended_intersection(line1, line2):
                # 解线性方程组，允许t和s超出[0,1]
                A1, B1, C1 = line1
                A2, B2, C2 = line2
                denominator = A1 * B2 - A2 * B1
                if abs(denominator) < 1e-6:
                    return None
                x = (B1 * C2 - B2 * C1) / denominator
                y = (C1 * A2 - C2 * A1) / denominator
                return (x, y)
            
            pl = extended_intersection(l1, l2)
            pr = extended_intersection(r1, r2)
            
            return pl, pr

        def get_offset_point(p, direction, distance):
            """给定点p，根据direction方向得到距离为distance的偏移点"""
            return (p[0] + direction[1] * distance, p[1] - direction[0] * distance)

        n = len(self.Pm)
        pl: List[Optional[Tuple[float, float]]] = [None] * n
        pr: List[Optional[Tuple[float, float]]] = [None] * n
        closed = self.closed
        
        for i in range(n):
            if i == 0:
                if not closed:
                    p1, p2 = pm[i], pm[i+1]
                    direction_vector = normalize((p2[0] - p1[0], p2[1] - p1[1]))
                    pl[i] = get_offset_point(p1, direction_vector, half_epsilon)
                    pr[i] = get_offset_point(p1, direction_vector, -half_epsilon)
                else:
                    # 处理闭合路径的第一个点
                    prev_point = pm[-2] if n >=2 else pm[0]
                    next_point = pm[i+1]
                    pl[i], pr[i] = calculate_intersections([prev_point, pm[i], next_point], half_epsilon)
            elif i == n - 1:
                if not closed:
                    p1, p2 = pm[i-1], pm[i]
                    direction_vector = normalize((p2[0] - p1[0], p2[1] - p1[1]))
                    pl[i] = get_offset_point(p2, direction_vector, half_epsilon)
                    pr[i] = get_offset_point(p2, direction_vector, -half_epsilon)
                else:
                    # 闭合路径的最后一个点等同于第一个点
                    pl[i] = pl[0]
                    pr[i] = pr[0]
            else:
                if closed or i < n - 1:
                    prev_point = pm[i-1]
                    current_point = pm[i]
                    next_point = pm[(i+1) % n] if closed else pm[i+1]
                    pl_val, pr_val = calculate_intersections([prev_point, current_point, next_point], half_epsilon)
                    # 处理交点不存在的情况
                    if pl_val is None:
                        direction = normalize((next_point[0] - current_point[0], next_point[1] - current_point[1]))
                        pl_val = get_offset_point(current_point, direction, half_epsilon)
                    if pr_val is None:
                        direction = normalize((next_point[0] - current_point[0], next_point[1] - current_point[1]))
                        pr_val = get_offset_point(current_point, direction, -half_epsilon)
                    pl[i] = pl_val
                    pr[i] = pr_val

        return pl, pr
      
    def calculate_new_position(self, theta_prime_action, length_prime_action):
        # === 关键修改：使用路径方向作为基准，而非累计角度 ===
        # 1. 获取当前点的路径方向
        path_angle = self._get_path_direction(self.current_position)
        
        # 2. 将动作角度视为相对路径方向的偏移
        effective_angle = path_angle + theta_prime_action * self.interpolation_period
        self._current_direction_angle = effective_angle
        # 3. 计算新位置（保留原有速度计算
        displacement = length_prime_action * self.interpolation_period
        x_next = self.current_position[0] + displacement * np.cos(effective_angle)
        y_next = self.current_position[1] + displacement * np.sin(effective_angle)
        
        return np.array([x_next, y_next])

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

    def _get_lookahead_indices(self):
        """Get indices of forward path points for closed or open paths"""
        current_segment = self._find_containing_segment(self.current_position)
        if current_segment < 0:
            current_segment = max(self.current_segment_idx, 0)
        start_idx = current_segment + 1
        indices = []
        total_points = len(self.Pm)

        for i in range(self.lookahead_points):
            idx = start_idx + i
            if self.closed:
                idx = idx % total_points
            else:
                idx = min(idx, total_points - 1)
            indices.append(idx)
        return indices

    def _compute_lookahead_features(self):
        """Transform forward path points to body frame as [x_body, y_body, d/ds]"""
        heading = self._current_direction_angle
        cos_h = self.fast_cos(heading)
        sin_h = self.fast_sin(heading)

        def world_to_body(delta):
            x_body = delta[0] * cos_h + delta[1] * sin_h
            y_body = -delta[0] * sin_h + delta[1] * cos_h
            return x_body, y_body

        features = []
        for idx in self._get_lookahead_indices():
            delta = np.array(self.Pm[idx]) - self.current_position
            x_local, y_local = world_to_body(delta)
            kappa_rate = self.curvature_rate_profile[idx] if idx < len(self.curvature_rate_profile) else 0.0
            features.extend([x_local, y_local, kappa_rate])
        return np.array(features, dtype=float)

    def _get_path_direction(self, pt):
        """Use cached heading array"""
        segment_index = self._find_containing_segment(pt)
        if segment_index >= 0 and segment_index < len(self.cache['segment_directions']):
            return self.cache['segment_directions'][segment_index]
        return self._current_direction_angle
    
    def _project_point_to_segment(self, pt, p1, p2):
        """将点投影到线段上的最近点"""
        vec_seg = p2 - p1
        vec_pt = pt - p1
        t = np.dot(vec_pt, vec_seg) / (np.linalg.norm(vec_seg)**2 + 1e-6)
        return p1 + t * vec_seg

    def calculate_direction_deviation(self, pt):
        path_direction = self._get_path_direction(pt)
        current_direction = self._current_direction_angle
        
        tau = current_direction - path_direction
        tau = (tau + np.pi) % (2 * np.pi) - np.pi
        return tau
    
    def calculate_reward(self):
        contour_error = self.get_contour_error(self.current_position)
        progress = float(self.state[4])
        end_point = np.array(self.Pm[-1])
        end_distance = np.linalg.norm(self.current_position - end_point)

        reward, components = self.reward_calculator.calculate_reward(
            contour_error=contour_error,
            progress=progress,
            velocity=self.velocity,
            action=(self.angular_vel, self.velocity),
            kcm_intervention=self.kcm_intervention,
            end_distance=end_distance,
            lap_completed=self.closed and self.lap_completed,
        )

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
            # 刚过转弯：加强误差修正奖
            error_correction = -30.0 * (error_ratio) ** 2
            direction_bonus = 20.0 * np.exp(-12 * abs(self.state[2]))  # 方向对齐奖励
            return error_correction + direction_bonus
            
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
        polygons = []
        n = len(self.Pm)
        closed = np.allclose(self.Pm[0], self.Pm[-1], atol=1e-6)
        
        for i in range(n-1):
            if self.cache['Pl'][i] is not None and self.cache['Pl'][i+1] is not None and \
               self.cache['Pr'][i] is not None and self.cache['Pr'][i+1] is not None:
                polygon = [self.cache['Pl'][i], self.cache['Pl'][i+1], self.cache['Pr'][i+1], self.cache['Pr'][i]]
                polygons.append(polygon)
        
        if closed and self.cache['Pl'][0] is not None and self.cache['Pr'][0] is not None:
            polygon = [self.cache['Pl'][-1], self.cache['Pl'][0], self.cache['Pr'][0], self.cache['Pr'][-1]]
            polygons.append(polygon)
        
        return polygons
    
    def get_contour_error(self, pt):
        """使用缓存的多边形信息"""
        segment_idx = self._find_containing_segment(pt)
        if segment_idx >= 0:
            p1 = np.array(self.Pm[segment_idx])
            p2 = np.array(self.Pm[segment_idx + 1])
            return self._helen_formula_distance(pt, p1, p2)
        return self._traditional_shortest_distance(pt)
 
    def _find_containing_segment(self, pt):
        """Use R-tree for faster queries"""
        x, y = pt
        candidate_idxs = list(self.rtree_idx.intersection((x, y, x, y)))
        
        # 先检查当前线
        if self.current_segment_idx in candidate_idxs:
            polygon = self.cache['polygons'][self.current_segment_idx]
            if polygon and self.is_point_in_polygon((x,y), polygon):
                return self.current_segment_idx
        
        # 检查候选线
        for idx in candidate_idxs:
            polygon = self.cache['polygons'][idx]
            if polygon and self.is_point_in_polygon((x,y), polygon):
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

    def _helen_formula_distance(self, pt, A, B):
        """Compute perpendicular distance from point to line"""
        # 向量AB
        AB = np.array(B) - np.array(A)
        
        # 向量AP
        AP = np.array(pt) - np.array(A)
        
        # 计算叉积的绝对|AB × AP|
        cross_abs = abs(AB[0]*AP[1] - AB[1]*AP[0])
        
        # 计算AB的长
        length_AB = np.linalg.norm(AB)
        
        # 避免除零错误
        if length_AB < 1e-6:
            return np.linalg.norm(AP)
        
        # 点到直线的距= |AB × AP| / |AB|
        return cross_abs / length_AB

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
                
            projection = self._project_point_to_segment(pt, p1, p2)
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
                projection = self._project_point_to_segment(pt, p1, p2)
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
    
    def is_point_in_polygon(self, point, polygon):
        """Optimized ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # 预先计算边界
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        # 快速排
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        
        # 使用整数运算加
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


    def is_done(self):
        contour_error = self.get_contour_error(self.current_position)
        
        # 基本结束条件：误差超限或步数过多
        # 修正：使half_epsilon，因为边界在 ±epsilon/2 
        if contour_error > self.half_epsilon or self.current_step >= self.max_steps:
            return True
        
        # 成功到达终点的判断（这些情况不应该被视为失败
        end_point = np.array(self.Pm[-1])
        end_distance = np.linalg.norm(self.current_position - end_point)
        progress = self.state[4]
        
        # 标记成功到达终点的情
        # 修正：使half_epsilon 作为基准
        if progress > 0.95 and end_distance < self.half_epsilon * 0.6:
            # 设置成功标志，便于后续奖励计
            self.reached_target = True
            return True
        
        # 闭合路径：完成一圈也算成
        if self.closed and self.lap_completed:
            self.reached_target = True
            return True
        
        return False
   
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = torch.nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = torch.nn.Linear(hidden_dim//4, 1)
        
        # 层归一
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim//2)
        self.ln3 = nn.LayerNorm(hidden_dim//4)
        
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def _initialize_weights(self):
        # 改进的权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.ln1(x)
        x = self.dropout(x)
        
        x = F.elu(self.fc2(x))
        x = self.ln2(x)
        x = self.dropout(x)
        
        x = F.elu(self.fc3(x))
        x = self.ln3(x)
        
        return self.fc4(x)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # 共享基础
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 增加共享
        
        # 角度控制分支
        self.angle_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.angle_mu = nn.Linear(hidden_dim//4, 1)
        self.angle_std = nn.Linear(hidden_dim//4, 1)
        
        # 速度控制分支
        self.speed_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.speed_mu = nn.Linear(hidden_dim//4, 1)
        self.speed_std = nn.Linear(hidden_dim//4, 1)
        
        # 层归一
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim//2)
        
        # 初始
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier初始
        nn.init.xavier_uniform_(self.angle_mu.weight)
        nn.init.xavier_uniform_(self.speed_mu.weight)
        nn.init.constant_(self.angle_std.bias, 0.3)  # 初始标准
        nn.init.constant_(self.speed_std.bias, 0.3)

    def forward(self, x):
        x = F.elu(self.shared_fc1(x))  # ELU激活函
        x = self.ln1(x)
        x = F.elu(self.shared_fc2(x))
        x = self.ln2(x)
        
        # 角度分支
        angle_feat = F.selu(self.angle_fc(x))  # SELU激
        angle_mu = self.angle_mu(angle_feat)
        angle_std = F.softplus(self.angle_std(angle_feat)) + 1e-3
        
        # 速度分支
        speed_feat = F.selu(self.speed_fc(x))
        speed_mu = torch.sigmoid(self.speed_mu(speed_feat))  # 速度限制在[0,1]
        speed_std = F.softplus(self.speed_std(speed_feat)) + 1e-3
        
        mu = torch.cat([angle_mu, speed_mu], dim=1)
        std = torch.cat([angle_std, speed_std], dim=1)
        
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device ):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim ).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        # 添加学习率调度器（注意：verbose 参数在新PyTorch 中已被移除）
        self.actor_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='min', factor=0.5, patience=10)
        self.critic_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='min', factor=0.5, patience=5)
    
    def take_action(self, state):
        # 使用更高效的方式创建Tensor
        if not hasattr(self, 'state_tensor'):
            self.state_tensor = torch.empty((1, len(state)), dtype=torch.float, device=self.device)
        self.state_tensor[0] = torch.tensor(state, dtype=torch.float)
        mu, sigma = self.actor(self.state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.squeeze().cpu().numpy().tolist()
    
    def update(self, transition_dict):
        # 使用更高效的方式创建Tensor
        states = torch.tensor(np.array(transition_dict['states']), 
                             dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), 
                              dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], 
                              dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), 
                                  dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], 
                            dtype=torch.float).view(-1, 1).to(self.device)
        
        # 奖励标准- 提高critic学习稳定性，但限制标准化范围
        if rewards.std() > 1e-6:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            # 限制标准化后的奖励范围，防止过度放大
            rewards = torch.clamp(normalized_rewards, min=-5.0, max=5.0)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        # 优势函数标准- 限制范围防止过大的advantage导致loss爆炸
        if advantage.std() > 1e-6:
            normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = torch.clamp(normalized_advantage, min=-10.0, max=10.0)
        
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            # 添加数值稳定性处
            std = torch.clamp(std, min=1e-4, max=1.0)  # 防止std过小或过
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            
            # 改进熵计- 分别处理角度和速度
            angle_entropy = action_dists.entropy()[:, 0].mean() * 0.01
            speed_entropy = action_dists.entropy()[:, 1].mean() * 0.01
            total_entropy = angle_entropy + speed_entropy
            
            # 计算概率比时添加数值稳定
            log_ratio = log_probs.sum(dim=1, keepdim=True) - old_log_probs.sum(dim=1, keepdim=True)
            # 限制log_ratio的范围，防止exp爆炸
            log_ratio = torch.clamp(log_ratio, min=-5, max=5)
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # 计算actor loss并添加数值限
            policy_loss = -torch.min(surr1, surr2).mean()
            actor_loss = policy_loss - total_entropy
            
            # 限制actor loss的范围，防止梯度爆炸
            actor_loss = torch.clamp(actor_loss, min=-10.0, max=10.0)
            
            # 改进critic损失计算
            critic_loss = F.smooth_l1_loss(self.critic(states), td_target.detach())

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # 学习率调
        self.actor_scheduler.step(abs(actor_loss.item()))
        self.critic_scheduler.step(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists('logs'):
        os.makedirs('logs')

    step_log = open('logs/step_log.csv', 'w', newline='')
    episode_log = open('logs/episode_log.csv', 'w', newline='')
    paper_metrics_log = open('logs/paper_metrics.csv', 'w', newline='')
    step_writer = csv.writer(step_log)
    episode_writer = csv.writer(episode_log)
    paper_metrics_writer = csv.writer(paper_metrics_log)
    step_writer.writerow(['episode', 'step', 'action_theta', 'action_vel', 'reward'])
    episode_writer.writerow(['episode', 'total_reward', 'avg_actor_loss', 'avg_critic_loss', 'epsilon'])
    paper_metrics_writer.writerow(['episode', 'rmse_error', 'mean_jerk', 'roughness_proxy', 
                                     'mean_velocity', 'max_error', 'mean_kcm_intervention', 'steps', 'progress'])

    # ===== 使用固定路径和允=====
    env_config = {
        "epsilon": 0.15,
        "Pm": np.array([[0.0, 0.0], [1.0,0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
        "interpolation_period": 0.01,
        "MAX_VEL": 1000,
        "MAX_ACC": 5000,
        "MAX_JERK": 50000,
        "MAX_ANG_VEL": math.pi * 2,
        "MAX_ANG_ACC": math.pi * 10,
        "MAX_ANG_JERK": math.pi * 100,
        "device": device,
        "max_steps": 10000
    }
    
    # 创建环境
    env = Env(**env_config)
    
    agent_config = {
        "state_dim": env.observation_dim,
        "hidden_dim": 512,  # 减小网络规模，提高稳定
        "action_dim": env.action_space_dim,
        "actor_lr": 1e-5,   # 进一步降低学习率
        "critic_lr": 5e-5,  # 降低critic学习
        "lmbda": 0.95,     
        "epochs": 10,       # 减少训练轮次，防止过拟合
        "eps": 0.1,         # 减小clip范围，更保守的更
        "gamma": 0.99,     
        "device": device
    }
    
    agent = PPOContinuous(**agent_config)
    
    total_episodes = 1500
    reward_history = []
    loss_history = []
    smoothed_rewards = []
    smoothing_factor = 0.2
    
    # 初始化论文指标统计器
    paper_metrics = PaperMetrics()
    avg_actor_loss = 0
    avg_critic_loss = 0
    with tqdm(total=total_episodes, desc="Training Progress") as pbar:
        for episode in range(total_episodes):
            # 重置论文指标统计
            paper_metrics.reset()
            
            transition_dict = {
                'states': [], 
                'positions': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
                'steps': []
            }
            
            state = env.reset()
            episode_reward = 0
            done = False
            step_counter = 0
            final_progress = 0
            
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, info = env.step(action)
                
                transition_dict['states'].append(state)
                transition_dict['positions'].append(info['position'])
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['steps'].append(info['step'])
                
                # 更新论文指标
                paper_metrics.update(
                    contour_error=info['contour_error'],
                    jerk=info['jerk'],
                    velocity=env.velocity,
                    kcm_intervention=info['kcm_intervention']
                )
                
                step_writer.writerow([
                    episode,
                    step_counter,
                    action[0],
                    action[1],
                    reward
                ])
                
                state = next_state
                episode_reward += reward
                step_counter += 1
                final_progress = info.get('progress',0)
                
                # 不再每步更新监控器，只在episode结束时检
            
            # 更新策略
            if len(transition_dict['states']) > 10:
                avg_actor_loss, avg_critic_loss = agent.update(transition_dict)
                loss_history.append((avg_actor_loss, avg_critic_loss))
            else:
                avg_actor_loss, avg_critic_loss = 0.0, 0.0
            
            reward_history.append(episode_reward)
            
            if not smoothed_rewards:
                smoothed_rewards.append(episode_reward)
            else:
                new_smoothed = smoothing_factor * episode_reward + (1 - smoothing_factor) * smoothed_rewards[-1]
                smoothed_rewards.append(new_smoothed)
            
            # 计算论文指标
            metrics = paper_metrics.compute()
            
            # 记录回合信息
            episode_writer.writerow([
                episode,
                episode_reward,
                avg_actor_loss,
                avg_critic_loss,
                env_config["epsilon"]
            ])
            
            # 记录论文指标到CSV
            paper_metrics_writer.writerow([
                episode,
                metrics['rmse_error'],
                metrics['mean_jerk'],
                metrics['roughness_proxy'],
                metrics['mean_velocity'],
                metrics['max_error'],
                metrics['mean_kcm_intervention'],
                metrics['steps'],
                final_progress
            ])
            
            # 0个episode打印详细指标
            if (episode + 1) % 50 == 0:
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1} - Paper Metrics Summary:")
                print(f"{'='*80}")
                print(f"  RMSE Error:              {metrics['rmse_error']:.6f}")
                print(f"  Mean Jerk:               {metrics['mean_jerk']:.6f}")
                print(f"  Roughness Proxy:         {metrics['roughness_proxy']:.6f}")
                print(f"  Mean Velocity:           {metrics['mean_velocity']:.4f}")
                print(f"  Max Error:               {metrics['max_error']:.6f}")
                print(f"  Mean KCM Intervention:   {metrics['mean_kcm_intervention']:.6f}")
                print(f"  Steps:                   {metrics['steps']}")
                print(f"  Progress:                {final_progress:.4f}")
                print(f"  Total Reward:            {episode_reward:.2f}")
                print(f"{'='*80}\n")
            
        # 更新进度
            pbar.set_postfix({
                'Reward': f'{episode_reward:.1f}',
                'Smoothed': f'{smoothed_rewards[-1]:.1f}',
                'Actor Loss': f'{avg_actor_loss:.2f}',
                'Critic Loss': f'{avg_critic_loss:.2f}'
            })
            pbar.update(1)
    
    step_log.close()
    episode_log.close()
    paper_metrics_log.close()
    
    print("\n" + "="*80)
    print("训练完成！论文指标已保存logs/paper_metrics.csv")
    print("="*80 + "\n")
    
    # 保存模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'config': agent_config
    }, f'tracking_model_final.pth')
    
    # 可视化最终轨
    print(f"\n可视化最终轨(ε={env_config['epsilon']:.3f})")
    visualize_final_path(env)

if __name__ == "__main__":
    run_training()












