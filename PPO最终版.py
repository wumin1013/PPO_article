import os
# 解决 OpenMP 多副本冲突问题 - 必须在所有其他导入之前设置
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
import rl_utils
import csv
from math import degrees,acos,sqrt
from typing import List, Tuple, Optional
from numba import jit
import time
from rtree import index

# 添加Numba JIT编译优化运动学约束计算
@jit(nopython=True)
def apply_kinematic_constraints(prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
                               vel_action, ang_vel_action, dt,
                               MAX_VEL, MAX_ACC, MAX_JERK,
                               MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK):
    # 线速度约束 - 使用min/max替代np.clip
    constrained_vel = vel_action
    if constrained_vel < 0.0:
        constrained_vel = 0.0
    elif constrained_vel > MAX_VEL:
        constrained_vel = MAX_VEL
    
    # 线加速度约束
    raw_acc = (constrained_vel - prev_vel) / dt
    # 使用min/max替代np.clip
    constrained_acc = raw_acc
    if constrained_acc < -MAX_ACC:
        constrained_acc = -MAX_ACC
    elif constrained_acc > MAX_ACC:
        constrained_acc = MAX_ACC
    
    # 线加加速度约束
    raw_jerk = (constrained_acc - prev_acc) / dt
    # 使用min/max替代np.clip
    constrained_jerk = raw_jerk
    if constrained_jerk < -MAX_JERK:
        constrained_jerk = -MAX_JERK
    elif constrained_jerk > MAX_JERK:
        constrained_jerk = MAX_JERK
    
    # 反向修正加速度和速度
    final_acc = prev_acc + constrained_jerk * dt
    final_vel = prev_vel + final_acc * dt
    # 确保最终速度在允许范围内
    if final_vel < 0.0:
        final_vel = 0.0
    elif final_vel > MAX_VEL:
        final_vel = MAX_VEL
    
    # 角速度约束 - 使用min/max替代np.clip
    constrained_ang_vel = ang_vel_action
    if constrained_ang_vel < -MAX_ANG_VEL:
        constrained_ang_vel = -MAX_ANG_VEL
    elif constrained_ang_vel > MAX_ANG_VEL:
        constrained_ang_vel = MAX_ANG_VEL
    
    # 角加速度约束
    raw_ang_acc = (constrained_ang_vel - prev_ang_vel) / dt
    # 使用min/max替代np.clip
    constrained_ang_acc = raw_ang_acc
    if constrained_ang_acc < -MAX_ANG_ACC:
        constrained_ang_acc = -MAX_ANG_ACC
    elif constrained_ang_acc > MAX_ANG_ACC:
        constrained_ang_acc = MAX_ANG_ACC
    
    # 角加加速度约束
    raw_ang_jerk = (constrained_ang_acc - prev_ang_acc) / dt
    # 使用min/max替代np.clip
    constrained_ang_jerk = raw_ang_jerk
    if constrained_ang_jerk < -MAX_ANG_JERK:
        constrained_ang_jerk = -MAX_ANG_JERK
    elif constrained_ang_jerk > MAX_ANG_JERK:
        constrained_ang_jerk = MAX_ANG_JERK
    
    # 反向修正
    final_ang_acc = prev_ang_acc + constrained_ang_jerk * dt
    final_ang_vel = prev_ang_vel + final_ang_acc * dt
    # 确保角速度在允许范围内
    if final_ang_vel < -MAX_ANG_VEL:
        final_ang_vel = -MAX_ANG_VEL
    elif final_ang_vel > MAX_ANG_VEL:
        final_ang_vel = MAX_ANG_VEL
            
    return (final_vel, final_acc, constrained_jerk,
            final_ang_vel, final_ang_acc, constrained_ang_jerk)

def configure_chinese_font():
    try:
        system_fonts = ['Microsoft YaHei', 'SimHei', 'FangSong', 'STSong']
        linux_fonts = ['WenQuanYi Micro Hei', 'AR PL UMing CN']
        font_list = list(dict.fromkeys(system_fonts + linux_fonts))
        mpl.rcParams['font.sans-serif'] = font_list + mpl.rcParams['font.sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False
        test_font = mpl.font_manager.FontProperties(family=font_list) # type: ignore
        if not test_font.get_name():
            raise RuntimeError("字体配置失败")
    except Exception as e:
        print(f"字体配置警告: {str(e)}")
        print("将使用默认字体显示，中文可能显示为方框")

configure_chinese_font()

def visualize_final_path(env):
    plt.figure(figsize=(10, 6), dpi=100)
    
    def clean_path(path):
        return np.array([p for p in path if p is not None and not np.isnan(p).any()])

    pm = clean_path(env.Pm)
    plt.plot(pm[:,0], pm[:,1], 'k--', linewidth=2.5, label='Reference Path (Pm)')
    plt.scatter(pm[:,0], pm[:,1], c='black', marker='*', s=150, edgecolor='gold', zorder=3)

    pl = clean_path(env.Pl)
    pr = clean_path(env.Pr)
    plt.plot(pl[:,0], pl[:,1], 'g--', linewidth=1.8, label='Left Boundary (Pl)', alpha=0.7)
    plt.plot(pr[:,0], pr[:,1], 'b--', linewidth=1.8, label='Right Boundary (Pr)', alpha=0.7)

    pt = np.array(env.trajectory)
    plt.plot(pt[:,0], pt[:,1], 'r-', linewidth=1.5, label='Actual Trajectory (Pt)')
    
    plt.scatter(pt[::20,0], pt[::20,1], c='purple', s=40, alpha=0.6, 
                edgecolor='white', label='Sampled Points', zorder=2)

    plt.annotate(f'Start\n({pm[0,0]:.1f}, {pm[0,1]:.1f})', 
                 xy=pm[0], xytext=(-20, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='gray', alpha=0.6))
    
    if len(pm) > 1:
        plt.annotate(f'End\n({pm[-1,0]:.1f}, {pm[-1,1]:.1f})', 
                     xy=pm[-1], xytext=(-40, 20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='gray', alpha=0.6))

    param_text = (
        f'ε = {env.epsilon:.2f}\n'
        f'MAX_VEL = {env.MAX_VEL:.1f}\n'
        f'Δt = {env.interpolation_period:.2f}s\n'
        f'Steps = {len(pt)}'
    )
    plt.gcf().text(0.88, 0.85, param_text, 
                   fontfamily='monospace', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.axis('equal')
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('Final Trajectory Tracking Performance', fontsize=14, pad=20)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, color='gray', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

class Env:
    def __init__(self, device, epsilon, interpolation_period, 
                 MAX_VEL , MAX_ACC,MAX_JERK,MAX_ANG_VEL,MAX_ANG_ACC,MAX_ANG_JERK,Pm,max_steps):
        self.observation_dim = 12
        self.action_space_dim = 2
        self.epsilon = epsilon  # 总带宽（Pl到Pr的距离）
        self.half_epsilon = epsilon / 2  # 单侧偏移距离（Pm到Pl或Pr的距离）
        self.rmax = 3 * epsilon
        self.device = device
        self.max_steps = max_steps
        self.interpolation_period = interpolation_period
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
        
        # 闭合路径追踪相关
        self.accumulated_distance = 0.0  # 累计距离
        self.previous_segment_idx = -1   # 上一个线段索引
        self.reached_final_segment = False  # 是否到达最后一段
        self.lap_completed = False       # 是否完成一圈
        self.segment_transition_history = []  # 线段转换历史
        
        # 轨迹误差历史记录（用于过弯后修正）
        self.error_history = []         # 最近的误差历史
        self.max_error_history = 20     # 保存最近20步的误差
        self.last_corner_segment = -1   # 上一个过弯的线段

        
        # 运动学状态变量
        self.velocity = 0.0       # 当前速度
        self.acceleration = 0.0   # 当前加速度
        self.jerk = 0.0           # 当前捷度
        self.angular_vel = 0.0  # 当前角速度
        self.angular_acc = 0.0  # 当前角加速度
        self.angular_jerk = 0.0  # 当前角加加速度

        self.Pm = [np.array(p) for p in Pm]
        # 检查路径是否闭合
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
            'segment_info': {}  # 存储每个线段的缓存信息
        }
        # 预计算并缓存所有几何特征
        self._precompute_and_cache_geometric_features()
        # 创建三角函数查找表
        self._create_trig_lookup_table()
        
        self.last_progress = 0.0
        
        # 添加新属性用于跟踪线段信息
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
        
        # 现在缓存已经填充，可以安全地设置归一化参数
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
            'angular_jerk': self.MAX_ANG_JERK
        }
        
        self.reset()
    
    def _create_trig_lookup_table(self):
        """创建三角函数查找表加速计算"""
        # 创建360个点的查找表（0-359度）
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
        """快速余弦计算"""
        # 尝试直接查找特殊角度
        if rad in self.COS_TABLE:
            return self.COS_TABLE[rad]
        
        # 转换为角度并查找
        deg = round(math.degrees(rad)) % 360
        return self.COS_TABLE.get(deg, math.cos(rad))
    
    def fast_sin(self, rad):
        """快速正弦计算"""
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
        
        # 缓存线段长度和角度
        self.cache['segment_lengths'] = segment_lengths
        self.cache['angles'] = angles
        
        # 计算线段方向（新增）
        segment_directions = []
        n = len(self.Pm)
        if self.closed:
            # 闭合路径：包括从最后一个点到第一个点的线段
            for i in range(n-1):
                p1 = np.array(self.Pm[i])
                p2 = np.array(self.Pm[(i + 1) % n])
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                direction = math.atan2(dy, dx)
                segment_directions.append(direction)
        else:
            # 非闭合路径
            for i in range(n - 1):
                p1 = np.array(self.Pm[i])
                p2 = np.array(self.Pm[i + 1])
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                direction = math.atan2(dy, dx)
                segment_directions.append(direction)
        
        self.cache['segment_directions'] = segment_directions
            
        # 计算总路径长度
        self.cache['total_path_length'] = sum(segment_lengths) if segment_lengths else 0.0
        
        # 创建多边形并缓存（重用_create_polygons函数）
        self.cache['polygons'] = self._create_polygons()
        
        # 预缓存线段信息
        n = len(self.Pm)
        for idx in range(len(segment_lengths)):
            length = segment_lengths[idx]
            
            # 获取当前线段的方向
            current_direction = segment_directions[idx] if idx < len(segment_directions) else 0.0
            
            # 获取多边形（如果有）
            polygon = self.cache['polygons'][idx] if idx < len(self.cache['polygons']) else None
            
            # 计算下一拐角角度（重用_get_next_angle逻辑）
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
        self.current_position = np.array(self.Pm[0])  # 确保是数组形式
        self.trajectory = [self.current_position.copy()]
        self.trajectory_states = [ ]
        self._current_direction_angle, self._current_step_length = self.initialize_starting_conditions()

        # 初始线段索引和进度
        self.current_segment_idx = 0
        distance_to_next_turn = self.cache['segment_lengths'][0] if self.cache['segment_lengths'] else 0.0
        overall_progress = 0.0
        
        # 重置闭合路径追踪状态
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
        tau_initial = 0.0  # 初始时应该完全对齐
        
        # 计算下一个转折点夹角
        next_angle = self._get_next_angle(self.current_segment_idx)
        
        # 运动学状态变量
        self.velocity = 0.0       # 当前速度
        self.acceleration = 0.0   # 当前加速度
        self.jerk = 0.0           # 当前捷度
        self.angular_vel = 0.0  # 当前角速度
        self.angular_acc = 0.0  # 当前角加速度
        self.angular_jerk = 0.0  # 当前角加加速度
        
        self.state = np.array([
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
            self.angular_jerk
        ])
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
        # 角运动约束处理
        prev_ang_vel = self.angular_vel
        prev_ang_acc = self.angular_acc
        #解包动作
        theta_prime, length_prime = action
        # 使用Numba优化的约束计算
        (self.velocity, self.acceleration, self.jerk,
         self.angular_vel, self.angular_acc, self.angular_jerk) = apply_kinematic_constraints(
            prev_vel, prev_acc, prev_ang_vel, prev_ang_acc,
            length_prime, theta_prime, self.interpolation_period,
            self.MAX_VEL, self.MAX_ACC, self.MAX_JERK,
            self.MAX_ANG_VEL, self.MAX_ANG_ACC, self.MAX_ANG_JERK
        )
        # === 使用修正后的动作执行状态转移 ===
        # 构建最终安全动作
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
        }
        normalized_state = self.normalize_state(next_state)
        self.state = next_state
        return normalized_state, reward, done, info
    
    def normalize_state(self, state):
        normalized = np.zeros_like(state)
        for i, key in enumerate([
            'theta_prime', 'length_prime', 'tau_next', 
            'distance_to_next_turn', 'overall_progress', 'next_angle',
            'velocity', 'acceleration', 'jerk',
            'angular_vel', 'angular_acc', 'angular_jerk'
        ]):
            max_val = self.normalization_params[key]
            # 特殊处理距离和进度
            if key == 'distance_to_next_turn':
                scaled = np.log1p(state[i]) / np.log1p(max_val)#对数缩放
                normalized[i] = np.clip(scaled, 0, 1)
            elif key == 'overall_progress':
                normalized[i] = state[i] 
            else:
                normalized[i] = np.clip(state[i] / max_val, -1, 1)
        return normalized
    
    def apply_action(self, action):
        theta_prime, length_prime = action
        # 获取当前路径方向
        path_angle = self._get_path_direction(self.current_position)
        # 计算有效角度
        
        effective_angle = path_angle + theta_prime * self.interpolation_period
        self._current_direction_angle = effective_angle
        # 计算新位置
        displacement = length_prime * self.interpolation_period
        cos_angle = self.fast_cos(effective_angle)
        sin_angle = self.fast_sin(effective_angle)
        x_next = self.current_position[0] + displacement * cos_angle
        y_next = self.current_position[1] + displacement * sin_angle
        
        # 更新位置
        self.current_position = np.array([x_next, y_next])
        self.trajectory.append(self.current_position.copy())
        # 计算方向偏差
        tau_next = self.calculate_direction_deviation(self.current_position)
        # 更新线段信息
        self.current_segment_idx, distance_to_next_turn = self._update_segment_info()
        # 使用改进的进度计算方法
        overall_progress = self._calculate_closed_path_progress(self.current_position) if self.closed else self._calculate_path_progress(self.current_position)
        # 计算下一个转折点夹角
        next_angle = self._get_next_angle(self.current_segment_idx)

        # 构建状态
        return np.array([theta_prime, length_prime, tau_next, distance_to_next_turn, overall_progress, 
                         next_angle,self.velocity, self.acceleration, self.jerk,
                         self.angular_vel, self.angular_acc, self.angular_jerk])
    
    def _update_segment_info(self):
        """使用缓存的线段信息"""
        pt = self.current_position
        segment_idx = self._find_containing_segment(pt)
        
        if segment_idx >= 0 and segment_idx in self.cache['segment_info']:
            # 获取线段终点
            next_turn_point = np.array(self.Pm[segment_idx + 1])
            # 计算到下一拐点的欧氏距离
            distance_to_next_turn = np.linalg.norm(next_turn_point - pt)
            return segment_idx, distance_to_next_turn
        
        return self.current_segment_idx, float('inf')
    
    def _get_next_angle(self, segment_idx):
        """从缓存中获取下一拐角角度"""
        if segment_idx in self.cache['segment_info']:
            return self.cache['segment_info'][segment_idx]['next_angle']
        return 0.0
    
    def _calculate_closed_path_progress(self, pt):
        """专门用于闭合路径的进度计算，使用线段跳转检测终点到达"""
        if not self.closed:
            return self._calculate_path_progress(pt)
        
        # 如果已完成一圈，直接返回1.0
        if self.lap_completed:
            return 1.0
        
        # 找到当前最近的线段
        current_segment_idx = self._find_nearest_segment_for_progress(pt)
        if current_segment_idx < 0:
            return 0.0
        
        # 检测线段跳转，判断是否完成一圈
        self._detect_lap_completion_by_segment_jump(current_segment_idx)
        
        if self.lap_completed:
            return 1.0
        
        # 正常进度计算（排除重复的最后一个点）
        total_segments = len(self.Pm) - 1  # 闭合路径实际只有n-1个线段  
        current_dist = 0.0
        
        # 累加到当前线段之前的所有长度
        for i in range(current_segment_idx):
            if i < len(self.cache['segment_lengths']):
                current_dist += self.cache['segment_lengths'][i]
        
        # 计算在当前线段上的投影位置
        if current_segment_idx < total_segments:
            p1 = np.array(self.Pm[current_segment_idx])
            p2 = np.array(self.Pm[current_segment_idx + 1])
        else:
            return 0.95  # 接近完成但还未完成
        
        # 投影计算
        seg_vec = p2 - p1
        pt_vec = pt - p1
        seg_length = np.linalg.norm(seg_vec)
        
        if seg_length > 1e-6:
            t = np.clip(np.dot(pt_vec, seg_vec) / (seg_length ** 2), 0, 1)
            segment_progress = t * seg_length
            current_dist += segment_progress
        
        # 计算真正的总路径长度（不包括回到起点的虚拟线段）
        actual_total_length = sum(self.cache['segment_lengths'][:total_segments])
        raw_progress = current_dist / actual_total_length if actual_total_length > 0 else 0.0
        
        # 确保进度在合理范围内，但不会重置为0
        return np.clip(raw_progress, 0.0, 0.99)  # 最大0.99，完成时才返回1.0
    
    def _detect_lap_completion(self, current_segment):
        """简化的套圈检测，仅用于防止无限循环"""
        # 只在非常明确的情况下才认为完成了一圈
        if current_segment != self.previous_segment:
            # 检测到回到起点附近且已经有很高的进度
            if (current_segment == 0 and self.last_raw_progress > 0.9 
                and self.previous_segment == len(self.Pm) - 2):
                
                self.completed_laps += 1
                
                # 防止过度套圈 - 立即停止
                if self.completed_laps >= 1:  # 只允许完成一圈
                    self.last_raw_progress = 1.0  # 设置为完成状态
            
            self.previous_segment = current_segment
    
    def _detect_lap_completion_by_segment_jump(self, current_segment_idx):
        """通过线段跳转检测是否完成一圈"""
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
            
            # 模式1：从最后一个线段跳转到第一个线段
            if (self.previous_segment_idx == total_segments - 1 and 
                current_segment_idx == 0):
                
                # 额外验证：确保已经访问过大部分线段
                visited_segments = set()
                for transition in self.segment_transition_history:
                    if transition['from'] >= 0:
                        visited_segments.add(transition['from'])
                    if transition['to'] >= 0:
                        visited_segments.add(transition['to'])
                
                # 如果访问了超过70%的线段，认为完成了一圈
                coverage_ratio = len(visited_segments) / total_segments
                if coverage_ratio > 0.7:
                    self.lap_completed = True
                    print(f"检测到完成一圈：从线段{self.previous_segment_idx}跳转到线段{current_segment_idx}，覆盖率{coverage_ratio:.2f}")
            
            # 模式2：连续访问模式检测
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
            计算平行线
            offset_distance: 偏移距离（单侧）
            """
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            # 计算单位法向量
            normal_vector = (-dy, dx)  # 左侧法线
            unit_normal_vector = normalize(normal_vector)
            
            A, B = unit_normal_vector  # 法线向量即为直线方程中的系数
            
            def line_equation(distance, point=p1):
                # 计算常数项 C
                C = -(A * point[0] + B * point[1]) + distance
                return A, B, C
            
            return line_equation(offset_distance), line_equation(-offset_distance)

        def find_intersection(line1, line2):
            A1, B1, C1 = line1
            A2, B2, C2 = line2
            
            determinant = A1 * B2 - A2 * B1
            
            if determinant == 0:
                return None  # 平行线或重合线，没有唯一交点
                
            x = (B1 * C2 - B2 * C1) / determinant  # 注意这里的顺序
            y = (C1 * A2 - C2 * A1) / determinant   # 确保 y 的符号正确
            
            return (x, y)

        def calculate_intersections(pm, offset_distance):
            """
            计算三个点的偏移路径交点
            offset_distance: 单侧偏移距离
            """
            p1, p2, p3 = pm
            # 获取P1P2的平行线（左右各偏移offset_distance）
            l1, r1 = get_parallel_lines(p1, p2, offset_distance)
            # 获取P2P3的平行线（左右各偏移offset_distance）
            l2, r2 = get_parallel_lines(p2, p3, offset_distance)
            
            # 寻找交点，允许延长线段
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
        
        # 2. 将动作角度视为相对路径方向的偏移量
        effective_angle = path_angle + theta_prime_action * self.interpolation_period
        self._current_direction_angle = effective_angle
        # 3. 计算新位置（保留原有速度计算）
        displacement = length_prime_action * self.interpolation_period
        x_next = self.current_position[0] + displacement * np.cos(effective_angle)
        y_next = self.current_position[1] + displacement * np.sin(effective_angle)
        
        return np.array([x_next, y_next])

    def _compute_geometric_features(self):
        """
        计算并存储几何特征：线段长度、角度
        返回一个包含三个元素的元组：(线段长度列表, 角度列表)
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
            # 闭合路径：最后一个点到第一个点的线段长度为0（因为它们是同一个点）
            # 不需要添加额外的线段长度
            pass
        
        
        # 计算角度
        angles = []
        
        if n >= 3:  # 至少需要3个点才能计算角度
            for i in range(n):
                # 对于闭合路径，所有点都有角度
                # 对于非闭合路径，只有中间点有角度
                if self.closed or (not self.closed and i > 0 and i < n - 1):
                    # 获取三个连续点（考虑闭合路径）
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
                    
                    # 避免除以零
                    if len1 < 1e-6 or len2 < 1e-6:
                        angle = 0.0
                    else:
                        # 计算角度（单位：弧度），确保逆时针为正，顺时针为负
                        dot_product = np.dot(vec1, vec2) / (len1 * len2)
                        cross_product = np.cross(vec1, vec2) / (len1 * len2)
                        
                        # 使用atan2确保正确的角度方向
                        angle = math.atan2(cross_product, dot_product)
                    
                    angles.append(angle)
        
        # 存储几何特征
        self.segment_lengths = segment_lengths
        self.angles = angles
        
        return segment_lengths, angles

    def _get_path_direction(self, pt):
        """使用缓存的方向数据"""
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
        """改进的奖励函数，标准化奖励范围并改善价值函数学习"""
        
        # 基础跟踪奖励 - 标准化到[-10, 10]范围
        distance = self.get_contour_error(self.current_position)
        # 修正：使用 half_epsilon 作为归一化基准，因为有效边界在 ±epsilon/2
        distance_ratio = np.clip(distance / self.half_epsilon, 0, 2)  # 限制最大比例
        
        # 使用更平滑的奖励函数
        tracking_reward = 10.0 * np.exp(-3 * distance_ratio) - 5.0  # 范围：[-5, 5]
        
        # 进度奖励 - 标准化
        progress = self.state[4]
        progress_diff = max(0, progress - self.last_progress)  # 只奖励正向进度
        progress_reward = 20.0 * progress_diff  # 根据进度差给奖励
        
        # 方向对齐奖励 - 标准化
        tau = abs(self.state[2])
        direction_reward = 5.0 * np.exp(-2 * tau) - 2.5  # 范围：[-2.5, 2.5]
        
        # 速度奖励 - 鼓励合理速度
        velocity_ratio = self.velocity / self.MAX_VEL
        velocity_reward = 2.0 * (1 - abs(velocity_ratio - 0.7))  # 鼓励70%最大速度
        
        # 平滑性奖励 - 惩罚过大的加速度变化
        jerk_penalty = -2.0 * np.clip(abs(self.jerk) / self.MAX_JERK, 0, 1)
        ang_jerk_penalty = -2.0 * np.clip(abs(self.angular_jerk) / self.MAX_ANG_JERK, 0, 1)
        smoothness_reward = jerk_penalty + ang_jerk_penalty
        
        # 终点到达奖励 - 大幅提升
        completion_reward = 0
        end_point = np.array(self.Pm[-1])
        end_distance = np.linalg.norm(self.current_position - end_point)
        # 修正：使用 half_epsilon 作为基准
        end_distance_ratio = end_distance / self.half_epsilon
        if progress > 0.8:
            # 高进度时的终点接近奖励
            proximity_bonus = 50.0 * (progress - 0.8) * np.exp(-5 * end_distance_ratio)
            completion_reward += proximity_bonus
            
            # 非常接近终点时的巨大奖励
            if end_distance < self.half_epsilon * 0.6:
                completion_reward += 100.0 * np.exp(-10 * end_distance_ratio)
                
                # 完全到达终点的最高奖励
                if end_distance < self.half_epsilon * 0.2:
                    completion_reward += 200.0
        
        # 闭合路径完成奖励
        if self.closed and self.lap_completed:
            completion_reward += 150.0
        
        # 存活奖励 - 防止早期终止
        survival_reward = 0.1  # 每步小额奖励
        
        # 组合总奖励，确保合理的数值范围
        total_reward = (tracking_reward + progress_reward + direction_reward + 
                       velocity_reward + smoothness_reward + completion_reward + survival_reward)
        
        # 限制总奖励范围，防止梯度爆炸
        total_reward = np.clip(total_reward, -20, 100)
        
        self.last_progress = progress
        return total_reward

    def _calculate_segment_adaptive_reward(self):
        """根据路径段类型计算自适应奖励，加强直线段跟踪精度"""
        current_segment = self.current_segment_idx
        distance_error = self.get_contour_error(self.current_position)
        # 修正：使用 half_epsilon 作为归一化基准
        error_ratio = distance_error / self.half_epsilon
        
        # 获取当前和下一个拐角的角度
        current_angle = self._get_current_segment_angle()
        next_angle = abs(self.state[5])  # 下一拐角角度
        distance_to_turn = self.state[3]  # 到下一拐点距离
        
        # 判断路径段类型
        turn_threshold = 0.8
        is_near_turn = distance_to_turn < turn_threshold
        is_sharp_turn = next_angle > np.pi / 8  # 22.5度以上为急转弯
        is_moderate_turn = np.pi / 16 < next_angle <= np.pi / 8  # 11.25-22.5度为中等转弯
        is_straight = next_angle <= np.pi / 16  # 11.25度以下为直线段
        
        # 判断是否刚过转弯
        just_passed_turn = False
        if current_segment > 0:
            prev_angle = abs(self.cache['angles'][current_segment - 1]) if current_segment - 1 < len(self.cache['angles']) else 0
            just_passed_turn = prev_angle > np.pi / 8 and distance_to_turn > 2.0
        
        if is_near_turn and is_sharp_turn:
            # 接近急转弯处：优先控制速度，适当放宽精度要求
            speed_penalty = -12.0 * (self.velocity / self.MAX_VEL) ** 2
            precision_bonus = 15.0 * np.exp(-8 * error_ratio)  # 相对宽松的精度要求
            return speed_penalty + precision_bonus
            
        elif is_near_turn and is_moderate_turn:
            # 接近中等转弯：平衡速度和精度
            speed_penalty = -6.0 * (self.velocity / self.MAX_VEL) ** 2
            precision_bonus = 20.0 * np.exp(-12 * error_ratio)
            return speed_penalty + precision_bonus
            
        elif just_passed_turn:
            # 刚过转弯：加强误差修正奖励
            error_correction = -30.0 * (error_ratio) ** 2
            direction_bonus = 20.0 * np.exp(-12 * abs(self.state[2]))  # 方向对齐奖励
            return error_correction + direction_bonus
            
        elif is_straight:
            # 直线段：极度重视跟踪精度，要求紧贴PM
            # 大幅加强精度惩罚
            if error_ratio > 0.3:  # 误差超过30%容差时严厉惩罚
                precision_penalty = -80.0 * (error_ratio) ** 3
            else:
                precision_penalty = -40.0 * (error_ratio) ** 2
            
            # 精度奖励：越接近PM越好
            precision_bonus = 50.0 * np.exp(-20 * error_ratio)  # 极高的精度要求
            
            # 速度奖励：在保证精度的前提下鼓励合理速度
            if error_ratio < 0.2:  # 误差小于20%时才给速度奖励
                speed_bonus = 12.0 * (self.velocity / self.MAX_VEL) * np.exp(-10 * error_ratio)
            else:
                speed_bonus = 0.0  # 误差大时不给速度奖励
            
            # 方向对齐奖励：直线段要求严格的方向对齐
            direction_bonus = 25.0 * np.exp(-15 * abs(self.state[2]))
            
            return precision_penalty + precision_bonus + speed_bonus + direction_bonus
            
        else:
            # 其他情况：平衡处理
            precision_penalty = -20.0 * (error_ratio) ** 2
            precision_bonus = 15.0 * np.exp(-10 * error_ratio)
            speed_bonus = 8.0 * (self.velocity / self.MAX_VEL)
            return precision_penalty + precision_bonus + speed_bonus

    def _calculate_velocity_reward(self):
        """速度奖励：直线段高速，拐弯处低速"""
        distance_to_turn = self.state[3]
        next_angle = abs(self.state[5])
        
        # 计算期望速度
        if distance_to_turn < 0.3 and next_angle > np.pi / 8:  # 接近急转弯
            expected_speed_ratio = 0.3  # 期望30%最大速度
        elif distance_to_turn < 0.6 and next_angle > np.pi / 12:  # 接近中等转弯
            expected_speed_ratio = 0.6  # 期望60%最大速度
        else:  # 直线段
            expected_speed_ratio = 0.9  # 期望90%最大速度
        
        current_speed_ratio = self.velocity / self.MAX_VEL
        speed_error = abs(current_speed_ratio - expected_speed_ratio)
        
        return -10.0 * speed_error

    def _calculate_smoothness_reward(self):
        """平滑性奖励"""
        jerk_penalty = -2.0 * (abs(self.jerk) / self.MAX_JERK) ** 2
        ang_jerk_penalty = -2.0 * (abs(self.angular_jerk) / self.MAX_ANG_JERK) ** 2
        return jerk_penalty + ang_jerk_penalty

    def _get_current_segment_angle(self):
        """获取当前线段的角度信息"""
        if self.current_segment_idx < len(self.cache['angles']):
            return abs(self.cache['angles'][self.current_segment_idx])
        return 0.0

    def _calculate_corner_correction_reward(self):
        """计算转弯后的轨迹修正奖励，使用误差历史判断修正趋势"""
        distance_from_path = self.get_contour_error(self.current_position)
        current_segment = self.current_segment_idx
        
        # 判断是否刚过转弯点
        if current_segment > 0:
            prev_angle = abs(self.cache['angles'][current_segment - 1]) if current_segment - 1 < len(self.cache['angles']) else 0
            
            # 如果上一个转弯角度较大，说明刚经过急转弯
            if prev_angle > np.pi / 8:  # 降低阈值到22.5度
                # 在转弯后的一段距离内加强修正奖励
                distance_to_turn = self.state[3]
                if distance_to_turn > 0.5 and distance_to_turn < 2.0:  # 转弯后适当距离内
                    
                    # 基础修正奖励：距离路径越近越好
                    # 修正：使用 half_epsilon 作为归一化基准
                    base_correction = 20.0 * np.exp(-8 * distance_from_path / self.half_epsilon)
                    
                    # 趋势奖励：如果误差在减少，给予额外奖励
                    trend_bonus = 0.0
                    if len(self.error_history) >= 3:
                        recent_errors = self.error_history[-3:]
                        if recent_errors[-1] < recent_errors[0]:  # 误差在减少
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
            
            # 前期步骤越小，要求精度越高
            step_factor = (50 - self.current_step) / 50.0  # 步数越少，因子越大
            
            # 距离路径越近，奖励越高
            precision_reward = 30.0 * step_factor * np.exp(-10 * distance_ratio)
            
            # 额外的方向对齐奖励
            tau = abs(self.state[2])
            direction_reward = 20.0 * step_factor * np.exp(-15 * tau)
            
            return precision_reward + direction_reward
        
        return 0.0
        
    def _calculate_path_progress(self, pt):
        """修复的路径进度计算"""
        n = len(self.Pm)
        total_length = self.cache['total_path_length'] or 1.0
        
        # 找到当前所在线段
        segment_index = self._find_containing_segment(pt)
        if segment_index >= 0:
            current_dist = 0.0
            
            # 累加之前所有线段长度
            for i in range(segment_index):
                current_dist += self.cache['segment_lengths'][i]
            
            # 计算在当前线段上的进度
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
        """使用 R-tree 加速查询"""
        x, y = pt
        candidate_idxs = list(self.rtree_idx.intersection((x, y, x, y)))
        
        # 先检查当前线段
        if self.current_segment_idx in candidate_idxs:
            polygon = self.cache['polygons'][self.current_segment_idx]
            if polygon and self.is_point_in_polygon((x,y), polygon):
                return self.current_segment_idx
        
        # 检查候选线段
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
            
            # 计算点到线段的投影
            seg_vec = p2 - p1
            pt_vec = pt - p1
            seg_length_sq = np.dot(seg_vec, seg_vec)
            
            # 避免除以零
            if seg_length_sq < 1e-6:
                continue
                
            # 计算投影参数 t
            t = np.dot(pt_vec, seg_vec) / seg_length_sq
            t = np.clip(t, 0, 1)
            projection = p1 + t * seg_vec
            
            # 计算距离
            dist = np.linalg.norm(pt - projection)
            
            # 更新最近线段
            if dist < min_dist:
                min_dist = dist
                nearest_segment_index = i
        
        return nearest_segment_index
    
    def _find_nearest_segment_for_progress(self, pt):
        """专门为进度计算找最近线段，避免套圈问题"""
        min_dist = float('inf')
        nearest_segment_index = -1
        n = len(self.Pm)
        
        # 只考虑实际的线段数量（闭合路径排除最后一个重复点）
        actual_segments = n - 1 if self.closed else n - 1
        
        for i in range(actual_segments):
            p1 = np.array(self.Pm[i])
            p2 = np.array(self.Pm[i + 1])
            
            # 计算点到线段的距离
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
            
            # 更新最近线段
            if dist < min_dist:
                min_dist = dist
                nearest_segment_index = i
        
        return nearest_segment_index

    def _helen_formula_distance(self, pt, A, B):
        """计算点到直线的距离（始终使用垂直距离）"""
        # 向量AB
        AB = np.array(B) - np.array(A)
        
        # 向量AP
        AP = np.array(pt) - np.array(A)
        
        # 计算叉积的绝对值 |AB × AP|
        cross_abs = abs(AB[0]*AP[1] - AB[1]*AP[0])
        
        # 计算AB的长度
        length_AB = np.linalg.norm(AB)
        
        # 避免除零错误
        if length_AB < 1e-6:
            return np.linalg.norm(AP)
        
        # 点到直线的距离 = |AB × AP| / |AB|
        return cross_abs / length_AB

    def _traditional_path_progress(self, pt, total_length, closed):
        """传统方法计算路径进度(当pnpoly方法失败时使用)"""
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
            
            # 修正：使用 half_epsilon 作为容差范围
            if dist_to_segment < self.half_epsilon:
                current_dist += np.linalg.norm(projection - p1)
                found = True
                break  # 找到最近的线段后退出循环
            # 不再累加不相关的线段长度
        
        # 检查闭合路径的最后一个线段
        if not found and closed:
            p1, p2 = self.Pm[-1], self.Pm[0]
            segment_length = np.linalg.norm(p2-p1)
            if segment_length > 1e-6:
                projection = self._project_point_to_segment(pt, p1, p2)
                dist_to_segment = np.linalg.norm(pt - projection)
                
                # 修正：使用 half_epsilon 作为容差范围
                if dist_to_segment < self.half_epsilon:
                    current_dist += np.linalg.norm(projection - p1)
                    found = True
        
        # 关键修改：未找到时返回0进度
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
        """优化的光线投射算法"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # 预先计算边界
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        # 快速排除
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        
        # 使用整数运算加速
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
        # 修正：使用 half_epsilon，因为边界在 ±epsilon/2 处
        if contour_error > self.half_epsilon or self.current_step >= self.max_steps:
            return True
        
        # 成功到达终点的判断（这些情况不应该被视为失败）
        end_point = np.array(self.Pm[-1])
        end_distance = np.linalg.norm(self.current_position - end_point)
        progress = self.state[4]
        
        # 标记成功到达终点的情况
        # 修正：使用 half_epsilon 作为基准
        if progress > 0.95 and end_distance < self.half_epsilon * 0.6:
            # 设置成功标志，便于后续奖励计算
            self.reached_target = True
            return True
        
        # 闭合路径：完成一圈也算成功
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
        
        # 层归一化
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
        # 共享基础层
        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 增加共享层
        
        # 角度控制分支
        self.angle_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.angle_mu = nn.Linear(hidden_dim//4, 1)
        self.angle_std = nn.Linear(hidden_dim//4, 1)
        
        # 速度控制分支
        self.speed_fc = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.speed_mu = nn.Linear(hidden_dim//4, 1)
        self.speed_std = nn.Linear(hidden_dim//4, 1)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim//2)
        
        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        # Xavier初始化
        nn.init.xavier_uniform_(self.angle_mu.weight)
        nn.init.xavier_uniform_(self.speed_mu.weight)
        nn.init.constant_(self.angle_std.bias, 0.3)  # 初始标准差
        nn.init.constant_(self.speed_std.bias, 0.3)

    def forward(self, x):
        x = F.elu(self.shared_fc1(x))  # ELU激活函数
        x = self.ln1(x)
        x = F.elu(self.shared_fc2(x))
        x = self.ln2(x)
        
        # 角度分支
        angle_feat = F.selu(self.angle_fc(x))  # SELU激活
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
        # 添加学习率调度器（注意：verbose 参数在新版 PyTorch 中已被移除）
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
        
        # 奖励标准化 - 提高critic学习稳定性，但限制标准化范围
        if rewards.std() > 1e-6:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            # 限制标准化后的奖励范围，防止过度放大
            rewards = torch.clamp(normalized_rewards, min=-5.0, max=5.0)
        
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        
        # 优势函数标准化 - 限制范围防止过大的advantage导致loss爆炸
        if advantage.std() > 1e-6:
            normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage = torch.clamp(normalized_advantage, min=-10.0, max=10.0)
        
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            # 添加数值稳定性处理
            std = torch.clamp(std, min=1e-4, max=1.0)  # 防止std过小或过大
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            
            # 改进熵计算 - 分别处理角度和速度熵
            angle_entropy = action_dists.entropy()[:, 0].mean() * 0.01
            speed_entropy = action_dists.entropy()[:, 1].mean() * 0.01
            total_entropy = angle_entropy + speed_entropy
            
            # 计算概率比时添加数值稳定性
            log_ratio = log_probs.sum(dim=1, keepdim=True) - old_log_probs.sum(dim=1, keepdim=True)
            # 限制log_ratio的范围，防止exp爆炸
            log_ratio = torch.clamp(log_ratio, min=-5, max=5)
            ratio = torch.exp(log_ratio)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            # 计算actor loss并添加数值限制
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
            
        # 学习率调度
        self.actor_scheduler.step(abs(actor_loss.item()))
        self.critic_scheduler.step(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()

# ===== 实时监控类 - 显示训练过程，只保存最佳结果 =====
class TrainingMonitor:
    def __init__(self, env, save_dir="best_results"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 跟踪最佳结果
        self.best_reward = float('-inf')
        self.best_trajectory = None
        self.best_episode = 0
        self.best_progress = 0.0
        self.best_actor_loss = 0.0
        self.best_critic_loss = 0.0
        
        # 实时显示相关
        self.episode_rewards = []
        self.episode_progress = []
        self.actor_losses = []
        self.critic_losses = []
        
        # 创建图表
        plt.ion()  # 开启交互模式
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Training Monitor', fontsize=16)
        
        # 统计信息
        self.episode_count = 0
        
    def clean_path(self, path):
        """清理路径数据，移除None和NaN值"""
        return np.array([p for p in path if p is not None and not np.isnan(p).any()])

    def update(self, episode, total_reward, progress, actor_loss, critic_loss, trajectory):
        """更新监控数据并检查最佳结果"""
        self.episode_count = episode
        
        # 更新实时数据
        self.episode_rewards.append(total_reward)
        self.episode_progress.append(progress)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        # 每10个episode更新一次显示
        if episode % 10 == 0:
            self.update_display(trajectory)
        
        # 评估当前结果的质量（综合考虑奖励、进度和终点距离）
        quality_score = self._calculate_quality_score(total_reward, progress, trajectory)
        best_quality = self._calculate_quality_score(self.best_reward, self.best_progress, self.best_trajectory)
        
        if quality_score > best_quality:
            self.best_reward = total_reward
            self.best_trajectory = trajectory.copy() if trajectory else None
            self.best_episode = episode
            self.best_progress = progress
            self.best_actor_loss = actor_loss
            self.best_critic_loss = critic_loss
            
            # 保存最佳轨迹图
            self.save_best_trajectory()
            
            # 计算终点距离并显示
            if trajectory and len(trajectory) > 0:
                final_position = np.array(trajectory[-1])
                target_point = np.array(self.env.Pm[-1])  # 统一使用真正的终点
                target_name = "终点"
                
                end_distance = np.linalg.norm(final_position - target_point)
                end_distance_ratio = end_distance / self.env.epsilon
                
                print(f"\n🎯 发现更好结果！Episode {episode}: Reward={total_reward:.1f}, Progress={progress:.3f}")
                print(f"   距离{target_name}: {end_distance:.4f} (容差比: {end_distance_ratio:.2f})")
                
                # 检查是否接近完成
                if end_distance < self.env.epsilon * 0.3:
                    print(f"   🎉 非常接近{target_name}！距离小于30%容差")
                elif end_distance < self.env.epsilon * 0.5:
                    print(f"   ⭐ 较接近{target_name}！距离小于50%容差")
            else:
                print(f"\n🎯 发现更好结果！Episode {episode}: Reward={total_reward:.1f}, Progress={progress:.3f}")
    
    def update_display(self, current_trajectory):
        """更新实时显示"""
        try:
            # 清除所有子图
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # 子图1：当前轨迹
            self.plot_trajectory(self.ax1, current_trajectory, "Current Trajectory")
            
            # 子图2：奖励曲线
            if self.episode_rewards:
                self.ax2.plot(self.episode_rewards, 'b-', linewidth=1, alpha=0.7, label='Episode Reward')
                if len(self.episode_rewards) > 10:
                    # 平滑曲线
                    window = min(50, len(self.episode_rewards))
                    smooth_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    self.ax2.plot(range(window-1, len(self.episode_rewards)), smooth_rewards, 'r-', linewidth=2, label='Smoothed')
                self.ax2.set_title('Training Rewards')
                self.ax2.set_xlabel('Episode')
                self.ax2.set_ylabel('Reward')
                self.ax2.legend()
                self.ax2.grid(True, alpha=0.3)
            
            # 子图3：进度曲线
            if self.episode_progress:
                self.ax3.plot(self.episode_progress, 'g-', linewidth=2)
                self.ax3.set_title('Path Progress')
                self.ax3.set_xlabel('Episode')
                self.ax3.set_ylabel('Progress')
                self.ax3.set_ylim(0, 1.1)
                self.ax3.grid(True, alpha=0.3)
            
            # 子图4：损失曲线
            if self.actor_losses and self.critic_losses:
                self.ax4.plot(self.actor_losses, 'r-', linewidth=1, label='Actor Loss', alpha=0.8)
                self.ax4.plot(self.critic_losses, 'b-', linewidth=1, label='Critic Loss', alpha=0.8)
                self.ax4.set_title('Training Losses')
                self.ax4.set_xlabel('Episode')
                self.ax4.set_ylabel('Loss')
                self.ax4.legend()
                self.ax4.grid(True, alpha=0.3)
            
            # 更新标题显示最佳结果
            self.fig.suptitle(f'Training Monitor - Episode {self.episode_count} | Best: Ep.{self.best_episode}, Reward={self.best_reward:.1f}, Progress={self.best_progress:.3f}', 
                             fontsize=14)
            
            plt.tight_layout()
            plt.pause(0.01)  # 短暂暂停以更新显示
            
        except Exception as e:
            print(f"Display update error: {e}")
    
    def plot_trajectory(self, ax, trajectory, title):
        """绘制轨迹图"""
        # 清理路径数据
        pm = self.clean_path(self.env.Pm)
        pl = self.clean_path(self.env.cache['Pl'])
        pr = self.clean_path(self.env.cache['Pr'])
        
        # 绘制参考路径和边界
        ax.plot(pm[:,0], pm[:,1], 'k--', linewidth=2, label='Reference Path', alpha=0.8)
        ax.scatter(pm[:,0], pm[:,1], c='black', marker='*', s=100, edgecolor='gold', zorder=3)
        ax.plot(pl[:,0], pl[:,1], 'g--', linewidth=1.5, label='Left Boundary', alpha=0.6)
        ax.plot(pr[:,0], pr[:,1], 'b--', linewidth=1.5, label='Right Boundary', alpha=0.6)
        
        # 绘制轨迹
        if trajectory and len(trajectory) > 0:
            pt = np.array(trajectory)
            ax.plot(pt[:,0], pt[:,1], 'r-', linewidth=1.5, label='Actual Trajectory', alpha=0.9)
            
            # 标记起点和终点
            if len(pt) > 0:
                ax.scatter(pt[0,0], pt[0,1], c='green', s=80, marker='o', label='Start', zorder=4)
                ax.scatter(pt[-1,0], pt[-1,1], c='red', s=80, marker='x', label='End', zorder=4)
        
        ax.axis('equal')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _calculate_quality_score(self, reward, progress, trajectory):
        """计算轨迹质量分数，考虑奖励、进度和终点距离"""
        if not trajectory or len(trajectory) == 0:
            return float('-inf')
        
        # 基础分数：奖励 + 进度权重
        base_score = reward + 1000 * progress
        
        # 终点距离奖励 - 统一使用真正的终点
        final_position = np.array(trajectory[-1])
        target_point = np.array(self.env.Pm[-1])  # 统一使用最后一个点作为终点
        
        end_distance = np.linalg.norm(final_position - target_point)
        end_distance_ratio = end_distance / self.env.epsilon
        
        # 距离奖励：越接近终点奖励越高
        distance_bonus = 500.0 * np.exp(-10 * end_distance_ratio)
        
        # 如果非常接近终点，给予巨大奖励
        if end_distance < self.env.epsilon * 0.2:
            distance_bonus += 1000.0 * np.exp(-50 * end_distance_ratio)
        
        return base_score + distance_bonus
    
    def save_best_trajectory(self):
        """保存最佳轨迹图"""
        plt.figure(figsize=(12, 8), dpi=150)
        
        # 清理路径数据
        pm = self.clean_path(self.env.Pm)
        pl = self.clean_path(self.env.cache['Pl'])
        pr = self.clean_path(self.env.cache['Pr'])
        
        # 绘制参考路径和边界
        plt.plot(pm[:,0], pm[:,1], 'k--', linewidth=3, label='Reference Path (Pm)', alpha=0.8)
        plt.scatter(pm[:,0], pm[:,1], c='black', marker='*', s=200, edgecolor='gold', zorder=3)
        plt.plot(pl[:,0], pl[:,1], 'g--', linewidth=2, label='Left Boundary (Pl)', alpha=0.7)
        plt.plot(pr[:,0], pr[:,1], 'b--', linewidth=2, label='Right Boundary (Pr)', alpha=0.7)
        
        # 绘制最佳轨迹
        if self.best_trajectory:
            pt = np.array(self.best_trajectory)
            plt.plot(pt[:,0], pt[:,1], 'r-', linewidth=2, label='Best Trajectory', alpha=0.9)
            
            # 标记起点和终点
            if len(pt) > 0:
                plt.scatter(pt[0,0], pt[0,1], c='green', s=150, marker='o', label='Start', zorder=4)
                plt.scatter(pt[-1,0], pt[-1,1], c='red', s=150, marker='x', label='End', zorder=4)
            
            # 计算误差统计
            errors = []
            for point in pt:
                error = self.env.get_contour_error(point)
                errors.append(error)
            
            max_error = max(errors) if errors else 0
            avg_error = sum(errors) / len(errors) if errors else 0
            
            # 计算终点距离
            final_position = np.array(pt[-1])
            target_point = np.array(self.env.Pm[-1])  # 统一使用最后一个点作为终点
            target_name = "End Point"
            
            end_distance = np.linalg.norm(final_position - target_point)
            end_distance_ratio = end_distance / self.env.epsilon
            
            # 添加详细信息
            info_text = (
                f'Episode: {self.best_episode}\n'
                f'Total Reward: {self.best_reward:.1f}\n'
                f'Progress: {self.best_progress:.3f}\n'
                f'Max Error: {max_error:.4f}\n'
                f'Avg Error: {avg_error:.4f}\n'
                f'Distance to {target_name}: {end_distance:.4f}\n'
                f'Distance Ratio: {end_distance_ratio:.2f}\n'
                f'ε = {self.env.epsilon:.3f}\n'
                f'Steps: {len(pt)}'
            )
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                    fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    fontsize=10)
        
        plt.axis('equal')
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(f'Best Trajectory - Episode {self.best_episode}', fontsize=14, pad=20)
        plt.legend(loc='lower right', framealpha=0.9)
        plt.grid(True, color='gray', linestyle=':', alpha=0.4)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.save_dir, f'best_trajectory_ep_{self.best_episode}.png'), 
                   bbox_inches='tight', facecolor='white')
        plt.close()  # 关闭图形释放内存
    
    def get_summary(self):
        """获取训练总结"""
        return {
            'best_episode': self.best_episode,
            'best_reward': self.best_reward,
            'best_progress': self.best_progress,
            'best_actor_loss': self.best_actor_loss,
            'best_critic_loss': self.best_critic_loss,
            'total_episodes': self.episode_count
        }
        
def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists('logs'):
        os.makedirs('logs')

    step_log = open('logs/step_log.csv', 'w', newline='')
    episode_log = open('logs/episode_log.csv', 'w', newline='')
    step_writer = csv.writer(step_log)
    episode_writer = csv.writer(episode_log)
    step_writer.writerow(['episode', 'step', 'action_theta', 'action_vel', 'reward'])
    episode_writer.writerow(['episode', 'total_reward', 'avg_actor_loss', 'avg_critic_loss', 'epsilon'])

    # ===== 使用固定路径和允差 =====
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
        "hidden_dim": 512,  # 减小网络规模，提高稳定性
        "action_dim": env.action_space_dim,
        "actor_lr": 1e-5,   # 进一步降低学习率
        "critic_lr": 5e-5,  # 降低critic学习率
        "lmbda": 0.95,     
        "epochs": 10,       # 减少训练轮次，防止过拟合
        "eps": 0.1,         # 减小clip范围，更保守的更新
        "gamma": 0.99,     
        "device": device
    }
    
    agent = PPOContinuous(**agent_config)
    
    total_episodes = 1500
    reward_history = []
    loss_history = []
    smoothed_rewards = []
    smoothing_factor = 0.2
    
    # 在创建环境后初始化监控器
    monitor = TrainingMonitor(env)
    avg_actor_loss = 0
    avg_critic_loss = 0
    with tqdm(total=total_episodes, desc="Training Progress") as pbar:
        for episode in range(total_episodes):
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
                
                # 不再每步更新监控器，只在episode结束时检查
            
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
            
            # 记录回合信息
            episode_writer.writerow([
                episode,
                episode_reward,
                avg_actor_loss,
                avg_critic_loss,
                env_config["epsilon"]
            ])
            
            # 在episode结束时更新监控数据
            monitor.update(
                episode=episode,
                total_reward=episode_reward,
                progress=final_progress,
                actor_loss=avg_actor_loss,
                critic_loss=avg_critic_loss,
                trajectory=env.trajectory.copy()
            )
            
            # 更新进度条
            pbar.set_postfix({
                'Reward': f'{episode_reward:.1f}',
                'Smoothed': f'{smoothed_rewards[-1]:.1f}',
                'Actor Loss': f'{avg_actor_loss:.2f}',
                'Critic Loss': f'{avg_critic_loss:.2f}'
            })
            pbar.update(1)
    
    step_log.close()
    episode_log.close()
    
    # 保存模型
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'config': agent_config
    }, f'tracking_model_final.pth')
    
    # 可视化最终轨迹
    print(f"\n可视化最终轨迹 (ε={env_config['epsilon']:.3f})")
    visualize_final_path(env)

if __name__ == "__main__":
    run_training()
