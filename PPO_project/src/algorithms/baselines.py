"""
基线算法模块
实现论文对比实验所需的基线方法

包含:
1. NNC (无Jerk约束的神经网络控制器)
2. Traditional S-Curve Planner (传统S型加减速规划器)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .ppo import PPOContinuous, PolicyNetContinuous, ValueNet


class NNCAgent(PPOContinuous):
    """
    NNC (Neural Network Controller without Jerk Constraint)
    
    基线方法1: 禁用KCM模块的PPO智能体
    网络输出直接作为动作，仅进行基础的速度裁剪
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 actor_lr: float,
                 critic_lr: float,
                 lmbda: float,
                 epochs: int,
                 eps: float,
                 gamma: float,
                 device: torch.device,
                 max_vel: float = 1.0,
                 max_ang_vel: float = 1.5,
                 observation_space=None,
                 action_space=None):
        """
        初始化NNC智能体
        
        Args:
            与PPOContinuous相同的参数
            max_vel: 最大线速度（用于基础裁剪）
            max_ang_vel: 最大角速度（用于基础裁剪）
        """
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lmbda=lmbda,
            epochs=epochs,
            eps=eps,
            gamma=gamma,
            device=device,
            observation_space=observation_space,
            action_space=action_space
        )
        
        self.max_vel = max_vel
        self.max_ang_vel = max_ang_vel
        self.enable_kcm = False  # 标记：不使用KCM
    
    def take_action(self, state):
        """
        选择动作（跳过KCM约束）
        
        Returns:
            动作元组 (angular_velocity, linear_velocity)
        """
        if not hasattr(self, 'state_tensor'):
            self.state_tensor = torch.empty((1, len(state)), dtype=torch.float, device=self.device)
        
        self.state_tensor[0] = torch.tensor(state, dtype=torch.float)
        mu, sigma = self.actor(self.state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        
        # 仅进行基础裁剪（不使用KCM的捷度约束）
        ang_vel = float(action[0, 0].cpu().numpy())
        lin_vel = float(action[0, 1].cpu().numpy())
        
        # 简单裁剪到允许范围
        ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
        lin_vel = np.clip(lin_vel, 0.0, self.max_vel)
        
        return [ang_vel, lin_vel]


class SCurvePlanner:
    """
    传统S型加减速规划器
    
    基线方法2: 非强化学习的传统CNC控制方法
    根据最大加速度和最大捷度预计算速度规划曲线
    """
    
    def __init__(self,
                 max_vel: float,
                 max_acc: float,
                 max_jerk: float,
                 dt: float):
        """
        初始化S型规划器
        
        Args:
            max_vel: 最大速度
            max_acc: 最大加速度
            max_jerk: 最大捷度
            dt: 时间步长
        """
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.dt = dt
        
        # 速度规划状态
        self.current_vel = 0.0
        self.current_acc = 0.0
        self.target_vel = 0.0
        
        # S曲线相关参数
        self.phase = 'accel_increasing'  # 加速度增加阶段
        self.phase_time = 0.0
    
    def reset(self):
        """重置规划器状态"""
        self.current_vel = 0.0
        self.current_acc = 0.0
        self.target_vel = 0.0
        self.phase = 'accel_increasing'
        self.phase_time = 0.0
    
    def compute_s_curve_profile(self, distance: float, initial_vel: float = 0.0) -> List[float]:
        """
        计算完整的S型速度曲线
        
        S型曲线包含7个阶段:
        1. 加速度增加（Jerk = +J_max）
        2. 匀加速（Jerk = 0, Acc = A_max）
        3. 加速度减小（Jerk = -J_max）
        4. 匀速（Jerk = 0, Acc = 0）
        5. 加速度减小（负方向，Jerk = -J_max）
        6. 匀减速（Jerk = 0, Acc = -A_max）
        7. 加速度增加（回零，Jerk = +J_max）
        
        Args:
            distance: 需要行驶的距离
            initial_vel: 初始速度
        
        Returns:
            速度序列
        """
        # 计算S型曲线的时间参数
        t_j = self.max_acc / self.max_jerk  # 捷度作用时间
        
        # 判断是否能达到最大速度
        s_acc = 0.5 * self.max_acc * t_j + self.max_acc * t_j  # 加速阶段距离估算
        
        if 2 * s_acc < distance:
            # 能达到最大速度
            v_max = self.max_vel
            t_acc = t_j + (v_max / self.max_acc)
            t_const = (distance - 2 * s_acc) / v_max
        else:
            # 不能达到最大速度，需要计算实际最大速度
            v_max = np.sqrt(self.max_acc * distance)
            t_acc = v_max / self.max_acc
            t_const = 0.0
        
        t_dec = t_acc  # 对称减速
        
        # 生成速度曲线
        velocity_profile = []
        total_time = 2 * t_acc + t_const
        num_steps = int(total_time / self.dt)
        
        for i in range(num_steps):
            t = i * self.dt
            
            if t < t_j:
                # 阶段1: 加速度增加
                jerk = self.max_jerk
                acc = jerk * t
                vel = initial_vel + 0.5 * jerk * t**2
            elif t < t_acc - t_j:
                # 阶段2: 匀加速
                t_phase = t - t_j
                acc = self.max_acc
                vel = initial_vel + 0.5 * self.max_jerk * t_j**2 + self.max_acc * t_phase
            elif t < t_acc:
                # 阶段3: 加速度减小
                t_phase = t - (t_acc - t_j)
                jerk = -self.max_jerk
                acc = self.max_acc + jerk * t_phase
                vel = v_max - 0.5 * self.max_jerk * (t_acc - t)**2
            elif t < t_acc + t_const:
                # 阶段4: 匀速
                vel = v_max
            elif t < t_acc + t_const + t_j:
                # 阶段5: 加速度减小（负方向）
                t_phase = t - (t_acc + t_const)
                jerk = -self.max_jerk
                acc = jerk * t_phase
                vel = v_max + 0.5 * jerk * t_phase**2
            elif t < total_time - t_j:
                # 阶段6: 匀减速
                t_phase = t - (t_acc + t_const + t_j)
                acc = -self.max_acc
                vel = v_max - 0.5 * self.max_jerk * t_j**2 - self.max_acc * t_phase
            else:
                # 阶段7: 加速度增加（回零）
                t_phase = t - (total_time - t_j)
                jerk = self.max_jerk
                acc = -self.max_acc + jerk * t_phase
                vel = 0.5 * self.max_jerk * (total_time - t)**2
            
            vel = np.clip(vel, 0.0, self.max_vel)
            velocity_profile.append(vel)
        
        return velocity_profile
    
    def plan_velocity(self, 
                     current_pos: np.ndarray,
                     target_pos: np.ndarray,
                     path_direction: np.ndarray) -> Tuple[float, float]:
        """
        规划单步速度（简化版本，用于在线控制）
        
        Args:
            current_pos: 当前位置
            target_pos: 目标位置
            path_direction: 路径方向向量
        
        Returns:
            (angular_velocity, linear_velocity)
        """
        # 计算到目标的距离
        distance = np.linalg.norm(target_pos - current_pos)
        
        # 简化的S型速度规划：根据距离调整目标速度
        if distance > 5.0:
            self.target_vel = self.max_vel
        elif distance > 2.0:
            self.target_vel = self.max_vel * 0.7
        elif distance > 0.5:
            self.target_vel = self.max_vel * 0.4
        else:
            self.target_vel = self.max_vel * 0.2
        
        # 使用S型曲线更新速度
        vel_error = self.target_vel - self.current_vel
        
        if abs(vel_error) < 0.01:
            # 已达到目标速度
            jerk = 0.0
            self.current_acc = 0.0
        else:
            # 需要调整速度
            desired_acc = np.sign(vel_error) * min(abs(vel_error) / self.dt, self.max_acc)
            acc_error = desired_acc - self.current_acc
            
            # 应用捷度限制
            jerk = np.clip(acc_error / self.dt, -self.max_jerk, self.max_jerk)
            self.current_acc = np.clip(
                self.current_acc + jerk * self.dt,
                -self.max_acc,
                self.max_acc
            )
        
        # 更新速度
        self.current_vel = np.clip(
            self.current_vel + self.current_acc * self.dt,
            0.0,
            self.max_vel
        )
        
        # 计算角速度（简单的方向对齐）
        direction_to_target = target_pos - current_pos
        if np.linalg.norm(direction_to_target) > 1e-6:
            direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        else:
            direction_to_target = path_direction
        
        # 计算角度差
        cross = np.cross(path_direction, direction_to_target)
        dot = np.dot(path_direction, direction_to_target)
        angle_error = np.arctan2(cross, dot)
        
        # 简单的P控制器
        angular_vel = np.clip(angle_error * 2.0, -1.5, 1.5)
        
        return angular_vel, self.current_vel
    
    def take_action(self, state: List[float]) -> List[float]:
        """
        根据状态选择动作（兼容强化学习接口）
        
        Args:
            state: 环境状态（包含位置、方向等信息）
        
        Returns:
            [angular_velocity, linear_velocity]
        """
        # 从状态中提取必要信息
        # 假设状态格式: [contour_error, direction_error, velocity, ...]
        
        # 简化版本：根据误差调整速度
        contour_error = abs(state[0]) if len(state) > 0 else 0.0
        
        # 根据误差调整目标速度
        if contour_error > 0.3:
            self.target_vel = self.max_vel * 0.3
        elif contour_error > 0.1:
            self.target_vel = self.max_vel * 0.6
        else:
            self.target_vel = self.max_vel
        
        # 更新速度（使用S型曲线）
        vel_error = self.target_vel - self.current_vel
        desired_acc = np.sign(vel_error) * min(abs(vel_error) / self.dt, self.max_acc)
        acc_error = desired_acc - self.current_acc
        
        jerk = np.clip(acc_error / self.dt, -self.max_jerk, self.max_jerk)
        self.current_acc = np.clip(
            self.current_acc + jerk * self.dt,
            -self.max_acc,
            self.max_acc
        )
        
        self.current_vel = np.clip(
            self.current_vel + self.current_acc * self.dt,
            0.0,
            self.max_vel
        )
        
        # 角速度控制（简单的比例控制）
        direction_error = state[1] if len(state) > 1 else 0.0
        angular_vel = np.clip(direction_error * 2.0, -1.5, 1.5)
        
        return [angular_vel, self.current_vel]


def create_baseline_agent(baseline_type: str,
                          config: dict,
                          device: torch.device):
    """
    工厂函数：创建基线算法智能体
    
    Args:
        baseline_type: 基线类型 ('nnc' 或 's_curve')
        config: 配置字典
        device: 计算设备
    
    Returns:
        基线智能体实例
    """
    if baseline_type == 'nnc':
        # 创建NNC智能体（无KCM约束）
        ppo_config = config['ppo']
        kcm_config = config['kinematic_constraints']
        
        agent = NNCAgent(
            state_dim=config.get('state_dim', 12),
            hidden_dim=ppo_config['hidden_dim'],
            action_dim=config.get('action_dim', 2),
            actor_lr=ppo_config['actor_lr'],
            critic_lr=ppo_config['critic_lr'],
            lmbda=ppo_config['lmbda'],
            epochs=ppo_config['epochs'],
            eps=ppo_config['eps'],
            gamma=ppo_config['gamma'],
            device=device,
            max_vel=kcm_config['MAX_VEL'],
            max_ang_vel=kcm_config['MAX_ANG_VEL'],
            observation_space=config.get('observation_space'),
            action_space=config.get('action_space')
        )
        
        return agent
    
    elif baseline_type == 's_curve':
        # 创建传统S型规划器
        kcm_config = config['kinematic_constraints']
        env_config = config['environment']
        
        planner = SCurvePlanner(
            max_vel=kcm_config['MAX_VEL'],
            max_acc=kcm_config['MAX_ACC'],
            max_jerk=kcm_config['MAX_JERK'],
            dt=env_config['interpolation_period']
        )
        
        return planner
    
    else:
        raise ValueError(f"未知的基线类型: {baseline_type}. 可用类型: ['nnc', 's_curve']")
