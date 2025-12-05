"""
奖励函数计算模块
实现论文中定义的奖励函数，使用配置化的权重
"""

import numpy as np
from typing import Dict


class RewardCalculator:
    """
    奖励计算器
    
    实现论文中的奖励函数:
    R = w_e * R_tracking + w_c * R_comfort + w_j * R_smoothness + ...
    """
    
    def __init__(self, weights: Dict[str, float], 
                 max_vel: float,
                 max_acc: float,
                 max_jerk: float,
                 max_ang_jerk: float,
                 half_epsilon: float):
        """
        初始化奖励计算器
        
        Args:
            weights: 奖励权重字典 {'w_e', 'w_c', 'w_j', ...}
            max_vel: 最大速度
            max_acc: 最大加速度
            max_jerk: 最大捷度
            max_ang_jerk: 最大角捷度
            half_epsilon: 单侧边界距离
        """
        self.weights = weights
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.max_ang_jerk = max_ang_jerk
        self.half_epsilon = half_epsilon
        
        self.last_progress = 0.0
    
    def calculate_reward(self, 
                        contour_error: float,
                        progress: float,
                        direction_error: float,
                        velocity: float,
                        jerk: float,
                        angular_jerk: float,
                        kcm_intervention: float,
                        end_distance: float,
                        lap_completed: bool = False) -> float:
        """
        计算总奖励
        
        Args:
            contour_error: 轮廓跟踪误差 (距离中心线的距离)
            progress: 当前进度 [0, 1]
            direction_error: 方向偏差 (tau)
            velocity: 当前速度
            jerk: 当前线性捷度
            angular_jerk: 当前角捷度
            kcm_intervention: KCM干预程度
            end_distance: 到终点的距离
            lap_completed: 是否完成一圈(闭合路径)
        
        Returns:
            total_reward: 总奖励值
        """
        # 1. 跟踪奖励 - 惩罚轮廓误差
        distance_ratio = np.clip(contour_error / self.half_epsilon, 0, 2)
        tracking_reward = 10.0 * np.exp(-3 * distance_ratio) - 5.0
        
        # 2. 进度奖励 - 鼓励前进
        progress_diff = max(0, progress - self.last_progress)
        progress_reward = 20.0 * progress_diff
        
        # 3. 方向对齐奖励 - 鼓励沿路径方向
        direction_reward = 5.0 * np.exp(-2 * abs(direction_error)) - 2.5
        
        # 4. 速度奖励 - 鼓励合理速度
        velocity_ratio = velocity / self.max_vel
        velocity_reward = 2.0 * (1 - abs(velocity_ratio - 0.7))
        
        # 5. 平滑性奖励 - 惩罚过大捷度
        jerk_penalty = -2.0 * np.clip(abs(jerk) / self.max_jerk, 0, 1)
        ang_jerk_penalty = -2.0 * np.clip(abs(angular_jerk) / self.max_ang_jerk, 0, 1)
        smoothness_reward = jerk_penalty + ang_jerk_penalty
        
        # 6. 约束干预惩罚
        constraint_penalty = -2.0 * kcm_intervention
        
        # 7. 完成奖励
        completion_reward = 0.0
        end_distance_ratio = end_distance / self.half_epsilon
        
        if progress > 0.8:
            proximity_bonus = 50.0 * (progress - 0.8) * np.exp(-5 * end_distance_ratio)
            completion_reward += proximity_bonus
            
            if end_distance < self.half_epsilon * 0.6:
                completion_reward += 100.0 * np.exp(-10 * end_distance_ratio)
                
                if end_distance < self.half_epsilon * 0.2:
                    completion_reward += 200.0
        
        # 闭合路径完成奖励
        if lap_completed:
            completion_reward += 150.0
        
        # 8. 存活奖励
        survival_reward = 0.1
        
        # 应用权重并组合
        total_reward = (
            self.weights.get('w_e', 1.0) * tracking_reward +
            self.weights.get('w_progress', 2.0) * progress_reward +
            self.weights.get('w_direction', 0.5) * direction_reward +
            self.weights.get('w_velocity', 0.2) * velocity_reward +
            self.weights.get('w_j', 0.3) * smoothness_reward +
            self.weights.get('w_constraint', 0.2) * constraint_penalty +
            completion_reward +
            survival_reward
        )
        
        # 限制奖励范围
        total_reward = np.clip(total_reward, -20, 100)
        
        # 更新进度
        self.last_progress = progress
        
        return total_reward
    
    def reset(self):
        """重置奖励计算器状态"""
        self.last_progress = 0.0
