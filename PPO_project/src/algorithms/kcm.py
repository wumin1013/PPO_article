"""
运动学约束模块 (Kinematic Constraint Module, KCM)
严格对应论文 Algorithm 1 实现

核心思想:
1. Top-down Constraint: Jerk优先 -> Acc反推 -> Vel反推
2. Bottom-up Integration: 从约束后的jerk积分回最终位姿
"""

import numpy as np
from typing import Tuple


class KinematicConstraintModule:
    """
    运动学约束模块
    
    实现论文Algorithm 1的运动学约束层,确保智能体输出的动作
    满足物理系统的速度、加速度和捷度约束。
    """
    
    def __init__(self, 
                 max_vel: float,
                 max_acc: float, 
                 max_jerk: float,
                 max_ang_vel: float,
                 max_ang_acc: float,
                 max_ang_jerk: float,
                 dt: float):
        """
        初始化运动学约束参数
        
        Args:
            max_vel: 最大线速度 (m/s)
            max_acc: 最大线加速度 (m/s²)
            max_jerk: 最大线捷度 (m/s³)
            max_ang_vel: 最大角速度 (rad/s)
            max_ang_acc: 最大角加速度 (rad/s²)
            max_ang_jerk: 最大角捷度 (rad/s³)
            dt: 时间步长 (s)
        """
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.max_ang_vel = max_ang_vel
        self.max_ang_acc = max_ang_acc
        self.max_ang_jerk = max_ang_jerk
        self.dt = dt
    
    def apply_constraints(self,
                         raw_action: Tuple[float, float],
                         prev_state: dict) -> Tuple[Tuple[float, float], float]:
        """
        应用运动学约束到原始动作
        
        Algorithm 1 实现:
        Input: a_raw (网络输出的意图动作 [v_intent, theta_intent])
        Process:
            1. Top-down Constraint: 
               - 优先计算Jerk限制
               - 基于Jerk限制反推Acc限制  
               - 基于Acc限制反推Vel限制
            2. Bottom-up Integration:
               - 使用积分公式从jerk_final反算最终位姿
               - v = v0 + a*t + 0.5*j*t^2
        Output: 
            - a_final (实际执行动作)
            - kcm_intervention (干预度指标)
        
        Args:
            raw_action: 网络输出的原始动作 (ang_vel_intent, lin_vel_intent)
            prev_state: 上一时刻的运动学状态
                - 'velocity': 线速度
                - 'acceleration': 线加速度  
                - 'angular_vel': 角速度
                - 'angular_acc': 角加速度
        
        Returns:
            final_action: 约束后的安全动作 (ang_vel_final, lin_vel_final)
            intervention: KCM干预程度 (用于论文分析)
        """
        ang_vel_intent, lin_vel_intent = raw_action
        
        # 提取前一状态
        prev_vel = prev_state['velocity']
        prev_acc = prev_state['acceleration']
        prev_ang_vel = prev_state['angular_vel']
        prev_ang_acc = prev_state['angular_acc']
        
        # ===== 线性运动约束 (Algorithm 1: Linear Motion) =====
        
        # Step 1: 计算意图加速度
        intent_acc = (lin_vel_intent - prev_vel) / self.dt
        
        # Step 2: 计算意图捷度 (Jerk优先级最高)
        intent_jerk = (intent_acc - prev_acc) / self.dt
        
        # Step 3: Top-down约束 - 从Jerk开始
        constrained_jerk = np.clip(intent_jerk, -self.max_jerk, self.max_jerk)
        
        # Step 4: 基于约束后的Jerk反推加速度限制
        constrained_acc = prev_acc + constrained_jerk * self.dt
        constrained_acc = np.clip(constrained_acc, -self.max_acc, self.max_acc)
        
        # Step 5: 反向修正Jerk (确保加速度约束得到满足)
        constrained_jerk = (constrained_acc - prev_acc) / self.dt
        
        # Step 6: Bottom-up积分 - 从Jerk积分到速度
        # 使用运动学公式: v = v0 + a*dt + 0.5*j*dt^2
        final_vel = prev_vel + prev_acc * self.dt + 0.5 * constrained_jerk * self.dt**2
        final_vel = np.clip(final_vel, 0.0, self.max_vel)  # 速度非负且有上限
        
        # Step 7: 最终一致性校正
        final_acc = (final_vel - prev_vel) / self.dt
        final_jerk = (final_acc - prev_acc) / self.dt
        
        # ===== 角运动约束 (Algorithm 1: Angular Motion) =====
        
        # Step 1: 计算意图角加速度
        intent_ang_acc = (ang_vel_intent - prev_ang_vel) / self.dt
        
        # Step 2: 计算意图角捷度
        intent_ang_jerk = (intent_ang_acc - prev_ang_acc) / self.dt
        
        # Step 3: Top-down约束 - 从角捷度开始
        constrained_ang_jerk = np.clip(intent_ang_jerk, -self.max_ang_jerk, self.max_ang_jerk)
        
        # Step 4: 反推角加速度限制
        constrained_ang_acc = prev_ang_acc + constrained_ang_jerk * self.dt
        constrained_ang_acc = np.clip(constrained_ang_acc, -self.max_ang_acc, self.max_ang_acc)
        
        # Step 5: 反向修正角捷度
        constrained_ang_jerk = (constrained_ang_acc - prev_ang_acc) / self.dt
        
        # Step 6: Bottom-up积分到角速度
        final_ang_vel = prev_ang_vel + prev_ang_acc * self.dt + 0.5 * constrained_ang_jerk * self.dt**2
        final_ang_vel = np.clip(final_ang_vel, -self.max_ang_vel, self.max_ang_vel)
        
        # Step 7: 最终一致性校正
        final_ang_acc = (final_ang_vel - prev_ang_vel) / self.dt
        final_ang_jerk = (final_ang_acc - prev_ang_acc) / self.dt
        
        # ===== 计算KCM干预度 =====
        # 衡量约束模块对原始意图的修正程度
        velocity_diff = abs(final_vel - lin_vel_intent)
        angular_diff = abs(final_ang_vel - ang_vel_intent)
        intervention = velocity_diff + angular_diff
        
        # 构建最终动作
        final_action = (final_ang_vel, final_vel)
        
        # 返回约束后的状态(用于环境更新)
        constrained_state = {
            'velocity': final_vel,
            'acceleration': final_acc,
            'jerk': final_jerk,
            'angular_vel': final_ang_vel,
            'angular_acc': final_ang_acc,
            'angular_jerk': final_ang_jerk
        }
        
        return final_action, intervention, constrained_state
    
    def get_state_dict(self) -> dict:
        """获取约束参数的字典表示"""
        return {
            'max_vel': self.max_vel,
            'max_acc': self.max_acc,
            'max_jerk': self.max_jerk,
            'max_ang_vel': self.max_ang_vel,
            'max_ang_acc': self.max_ang_acc,
            'max_ang_jerk': self.max_ang_jerk,
            'dt': self.dt
        }
