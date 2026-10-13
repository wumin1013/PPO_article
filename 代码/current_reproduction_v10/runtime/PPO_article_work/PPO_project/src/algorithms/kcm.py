"""
运动学约束模块 (Kinematic Constraint Module, KCM)
与环境共用离散 KCM 实现

核心思想:
在离散加速度可行区间内投影，并为触及速度上限预留捷度制动余量。
"""

import numpy as np
from typing import Tuple


class KinematicConstraintModule:
    """
    运动学约束模块
    
    实现标量进给与完整航向意图的离散约束层。
    笛卡尔各轴和实际位置指令仍需独立检查。
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
            max_vel: 最大线速度 (path-unit/s)
            max_acc: 最大线加速度 (path-unit/s²)
            max_jerk: 最大线捷度 (path-unit/s³)
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
    
    def apply_constraints(self, raw_action: Tuple[float, float], prev_state: dict):
        """Compatibility wrapper for the environment's discrete scalar KCM.

        Returns (angular/feed velocity action, normalized intervention, state).
        Position uses x[k] = x[k-1] + dt*v[k]*direction[k]. Callers supply
        the complete heading intention, including any reference correction,
        before projection. Cartesian constraints require an independent audit.
        """
        from src.environment.kinematics import apply_kinematic_constraints
        ang_intent, vel_intent = raw_action
        velocity, acceleration, jerk, omega, alpha, angular_jerk = apply_kinematic_constraints(
            prev_state["velocity"], prev_state["acceleration"],
            prev_state["angular_vel"], prev_state["angular_acc"],
            vel_intent, ang_intent, self.dt, self.max_vel, self.max_acc,
            self.max_jerk, self.max_ang_vel, self.max_ang_acc, self.max_ang_jerk,
        )
        intervention = (abs(velocity - vel_intent) / self.max_vel
                        + abs(omega - ang_intent) / self.max_ang_vel)
        state = dict(velocity=velocity, acceleration=acceleration, jerk=jerk,
                     angular_vel=omega, angular_acc=alpha, angular_jerk=angular_jerk)
        return (omega, velocity), float(intervention), state

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
