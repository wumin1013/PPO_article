# -*- coding: utf-8 -*-
"""测试运动学约束 - 模拟进弯场景"""
from src.environment.kinematics import apply_kinematic_constraints

MAX_ANG_VEL = 6.283185307179586
MAX_ANG_ACC = 100.0
MAX_ANG_JERK = 1000.0
dt = 0.01

# 模拟从加速状态进入弯道
# 进弯前：omega_cmd=1.0 (目标 6.28)，角速度在加速
# 进弯后：omega_cmd=0.25 (目标 1.57)，但角速度可能继续加速

print('模拟进弯场景：')
print('step,omega_cmd,prev_omega,prev_acc,final_omega,final_acc')

prev_omega = 0.0
prev_acc = 0.0

# Step 32-40: omega_cmd = 1.0 (目标 6.28)
for i in range(9):
    omega_cmd = 6.28  # omega_ratio_cmd=1.0
    r = apply_kinematic_constraints(
        22.0, 0.0, prev_omega, prev_acc,
        22.0, omega_cmd, dt,
        1000.0, 5000.0, 50000.0,
        MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK
    )
    print(f'{32+i},1.0,{prev_omega:.4f},{prev_acc:.2f},{r[3]:.4f},{r[4]:.2f}')
    prev_omega = r[3]
    prev_acc = r[4]

# Step 41-46: omega_cmd = 0.25 (目标 1.57)
for i in range(6):
    omega_cmd = 0.25 * 6.28  # omega_ratio_cmd=0.25
    r = apply_kinematic_constraints(
        22.0, 0.0, prev_omega, prev_acc,
        22.0, omega_cmd, dt,
        1000.0, 5000.0, 50000.0,
        MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK
    )
    print(f'{41+i},0.25,{prev_omega:.4f},{prev_acc:.2f},{r[3]:.4f},{r[4]:.2f}')
    prev_omega = r[3]
    prev_acc = r[4]

# Step 47+: omega_cmd = 负值（恢复态）
for i in range(5):
    omega_cmd = -0.247 * 6.28  # omega_ratio_cmd=-0.247
    r = apply_kinematic_constraints(
        22.0, 0.0, prev_omega, prev_acc,
        22.0, omega_cmd, dt,
        1000.0, 5000.0, 50000.0,
        MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK
    )
    print(f'{47+i},-0.25,{prev_omega:.4f},{prev_acc:.2f},{r[3]:.4f},{r[4]:.2f}')
    prev_omega = r[3]
    prev_acc = r[4]
