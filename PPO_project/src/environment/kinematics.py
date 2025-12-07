"""
运动学约束相关工具函数。
"""
from numba import jit


@jit(nopython=True)
def apply_kinematic_constraints(
    prev_vel,
    prev_acc,
    prev_ang_vel,
    prev_ang_acc,
    vel_action,
    ang_vel_action,
    dt,
    MAX_VEL,
    MAX_ACC,
    MAX_JERK,
    MAX_ANG_VEL,
    MAX_ANG_ACC,
    MAX_ANG_JERK,
):
    """
    基于加速度/捷度等约束对线速度和角速度意图进行裁剪。
    """
    constrained_vel = vel_action
    if constrained_vel < 0.0:
        constrained_vel = 0.0
    elif constrained_vel > MAX_VEL:
        constrained_vel = MAX_VEL

    raw_acc = (constrained_vel - prev_vel) / dt
    constrained_acc = raw_acc
    if constrained_acc < -MAX_ACC:
        constrained_acc = -MAX_ACC
    elif constrained_acc > MAX_ACC:
        constrained_acc = MAX_ACC

    raw_jerk = (constrained_acc - prev_acc) / dt
    constrained_jerk = raw_jerk
    if constrained_jerk < -MAX_JERK:
        constrained_jerk = -MAX_JERK
    elif constrained_jerk > MAX_JERK:
        constrained_jerk = MAX_JERK

    final_acc = prev_acc + constrained_jerk * dt
    final_vel = prev_vel + final_acc * dt
    if final_vel < 0.0:
        final_vel = 0.0
    elif final_vel > MAX_VEL:
        final_vel = MAX_VEL

    constrained_ang_vel = ang_vel_action
    if constrained_ang_vel < -MAX_ANG_VEL:
        constrained_ang_vel = -MAX_ANG_VEL
    elif constrained_ang_vel > MAX_ANG_VEL:
        constrained_ang_vel = MAX_ANG_VEL

    raw_ang_acc = (constrained_ang_vel - prev_ang_vel) / dt
    constrained_ang_acc = raw_ang_acc
    if constrained_ang_acc < -MAX_ANG_ACC:
        constrained_ang_acc = -MAX_ANG_ACC
    elif constrained_ang_acc > MAX_ANG_ACC:
        constrained_ang_acc = MAX_ANG_ACC

    raw_ang_jerk = (constrained_ang_acc - prev_ang_acc) / dt
    constrained_ang_jerk = raw_ang_jerk
    if constrained_ang_jerk < -MAX_ANG_JERK:
        constrained_ang_jerk = -MAX_ANG_JERK
    elif constrained_ang_jerk > MAX_ANG_JERK:
        constrained_ang_jerk = MAX_ANG_JERK

    final_ang_acc = prev_ang_acc + constrained_ang_jerk * dt
    final_ang_vel = prev_ang_vel + final_ang_acc * dt
    if final_ang_vel < -MAX_ANG_VEL:
        final_ang_vel = -MAX_ANG_VEL
    elif final_ang_vel > MAX_ANG_VEL:
        final_ang_vel = MAX_ANG_VEL

    return (
        final_vel,
        final_acc,
        constrained_jerk,
        final_ang_vel,
        final_ang_acc,
        constrained_ang_jerk,
    )


__all__ = ["apply_kinematic_constraints"]
