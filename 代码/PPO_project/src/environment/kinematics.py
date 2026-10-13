"""Discrete scalar feed/heading constraints (backward Euler sampling).

The environment supplies the complete heading intention, including reference
guidance, before this projection. No heading correction follows the KCM.
These scalar and heading bounds do not impose Cartesian axis constraints.
"""
from numba import jit
import math


@jit(nopython=True)
def _stopping_velocity_increment(acceleration, jerk_step, dt):
    """Velocity increment when applying a now, then reducing it by J*dt."""
    if acceleration <= 0.0:
        return 0.0
    n = math.ceil(acceleration / jerk_step)
    return dt * (n * acceleration - jerk_step * n * (n - 1) / 2.0)


@jit(nopython=True)
def _viable_acceleration(margin, max_acc, jerk_step, dt):
    """Largest acceleration that leaves enough velocity margin to ramp to zero."""
    low, high = 0.0, max_acc
    for _ in range(45):
        middle = 0.5 * (low + high)
        if _stopping_velocity_increment(middle, jerk_step, dt) <= max(margin, 0.0):
            low = middle
        else:
            high = middle
    return low


@jit(nopython=True)
def _project_velocity(prev_vel, prev_acc, intent, dt, lower, upper, max_acc, max_jerk):
    jerk_step = max_jerk * dt
    positive_bound = _viable_acceleration(upper - prev_vel, max_acc, jerk_step, dt)
    negative_bound = _viable_acceleration(prev_vel - lower, max_acc, jerk_step, dt)
    a_low = max(-max_acc, prev_acc - jerk_step, -negative_bound)
    a_high = min(max_acc, prev_acc + jerk_step, positive_bound)
    if a_low > a_high + 1e-8:
        raise ValueError("KCM received a non-viable state or an abrupt hard speed-limit change")
    acceleration = min(max((intent - prev_vel) / dt, a_low), a_high)
    velocity = prev_vel + acceleration * dt
    # Report derivatives of the actual output; no independent final speed clip.
    acceleration = (velocity - prev_vel) / dt
    jerk = (acceleration - prev_acc) / dt
    return velocity, acceleration, jerk


@jit(nopython=True)
def apply_kinematic_constraints(
    prev_vel, prev_acc, prev_ang_vel, prev_ang_acc, vel_action, ang_vel_action,
    dt, MAX_VEL, MAX_ACC, MAX_JERK, MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK,
):
    """Project onto feasible scalar acceleration intervals before integrating.

    v[k] = v[k-1] + a[k]*dt, a[k] = a[k-1] + j[k]*dt. A discrete
    braking-margin bound prevents a later velocity clip from invalidating jerk.
    The hard limits must remain constant during an episode; geometric speed caps
    are intent targets, whose instantaneous attainment is not guaranteed.
    """
    if min(dt, MAX_VEL, MAX_ACC, MAX_JERK, MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK) <= 0.0:
        raise ValueError("KCM dt and motion limits must be positive")
    velocity, acceleration, jerk = _project_velocity(
        prev_vel, prev_acc, vel_action, dt, 0.0, MAX_VEL, MAX_ACC, MAX_JERK)
    angular_vel, angular_acc, angular_jerk = _project_velocity(
        prev_ang_vel, prev_ang_acc, ang_vel_action, dt,
        -MAX_ANG_VEL, MAX_ANG_VEL, MAX_ANG_ACC, MAX_ANG_JERK)
    return velocity, acceleration, jerk, angular_vel, angular_acc, angular_jerk


__all__ = ["apply_kinematic_constraints"]
