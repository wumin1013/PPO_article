"""Independent feed-target tracking candidate; frozen KCM remains unchanged.

The original hard-bound projection preserves velocity viability. This candidate
also starts releasing acceleration before reaching the current target speed.
Targets can change arbitrarily; an unreachable new target can still be crossed
while inherited acceleration is released at the allowed jerk. No speed floor,
output clipping, heading change, environment installation or trajectory repair.
"""
from __future__ import annotations

import math

from src.environment.kinematics import _project_velocity, _viable_acceleration


def project_feed_target(prev_velocity, prev_acceleration, target_velocity, *,
                        dt, max_vel, max_acc, max_jerk):
    """Return actual end-of-step (velocity, acceleration, jerk).

    For a constant target, cap acceleration toward it by the velocity change
    required to release that acceleration with maximum jerk. The resulting
    one-tick intent is projected by the original hard-bound KCM, which retains
    authority when a new target is temporarily unreachable. This is a public
    feed tracking change shared by fixed/adaptive candidates, not learned
    lookahead or a guarantee against every possible target-jump overshoot.
    """
    values = (prev_velocity, prev_acceleration, target_velocity,
              dt, max_vel, max_acc, max_jerk)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Finite state, target and motion limits required')
    if min(dt, max_vel, max_acc, max_jerk) <= 0:
        raise ValueError('Positive sampling interval and motion limits required')
    if (prev_velocity < 0 or prev_velocity > max_vel*(1+1e-9)
            or abs(prev_acceleration) > max_acc*(1+1e-9)):
        raise ValueError('Initial feed state is outside hard motion bounds')
    target = min(max(float(target_velocity), 0.), max_vel)
    error = target-prev_velocity
    magnitude = _viable_acceleration(abs(error), max_acc, max_jerk*dt, dt)
    target_acceleration = math.copysign(magnitude, error) if error else 0.
    guarded_intent = prev_velocity+target_acceleration*dt
    return _project_velocity(prev_velocity, prev_acceleration, guarded_intent,
                             dt, 0., max_vel, max_acc, max_jerk)


__all__ = ['project_feed_target']
