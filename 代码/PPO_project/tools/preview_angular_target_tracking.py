"""Experimental target-aware angular intention before the unchanged KCM.

The original KCM reserves acceleration-release margin at hard velocity bounds.
This wrapper additionally releases acceleration when approaching the current
angular target, using the same discrete viability kernel as feed tracking.
It selects no preview distance and never changes an executed trajectory.
"""
from contextlib import contextmanager
import math

from src.environment.cnc_env import Env
from src.environment.kinematics import _viable_acceleration


def guarded_angular_intent(previous_rate, target_rate, *, dt, max_rate,
                           max_acceleration, max_jerk):
    values = (previous_rate, target_rate, dt, max_rate, max_acceleration, max_jerk)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('Finite angular state, target and limits required')
    if min(dt, max_rate, max_acceleration, max_jerk) <= 0:
        raise ValueError('Positive angular limits and sampling period required')
    if abs(previous_rate) > max_rate * (1 + 1e-9):
        raise ValueError('Previous angular rate is outside hard bounds')
    target = min(max(target_rate, -max_rate), max_rate)
    error = target - previous_rate
    magnitude = _viable_acceleration(abs(error), max_acceleration, max_jerk * dt, dt)
    acceleration = math.copysign(magnitude, error) if error else 0.
    return previous_rate + acceleration * dt


@contextmanager
def angular_target_execution():
    """Wrap the active guidance; the sole original angular KCM still executes."""
    original = Env._resolve_heading_intent
    if getattr(original, '_angular_target_wrapper', False):
        raise RuntimeError('Angular target wrapper already installed')

    def resolve(self, policy_omega, velocity_intent):
        target = float(original(self, policy_omega, velocity_intent))
        intent = guarded_angular_intent(float(self.angular_vel), target,
            dt=float(self.interpolation_period), max_rate=float(self.MAX_ANG_VEL),
            max_acceleration=float(self.MAX_ANG_ACC), max_jerk=float(self.MAX_ANG_JERK))
        self._heading_intent_status.update(heading_rate_target=target,
            heading_rate_intent=intent, angular_target_release_active=float(abs(intent-target) > 1e-12))
        return intent

    resolve._angular_target_wrapper = True
    Env._resolve_heading_intent = resolve
    try:
        yield
    finally:
        Env._resolve_heading_intent = original
