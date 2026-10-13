"""Discrete jerk-bounded terminal braking with a variable-duration stop pulse.

End-of-step integration: v[k]=v[k-1]+a[k]dt. First release the inherited
acceleration at the original jerk bound, then remove the remaining velocity
with a triangular or trapezoidal acceleration pulse. Lengthening that pulse preserves its
sign, unlike a single cubic with a fixed nonzero initial acceleration.
No trajectory geometry is repaired here; audit the entire joined command.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BrakingPolicy:
    hold_s: float = .020
    max_braking_s: float = 2.0
    angular_lead_steps: int = 10
    limit_fraction: float = .98


def _derivatives(velocity, initial_acceleration, dt):
    acceleration = np.r_[initial_acceleration, np.diff(velocity)/dt]
    jerk = np.r_[0., np.diff(acceleration)/dt]
    return velocity, acceleration, jerk


def _check(profile, limit_v, limit_a, limit_j, *, nonnegative):
    v, a, j = profile
    # Floating arithmetic of differences only; this is stricter than the
    # existing independent output audit, whose thresholds remain unchanged.
    tolerance = 1e-9
    return (all(np.isfinite(x).all() for x in profile)
            and (not nonnegative or np.min(v) >= 0.)
            and np.max(np.abs(v)) <= limit_v*(1+tolerance)
            and np.max(np.abs(a)) <= limit_a*(1+tolerance)
            and np.max(np.abs(j)) <= limit_j*(1+tolerance))


def discrete_jerk_stop_profile(velocity, acceleration, dt, *, limit_v, limit_a,
                              limit_j, nonnegative, min_steps=2,
                              max_steps=2000, jerk_fraction=.98):
    """Return (v,a,j) and release/pulse durations, including splice state.

    The release uses the original full jerk bound: reducing it would exclude
    boundary states made viable by KCM using that bound. ``jerk_fraction``
    reserves numerical margin for the subsequent pulse only. This finite
    constructive family is not a proof of globally time-optimal braking.
    """
    values = (velocity, acceleration, dt, limit_v, limit_a, limit_j, jerk_fraction)
    if not np.isfinite(values).all() or min(dt, limit_v, limit_a, limit_j) <= 0:
        raise ValueError('Finite state and positive sampling/limits required')
    if not 0 < jerk_fraction <= 1 or min_steps < 0 or max_steps < 0:
        raise ValueError('Invalid braking duration or jerk fraction')
    if int(min_steps) != min_steps or int(max_steps) != max_steps:
        raise ValueError('Braking durations must be integer sample counts')
    initial = _derivatives(np.array([float(velocity)]), float(acceleration), dt)
    if not _check(initial, limit_v, limit_a, limit_j, nonnegative=nonnegative):
        raise ValueError('Initial velocity/acceleration is outside the motion bounds')
    if velocity == 0 and acceleration == 0:
        return dict(profile=initial, release_steps=0, pulse_steps=0, stop_steps=0)

    release_steps = int(math.ceil(abs(acceleration)/(limit_j*dt)))
    if release_steps > max_steps:
        raise ValueError('Acceleration release exceeds the registered horizon')
    k = np.arange(1, release_steps+1, dtype=float)
    release_a = np.sign(acceleration)*np.maximum(abs(acceleration)-limit_j*dt*k, 0.)
    release_v = np.r_[float(velocity), float(velocity)+dt*np.cumsum(release_a)]
    release_profile = _derivatives(release_v, float(acceleration), dt)
    if not _check(release_profile, limit_v, limit_a, limit_j, nonnegative=nonnegative):
        raise ValueError('Initial state cannot release acceleration without crossing a velocity bound')
    remaining = float(release_v[-1])
    if remaining == 0:
        return dict(profile=release_profile, release_steps=release_steps,
                    pulse_steps=0, stop_steps=release_steps)
    # Largest acceleration envelope with a[0]=a[n]=0 for this duration:
    # min(k*J*dt,(n-k)*J*dt,A). Scale its area to remove the remaining speed.
    # An acceleration plateau avoids unnecessarily long triangular stops at
    # large speed. The final TWO velocity samples are exactly zero.
    lower = max(2, int(min_steps)-release_steps,
                int(math.ceil(2*math.sqrt(abs(remaining)/(limit_j*jerk_fraction*dt*dt)))),
                int(math.ceil(abs(remaining)/(limit_a*dt)))+1)
    for pulse_steps in range(lower, int(max_steps)-release_steps+1):
        ramp = limit_j*jerk_fraction*dt*np.minimum(np.arange(pulse_steps+1), np.arange(pulse_steps, -1, -1))
        envelope = np.minimum(ramp, limit_a)
        cumulative = np.cumsum(envelope, dtype=float)
        area = float(cumulative[-1])
        if dt*area < abs(remaining):
            continue
        pulse_v = remaining*(1.-cumulative/area)
        samples = np.r_[release_v, pulse_v[1:]]
        profile = _derivatives(samples, float(acceleration), dt)
        if _check(profile, limit_v, limit_a, limit_j, nonnegative=nonnegative):
            return dict(profile=profile, release_steps=release_steps,
                        pulse_steps=pulse_steps, stop_steps=len(samples)-1)
    raise ValueError('No discrete jerk stop within the registered horizon')


def plan_terminal_brake(position, heading, velocity, acceleration,
                        angular_velocity, angular_acceleration, *, dt, limits,
                        policy=None):
    """Drop-in tail fields for the v2 planner, preserving its unchanged splice.

    An initially moving angular channel reaches v=a=0 at least ten samples
    before the feed channel. Feed pulse duration is increased only as required
    by that coupling; there is no added high-speed coast or endpoint snapping.
    A feed channel already at rest cannot satisfy angular-before-feed without
    introducing new motion; that case is reported as infeasible for this family.
    """
    policy = BrakingPolicy() if policy is None else policy
    point = np.asarray(position, dtype=float)
    names = ('MAX_VEL','MAX_ACC','MAX_JERK','MAX_ANG_VEL','MAX_ANG_ACC','MAX_ANG_JERK')
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('position must be a finite two-vector')
    values = (heading, velocity, acceleration, angular_velocity, angular_acceleration, dt,
              *(limits[k] for k in names), policy.hold_s, policy.max_braking_s, policy.limit_fraction)
    if not np.isfinite(values).all() or dt <= 0 or min(limits[k] for k in names) <= 0:
        raise ValueError('Finite state, positive dt and positive motion bounds required')
    if (not 0 < policy.limit_fraction <= 1 or policy.hold_s < 0 or policy.max_braking_s <= 0
            or policy.angular_lead_steps < 0 or int(policy.angular_lead_steps) != policy.angular_lead_steps):
        raise ValueError('Invalid braking policy')
    maximum = int(math.floor(policy.max_braking_s/dt))
    angular_moving = angular_velocity != 0 or angular_acceleration != 0
    angular = discrete_jerk_stop_profile(angular_velocity, angular_acceleration, dt,
        limit_v=limits['MAX_ANG_VEL'], limit_a=limits['MAX_ANG_ACC'], limit_j=limits['MAX_ANG_JERK'],
        nonnegative=False, max_steps=maximum, jerk_fraction=policy.limit_fraction)
    minimum_feed = angular['stop_steps']+policy.angular_lead_steps if angular_moving else 2
    feed = discrete_jerk_stop_profile(velocity, acceleration, dt,
        limit_v=limits['MAX_VEL'], limit_a=limits['MAX_ACC'], limit_j=limits['MAX_JERK'],
        nonnegative=True, min_steps=minimum_feed, max_steps=maximum, jerk_fraction=policy.limit_fraction)
    if angular_moving and feed['stop_steps'] < minimum_feed:
        raise ValueError('Feed would already be stationary before the required angular lead')
    steps = max(feed['stop_steps'], angular['stop_steps'])
    hold_steps = int(math.ceil(policy.hold_s/dt))+3
    size = steps+hold_steps+1
    columns = dict(time_s=np.arange(size)*dt)
    for fields, result in ((('v_mm_s','a_mm_s2','j_mm_s3'),feed),
                           (('omega_rad_s','alpha_rad_s2','angular_jerk_rad_s3'),angular)):
        for name, samples in zip(fields, result['profile']):
            columns[name] = np.pad(samples, (0, size-len(samples)))
    headings = heading+np.r_[0., np.cumsum(columns['omega_rad_s'][1:]*dt)]
    increments = dt*columns['v_mm_s'][1:,None]*np.c_[np.cos(headings[1:]),np.sin(headings[1:])]
    positions = np.vstack([point, point+np.cumsum(increments,axis=0)])
    columns.update(x_mm=positions[:,0], y_mm=positions[:,1], heading_rad=headings)
    return dict(columns=columns, braking_steps=steps, angular_braking_steps=angular['stop_steps'],
        feed_braking_steps=feed['stop_steps'], hold_steps=hold_steps,
        braking_time_s=steps*dt, total_tail_time_s=(size-1)*dt,
        internal_terminal_rest=all(columns[k][-1] == 0 for k in ('v_mm_s','a_mm_s2','omega_rad_s','alpha_rad_s2')),
        endpoint_policy='Stop within independently audited geometric tolerance; no exact endpoint snap',
        braking_version='discrete-jerk-stop-v3', angular_release_steps=angular['release_steps'],
        feed_release_steps=feed['release_steps'], angular_pulse_steps=angular['pulse_steps'],
        feed_pulse_steps=feed['pulse_steps'],
        initial_acceleration_release='Original full jerk bound; margin applies to the triangular/trapezoidal pulse only',
        optimality_scope='Shortest admissible integer duration in this staged pulse family, not a global minimum')


__all__ = ['BrakingPolicy', 'discrete_jerk_stop_profile', 'plan_terminal_brake']
