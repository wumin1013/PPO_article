"""Finite discrete braking after geometric arrival, for simulation commands.

This module does not change a learned preview or repair an infeasible prefix.
Velocity samples obey the same backward-Euler convention as KCM. Polynomial
durations are chosen from motion limits, not from per-path performance. The
complete joined position command must pass an independent geometry/rest audit.
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


def discrete_stop_profile(velocity, acceleration, dt, count):
    """Cubic samples: u[-1]=u[0]-a[0]dt; u[n-1]=u[n]=0.

    Two terminal zero samples make final backward acceleration exactly zero.
    The final transition of jerk is included in the feasibility check.
    """
    if count < 2 or dt <= 0:
        raise ValueError("Need positive dt and at least two braking steps")
    n = int(count)
    k = np.arange(n + 1, dtype=float)
    c = float(velocity) / (n * (n - 1))
    d = c - (float(velocity) - float(acceleration) * dt) / (n * (n + 1))
    u = (n - k) * (n - 1 - k) * (c + d * k)
    u[0], u[-2:] = velocity, 0.
    a = np.r_[acceleration, np.diff(u) / dt]
    j = np.r_[0., np.diff(a) / dt]
    return u, a, j


def _feasible(profile, limit_v, limit_a, limit_j, *, nonnegative):
    v, a, j = profile
    return (np.isfinite(v).all() and (not nonnegative or v.min() >= 0.)
            and np.max(np.abs(v[1:])) <= limit_v
            and np.max(np.abs(a[1:])) <= limit_a
            and np.max(np.abs(j[1:])) <= limit_j)


def plan_terminal_brake(position, heading, velocity, acceleration,
                        angular_velocity, angular_acceleration, *, dt, limits,
                        policy=None):
    """Return integrated tail INCLUDING its unchanged splice state.

    Heading stops before feed, so stopping feed never hides a rotating state.
    No post-integration position/velocity clipping or target snapping occurs.
    Failure means this bounded cubic family found no feasible stop; this is
    not a proof that all possible braking profiles are infeasible.
    """
    policy = BrakingPolicy() if policy is None else policy
    values = np.r_[position, heading, velocity, acceleration, angular_velocity,
                   angular_acceleration, dt, list(limits.values())]
    if not np.isfinite(values).all() or dt <= 0 or velocity < 0:
        raise ValueError("Need finite state, nonnegative feed and positive dt")
    if not (0 < policy.limit_fraction <= 1 and policy.hold_s >= 0
            and policy.max_braking_s > 0 and policy.angular_lead_steps >= 0):
        raise ValueError("Invalid braking policy")
    maximum = int(math.floor(policy.max_braking_s / dt))
    margin = policy.limit_fraction
    omega_profile = None
    for angular_n in range(2, maximum + 1):
        trial = discrete_stop_profile(angular_velocity, angular_acceleration, dt, angular_n)
        if _feasible(trial, limits['MAX_ANG_VEL'],
                     limits['MAX_ANG_ACC'], limits['MAX_ANG_JERK'] * margin,
                     nonnegative=False):
            omega_profile = trial
            break
    if omega_profile is None:
        raise ValueError("No angular braking profile within registered horizon")
    feed_profile = None
    minimum = angular_n + policy.angular_lead_steps if abs(angular_velocity) + abs(angular_acceleration) > 0 else 2
    for feed_n in range(max(2, minimum), maximum + 1):
        trial = discrete_stop_profile(velocity, acceleration, dt, feed_n)
        if _feasible(trial, limits['MAX_VEL'],
                     limits['MAX_ACC'], limits['MAX_JERK'] * margin,
                     nonnegative=True):
            feed_profile = trial
            break
    if feed_profile is None:
        raise ValueError("No nonnegative feed braking profile within registered horizon")
    steps = max(feed_n, angular_n)
    # Three derivative-history samples before the full stationary hold window.
    hold_steps = int(math.ceil(policy.hold_s / dt)) + 3
    size = steps + hold_steps + 1
    columns = {'time_s': np.arange(size) * dt}
    for names, profile in [(('v_mm_s','a_mm_s2','j_mm_s3'),feed_profile),
                           (('omega_rad_s','alpha_rad_s2','angular_jerk_rad_s3'),omega_profile)]:
        for name, samples in zip(names, profile):
            columns[name] = np.pad(samples, (0, size-len(samples)))
    headings = heading + np.r_[0., np.cumsum(columns['omega_rad_s'][1:] * dt)]
    directions = np.c_[np.cos(headings[1:]), np.sin(headings[1:])]
    increments = dt * columns['v_mm_s'][1:,None] * directions
    positions = np.vstack([np.asarray(position,float), np.asarray(position,float) + np.cumsum(increments,axis=0)])
    columns.update(x_mm=positions[:,0], y_mm=positions[:,1], heading_rad=headings)
    return dict(columns=columns, braking_steps=steps, angular_braking_steps=angular_n,
                feed_braking_steps=feed_n, hold_steps=hold_steps,
                braking_time_s=steps*dt, total_tail_time_s=(size-1)*dt,
                internal_terminal_rest=all(columns[k][-1] == 0 for k in
                    ('v_mm_s','a_mm_s2','omega_rad_s','alpha_rad_s2')),
                endpoint_policy='Stop within audited geometric tolerance; no exact endpoint snap')
