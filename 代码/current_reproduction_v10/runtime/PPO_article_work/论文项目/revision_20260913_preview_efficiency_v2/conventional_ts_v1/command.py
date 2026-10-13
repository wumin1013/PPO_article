"""Sample the conventional feed law on an analytical arc-length geometry.

No geometry construction, fitting, optimization, registration or old-result
loading occurs here. A registered caller supplies its verified geometry and
schedule. Complete Cartesian constraints still require the independent audit.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np


DT_S = .001
STATIONARY_HOLD_SAMPLES = 25
MAX_COMPLETE_TIME_S = 1200.
POSITION_QUERY_CHUNK = 32768


def assemble(schedule, geometry):
    if not schedule.phases or schedule.total_time_s <= 0:
        raise ValueError('A complete experiment requires positive traversal time')
    if schedule.phases[0].start_speed_mm_s != 0. or schedule.phases[-1].end_speed_mm_s != 0.:
        raise ValueError('The shared experiment starts and ends at rest')
    if abs(float(geometry.length)-schedule.total_length_mm) > 1e-7:
        raise ValueError('Geometry length and integrated feed schedule differ')
    last_moving_tick = math.ceil(schedule.total_time_s/DT_S)
    last_tick = last_moving_tick+STATIONARY_HOLD_SAMPLES
    if last_tick*DT_S > MAX_COMPLETE_TIME_S:
        raise ValueError('The finite complete-command time budget is exceeded')
    t = np.arange(last_tick+1, dtype=np.float64)*DT_S
    sigma = np.empty_like(t)
    speed = np.empty_like(t)
    tangential_acceleration = np.empty_like(t)
    tangential_jerk = np.empty_like(t)
    for index, phase in enumerate(schedule.phases):
        end_time = schedule.phase_starts_s[index+1] if index+1 < len(schedule.phases) else schedule.total_time_s
        selected = slice(int(np.searchsorted(t,phase.start_time_s,side='left')),
                         int(np.searchsorted(t,end_time,side='left')))
        q = np.clip((t[selected]-phase.start_time_s)/phase.duration_s,0.,1.)
        v0, difference = phase.start_speed_mm_s, phase.end_speed_mm_s-phase.start_speed_mm_s
        sigma[selected] = phase.start_length_mm+phase.duration_s*(v0*q+difference*(2.5*q**4-3.*q**5+q**6))
        speed[selected] = v0+difference*(10.*q**3-15.*q**4+6.*q**5)
        tangential_acceleration[selected] = difference/phase.duration_s*(30.*q*q-60.*q**3+30.*q**4)
        tangential_jerk[selected] = difference/phase.duration_s**2*(60.*q-180.*q*q+120.*q**3)
    stationary = t >= schedule.total_time_s
    sigma[stationary] = schedule.total_length_mm
    speed[stationary] = tangential_acceleration[stationary] = tangential_jerk[stationary] = 0.
    if (sigma[0] != 0. or not np.isfinite(np.column_stack((sigma,speed,tangential_acceleration,tangential_jerk))).all()
            or np.any(np.diff(sigma) < -1e-10) or np.any(sigma < 0.)
            or np.any(sigma > schedule.total_length_mm+1e-10)):
        raise ValueError('Sampled feed law is nonfinite or loses ordered arc-length progress')
    pos = np.empty((len(t),2), dtype=np.float64)
    for start in range(0,len(t),POSITION_QUERY_CHUNK):
        end = min(len(t),start+POSITION_QUERY_CHUNK)
        # The supplied geometry evaluates its true curves using inverse arc
        # length; this module never interpolates a sampled XY polyline.
        pos[start:end] = geometry.position(sigma[start:end])
    if not np.isfinite(pos).all():
        raise ValueError('Analytical geometry returned a nonfinite command')
    return dict(t=t,pos=pos,sigma=sigma,
        planned_feed=speed,planned_tangential_acceleration=tangential_acceleration,
        planned_tangential_jerk=tangential_jerk,
        metadata=dict(continuous_motion_duration_s=schedule.total_time_s,
            complete_command_time_s=float(t[-1]), sample_count=len(t), dt_s=DT_S,
            stationary_hold_samples=STATIONARY_HOLD_SAMPLES,
            maximum_complete_time_s=MAX_COMPLETE_TIME_S,
            fixed_geometry_length_mm=float(geometry.length),
            compatibility_sweeps=schedule.compatibility_sweeps,
            curve_cruise_speeds_mm_s=list(schedule.curve_speeds_mm_s),
            straight_peak_speeds_mm_s=list(schedule.straight_peak_speeds_mm_s),
            scheduled_phases=[asdict(phase) for phase in schedule.phases],
            boundary_policy='The analytic feed law reaches zero speed and acceleration at its natural endpoint; the same endpoint position is held after that time. No instantaneous endpoint reset.',
            attribution='Curve-constant feed and straight quintic transitions with adjacent length compatibility; no SLSQP or global time multiplier.',
            planned_scalar_derivatives_are_not_Cartesian_audit=True,
            original_command_geometry_audit_required=True))
