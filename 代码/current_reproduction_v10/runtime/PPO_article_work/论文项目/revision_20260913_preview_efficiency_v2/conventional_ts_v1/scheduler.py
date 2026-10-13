"""Sencer-style fixed-curve cruise and straight-line quintic feed scheduling.

Source: Sencer et al. (2015), corrected Eqs.29--37, as recorded in
ts_local_revision_v1/sencer2015_source_mapping_audit_v1.md. Deterministic
compatibility sweeps and bisection replace unspecified increment sizes.
This module does not construct geometry, import a policy, or run an experiment.
Curve caps must already include BOTH Cartesian acceleration and Cartesian jerk.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math


DISTANCE_TOLERANCE_MM = 1e-9
SPEED_TOLERANCE_MM_S = 1e-10
BISECTION_STEPS = 60
MAX_COMPATIBILITY_SWEEPS = 256


def _positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and ' + ('nonnegative' if zero else 'positive'))
    return value


def transition_duration(v0, v1, acceleration, jerk):
    """Minimum duration of THIS quintic velocity transition on a straight.

The true normalized acceleration/jerk peaks are 15/8 and 10/sqrt(3).
This is not a minimum-time claim over all possible velocity functions.
"""
    difference = abs(v1-v0)
    return max(15.*difference/(8.*acceleration),
               math.sqrt((10./math.sqrt(3.))*difference/jerk))


def transition_distance(v0, v1, acceleration, jerk):
    return .5*(v0+v1)*transition_duration(v0, v1, acceleration, jerk)


def _reachable_high(low, high, length, acceleration, jerk):
    """Only the greater endpoint is lowered; every bisection retains a feasible low bound."""
    if transition_distance(low, high, acceleration, jerk) <= length:
        return high
    if length == 0:
        return low
    feasible, infeasible = low, high
    for _ in range(BISECTION_STEPS):
        middle = .5*(feasible+infeasible)
        if transition_distance(low, middle, acceleration, jerk) <= length:
            feasible = middle
        else:
            infeasible = middle
    return feasible


@dataclass(frozen=True)
class Phase:
    kind: str
    source_index: int
    start_time_s: float
    duration_s: float
    start_length_mm: float
    length_mm: float
    start_speed_mm_s: float
    end_speed_mm_s: float

    def at(self, elapsed):
        """Arc length and tangential v/a/j; geometry supplies the Cartesian terms."""
        q = min(1., max(0., elapsed/self.duration_s))
        v0, change = self.start_speed_mm_s, self.end_speed_mm_s-self.start_speed_mm_s
        if change == 0:
            return self.start_length_mm + v0*q*self.duration_s, v0, 0., 0.
        length = self.duration_s*(v0*q+change*(2.5*q**4-3.*q**5+q**6))
        speed = v0+change*(10.*q**3-15.*q**4+6.*q**5)
        acceleration = change/self.duration_s*(30.*q*q-60.*q**3+30.*q**4)
        jerk = change/self.duration_s**2*(60.*q-180.*q*q+120.*q**3)
        return self.start_length_mm+length, speed, acceleration, jerk


@dataclass(frozen=True)
class Schedule:
    phases: tuple[Phase, ...]
    phase_starts_s: tuple[float, ...]
    curve_speeds_mm_s: tuple[float, ...]
    straight_peak_speeds_mm_s: tuple[float, ...]
    total_length_mm: float
    total_time_s: float
    compatibility_sweeps: int

    def at(self, elapsed):
        if not math.isfinite(elapsed):
            raise ValueError('Query time must be finite')
        if not self.phases:
            return 0., 0., 0., 0.
        if elapsed <= 0:
            return self.phases[0].at(0.)
        if elapsed >= self.total_time_s:
            return self.total_length_mm, self.phases[-1].end_speed_mm_s, 0., 0.
        index = bisect_right(self.phase_starts_s, elapsed)-1
        phase = self.phases[index]
        return phase.at(elapsed-phase.start_time_s)


def schedule(curve_lengths_mm, curve_speed_caps_mm_s, straight_lengths_mm, *,
             start_speed=0., end_speed=0., F=100., A=2000., J=62500.):
    """Build one deterministic, adjacent-compatible feed schedule.

Input order is line0, curve0, line1, ..., curveN-1, lineN. Curve caps
are geometric cruise bounds and may only decrease for adjacent compatibility.
There is no nonlinear time optimizer or whole-path slowdown factor.
"""
    F, A, J = (_positive(value, name) for value, name in ((F,'F'),(A,'A'),(J,'J')))
    curves = tuple(_positive(value, 'curve length') for value in curve_lengths_mm)
    caps = tuple(min(F, _positive(value, 'curve speed cap')) for value in curve_speed_caps_mm_s)
    lines = tuple(_positive(value, 'straight length', zero=True) for value in straight_lengths_mm)
    start_speed, end_speed = (_positive(value, name, zero=True)
                              for value, name in ((start_speed,'start speed'),(end_speed,'end speed')))
    if len(caps) != len(curves) or len(lines) != len(curves)+1:
        raise ValueError('Require N curve lengths/caps and N+1 straight lengths')
    if start_speed > F or end_speed > F:
        raise ValueError('An endpoint speed exceeds the program feed')
    speeds = [start_speed, *caps, end_speed]
    count = len(lines)
    converged = False
    for sweep in range(1, MAX_COMPATIBILITY_SWEEPS+1):
        for indices in (range(count), range(count-1, -1, -1)):
            for i in indices:
                left, right = speeds[i], speeds[i+1]
                if transition_distance(left, right, A, J) <= lines[i]:
                    continue
                high_index, low_index = (i,i+1) if left > right else (i+1,i)
                lowered = _reachable_high(speeds[low_index], speeds[high_index], lines[i], A, J)
                if high_index in (0,len(speeds)-1):
                    raise ValueError('The prescribed boundary speed cannot fit the adjacent straight')
                speeds[high_index] = lowered
        residual = max((transition_distance(speeds[i],speeds[i+1],A,J)-length
                        for i,length in enumerate(lines)), default=0.)
        if residual <= DISTANCE_TOLERANCE_MM:
            converged = True
            break
    if not converged:
        raise ValueError('Adjacent compatibility did not converge within its finite sweep budget')
    if any(value <= 0 for value in speeds[1:-1]):
        raise ValueError('A positive-length curve has zero cruise speed; this representation is not traversable')

    phases = []
    peaks = []
    time_now = length_now = 0.

    def append(kind, source_index, length, v0, v1):
        nonlocal time_now, length_now
        if length == 0 and v0 == v1:
            return
        duration = length/v0 if v0 == v1 and v0 > 0 else transition_duration(v0,v1,A,J)
        if duration <= 0 or not math.isfinite(duration):
            raise ValueError('Invalid phase duration')
        phases.append(Phase(kind,source_index,time_now,duration,length_now,length,v0,v1))
        time_now += duration
        length_now += length

    for index, length in enumerate(lines):
        v0, v1 = speeds[index:index+2]
        lower, upper = max(v0,v1), F

        def needed(peak):
            return transition_distance(v0,peak,A,J)+transition_distance(peak,v1,A,J)

        if needed(lower) > length+DISTANCE_TOLERANCE_MM:
            raise ValueError('Internal adjacent compatibility verification failed')
        if needed(F) <= length:
            peak = F
        else:
            for _ in range(BISECTION_STEPS):
                middle = .5*(lower+upper)
                if needed(middle) <= length:
                    lower = middle
                else:
                    upper = middle
            peak = lower
        peaks.append(peak)
        incoming, outgoing = transition_distance(v0,peak,A,J), transition_distance(peak,v1,A,J)
        cruise = length-incoming-outgoing
        if cruise < -DISTANCE_TOLERANCE_MM:
            raise ValueError('Transitions exceed the straight length')
        append('straight_transition',index,incoming,v0,peak)
        append('straight_cruise',index,max(0.,cruise),peak,peak)
        append('straight_transition',index,outgoing,peak,v1)
        if index < len(curves):
            append('curve_cruise',index,curves[index],speeds[index+1],speeds[index+1])
    expected_length = math.fsum((*curves,*lines))
    if abs(length_now-expected_length) > DISTANCE_TOLERANCE_MM*max(1,len(lines)):
        raise ValueError('Integrated schedule length differs from the fixed geometry')
    return Schedule(tuple(phases),tuple(phase.start_time_s for phase in phases),
                    tuple(speeds[1:-1]),tuple(peaks),length_now,time_now,sweep)
