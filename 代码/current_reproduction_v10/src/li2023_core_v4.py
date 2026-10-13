"""Independent kinematic primitives supporting the Li et al. (2023) audit.

This is NOT a reproduction of that paper's full optimizer. Geometry derivatives
are independently derived by the chain rule; asymmetric scalar profiles follow
exact constant-jerk integration. Source ambiguities and missing original-example
inputs are recorded in audit/li2023_reproduction_feasibility.md.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def advance(state: np.ndarray, jerk: float, duration: float) -> np.ndarray:
    """Exactly integrate one constant-jerk scalar phase."""
    s, v, a = state
    h = duration
    return np.array([s + v*h + a*h*h/2 + jerk*h**3/6,
                     v + a*h + jerk*h*h/2, a + jerk*h])


@dataclass
class ScalarProfile:
    durations: np.ndarray
    jerks: np.ndarray
    events: np.ndarray
    states: np.ndarray
    length: float
    peak_speed: float

    @property
    def duration(self) -> float:
        return float(self.events[-1])

    def evaluate(self, time: float) -> np.ndarray:
        """Return (s,v,a,j); a resting continuation applies outside [0,T].

        At an internal event the jerk is the right-hand limit. Acceleration,
        velocity and position are continuous; jerk need not be continuous.
        """
        if time < 0:
            return np.zeros(4)
        if time >= self.duration:
            return np.array([self.length, 0., 0., 0.])
        index = min(int(np.searchsorted(self.events, time, side="right")-1), 6)
        state = advance(self.states[index], self.jerks[index], time-self.events[index])
        return np.r_[state, self.jerks[index]]


def asymmetric_profile(length: float, velocity: float, acceleration: float,
                       jerk_accel: float, jerk_decel: float) -> ScalarProfile:
    """Construct an exact scalar rest-to-rest profile with independent jerks.

    Scalar limits are not Cartesian limits on curved primitives. Callers must
    independently verify Cartesian derivatives and overlap feasibility.
    """
    values = (length, velocity, acceleration, jerk_accel, jerk_decel)
    if not all(np.isfinite(values)) or min(values) <= 0:
        raise ValueError("length and scalar limits must be finite and positive")

    def phases(speed: float, jerk: float) -> tuple[float, float]:
        tj = min(np.sqrt(speed/jerk), acceleration/jerk)
        return float(tj), float(max(0., speed/(jerk*tj)-tj))

    def moving_distance(speed: float) -> float:
        if speed == 0:
            return 0.
        tja, taa = phases(speed, jerk_accel)
        tjd, tad = phases(speed, jerk_decel)
        return speed*(2*tja+taa+2*tjd+tad)/2

    speed = velocity
    if moving_distance(speed) > length:
        low, high = 0., velocity
        for _ in range(100):
            mid = (low+high)/2
            if moving_distance(mid) > length:
                high = mid
            else:
                low = mid
        speed = (low+high)/2
    tja, taa = phases(speed, jerk_accel)
    tjd, tad = phases(speed, jerk_decel)
    cruise = max(0., (length-moving_distance(speed))/speed)
    durations = np.array([tja, taa, tja, cruise, tjd, tad, tjd])
    jerks = np.array([jerk_accel, 0., -jerk_accel, 0., -jerk_decel, 0., jerk_decel])
    events = np.r_[0., np.cumsum(durations)]
    states = [np.zeros(3)]
    for h, j in zip(durations, jerks):
        states.append(advance(states[-1], j, h))
    return ScalarProfile(durations, jerks, events, np.array(states), length, speed)


@dataclass
class Primitive:
    origin: np.ndarray
    tangent_angle: float
    curvature: float = 0.

    def evaluate(self, scalar: np.ndarray) -> np.ndarray:
        """Return rows (position, velocity, acceleration, jerk), each in R².

        Positive curvature is counterclockwise. The signed-curvature form also
        includes clockwise arcs and a line at zero curvature without branches
        in the derivative formulas.
        """
        s, f, a, j = scalar
        k = self.curvature
        angle = self.tangent_angle+k*s
        tangent = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        t0 = np.array([np.cos(self.tangent_angle), np.sin(self.tangent_angle)])
        n0 = np.array([-t0[1], t0[0]])
        # sinc and sin² keep the near-line limit well conditioned.
        position = self.origin + s*np.sinc(k*s/np.pi)*t0
        if k != 0:
            position = position + 2*np.sin(k*s/2)**2/k*n0
        return np.array([position, f*tangent, a*tangent+k*f*f*normal,
                         (j-k*k*f**3)*tangent+3*k*f*a*normal])


def overlap(first: Primitive, p1: ScalarProfile, second: Primitive,
            p2: ScalarProfile, overlap_time: float, time: float) -> np.ndarray:
    """Add two complete displacement increments under a shifted local clock.

    This superposition kernel does not select overlap or enforce a corridor.
    It requires connected primitive endpoints and excludes triple activity.
    """
    if not 0 <= overlap_time <= min(p1.duration, p2.duration):
        raise ValueError("overlap must lie within both primitive durations")
    end1 = first.evaluate(np.array([p1.length, 0., 0., 0.]))[0]
    if not np.allclose(end1, second.origin, atol=1e-9, rtol=0):
        raise ValueError("primitive endpoints must be connected")
    value = first.evaluate(p1.evaluate(time))
    second_time = time-(p1.duration-overlap_time)
    # Preserve the exact global final-event convention despite cancellation
    # when translating to the second primitive's local clock.
    if time >= p1.duration+p2.duration-overlap_time:
        second_time = p2.duration
    value2 = second.evaluate(p2.evaluate(second_time))
    value2[0] -= second.origin
    return value+value2


def selftest() -> dict:
    records = []

    def check(name: str, error: float, tolerance: float, **details):
        records.append(dict(name=name, error=float(error), tolerance=tolerance,
                            passed=bool(np.isfinite(error) and error <= tolerance), **details))

    state = np.array([1.2, 3.4, -.7])
    check("constant_jerk_semigroup", np.max(np.abs(advance(advance(state, 12., .17), 12., .23)-advance(state, 12., .4))), 1e-12)
    scalar = np.array([2., 5., 0., 0.])
    for k in [.25, -.25]:
        result = Primitive(np.array([1., -2.]), .3, k).evaluate(scalar)
        check(f"uniform_circle_acceleration_{k}", abs(np.linalg.norm(result[2])-abs(k)*25), 1e-12)
        check(f"uniform_circle_jerk_{k}", abs(np.linalg.norm(result[3])-k*k*125), 1e-12)
    primitive = Primitive(np.array([1., -2.]), .3, -.2)
    scalar = np.array([2., 5., -.8, 1.1])
    value = primitive.evaluate(scalar)
    check("cartesian_acceleration_norm_identity", abs(np.dot(value[2],value[2])-(scalar[2]**2+.2**2*scalar[1]**4)), 1e-11)
    expected_j2 = (scalar[3]-.2**2*scalar[1]**3)**2+(3*.2*scalar[1]*scalar[2])**2
    check("cartesian_jerk_norm_identity", abs(np.dot(value[3],value[3])-expected_j2), 1e-11)
    check("resting_derivatives", np.max(np.abs(primitive.evaluate(np.array([2., 0., 0., 0.]))[1:])), 1e-12)
    line = Primitive(primitive.origin, primitive.tangent_angle, 0.)
    near = Primitive(primitive.origin, primitive.tangent_angle, 1e-10)
    check("large_radius_line_limit", np.max(np.abs(line.evaluate(scalar)-near.evaluate(scalar))), 2e-8)
    phi = .73
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    offset = np.array([7., -9.])
    transformed = Primitive(rot@primitive.origin+offset, primitive.tangent_angle+phi, primitive.curvature).evaluate(scalar)
    target = value@rot.T
    target[0] += offset
    check("rigid_transform_covariance", np.max(np.abs(transformed-target)), 1e-12)
    reversed_value = primitive.evaluate(scalar*np.array([1., -1., 1., -1.]))
    check("time_reversal_derivatives", np.max(np.abs(reversed_value-value*np.array([1.,-1.,1.,-1.])[:,None])), 1e-12)
    # Differentiate each independent analytic order along the scalar cubic.
    h = 1e-5
    left = primitive.evaluate(np.r_[advance(scalar[:3], scalar[3], -h), scalar[3]])
    right = primitive.evaluate(np.r_[advance(scalar[:3], scalar[3], h), scalar[3]])
    check("chain_rule_central_differences", np.max(np.abs((right[:3]-left[:3])/(2*h)-value[1:])), 2e-8, step=h)

    inputs = [(1e-3, 40.,400.,4000.,1458.), (1.,40.,400.,4000.,1458.),
              (50.,40.,400.,3250.,1458.), (200.,10.,4.,100.,.8)]
    profile_records = []
    for i, args in enumerate(inputs):
        p = asymmetric_profile(*args)
        values = np.array([p.evaluate(t) for t in np.linspace(0,p.duration,10001)])
        check(f"profile_{i}_endpoint_integral", np.max(np.abs(p.states[-1]-[args[0],0.,0.])), 1e-10)
        check(f"profile_{i}_nonnegative_durations", max(0.,-p.durations.min()), 0.)
        check(f"profile_{i}_nonnegative_speed", max(0.,-values[:,1].min()), 1e-10)
        check(f"profile_{i}_speed_limit", max(0.,values[:,1].max()-args[1]), 1e-10)
        check(f"profile_{i}_acceleration_limit", max(0.,np.abs(values[:,2]).max()-args[2]), 1e-10)
        check(f"profile_{i}_event_jerk_limit", max(0.,np.abs(p.jerks).max()-max(args[3:])), 0.)
        residuals = [np.max(np.abs(advance(p.states[q],p.jerks[q],p.durations[q])-p.states[q+1])) for q in range(7)]
        check(f"profile_{i}_phase_continuity", max(residuals), 1e-12)
        profile_records.append(dict(inputs=list(args), duration=p.duration, peak_speed=p.peak_speed,
                                    phase_durations=p.durations.tolist(), events=p.events.tolist(), states=p.states.tolist(), jerks=p.jerks.tolist()))

    # Four geometric adjacency types, both turn signs. These are prescribed
    # property cases, not the unavailable Fig. 7 numerical reproduction.
    p1 = asymmetric_profile(5.,10.,40.,400.,200.)
    p2 = asymmetric_profile(3.,10.,40.,170.,300.)
    for k1 in [0., .15, -.15]:
        for k2 in [0., .2, -.2]:
            first = Primitive(np.array([1.,2.]), .2, k1)
            second = Primitive(first.evaluate([5.,0.,0.,0.])[0], -.4, k2)
            ov = .3*min(p1.duration,p2.duration)
            total = p1.duration+p2.duration-ov
            start = overlap(first,p1,second,p2,ov,0.)
            end = overlap(first,p1,second,p2,ov,total)
            check(f"overlap_{k1}_{k2}_endpoints", max(np.max(np.abs(start[0]-first.origin)), np.max(np.abs(end[0]-second.evaluate([3.,0.,0.,0.])[0]))),1e-12)
            check(f"overlap_{k1}_{k2}_rest_boundaries", max(np.max(np.abs(start[1:3])),np.max(np.abs(end[1:]))),1e-12)
            time = p1.duration-.5*ov
            sample = overlap(first,p1,second,p2,ov,time)
            minus = overlap(first,p1,second,p2,ov,time-h)
            plus = overlap(first,p1,second,p2,ov,time+h)
            check(f"overlap_{k1}_{k2}_derivative_consistency",np.max(np.abs((plus[:3]-minus[:3])/(2*h)-sample[1:])),2e-5)

    return dict(status="passed" if all(row["passed"] for row in records) else "failed",
                scope="Independent mathematical kernel properties; not original examples, optimizer reproduction, corridor feasibility, or project baseline results.",
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                test_count=len(records), tests=records, scalar_profiles=profile_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = selftest()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"status":report["status"],"test_count":report["test_count"],"output":str(options.output)}))
    raise SystemExit(0 if report["status"] == "passed" else 1)
