"""Common 2-D command-vector audit and finite terminal brake.

Only Euclidean translational velocity, acceleration and jerk are constrained.
Angular derivatives and scalar feed derivatives are diagnostics. All dynamics
are reconstructed from the complete position commands by backward differences;
normal acceleration/jerk at turns are therefore included. Historical six-channel
sources and results are not modified or re-labelled by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PPO_project"))
from src.utils.revision_metrics import evaluate_trajectory, independent_kinematics

VERSION = "planar-vector-command-metrics-v1"
LIMITS = dict(MAX_VEL=100., MAX_ACC=2000., MAX_JERK=62500.)
CHANNELS = dict(planar_velocity_mm_s=("MAX_VEL", 1),
                planar_acceleration_mm_s2=("MAX_ACC", 2),
                planar_jerk_mm_s3=("MAX_JERK", 3))
DEFAULT_REST_POLICY = dict(window_s=.020, feed_velocity_mm_s=.001,
                           feed_acceleration_mm_s2=.01)


def _limits(value):
    limits = {name: float(value[name]) for name in LIMITS}
    if any(not np.isfinite(x) or x <= 0 for x in limits.values()):
        raise ValueError("Three finite positive planar motion limits are required")
    return limits


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(name + " must be a finite two-component vector")
    return array


def independent_planar_kinematics(positions, timestamps, *, initial_velocity=(0., 0.),
        initial_acceleration=(0., 0.), initial_heading=0., initial_angular_velocity=0.,
        initial_angular_acceleration=0.):
    """Original aligned vector differences, with explicit Euclidean norms.

Initial velocity/acceleration are physical boundary conditions, not extrapolated
samples. Jerk at the initial sample is unavailable and never invented.
"""
    initial_velocity = _vector(initial_velocity, "initial_velocity")
    initial_acceleration = _vector(initial_acceleration, "initial_acceleration")
    kin = independent_kinematics(positions, timestamps,
        initial_velocity=initial_velocity, initial_acceleration=initial_acceleration,
        initial_heading=initial_heading, initial_angular_velocity=initial_angular_velocity,
        initial_angular_acceleration=initial_angular_acceleration)
    for planar, axis in (("planar_velocity_mm_s", "axis_velocity_mm_s"),
                         ("planar_acceleration_mm_s2", "axis_acceleration_mm_s2"),
                         ("planar_jerk_mm_s3", "axis_jerk_mm_s3")):
        kin[planar] = np.linalg.norm(kin[axis], axis=1)
    return kin


def _check(values, limit, *, rtol=1e-5):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    magnitudes = np.abs(values[finite])
    violations = magnitudes > limit * (1 + rtol) + 1e-8
    return dict(peak=float(magnitudes.max()) if len(magnitudes) else None,
        limit=float(limit), checked_samples=len(values), finite_samples=int(finite.sum()),
        violating_samples=int(violations.sum()),
        violation_fraction=float(violations.mean()) if len(magnitudes) else None,
        satisfied=bool(len(values) and finite.all() and not violations.any()))


def _diagnostic(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return dict(peak=float(np.abs(values[finite]).max()) if finite.any() else None,
                finite_samples=int(finite.sum()), limit=None, enforced=False)


def evaluate_planar_terminal_trajectory(reference: Any, positions: Any, timestamps: Any, *,
        tolerance_mm: float, limits: Mapping[str, Any], initial_heading: float,
        initial_velocity: Any = (0., 0.), initial_acceleration: Any = (0., 0.),
        initial_angular_velocity: float | None = 0.,
        initial_angular_acceleration: float | None = 0.,
        rest_policy: Mapping[str, Any] | None = None,
        termination_reason: str = "completed", claimed_success: bool = True,
        witness_spacing_mm: float | None = None):
    """Unchanged ordered/tolerance audit plus three vector limits and full rest.

The legacy evaluator is used with an empty limit mapping only for geometry,
termination and trace validation. Its scalar/angle constraint fields are then
replaced, never disabled with infinities. Angular rest is diagnostic only.
"""
    active_limits = _limits(limits)
    initial = dict(initial_velocity=initial_velocity, initial_acceleration=initial_acceleration,
        initial_heading=initial_heading, initial_angular_velocity=initial_angular_velocity,
        initial_angular_acceleration=initial_angular_acceleration)
    result = evaluate_trajectory(reference, positions, timestamps,
        tolerance_mm=tolerance_mm, limits={}, termination_reason=termination_reason,
        claimed_success=claimed_success, constraint_relative_tolerance=1e-5,
        witness_spacing_mm=witness_spacing_mm, **initial)
    policy = dict(DEFAULT_REST_POLICY)
    if rest_policy is not None:
        unknown = set(rest_policy) - set(policy) - {"angular_velocity_rad_s", "angular_acceleration_rad_s2"}
        if unknown:
            raise ValueError("Unknown planar rest fields: " + str(sorted(unknown)))
        policy.update({key: float(value) for key, value in rest_policy.items() if key in policy})
    if any(not np.isfinite(value) or value <= 0 for value in policy.values()):
        raise ValueError("Planar rest thresholds and window must be finite and positive")
    result.update(schema_version=VERSION, terminal_metrics_version=VERSION,
        active_limits=active_limits, angular_constraints_enforced=False,
        planar_constraint_definition="Euclidean norms of complete 2-D position backward differences, including normal dynamics",
        rest_policy=policy, terminal_rest_checked=True, terminal_rest_satisfied=False,
        full_rest_satisfied=False, terminal_rest_checks={}, rest_failure_reasons=[],
        axis_constraints_implemented_by_evaluator=False,
        derivative_alignment="Backward 2-D differences at t[k]; explicit initial v/a; first jerk unavailable; discrete commands, not continuous servo guarantees")
    if not result["valid_trace"]:
        return result
    try:
        kin = independent_planar_kinematics(positions, timestamps, **initial)
    except (ValueError, TypeError) as exc:
        result.update(valid_trace=False, success=False, invalid_trace_detail=str(exc))
        result["failure_reasons"].append("invalid_planar_initial_state")
        return result
    # Only the initial jerk is structurally unavailable. Missing later values
    # are failures, not silently dropped finite samples.
    checks = {key: _check(kin[key][1:] if order == 3 else kin[key], active_limits[name])
              for key, (name, order) in CHANNELS.items()}
    result["constraints"] = checks
    result["kinematics_diagnostics"] = {key: _diagnostic(kin[key]) for key in
        ("feed_mm_s", "feed_acceleration_mm_s2", "feed_jerk_mm_s3",
         "angular_velocity_rad_s", "angular_acceleration_rad_s2", "angular_jerk_rad_s3")}
    reasons = list(result["failure_reasons"])
    reasons.extend("constraint_" + key for key, check in checks.items() if not check["satisfied"])
    result["geometric_arrival_success"] = bool(result["path_completed"] and result["geometry_feasible"])
    t = kin["timestamp_s"]
    roundoff = 16 * np.finfo(float).eps * max(1., abs(float(t[-1])))
    boundary = float(t[-1] - policy["window_s"])
    start = max(0, int(np.searchsorted(t, boundary + roundoff, side="right")) - 1)
    duration = float(t[-1] - t[start])
    complete = bool(len(t)-start >= 2 and duration + roundoff >= policy["window_s"])
    result["terminal_rest_window"] = dict(requested_duration_s=policy["window_s"],
        observed_duration_s=duration, start_index=start, start_timestamp_s=float(t[start]),
        end_timestamp_s=float(t[-1]), sample_count=len(t)-start, complete=complete)
    rest_checks = {key: _check(kin[key][start:], threshold, rtol=0.) for key, threshold in
        (("planar_velocity_mm_s", policy["feed_velocity_mm_s"]),
         ("planar_acceleration_mm_s2", policy["feed_acceleration_mm_s2"]))}
    for key, check in rest_checks.items():
        values = kin[key][start:]
        # Physical rest thresholds have no constraint tolerance or additive slack.
        check["satisfied"] = bool(len(values) and np.isfinite(values).all() and np.all(values <= check["limit"]))
        check["window_peak"] = check["peak"]
        check["terminal_value"] = float(values[-1]) if np.isfinite(values[-1]) else None
    result["terminal_rest_checks"] = rest_checks
    result["terminal_angular_diagnostics"] = {key: _diagnostic(kin[key][start:]) for key in
        ("angular_velocity_rad_s", "angular_acceleration_rad_s2", "angular_jerk_rad_s3")}
    rest_reasons = [] if complete else ["rest_window_incomplete"]
    rest_reasons.extend("rest_" + key for key, check in rest_checks.items() if not check["satisfied"])
    reasons.extend(rest_reasons)
    result.update(success=not reasons, failure_reasons=reasons, rest_failure_reasons=rest_reasons,
        terminal_rest_satisfied=not rest_reasons, full_rest_satisfied=not rest_reasons,
        terminal_position_definition="Within unchanged endpoint tolerance; no exact endpoint stop claim.")
    return result


@dataclass(frozen=True)
class PlanarBrakingPolicy:
    hold_s: float = .020
    max_braking_s: float = 2.
    limit_fraction: float = .98


def discrete_planar_stop_profile(velocity, acceleration, dt, count):
    """Vector cubic samples matching v[-1]=v[0]-a[0]dt and v[n-1]=v[n]=0."""
    velocity, acceleration = _vector(velocity, "velocity"), _vector(acceleration, "acceleration")
    if int(count) != count or count < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Need positive finite dt and >=2 integer braking steps")
    n = int(count)
    k = np.arange(n+1, dtype=float)[:, None]
    c = velocity/(n*(n-1))
    d = c-(velocity-acceleration*dt)/(n*(n+1))
    v = (n-k)*(n-1-k)*(c+d*k)
    v[0], v[-2:] = velocity, 0.
    a = np.vstack([acceleration, np.diff(v, axis=0)/dt])
    j = np.vstack([np.full(2, np.nan), np.diff(a, axis=0)/dt])
    return v, a, j


def plan_planar_terminal_brake(position, velocity_xy, acceleration_xy, *, dt, limits, policy=None):
    """Finite vector brake including unchanged splice; never repairs a prefix.

The first feasible duration of a fixed cubic family is selected using v/a/j
vector limits. Splice normal acceleration is preserved, so a curved state may
have a short curved stop; a straight state remains straight. There is no snap,
heading hard reset, extra angle limit, or per-path performance parameter.
The caller must audit the whole joined command, including geometric tolerance.
"""
    position = _vector(position, "position")
    velocity = _vector(velocity_xy, "velocity_xy")
    acceleration = _vector(acceleration_xy, "acceleration_xy")
    limits = _limits(limits)
    policy = PlanarBrakingPolicy() if policy is None else policy
    if (not np.isfinite(dt) or dt <= 0 or not 0 < policy.limit_fraction <= 1
            or not np.isfinite(policy.max_braking_s) or policy.max_braking_s <= 0
            or not np.isfinite(policy.hold_s) or policy.hold_s < .020):
        raise ValueError("Invalid planar braking dt/policy; >=20ms hold is mandatory")
    if np.linalg.norm(velocity) > limits["MAX_VEL"]*(1+1e-5)+1e-8 or np.linalg.norm(acceleration) > limits["MAX_ACC"]*(1+1e-5)+1e-8:
        raise ValueError("Infeasible planar splice state; a brake cannot repair prefix violations")
    profile = None
    maximum = int(math.floor(policy.max_braking_s/dt))
    for count in range(2, maximum+1):
        v, a, j = discrete_planar_stop_profile(velocity, acceleration, dt, count)
        nonreversal = bool(np.linalg.norm(velocity) <= 1e-12 or np.all(v @ velocity >= -1e-12))
        if (nonreversal and np.linalg.norm(v[1:], axis=1).max() <= limits["MAX_VEL"]
                and np.linalg.norm(a[1:], axis=1).max() <= limits["MAX_ACC"]
                and np.linalg.norm(j[1:], axis=1).max() <= limits["MAX_JERK"]*policy.limit_fraction):
            profile = v, a, j
            break
    if profile is None:
        raise ValueError("No feasible nonreversing vector cubic brake within registered horizon")
    hold_steps = int(math.ceil(policy.hold_s/dt))+3
    size = count+hold_steps+1
    v, a, j = [np.pad(x, ((0, size-len(x)), (0, 0))) for x in profile]
    positions = np.vstack([position, position+np.cumsum(v[1:]*dt, axis=0)])
    columns = dict(time_s=np.arange(size)*dt, x_mm=positions[:, 0], y_mm=positions[:, 1],
        velocity_x_mm_s=v[:, 0], velocity_y_mm_s=v[:, 1],
        acceleration_x_mm_s2=a[:, 0], acceleration_y_mm_s2=a[:, 1],
        jerk_x_mm_s3=j[:, 0], jerk_y_mm_s3=j[:, 1],
        v_mm_s=np.linalg.norm(v, axis=1), a_norm_mm_s2=np.linalg.norm(a, axis=1),
        j_norm_mm_s3=np.linalg.norm(j, axis=1))
    return dict(version=VERSION, columns=columns, braking_steps=count, hold_steps=hold_steps,
        braking_time_s=count*dt, total_tail_time_s=(size-1)*dt,
        internal_terminal_rest=bool(np.all(v[-1] == 0.) and np.all(a[-1] == 0.)),
        angular_constraints_enforced=False, selection="First feasible duration in fixed bounded vector cubic family",
        endpoint_policy="Full joined output must pass original geometric tolerance; no exact endpoint snap")
