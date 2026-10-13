"""Independent planar KCM with an inward normal-candidate roundoff reserve.

This source is not connected to an environment or policy. Safety checks use the
original disk radii and an explicit 1e-10 velocity-unit floating-point tolerance;
Input/backup checks and every physical limit remain unchanged. Only normal
projection/line-search acceptance is tightened; see design.md for the binary64
32-step error budget. This draft has not undergone numerical validation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np


DT = .001
VMAX = 100.
AMAX_INTERNAL = 1998.
JMAX_INTERNAL = 62437.5
AMAX_PHYSICAL = 2000.
JMAX_PHYSICAL = 62500.
VELOCITY_TOLERANCE = 1e-10
PHYSICAL_RATIO_TOLERANCE = 1.000001
MAX_HALVINGS = 24
MAX_BISECTIONS = 16
MAX_RELEASE_TERMS = 32
# These are inward normal-candidate margins, never tolerance/radius increases.
NORMAL_VELOCITY_MARGIN = 1e-8  # mm/s, for current and release endpoint velocity
NORMAL_ACCELERATION_MARGIN = 1e-7  # mm/s^2, guarantees 32-step release termination

_DISKS_PATH = Path(__file__).resolve().parents[1] / "kcm_projection_audit_v1/disks.py"
_DISKS_SHA = "bd831775ec019f8898324f57f1960ea7b32805d63fe48e0051d1235a0346cbcd"
if hashlib.sha256(_DISKS_PATH.read_bytes()).hexdigest() != _DISKS_SHA:
    raise ValueError("The audited finite-disk projection source changed")
_spec = importlib.util.spec_from_file_location("recursive_kcm_audited_disks_v1", _DISKS_PATH)
_disks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_disks)


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite planar vector")
    return result


def _norm(value):
    result = math.hypot(float(value[0]), float(value[1]))
    if not math.isfinite(result):
        raise ValueError("Nonfinite planar vector norm")
    return result


def _dt(value):
    if not math.isfinite(value) or value != DT:
        raise ValueError("This independent projector requires the original 1 ms timestep")
    return value


def _clip(vector, radius):
    length = _norm(vector)
    return vector.copy() if length <= radius else vector * (radius / length)


def release_endpoint(velocity, acceleration, dt=DT):
    """Velocity after reducing acceleration by J*dt per subsequent step.

All release velocities lie on the segment from current velocity to this
endpoint in real arithmetic. Each implemented candidate is checked separately.
"""
    velocity = _vector(velocity, "velocity")
    acceleration = _vector(acceleration, "acceleration")
    dt = _dt(dt)
    magnitude = _norm(acceleration)
    if magnitude == 0.:
        return velocity.copy()
    if (magnitude - AMAX_INTERNAL) * dt > VELOCITY_TOLERANCE:
        raise ValueError("Acceleration exceeds the registered release-sum domain")
    step = JMAX_INTERNAL * dt
    # A_internal/(J_internal*dt)=32. The positive-term form is the same
    # finite release sum without cancellation or a floor decision at an ULP
    # boundary. The explicit input roundoff allowance cannot create term 33.
    delta = dt * math.fsum(max(magnitude - k * step, 0.) for k in range(1, MAX_RELEASE_TERMS + 1))
    if not math.isfinite(delta) or delta < 0.:
        raise ValueError("Invalid acceleration-release sum")
    return velocity + (delta / magnitude) * acceleration


def _state_residuals(velocity, acceleration, dt):
    endpoint = release_endpoint(velocity, acceleration, dt)
    return dict(velocity=_norm(velocity) - VMAX,
                acceleration_in_velocity_units=(_norm(acceleration) - AMAX_INTERNAL) * dt,
                release_endpoint=_norm(endpoint) - VMAX), endpoint


def _transition(velocity, acceleration, candidate, centers, radii, dt):
    """Check original disks, original physical ratios, and release reserve."""
    next_a = (candidate - velocity) / dt
    next_j = (next_a - acceleration) / dt
    residuals = np.linalg.norm(candidate[None, :] - centers, axis=1) - radii
    ratios = np.array([_norm(candidate) / VMAX, _norm(next_a) / AMAX_PHYSICAL,
                       _norm(next_j) / JMAX_PHYSICAL])
    state_residuals, endpoint = _state_residuals(candidate, next_a, dt)
    finite = bool(np.isfinite(np.r_[candidate, next_a, next_j, residuals, ratios, endpoint]).all())
    disks_ok = finite and float(residuals.max()) <= VELOCITY_TOLERANCE
    physical_ok = finite and float(ratios.max()) <= PHYSICAL_RATIO_TOLERANCE
    reserve_ok = finite and max(state_residuals.values()) <= VELOCITY_TOLERANCE
    normal_reserve_ok = finite and (
        state_residuals["velocity"] <= -NORMAL_VELOCITY_MARGIN
        and state_residuals["release_endpoint"] <= -NORMAL_VELOCITY_MARGIN
        and state_residuals["acceleration_in_velocity_units"] <= -NORMAL_ACCELERATION_MARGIN * dt
    )
    return dict(velocity=candidate.copy(), acceleration=next_a, jerk=next_j,
                release_endpoint=endpoint, disk_residuals=residuals,
                physical_ratios=ratios, state_residuals=state_residuals,
                disks_ok=disks_ok, physical_ok=physical_ok, reserve_ok=reserve_ok,
                safe=disks_ok and physical_ok and reserve_ok,
                normal_reserve_ok=normal_reserve_ok,
                normal_safe=disks_ok and physical_ok and reserve_ok and normal_reserve_ok)


def enforce_kinematic_constraints(velocity, acceleration, desired_jerk, dt=DT):
    """Return next v/a/j and small diagnostics without changing original limits.

The disk projection targets the original clipped jerk/acceleration request.
If that projected state lacks the inward normal reserve, search from a separately
verified release backup toward it. Only previously verified safe points can be
returned; no claim is made that the chosen lambda is globally maximal.
"""
    velocity = _vector(velocity, "velocity")
    acceleration = _vector(acceleration, "acceleration")
    desired_jerk = _vector(desired_jerk, "desired_jerk")
    dt = _dt(dt)
    input_residuals, input_endpoint = _state_residuals(velocity, acceleration, dt)
    if max(input_residuals.values()) > VELOCITY_TOLERANCE:
        raise ValueError("Input state has no verified acceleration-release reserve: " + str(input_residuals))

    centers = np.asarray([np.zeros(2), velocity, velocity + acceleration * dt])
    radii = np.asarray([VMAX, AMAX_INTERNAL * dt, JMAX_INTERNAL * dt * dt])
    magnitude = _norm(acceleration)
    backup_a = acceleration * (max(magnitude - JMAX_INTERNAL * dt, 0.) / magnitude) if magnitude else np.zeros(2)
    backup = _transition(velocity, acceleration, velocity + dt * backup_a, centers, radii, dt)
    if not backup["safe"]:
        raise ValueError("Release backup failed numerical checks; limits unchanged: " + str(dict(
            disk_residuals=backup["disk_residuals"].tolist(), physical_ratios=backup["physical_ratios"].tolist(),
            state_residuals=backup["state_residuals"])))

    candidate_a = _clip(acceleration + dt * _clip(desired_jerk, JMAX_INTERNAL), AMAX_INTERNAL)
    request = velocity + dt * candidate_a
    projection = _disks.project_disks(request, centers, radii, tolerance=VELOCITY_TOLERANCE)
    projected, projection_failure = None, None
    if not projection["feasible_within_tolerance"]:
        projection_failure = "finite_disk_enumeration_found_no_candidate"
    else:
        try:
            projected = _transition(velocity, acceleration, np.asarray(projection["selected"]["point"], float), centers, radii, dt)
            if not projected["disks_ok"] or not projected["physical_ok"]:
                projection_failure = "selected_projection_failed_original_disk_or_physical_checks"
        except ValueError as error:
            projection_failure = "selected_projection_numerical_check_failed: " + str(error)

    result, chosen_lambda = (backup, 0.) if projection_failure else (projected, 1.)
    halvings = bisections = 0
    if not projection_failure and not projected["normal_safe"]:
        result, chosen_lambda = backup, 0.
        upper = 1.
        direction = projected["velocity"] - backup["velocity"]
        for _ in range(MAX_HALVINGS):
            trial_lambda = upper / 2.
            try:
                trial = _transition(velocity, acceleration, backup["velocity"] + trial_lambda * direction, centers, radii, dt)
            except ValueError:
                trial = {"normal_safe": False}
            halvings += 1
            if trial["normal_safe"]:
                result, chosen_lambda = trial, trial_lambda
                break
            upper = trial_lambda
        if chosen_lambda > 0.:
            for _ in range(MAX_BISECTIONS):
                trial_lambda = (chosen_lambda + upper) / 2.
                try:
                    trial = _transition(velocity, acceleration, backup["velocity"] + trial_lambda * direction, centers, radii, dt)
                except ValueError:
                    trial = {"normal_safe": False}
                bisections += 1
                if trial["normal_safe"]:
                    result, chosen_lambda = trial, trial_lambda
                else:
                    upper = trial_lambda

    if not result["safe"]:
        raise ValueError("No verified safe result; original limits were not relaxed")
    diagnostics = dict(guard_used=chosen_lambda < 1., guard_lambda=chosen_lambda,
        input_release_endpoint_norm_mm_s=_norm(input_endpoint),
        input_release_endpoint_residual_mm_s=input_residuals["release_endpoint"],
        projected_release_endpoint_norm_mm_s=_norm(projected["release_endpoint"]) if projected is not None else None,
        release_endpoint_norm_mm_s=_norm(result["release_endpoint"]),
        release_endpoint_residual_mm_s=result["state_residuals"]["release_endpoint"],
        backup_velocity_mm_s=backup["velocity"].tolist(), backup_acceleration_mm_s2=backup["acceleration"].tolist(),
        backup_release_endpoint_norm_mm_s=_norm(backup["release_endpoint"]), backup_verified=True,
        disk_residuals_mm_s=result["disk_residuals"].tolist(), physical_ratios=result["physical_ratios"].tolist(),
        halving_checks=halvings, bisection_checks=bisections,
        velocity_roundoff_tolerance_mm_s=VELOCITY_TOLERANCE,
        physical_ratio_tolerance=PHYSICAL_RATIO_TOLERANCE,
        normal_velocity_margin_mm_s=NORMAL_VELOCITY_MARGIN,
        normal_acceleration_margin_mm_s2=NORMAL_ACCELERATION_MARGIN,
        selected_normal_reserve_ok=bool(result["normal_reserve_ok"]),
        projection_candidate_kind=projection["selected"]["kind"] if projection["selected"] else None,
        projection_candidates=projection["candidate_count"], projection_failure=projection_failure,
        line_search_is_global_optimum=False)
    return result["velocity"], result["acceleration"], result["jerk"], diagnostics
