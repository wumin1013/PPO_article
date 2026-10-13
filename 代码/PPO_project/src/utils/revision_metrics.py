"""Independent ordered-geometry and output-command audit; no controller internals.

Position p[k] is a command at t[k]. Backward differences are aligned at t[k],
matching the environment's endpoint (semi-implicit Euler) integration convention.
These are discrete command derivatives, not continuous servo-motion guarantees.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .comparison_metrics import project_points_to_polyline


def _trajectory(positions: Any, timestamps: Any) -> tuple[np.ndarray, np.ndarray]:
    p, t = np.asarray(positions, float), np.asarray(timestamps, float)
    if p.ndim != 2 or p.shape[1] != 2 or t.ndim != 1 or len(p) != len(t) or len(t) < 2:
        raise ValueError("Need matching Nx2 command positions and N timestamps, including initial point")
    if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Command positions/timestamps must be finite, with strictly increasing timestamps")
    return p, t


def _backward(values: np.ndarray, t: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    dt = np.diff(t)
    output[1:] = np.diff(values, axis=0) / (dt[:, None] if values.ndim == 2 else dt)
    return output


def independent_kinematics(positions: Any, timestamps: Any, *, initial_velocity: Any = None,
                           initial_acceleration: Any = None, initial_heading: float | None = None,
                           initial_angular_velocity: float | None = None,
                           initial_angular_acceleration: float | None = None) -> dict[str, np.ndarray]:
    p, t = _trajectory(positions, timestamps)
    axis_v = _backward(p, t)
    if initial_velocity is not None:
        axis_v[0] = np.asarray(initial_velocity, float)
    axis_a = _backward(axis_v, t)
    if initial_acceleration is not None:
        axis_a[0] = np.asarray(initial_acceleration, float)
    axis_j = _backward(axis_a, t)
    feed = np.linalg.norm(axis_v, axis=1)
    acceleration = _backward(feed, t)
    if initial_velocity is not None and initial_acceleration is not None:
        v0 = np.asarray(initial_velocity, float)
        acceleration[0] = float(np.asarray(initial_acceleration, float) @ v0 / max(np.linalg.norm(v0), 1e-30))
    jerk = _backward(acceleration, t)
    heading = np.full(len(t), np.nan)
    moving = np.isfinite(feed) & (feed > 1e-9)
    if initial_heading is not None:
        heading[0] = float(initial_heading)
    heading[moving] = np.arctan2(axis_v[moving, 1], axis_v[moving, 0])
    # Carry the last known heading at rest; do not invent a heading before first motion.
    for i in range(1, len(t)):
        if not np.isfinite(heading[i]):
            heading[i] = heading[i - 1]
    valid = np.isfinite(heading)
    heading[valid] = np.unwrap(heading[valid])
    angular_v = _backward(heading, t)
    if initial_angular_velocity is not None:
        angular_v[0] = float(initial_angular_velocity)
    angular_a = _backward(angular_v, t)
    if initial_angular_acceleration is not None:
        angular_a[0] = float(initial_angular_acceleration)
    angular_j = _backward(angular_a, t)
    return {"timestamp_s": t, "axis_velocity_mm_s": axis_v, "axis_acceleration_mm_s2": axis_a,
            "axis_jerk_mm_s3": axis_j, "feed_mm_s": feed, "feed_acceleration_mm_s2": acceleration,
            "feed_jerk_mm_s3": jerk, "heading_rad": heading, "angular_velocity_rad_s": angular_v,
            "angular_acceleration_rad_s2": angular_a, "angular_jerk_rad_s3": angular_j}


def _reference_witnesses(reference: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.diff(reference, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("Reference contains adjacent repeated points")
    witnesses, arcs = [reference[0]], [0.0]
    total = 0.0
    for a, vector, length in zip(reference[:-1], vectors, lengths):
        n = max(1, int(np.ceil(length / spacing)))
        for fraction in np.linspace(0, 1, n + 1)[1:]:
            witnesses.append(a + fraction * vector)
            arcs.append(total + fraction * length)
        total += length
    return np.asarray(witnesses), np.asarray(arcs)


def ordered_projection(reference: Any, positions: Any, *, tolerance_mm: float,
                       witness_spacing_mm: float | None = None) -> dict[str, Any]:
    """Require reference witnesses to be visited in order by the command polyline.

    Each witness is matched in increasing command-segment time using a point-to-
    segment tolerance interval. Backtracking cannot award a witness twice; a
    closed-path start match cannot wrap progress to one. Witness spacing/2 is
    subtracted from the matching radius to conservatively cover between witnesses.
    This is an order/coverage certificate, not an exact continuous Frechet solver.
    """
    reference, p = np.asarray(reference, float), np.asarray(positions, float)
    if tolerance_mm <= 0 or reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2:
        raise ValueError("Need a 2-D reference and positive one-sided tolerance")
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 1 or not np.isfinite(p).all():
        raise ValueError("Need finite Nx2 trajectory")
    spacing = min(.25, tolerance_mm / 2) if witness_spacing_mm is None else float(witness_spacing_mm)
    if not 0 < spacing < 2 * tolerance_mm:
        raise ValueError("Witness spacing must be positive and smaller than twice tolerance")
    radius = tolerance_mm - spacing / 2
    witnesses, arcs = _reference_witnesses(reference, spacing)
    frontier, progress = 0, np.zeros(len(p))
    segment_indices = np.zeros(len(p), dtype=int)
    for k in range(len(p)):
        a = p[max(0, k - 1)]
        vector = p[k] - a
        length2 = float(vector @ vector)
        earliest = 0.0
        while frontier < len(witnesses):
            relative = witnesses[frontier] - a
            if length2 <= 1e-24:
                if np.linalg.norm(relative) > radius + 1e-12:
                    break
            else:
                center = float(relative @ vector / length2)
                perpendicular2 = float(relative @ relative - center * center * length2)
                discriminant = radius * radius - max(0.0, perpendicular2)
                if discriminant < -1e-12:
                    break
                half_width = np.sqrt(max(0.0, discriminant) / length2)
                lower, upper = max(0.0, center - half_width), min(1.0, center + half_width)
                earliest = max(earliest, lower)
                if earliest > upper + 1e-12:
                    break
            frontier += 1
        covered_s = arcs[frontier - 1] if frontier else 0.0
        progress[k] = float(covered_s / arcs[-1])
        segment_indices[k] = frontier
    nearest = project_points_to_polyline(p, reference, closed=False)
    return {"contour_error_mm": nearest.contour_error, "ordered_progress": progress,
            "witness_frontier": segment_indices, "reference_length_mm": float(arcs[-1]),
            "witness_count": len(witnesses), "covered_witness_count": frontier,
            "witness_spacing_mm": spacing, "witness_match_radius_mm": radius,
            "ordered_coverage_complete": bool(frontier == len(witnesses)),
            "start_error_mm": float(np.linalg.norm(p[0] - reference[0])),
            "endpoint_error_mm": float(np.linalg.norm(p[-1] - reference[-1]))}


def _peak(array: np.ndarray) -> float | None:
    finite = np.abs(array[np.isfinite(array)])
    return float(finite.max()) if len(finite) else None


def _limit_check(values: np.ndarray, limit: float | None, rtol: float) -> dict[str, Any]:
    finite = np.abs(values[np.isfinite(values)])
    available = limit is not None and float(limit) > 0
    violating = finite > float(limit) * (1 + rtol) + 1e-8 if available else np.zeros(len(finite), bool)
    return {"peak": _peak(values), "limit": float(limit) if available else None,
            "checked_samples": len(finite), "violating_samples": int(violating.sum()) if available else None,
            "violation_fraction": float(violating.mean()) if available and len(finite) else None,
            "satisfied": bool(not violating.any()) if available and len(finite) else None}


def evaluate_trajectory(reference: Any, positions: Any, timestamps: Any, *, tolerance_mm: float,
                        limits: Mapping[str, Any] | None = None, termination_reason: str = "",
                        claimed_success: bool | None = None, final_speed_tolerance_mm_s: float | None = None,
                        initial_velocity: Any = None, initial_acceleration: Any = None,
                        initial_heading: float | None = None, initial_angular_velocity: float | None = None,
                        initial_angular_acceleration: float | None = None,
                        constraint_relative_tolerance: float = 1e-5,
                        witness_spacing_mm: float | None = None) -> dict[str, Any]:
    """Never discard failed rollouts. Missing/invalid traces return an explicit failure."""
    try:
        p, t = _trajectory(positions, timestamps)
        projection_options = {} if witness_spacing_mm is None else {"witness_spacing_mm": witness_spacing_mm}
        geometry = ordered_projection(reference, p, tolerance_mm=tolerance_mm, **projection_options)
        kin = independent_kinematics(p, t, initial_velocity=initial_velocity,
                                    initial_acceleration=initial_acceleration, initial_heading=initial_heading,
                                    initial_angular_velocity=initial_angular_velocity,
                                    initial_angular_acceleration=initial_angular_acceleration)
    except (ValueError, TypeError, IndexError) as exc:
        return {"valid_trace": False, "success": False, "path_completed": False,
                "failure_reasons": ["invalid_trace"], "invalid_trace_detail": str(exc),
                "termination_reason": termination_reason, "claimed_success": claimed_success}
    limits = dict(limits or {})
    map_limits = {"feed_mm_s": "MAX_VEL", "feed_acceleration_mm_s2": "MAX_ACC", "feed_jerk_mm_s3": "MAX_JERK",
                  "angular_velocity_rad_s": "MAX_ANG_VEL", "angular_acceleration_rad_s2": "MAX_ANG_ACC",
                  "angular_jerk_rad_s3": "MAX_ANG_JERK"}
    checks = {key: _limit_check(kin[key], limits.get(limit_name), constraint_relative_tolerance)
              for key, limit_name in map_limits.items()}
    axis_checks = {}
    for suffix, limit_name in [("velocity_mm_s", "AXIS_MAX_VEL"), ("acceleration_mm_s2", "AXIS_MAX_ACC"), ("jerk_mm_s3", "AXIS_MAX_JERK")]:
        limit = limits.get(limit_name)
        axis_limits = [None, None] if limit is None else np.broadcast_to(np.asarray(limit, float), (2,))
        for axis, axis_name in enumerate(["x", "y"]):
            axis_checks[f"{axis_name}_{suffix}"] = _limit_check(kin[f"axis_{suffix}"][:, axis], axis_limits[axis], constraint_relative_tolerance)
    error = geometry["contour_error_mm"]
    geometry_ok = bool(np.max(error) <= tolerance_mm + 1e-8)
    completed = bool(geometry["ordered_coverage_complete"] and geometry["endpoint_error_mm"] <= tolerance_mm and geometry["start_error_mm"] <= tolerance_mm)
    terminal_feed = float(kin["feed_mm_s"][-1])
    stop_ok = None if final_speed_tolerance_mm_s is None else bool(terminal_feed <= final_speed_tolerance_mm_s)
    reasons = []
    if not completed:
        reasons.append("ordered_path_incomplete")
    if not geometry_ok:
        reasons.append("contour_tolerance_exceeded")
    if stop_ok is False:
        reasons.append("terminal_speed_exceeded")
    for key, check in checks.items():
        if check["satisfied"] is False:
            reasons.append(f"constraint_{key}")
    for key, check in axis_checks.items():
        if check["satisfied"] is False:
            reasons.append(f"constraint_{key}")
    if termination_reason and termination_reason not in {"success", "completed", "path_completed", "done"}:
        reasons.append(f"termination_{termination_reason}")
    if claimed_success is False:
        reasons.append("environment_reported_failure")
    return {"schema_version": "revision-metrics-v1", "valid_trace": True, "success": not reasons,
            "path_completed": completed, "ordered_progress": float(geometry["ordered_progress"][-1]),
            "geometry_feasible": geometry_ok, "failure_reasons": reasons,
            "termination_reason": termination_reason, "claimed_success": claimed_success,
            "elapsed_time_s": float(t[-1] - t[0]), "sample_count": len(t),
            "contour_error_max_mm": float(np.max(error)), "contour_error_mean_mm": float(np.mean(error)),
            "contour_error_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
            "reference_length_mm": geometry["reference_length_mm"],
            "command_travel_mm": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
            "start_error_mm": geometry["start_error_mm"], "endpoint_error_mm": geometry["endpoint_error_mm"],
            "terminal_feed_mm_s": terminal_feed, "terminal_rest_checked": stop_ok is not None,
            "terminal_rest_satisfied": stop_ok, "constraints": checks, "axis_diagnostics": axis_checks,
            "axis_constraints_implemented_by_evaluator": False,
            "derivative_alignment": "Backward differences at t[k], no invented leading samples; position command diagnostics, not servo guarantees.",
            "initial_velocity_supplied": initial_velocity is not None,
            "initial_acceleration_supplied": initial_acceleration is not None,
            "witness_spacing_mm": geometry["witness_spacing_mm"], "witness_match_radius_mm": geometry["witness_match_radius_mm"],
            "constraint_relative_tolerance": constraint_relative_tolerance}


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level descriptive summary; never use interpolation points as replicates."""
    successes = [row for row in rows if row.get("success") is True]
    times = np.array([row["elapsed_time_s"] for row in successes], float)
    return {"run_count": len(rows), "successful_run_count": len(successes), "failed_run_count": len(rows) - len(successes),
            "completion_rate": float(np.mean([r.get("path_completed", False) for r in rows])) if rows else None,
            "success_rate": len(successes) / len(rows) if rows else None,
            "successful_time_mean_s": float(times.mean()) if len(times) else None,
            "successful_time_std_s": float(times.std(ddof=1)) if len(times) > 1 else None,
            "time_sample_definition": "Only individually successful rollouts; pairwise time comparisons require common successful seed/path IDs.",
            "failure_counts": {reason: sum(reason in row.get("failure_reasons", []) for row in rows)
                               for reason in sorted({reason for row in rows for reason in row.get("failure_reasons", [])})}}
