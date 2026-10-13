"""Full-window rest audit added to the immutable ordered trajectory evaluator.

All derivatives are independently reconstructed from position commands. At zero
displacement the last observable heading is carried forward, so this audit does
not observe a controller's internal heading changes while stationary.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.utils.revision_metrics import evaluate_trajectory, independent_kinematics


DEFAULT_REST_POLICY = {
    "window_s": 0.020,
    "feed_velocity_mm_s": 0.001,
    "feed_acceleration_mm_s2": 0.01,
    "angular_velocity_rad_s": 1e-4,
    "angular_acceleration_rad_s2": 1e-3,
}


def _rest_policy(value: Mapping[str, Any] | None) -> dict[str, float]:
    policy = dict(DEFAULT_REST_POLICY)
    if value is not None:
        unknown = set(value) - set(policy)
        if unknown:
            raise ValueError(f"Unknown rest policy fields: {sorted(unknown)}")
        policy.update(value)
    policy = {key: float(number) for key, number in policy.items()}
    if any(not np.isfinite(number) or number <= 0 for number in policy.values()):
        raise ValueError("Rest window and physical thresholds must be finite and positive")
    return policy


def _rest_check(values: np.ndarray, threshold: float) -> dict[str, Any]:
    finite = np.isfinite(values)
    magnitude = np.abs(values)
    return {
        "threshold": threshold,
        "sample_count": len(values),
        "finite_sample_count": int(finite.sum()),
        "window_peak": float(magnitude[finite].max()) if finite.any() else None,
        "terminal_value": float(values[-1]) if len(values) and finite[-1] else None,
        "satisfied": bool(len(values) and finite.all() and np.all(magnitude <= threshold)),
    }


def evaluate_terminal_trajectory(
    reference: Any, positions: Any, timestamps: Any, *, tolerance_mm: float,
    limits: Mapping[str, Any], initial_heading: float,
    initial_velocity: Any = None, initial_acceleration: Any = None,
    initial_angular_velocity: float | None = 0.0,
    initial_angular_acceleration: float | None = 0.0,
    rest_policy: Mapping[str, Any] | None = None,
    termination_reason: str = "completed", claimed_success: bool = True,
    witness_spacing_mm: float | None = None,
) -> dict[str, Any]:
    """Require the unchanged full-path audit AND a complete terminal rest window.

    The terminal window includes the sample at or immediately before its left
    boundary, conservatively covering nonuniform sampling. Physical rest limits
    are tested directly; full-path constraint tolerance remains fixed at 1e-5.
    Stopping within the original endpoint tolerance does not imply an exact
    endpoint stop. Per-axis rest quantities are supplementary diagnostics.
    """
    policy = _rest_policy(rest_policy)
    initial = dict(initial_velocity=initial_velocity,
                   initial_acceleration=initial_acceleration,
                   initial_heading=initial_heading,
                   initial_angular_velocity=initial_angular_velocity,
                   initial_angular_acceleration=initial_angular_acceleration)
    result = evaluate_trajectory(
        reference, positions, timestamps, tolerance_mm=tolerance_mm, limits=limits,
        termination_reason=termination_reason, claimed_success=claimed_success,
        constraint_relative_tolerance=1e-5, witness_spacing_mm=witness_spacing_mm,
        **initial,
    )
    result.update(
        terminal_metrics_version="terminal-metrics-v1",
        rest_policy=policy,
        geometric_arrival_success=bool(result["success"]),
        terminal_rest_checked=True,
        terminal_rest_satisfied=False,
        full_rest_satisfied=False,
        terminal_rest_checks={},
        terminal_axis_rest_checks={},
        rest_failure_reasons=[],
        terminal_position_definition="Within unchanged endpoint tolerance; not an exact endpoint claim.",
        rest_observation_scope="Backward differences of position commands; internal heading motion at zero displacement is unobservable.",
        terminal_axis_rest_diagnostic_only=True,
    )
    if not result["valid_trace"]:
        result["rest_failure_reasons"] = ["rest_invalid_trace"]
        result["failure_reasons"].append("rest_invalid_trace")
        return result

    kin = independent_kinematics(positions, timestamps, **initial)
    t = kin["timestamp_s"]
    # This allowance only handles floating representation of the time boundary.
    time_roundoff = 16 * np.finfo(float).eps * max(1.0, abs(float(t[-1])))
    boundary = float(t[-1] - policy["window_s"])
    start = max(0, int(np.searchsorted(t, boundary + time_roundoff, side="right")) - 1)
    duration = float(t[-1] - t[start])
    complete = bool(len(t) - start >= 2 and duration + time_roundoff >= policy["window_s"])
    result["terminal_rest_window"] = {
        "requested_duration_s": policy["window_s"], "observed_duration_s": duration,
        "start_index": start, "start_timestamp_s": float(t[start]),
        "end_timestamp_s": float(t[-1]), "sample_count": len(t) - start,
        "complete": complete,
    }
    fields = {
        "feed_mm_s": "feed_velocity_mm_s",
        "feed_acceleration_mm_s2": "feed_acceleration_mm_s2",
        "angular_velocity_rad_s": "angular_velocity_rad_s",
        "angular_acceleration_rad_s2": "angular_acceleration_rad_s2",
    }
    checks = {key: _rest_check(kin[key][start:], policy[policy_key])
              for key, policy_key in fields.items()}
    result["terminal_rest_checks"] = checks
    for axis, name in enumerate(("x", "y")):
        for suffix, policy_key in (("velocity_mm_s", "feed_velocity_mm_s"),
                                   ("acceleration_mm_s2", "feed_acceleration_mm_s2")):
            result["terminal_axis_rest_checks"][f"{name}_{suffix}"] = _rest_check(
                kin[f"axis_{suffix}"][start:, axis], policy[policy_key])
    reasons = [] if complete else ["rest_window_incomplete"]
    reasons.extend(f"rest_{key}" for key, check in checks.items() if not check["satisfied"])
    rest_ok = not reasons
    result.update(full_rest_satisfied=rest_ok, terminal_rest_satisfied=rest_ok,
                  rest_failure_reasons=reasons, success=bool(result["success"] and rest_ok))
    result["failure_reasons"].extend(reasons)
    return result
