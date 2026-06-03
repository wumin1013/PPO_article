"""Traditional two-step CNC baseline.

The baseline follows a serial pipeline: first build a fixed smoothed path,
then run jerk-limited feedrate scheduling along that fixed path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from src.utils.comparison_metrics import (
    as_point_array,
    compute_comparison_metrics,
    interpolate_polyline,
    polyline_length,
    project_points_to_polyline,
    resample_polyline,
)


@dataclass(frozen=True)
class TwoStepConstraints:
    max_vel: float
    max_acc: float
    max_jerk: float
    max_ang_vel: float = 0.0
    max_ang_acc: float = 0.0
    max_ang_jerk: float = 0.0


@dataclass(frozen=True)
class SmoothingParams:
    transition_ratio: float = 0.25
    safety_factor: float = 0.70
    angle_threshold_deg: float = 8.0
    min_segment_length: float = 1e-6
    max_retries: int = 5


@dataclass(frozen=True)
class SchedulerParams:
    feedrate_safety_ratio: float = 0.18
    acc_safety_ratio: float = 0.35
    jerk_safety_ratio: float = 0.25
    a_effective_ratio: float = 0.65
    corner_speed_ratio: float = 0.35
    corner_influence_scale: float = 2.5
    brake_acc_ratio: float = 0.85
    curvature_eps: float = 1e-8
    stop_velocity_ratio: float = 1e-4
    target_spacing_mm: float | None = None


@dataclass(frozen=True)
class SmoothedPathResult:
    original_path: np.ndarray
    smoothed_path: np.ndarray
    closed: bool
    corners: list[dict[str, float]]
    transition_ratio: float
    safety_factor: float
    max_smoothed_path_error_mm: float
    boundary_violation_flag: bool
    retry_count: int


@dataclass(frozen=True)
class TwoStepRunResult:
    path_name: str
    smoothed: SmoothedPathResult
    trace_rows: list[dict[str, Any]]
    metrics: dict[str, Any]


def _closed_core(points: np.ndarray, closed: bool) -> np.ndarray:
    if closed and points.shape[0] > 2 and np.linalg.norm(points[0] - points[-1]) <= 1e-8:
        return points[:-1]
    return points


def _turn_angle(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    v_in = p - p_prev
    v_out = p_next - p
    len_in = float(np.linalg.norm(v_in))
    len_out = float(np.linalg.norm(v_out))
    if len_in <= 1e-12 or len_out <= 1e-12:
        return 0.0
    return float(math.acos(float(np.clip(np.dot(v_in / len_in, v_out / len_out), -1.0, 1.0))))


def _remove_neighbor_duplicates(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    arr = as_point_array(points)
    if arr.shape[0] <= 1:
        return arr
    out = [arr[0].copy()]
    for point in arr[1:]:
        if np.linalg.norm(point - out[-1]) > tol:
            out.append(point.copy())
    return np.asarray(out, dtype=float)


def _macro_vertices_for_corner_smoothing(core: np.ndarray, closed: bool, params: SmoothingParams) -> np.ndarray:
    """Compress dense polyline samples only when there are clear sparse corners.

    The square path contains many samples on each straight edge. Using adjacent
    sample lengths would make the smoothing radius unrealistically tiny. This
    helper keeps the start anchor and significant direction changes, but falls
    back to the original curve for continuous-curvature paths.
    """
    clean = _remove_neighbor_duplicates(core)
    n = int(clean.shape[0])
    if n < 4:
        return clean

    threshold = math.radians(float(params.angle_threshold_deg))
    significant: list[int] = []
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        if i == 0:
            continue
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        angle = _turn_angle(clean[prev_i], clean[i], clean[next_i])
        if angle >= threshold:
            significant.append(i)

    if not significant:
        return clean

    if closed:
        keep = sorted(set([0] + significant))
        compressed = clean[keep]
        compression_ratio = n / max(len(compressed), 1)
        # Apply macro smoothing to polygon-like paths, not to dense smooth curves.
        if len(compressed) <= 16 and compression_ratio >= 8.0:
            return compressed
        return clean

    keep = sorted(set([0] + significant + [n - 1]))
    compressed = clean[keep]
    compression_ratio = n / max(len(compressed), 1)
    if len(compressed) <= 16 and compression_ratio >= 8.0:
        return compressed
    return clean


def _append_point(out: list[np.ndarray], point: np.ndarray, tol: float = 1e-10) -> None:
    p = np.asarray(point, dtype=float).reshape(2)
    if out and np.linalg.norm(out[-1] - p) <= tol:
        return
    out.append(p.copy())


def _append_line(out: list[np.ndarray], start: np.ndarray, end: np.ndarray, spacing: float) -> None:
    length = float(np.linalg.norm(end - start))
    if length <= 1e-12:
        _append_point(out, end)
        return
    count = max(1, int(math.ceil(length / max(spacing, 1e-6))))
    for t in np.linspace(0.0, 1.0, count + 1):
        _append_point(out, start + float(t) * (end - start))


def _append_quadratic_bezier(
    out: list[np.ndarray],
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    spacing: float,
) -> None:
    approx_len = float(np.linalg.norm(control - start) + np.linalg.norm(end - control))
    count = max(8, int(math.ceil(approx_len / max(spacing, 1e-6))))
    for t in np.linspace(0.0, 1.0, count + 1):
        u = 1.0 - float(t)
        p = u * u * start + 2.0 * u * float(t) * control + float(t) * float(t) * end
        _append_point(out, p)


def _detect_corner_geometry(
    core: np.ndarray,
    *,
    closed: bool,
    transition_ratio: float,
    safety_factor: float,
    half_epsilon: float,
    params: SmoothingParams,
) -> list[dict[str, Any]]:
    n = int(core.shape[0])
    corners: list[dict[str, Any]] = []
    if n < 3:
        return corners
    angle_threshold = math.radians(float(params.angle_threshold_deg))
    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        if not closed and (i <= 0 or i >= n - 1):
            continue
        p_prev = core[prev_i]
        p = core[i]
        p_next = core[next_i]
        v_in = p - p_prev
        v_out = p_next - p
        len_in = float(np.linalg.norm(v_in))
        len_out = float(np.linalg.norm(v_out))
        if len_in <= params.min_segment_length or len_out <= params.min_segment_length:
            continue
        u_in = v_in / len_in
        u_out = v_out / len_out
        turn_angle = float(math.acos(float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))))
        if turn_angle < angle_threshold:
            continue
        d = min(
            float(transition_ratio) * len_in,
            float(transition_ratio) * len_out,
            float(safety_factor) * float(half_epsilon) / max(math.sin(turn_angle / 2.0), 1e-6),
        )
        if d <= params.min_segment_length:
            continue
        corners.append(
            {
                "index": int(i),
                "point": p.copy(),
                "u_in": u_in.copy(),
                "u_out": u_out.copy(),
                "turn_angle_rad": turn_angle,
                "turn_angle_deg": math.degrees(turn_angle),
                "transition_length_mm": float(d),
                "enter": p - d * u_in,
                "exit": p + d * u_out,
            }
        )
    return corners


def _smooth_once(
    reference_path: np.ndarray,
    *,
    half_epsilon: float,
    closed: bool,
    transition_ratio: float,
    safety_factor: float,
    params: SmoothingParams,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    core = _macro_vertices_for_corner_smoothing(_closed_core(reference_path, closed), closed, params)
    total = polyline_length(reference_path, closed=closed)
    nominal_spacing = total / max(int(reference_path.shape[0]) - 1, 1)
    spacing = float(np.clip(nominal_spacing, 0.05, 1.0))
    corners_raw = _detect_corner_geometry(
        core,
        closed=closed,
        transition_ratio=transition_ratio,
        safety_factor=safety_factor,
        half_epsilon=half_epsilon,
        params=params,
    )

    if not corners_raw:
        count = max(int(reference_path.shape[0]), int(math.ceil(total / spacing)) + 1)
        smoothed = resample_polyline(reference_path, count, closed=closed)
        return smoothed, []

    n = int(core.shape[0])
    enter = [core[i].copy() for i in range(n)]
    exit_ = [core[i].copy() for i in range(n)]
    corner_by_index: dict[int, dict[str, Any]] = {}
    for item in corners_raw:
        idx = int(item["index"])
        enter[idx] = np.asarray(item["enter"], dtype=float)
        exit_[idx] = np.asarray(item["exit"], dtype=float)
        corner_by_index[idx] = item

    out: list[np.ndarray] = []
    if closed:
        for i in range(n):
            next_i = (i + 1) % n
            _append_point(out, exit_[i])
            _append_line(out, exit_[i], enter[next_i], spacing)
            if next_i in corner_by_index:
                _append_quadratic_bezier(out, enter[next_i], core[next_i], exit_[next_i], spacing)
        if out:
            _append_point(out, out[0])
    else:
        _append_point(out, core[0])
        for i in range(n - 1):
            next_i = i + 1
            start = exit_[i] if i in corner_by_index else core[i]
            end = enter[next_i] if next_i in corner_by_index else core[next_i]
            _append_line(out, out[-1], start, spacing)
            _append_line(out, start, end, spacing)
            if next_i in corner_by_index:
                _append_quadratic_bezier(out, enter[next_i], core[next_i], exit_[next_i], spacing)
        _append_line(out, out[-1], core[-1], spacing)

    smoothed = np.asarray(out, dtype=float)
    if smoothed.shape[0] < 2:
        smoothed = reference_path.copy()

    corner_meta: list[dict[str, float]] = []
    for item in corners_raw:
        proj = project_points_to_polyline([item["point"]], smoothed, closed=closed)
        s_center = float(proj.raw_s[0]) if proj.raw_s.size else 0.0
        d = float(item["transition_length_mm"])
        corner_meta.append(
            {
                "index": float(item["index"]),
                "s_center_mm": s_center,
                "influence_radius_mm": max(2.5 * d, 1.0),
                "transition_length_mm": d,
                "turn_angle_deg": float(item["turn_angle_deg"]),
            }
        )
    return smoothed, corner_meta


def smooth_reference_path(
    reference_path: Sequence[Sequence[float]] | np.ndarray,
    *,
    half_epsilon: float,
    closed: bool = True,
    params: SmoothingParams | None = None,
) -> SmoothedPathResult:
    """Build a fixed smoothed path and verify its contour deviation."""
    smoothing_params = params or SmoothingParams()
    original = as_point_array(reference_path)
    if original.shape[0] < 2:
        raise ValueError("reference_path must contain at least two points")

    best: tuple[np.ndarray, list[dict[str, float]], float, float, int, float] | None = None
    max_retries = max(0, int(smoothing_params.max_retries))
    for attempt in range(max_retries + 1):
        factor = 0.70**attempt
        transition_ratio = float(smoothing_params.transition_ratio) * factor
        safety_factor = float(smoothing_params.safety_factor) * factor
        smoothed, corners = _smooth_once(
            original,
            half_epsilon=half_epsilon,
            closed=closed,
            transition_ratio=transition_ratio,
            safety_factor=safety_factor,
            params=smoothing_params,
        )
        errors = project_points_to_polyline(smoothed, original, closed=closed).contour_error
        max_error = float(np.max(errors)) if errors.size else 0.0
        best = (smoothed, corners, transition_ratio, safety_factor, attempt, max_error)
        if max_error <= float(half_epsilon) + 1e-9:
            break

    assert best is not None
    smoothed, corners, transition_ratio, safety_factor, retry_count, max_error = best
    return SmoothedPathResult(
        original_path=original,
        smoothed_path=smoothed,
        closed=bool(closed),
        corners=corners,
        transition_ratio=float(transition_ratio),
        safety_factor=float(safety_factor),
        max_smoothed_path_error_mm=float(max_error),
        boundary_violation_flag=bool(max_error > float(half_epsilon) + 1e-9),
        retry_count=int(retry_count),
    )


def _curvature_profile(points: np.ndarray, closed: bool) -> np.ndarray:
    arr = as_point_array(points)
    n = int(arr.shape[0])
    if n < 3:
        return np.zeros((n,), dtype=float)
    core = _closed_core(arr, closed)
    m = int(core.shape[0])
    curv = np.zeros((m,), dtype=float)
    for i in range(m):
        if not closed and (i == 0 or i == m - 1):
            continue
        p0 = core[(i - 1) % m]
        p1 = core[i]
        p2 = core[(i + 1) % m]
        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        denom = a * b * c
        if denom <= 1e-12:
            continue
        area2 = abs(float(np.cross(p1 - p0, p2 - p0)))
        curv[i] = 2.0 * area2 / denom
    if closed and arr.shape[0] > core.shape[0]:
        curv = np.concatenate([curv, curv[:1]])
    return curv


def _cumulative_for_points(points: np.ndarray, closed: bool) -> np.ndarray:
    arr = as_point_array(points)
    if arr.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if closed and np.linalg.norm(arr[0] - arr[-1]) <= 1e-8:
        work = arr
    elif closed:
        work = np.vstack([arr, arr[0]])
    else:
        work = arr
    if work.shape[0] < 2:
        return np.zeros((work.shape[0],), dtype=float)
    seg = np.linalg.norm(np.diff(work, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _corner_speed_cap(s_value: float, corners: list[dict[str, float]], total: float, closed: bool, ratio: float, max_vel: float) -> float:
    if not corners:
        return float(max_vel)
    cap = float(max_vel)
    for item in corners:
        center = float(item.get("s_center_mm", 0.0))
        radius = max(float(item.get("influence_radius_mm", 0.0)), 1e-9)
        dist = abs(float(s_value) - center)
        if closed and total > 1e-12:
            dist = min(dist, total - dist)
        if dist <= radius:
            cap = min(cap, float(ratio) * float(max_vel))
    return cap


def run_two_step_baseline(
    *,
    path_name: str,
    reference_path: Sequence[Sequence[float]] | np.ndarray,
    constraints: TwoStepConstraints,
    dt: float,
    half_epsilon: float,
    closed: bool,
    max_steps: int,
    smoothing_params: SmoothingParams | None = None,
    scheduler_params: SchedulerParams | None = None,
) -> TwoStepRunResult:
    """Run the traditional two-step baseline for one reference path."""
    sched = scheduler_params or SchedulerParams()
    smoothed = smooth_reference_path(reference_path, half_epsilon=half_epsilon, closed=closed, params=smoothing_params)
    smooth_path = smoothed.smoothed_path
    smooth_total = polyline_length(smooth_path, closed=False)
    point_s = _cumulative_for_points(smooth_path, closed=False)
    curvature = _curvature_profile(smooth_path, closed=False)
    if curvature.size != point_s.size:
        curvature = np.resize(curvature, point_s.size)

    physical_max_vel = float(constraints.max_vel)
    physical_max_acc = float(constraints.max_acc)
    physical_max_jerk = float(constraints.max_jerk)
    max_vel = physical_max_vel * float(np.clip(sched.feedrate_safety_ratio, 0.01, 1.0))
    max_acc = physical_max_acc * float(np.clip(sched.acc_safety_ratio, 0.01, 1.0))
    max_jerk = physical_max_jerk * float(np.clip(sched.jerk_safety_ratio, 0.01, 1.0))
    dt_value = max(float(dt), 1e-9)
    a_effective = float(sched.a_effective_ratio) * max_acc
    brake_acc = float(sched.brake_acc_ratio) * max_acc
    stop_velocity = max(max_vel * float(sched.stop_velocity_ratio), 1e-5)

    rows_core: list[dict[str, Any]] = []
    s = 0.0
    v = 0.0
    a = 0.0
    termination_status = "max_steps"

    for step in range(1, int(max_steps) + 1):
        remaining = max(0.0, smooth_total - s)
        if remaining <= 1e-9 and v <= stop_velocity:
            termination_status = "success"
            break

        kappa = float(np.interp(min(s, smooth_total), point_s, curvature)) if point_s.size else 0.0
        if not np.isfinite(kappa):
            termination_status = "numerical_error"
            break
        if abs(kappa) <= float(sched.curvature_eps):
            curve_cap = max_vel
        else:
            curve_cap = min(max_vel, math.sqrt(max(a_effective, 0.0) / max(abs(kappa), float(sched.curvature_eps))))
        corner_cap = _corner_speed_cap(
            s,
            smoothed.corners,
            smooth_total,
            bool(closed),
            float(sched.corner_speed_ratio),
            max_vel,
        )
        brake_cap = math.sqrt(max(0.0, 2.0 * brake_acc * remaining))
        v_cap = max(0.0, min(max_vel, curve_cap, corner_cap, brake_cap))

        a_des = float(np.clip((v_cap - v) / dt_value, -max_acc, max_acc))
        j = float(np.clip((a_des - a) / dt_value, -max_jerk, max_jerk))
        a_next = float(np.clip(a + j * dt_value, -max_acc, max_acc))
        v_next = float(np.clip(v + a_next * dt_value, 0.0, v_cap))
        s_next = min(smooth_total, s + v_next * dt_value)
        pos = interpolate_polyline(smooth_path, s_next, closed=False)
        if not (np.isfinite(s_next) and np.all(np.isfinite(pos)) and np.isfinite(v_next) and np.isfinite(a_next)):
            termination_status = "numerical_error"
            break

        rows_core.append(
            {
                "step": int(step),
                "time": float(step) * dt_value,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "v": float(v_next),
                "a": float(a_next),
                "j": float(j),
                "s_smoothed": float(s_next),
                "v_cap": float(v_cap),
            }
        )
        s = s_next
        v = v_next
        a = a_next

    if termination_status == "max_steps" and s >= smooth_total - 1e-8 and v <= stop_velocity:
        termination_status = "success"

    points = np.asarray([[row["x"], row["y"]] for row in rows_core], dtype=float) if rows_core else np.empty((0, 2))
    projection = project_points_to_polyline(points, reference_path, closed=closed)
    contour_error = projection.contour_error
    progress = projection.progress
    active_mask = []
    for idx, row in enumerate(rows_core):
        err = float(contour_error[idx]) if idx < contour_error.size else float("nan")
        prog = float(progress[idx]) if idx < progress.size else 0.0
        v_over = float(row["v"]) / max(physical_max_vel, 1e-12)
        j_over = abs(float(row["j"])) / max(physical_max_jerk, 1e-12)
        active = bool(abs(float(row["a"])) > 0.05 * physical_max_acc or abs(float(row["j"])) > 0.05 * physical_max_jerk)
        active_mask.append(active)
        row.update(
            {
                "contour_error": err,
                "progress": prog,
                "v_over_vmax": v_over,
                "abs_j_over_jmax": j_over,
                "active_accdec_window": active,
            }
        )

    if contour_error.size and np.nanmax(contour_error) > float(half_epsilon) + 1e-9 and termination_status == "success":
        # The task asks to record this condition without forcing termination.
        termination_status = "success"

    metrics = compute_comparison_metrics(
        reference_path=reference_path,
        trajectory=points,
        time=[row["time"] for row in rows_core],
        velocity=[row["v"] for row in rows_core],
        acceleration=[row["a"] for row in rows_core],
        jerk=[row["j"] for row in rows_core],
        max_vel=physical_max_vel,
        max_acc=physical_max_acc,
        max_jerk=physical_max_jerk,
        dt=dt_value,
        termination_status=termination_status,
        closed=closed,
        half_epsilon=half_epsilon,
        progress=[row["progress"] for row in rows_core],
        extra={
            "method": "Traditional two-step",
            "path": str(path_name),
            "max_smoothed_path_error_mm": float(smoothed.max_smoothed_path_error_mm),
            "smoothed_path_length_mm": float(smooth_total),
            "feedrate_safety_ratio": float(sched.feedrate_safety_ratio),
            "acc_safety_ratio": float(sched.acc_safety_ratio),
            "jerk_safety_ratio": float(sched.jerk_safety_ratio),
        },
    )
    if smoothed.boundary_violation_flag:
        metrics["boundary_violation_flag"] = True
    return TwoStepRunResult(path_name=str(path_name), smoothed=smoothed, trace_rows=rows_core, metrics=metrics)


__all__ = [
    "SchedulerParams",
    "SmoothedPathResult",
    "SmoothingParams",
    "TwoStepConstraints",
    "TwoStepRunResult",
    "run_two_step_baseline",
    "smooth_reference_path",
]
