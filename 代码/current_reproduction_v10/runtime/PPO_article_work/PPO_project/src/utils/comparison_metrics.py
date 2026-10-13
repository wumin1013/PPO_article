"""Unified comparison metrics for paper-level trajectory rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PolylineProjection:
    """Projection result of trajectory points onto a reference polyline."""

    contour_error: np.ndarray
    raw_s: np.ndarray
    unwrapped_s: np.ndarray
    progress: np.ndarray
    original_path_length: float


def as_point_array(points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert any point sequence into an Nx2 float array."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim == 1:
        if arr.size < 2:
            return np.empty((0, 2), dtype=float)
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return np.empty((0, 2), dtype=float)
    return np.asarray(arr[:, :2], dtype=float)


def infer_closed(points: Sequence[Sequence[float]] | np.ndarray, closed: bool | None = None) -> bool:
    """Infer whether a path is closed unless the caller provides the flag."""
    if closed is not None:
        return bool(closed)
    arr = as_point_array(points)
    return bool(arr.shape[0] > 2 and np.linalg.norm(arr[0] - arr[-1]) <= 1e-8)


def polyline_segments(
    points: Sequence[Sequence[float]] | np.ndarray,
    closed: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """Return starts, ends, segment lengths, cumulative starts, total length, closed flag."""
    arr = as_point_array(points)
    closed_flag = infer_closed(arr, closed)
    if arr.shape[0] < 2:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), 0.0, closed_flag

    if closed_flag and np.linalg.norm(arr[0] - arr[-1]) <= 1e-8:
        core = arr[:-1]
    else:
        core = arr

    if closed_flag:
        if core.shape[0] < 2:
            empty = np.empty((0, 2), dtype=float)
            return empty, empty, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), 0.0, closed_flag
        starts = core
        ends = np.roll(core, -1, axis=0)
    else:
        if core.shape[0] < 2:
            empty = np.empty((0, 2), dtype=float)
            return empty, empty, np.zeros((0,), dtype=float), np.zeros((0,), dtype=float), 0.0, closed_flag
        starts = core[:-1]
        ends = core[1:]

    seg_len = np.linalg.norm(ends - starts, axis=1)
    keep = seg_len > 1e-12
    starts = starts[keep]
    ends = ends[keep]
    seg_len = seg_len[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumulative[-1]) if cumulative.size else 0.0
    return starts, ends, seg_len, cumulative[:-1], total, closed_flag


def polyline_length(points: Sequence[Sequence[float]] | np.ndarray, closed: bool | None = None) -> float:
    """Compute path length with duplicate closed endpoints handled once."""
    return polyline_segments(points, closed=closed)[4]


def interpolate_polyline(
    points: Sequence[Sequence[float]] | np.ndarray,
    s_value: float,
    closed: bool | None = None,
) -> np.ndarray:
    """Interpolate a point by arclength along a polyline."""
    starts, ends, seg_len, seg_start, total, closed_flag = polyline_segments(points, closed=closed)
    if total <= 1e-12 or starts.shape[0] == 0:
        arr = as_point_array(points)
        return arr[0].copy() if arr.shape[0] else np.zeros((2,), dtype=float)

    s = float(s_value)
    if closed_flag:
        s = s % total
    else:
        s = float(np.clip(s, 0.0, total))

    idx = int(np.searchsorted(seg_start + seg_len, s, side="left"))
    idx = int(np.clip(idx, 0, seg_len.size - 1))
    length = float(seg_len[idx])
    if length <= 1e-12:
        return starts[idx].copy()
    t = float(np.clip((s - seg_start[idx]) / length, 0.0, 1.0))
    return starts[idx] + t * (ends[idx] - starts[idx])


def resample_polyline(
    points: Sequence[Sequence[float]] | np.ndarray,
    num_points: int,
    closed: bool | None = None,
) -> np.ndarray:
    """Resample a polyline by arclength."""
    n = max(2, int(num_points))
    total = polyline_length(points, closed=closed)
    closed_flag = infer_closed(points, closed)
    if total <= 1e-12:
        arr = as_point_array(points)
        base = arr[0].copy() if arr.shape[0] else np.zeros((2,), dtype=float)
        return np.repeat(base.reshape(1, 2), n, axis=0)
    if closed_flag:
        targets = np.linspace(0.0, total, n, endpoint=True)
    else:
        targets = np.linspace(0.0, total, n)
    out = np.vstack([interpolate_polyline(points, s, closed=closed_flag) for s in targets])
    if closed_flag:
        out[-1] = out[0]
    return out


def project_points_to_polyline(
    points: Sequence[Sequence[float]] | np.ndarray,
    reference_path: Sequence[Sequence[float]] | np.ndarray,
    closed: bool | None = None,
    chunk_size: int = 1024,
) -> PolylineProjection:
    """Project trajectory points to the closest reference segment.

    Closed paths are unwrapped over time, so a trajectory returning to the
    start of a loop reports progress near 1.0 instead of jumping back to 0.0.
    """
    pts = as_point_array(points)
    starts, ends, seg_len, seg_start, total, closed_flag = polyline_segments(reference_path, closed=closed)
    if pts.shape[0] == 0 or starts.shape[0] == 0 or total <= 1e-12:
        zeros = np.zeros((pts.shape[0],), dtype=float)
        return PolylineProjection(zeros, zeros, zeros, zeros, float(total))

    seg = ends - starts
    denom = np.sum(seg * seg, axis=1)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    best_dist = np.full((pts.shape[0],), np.inf, dtype=float)
    best_s = np.zeros((pts.shape[0],), dtype=float)

    for start_idx in range(0, pts.shape[0], int(chunk_size)):
        chunk = pts[start_idx : start_idx + int(chunk_size)]
        rel = chunk[:, None, :] - starts[None, :, :]
        t = np.clip(np.sum(rel * seg[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
        projection = starts[None, :, :] + t[:, :, None] * seg[None, :, :]
        dist2 = np.sum((chunk[:, None, :] - projection) ** 2, axis=2)
        idx = np.argmin(dist2, axis=1)
        rows = np.arange(chunk.shape[0])
        best_dist[start_idx : start_idx + chunk.shape[0]] = np.sqrt(dist2[rows, idx])
        best_s[start_idx : start_idx + chunk.shape[0]] = seg_start[idx] + t[rows, idx] * seg_len[idx]

    unwrapped = best_s.copy()
    if closed_flag and best_s.size:
        prev = float(best_s[0])
        unwrapped[0] = prev
        for i in range(1, best_s.size):
            s = float(best_s[i])
            base_turn = np.floor(prev / total) * total
            candidates = np.array([s + base_turn - total, s + base_turn, s + base_turn + total], dtype=float)
            chosen = float(candidates[int(np.argmin(np.abs(candidates - prev)))])
            if chosen < prev and (prev - chosen) > max(1e-6, 0.05 * total):
                chosen += total
            unwrapped[i] = chosen
            prev = chosen

    progress = np.clip(unwrapped / max(total, 1e-12), 0.0, 1.0)
    return PolylineProjection(best_dist, best_s, unwrapped, progress, float(total))


def _as_1d(values: Sequence[float] | np.ndarray | None, n: int, fill: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full((n,), fill, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= n:
        return arr[:n]
    out = np.full((n,), fill, dtype=float)
    out[: arr.size] = arr
    return out


def _finite_last(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def _safe_percentile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def compute_comparison_metrics(
    *,
    reference_path: Sequence[Sequence[float]] | np.ndarray,
    trajectory: Sequence[Sequence[float]] | np.ndarray,
    time: Sequence[float] | np.ndarray | None = None,
    velocity: Sequence[float] | np.ndarray | None = None,
    acceleration: Sequence[float] | np.ndarray | None = None,
    jerk: Sequence[float] | np.ndarray | None = None,
    max_vel: float,
    max_acc: float,
    max_jerk: float,
    dt: float,
    termination_status: str,
    closed: bool | None = None,
    half_epsilon: float | None = None,
    progress: Sequence[float] | np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the unified metrics used by the comparison tables."""
    pts = as_point_array(trajectory)
    n = int(pts.shape[0])
    v = _as_1d(velocity, n, fill=0.0)
    a = _as_1d(acceleration, n, fill=0.0)
    j = _as_1d(jerk, n, fill=0.0)
    t = _as_1d(time, n, fill=np.nan)
    projection = project_points_to_polyline(pts, reference_path, closed=closed)

    if progress is not None:
        progress_arr = _as_1d(progress, n, fill=np.nan)
        final_progress_value = _finite_last(progress_arr)
        final_progress = float(np.clip(final_progress_value, 0.0, 1.0)) if final_progress_value is not None else 0.0
    else:
        final_progress = float(projection.progress[-1]) if projection.progress.size else 0.0

    valid_motion = np.isfinite(v) & np.isfinite(a) & np.isfinite(j)
    valid_v = v[np.isfinite(v)]
    denom_v = max(float(max_vel), 1e-12)
    denom_a = max(float(max_acc), 1e-12)
    denom_j = max(float(max_jerk), 1e-12)
    v_util = valid_v / denom_v if valid_v.size else np.zeros((0,), dtype=float)
    j_util_all = np.abs(j[np.isfinite(j)]) / denom_j if np.isfinite(j).any() else np.zeros((0,), dtype=float)

    active_mask = valid_motion & ((np.abs(a) > 0.05 * denom_a) | (np.abs(j) > 0.05 * denom_j))
    active_count = int(np.count_nonzero(active_mask))
    if active_count > 0:
        active_j_util = np.abs(j[active_mask]) / denom_j
        reach_mask = (active_j_util >= 0.8) & (active_j_util <= 1.0 + 1e-6)
        jerk_reach = float(np.count_nonzero(reach_mask) / active_count)
    else:
        jerk_reach = float("nan")

    contour_error = projection.contour_error[np.isfinite(projection.contour_error)]
    max_contour = float(np.max(contour_error)) if contour_error.size else float("nan")
    mean_contour = float(np.mean(contour_error)) if contour_error.size else float("nan")
    termination_time = float(n) * float(dt)
    if n > 0 and not np.isfinite(termination_time):
        finite_t = t[np.isfinite(t)]
        termination_time = float(finite_t[-1]) if finite_t.size else 0.0

    max_relative_exceedance = float(np.max(np.maximum(np.abs(j_util_all) - 1.0, 0.0))) if j_util_all.size else 0.0
    effective_speed = (
        float(final_progress) * float(projection.original_path_length) / termination_time
        if termination_time > 1e-12
        else float("nan")
    )
    completion_efficiency = float(final_progress) / termination_time if termination_time > 1e-12 else float("nan")
    active_step_rate = float(active_count / n) if n > 0 else float("nan")

    metrics: dict[str, Any] = {
        "termination_status": str(termination_status or "unknown"),
        "final_progress": float(np.clip(final_progress, 0.0, 1.0)),
        "max_contour_error_mm": max_contour,
        "mean_contour_error_mm": mean_contour,
        "max_relative_linear_jerk_exceedance": max_relative_exceedance,
        "termination_time_s": termination_time,
        "mean_feedrate_utilization": float(np.mean(v_util)) if v_util.size else float("nan"),
        "high_feedrate_rate_80": float(np.count_nonzero(v_util >= 0.8) / v_util.size) if v_util.size else float("nan"),
        "p95_linear_jerk_utilization": _safe_percentile(j_util_all, 95.0),
        "jerk_reach_rate_80_active": jerk_reach,
        "effective_path_speed_mm_s": effective_speed,
        "path_completion_efficiency": completion_efficiency,
        "max_feedrate_utilization": float(np.max(v_util)) if v_util.size else float("nan"),
        "p50_linear_jerk_utilization": _safe_percentile(j_util_all, 50.0),
        "p99_linear_jerk_utilization": _safe_percentile(j_util_all, 99.0),
        "active_step_rate": active_step_rate,
        "boundary_violation_flag": bool(
            half_epsilon is not None and np.isfinite(max_contour) and max_contour > float(half_epsilon) + 1e-9
        ),
        "original_path_length_mm": float(projection.original_path_length),
    }
    if extra:
        metrics.update(dict(extra))
    return metrics


__all__ = [
    "PolylineProjection",
    "as_point_array",
    "compute_comparison_metrics",
    "infer_closed",
    "interpolate_polyline",
    "polyline_length",
    "polyline_segments",
    "project_points_to_polyline",
    "resample_polyline",
]
