"""
几何计算工具模块：集中管理路径偏移、点线关系等计算。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np


def normalize_vector(v: Sequence[float]) -> np.ndarray:
    """归一化向量，长度过小则返回零向量。"""
    vec = np.asarray(v, dtype=float)
    length = np.linalg.norm(vec)
    if length < 1e-6:
        return np.zeros_like(vec)
    return vec / length


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的夹角（弧度），逆时针为正。"""
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    dot_product = np.dot(v1, v2) / (len1 * len2)
    cross_product = np.cross(v1, v2) / (len1 * len2)
    return math.atan2(cross_product, dot_product)


def find_intersection(line1: Tuple[float, float, float], line2: Tuple[float, float, float]) -> Optional[np.ndarray]:
    """求两条直线的交点，直线以 (A, B, C) 形式表示 Ax + By + C = 0。"""
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return np.array([x, y], dtype=float)


def cross2(a: Sequence[float], b: Sequence[float]) -> float:
    """2D 叉积标量：a.x*b.y - a.y*b.x。"""
    return float(a[0] * b[1] - a[1] * b[0])


def left_normal(t: Sequence[float]) -> np.ndarray:
    """左法向：给定切向 t=(tx,ty)，返回 nL=(-ty,tx)。"""
    tx, ty = float(t[0]), float(t[1])
    return np.array([-ty, tx], dtype=float)


def right_normal(t: Sequence[float]) -> np.ndarray:
    """右法向：给定切向 t=(tx,ty)，返回 nR=(ty,-tx)。"""
    tx, ty = float(t[0]), float(t[1])
    return np.array([ty, -tx], dtype=float)


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    length = float(np.linalg.norm(v))
    if length < eps:
        return np.zeros_like(v, dtype=float)
    return v / length


def _iter_segments(points: Sequence[np.ndarray], closed: bool) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return []
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    if closed and len(points) > 2:
        segments.append((points[-1], points[0]))
    return segments


def _as_point_list(points: Sequence[Sequence[float]]) -> List[np.ndarray]:
    return [np.asarray(p, dtype=float) for p in points]


def _resolve_closed_core(
    points: List[np.ndarray],
    closed: bool | None,
    eps_len: float,
) -> Tuple[List[np.ndarray], bool, bool]:
    if not points:
        return [], False, False
    inferred_closed = len(points) > 2 and np.allclose(points[0], points[-1], atol=eps_len)
    closed_effective = bool(closed) if closed is not None else inferred_closed
    has_duplicate_last = closed_effective and inferred_closed
    if has_duplicate_last:
        return points[:-1], closed_effective, True
    return points, closed_effective, False


def _find_prev_distinct(points: Sequence[np.ndarray], i: int, closed: bool, eps_len: float) -> Optional[int]:
    n = len(points)
    if n == 0:
        return None
    steps = 0
    j = (i - 1) % n if closed else i - 1
    while 0 <= j < n and steps < n:
        if np.linalg.norm(points[i] - points[j]) >= eps_len:
            return j
        if not closed and j == 0:
            break
        j = (j - 1) % n if closed else j - 1
        steps += 1
    return None


def _find_next_distinct(points: Sequence[np.ndarray], i: int, closed: bool, eps_len: float) -> Optional[int]:
    n = len(points)
    if n == 0:
        return None
    steps = 0
    j = (i + 1) % n if closed else i + 1
    while 0 <= j < n and steps < n:
        if np.linalg.norm(points[j] - points[i]) >= eps_len:
            return j
        if not closed and j == n - 1:
            break
        j = (j + 1) % n if closed else j + 1
        steps += 1
    return None


def generate_offset_paths(
    Pm: Sequence[Sequence[float]],
    epsilon: float,
    closed: bool | None = None,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    生成偏移路径，返回左/右边界点列表。

    Args:
        Pm: 中心路径点序列。
        epsilon: 单侧偏移距离（Pl/Pr 到 Pm 的距离）。
        closed: 可选，显式指定是否闭合；默认按首尾点判断。
    """
    offset = float(epsilon)
    if offset <= 0.0:
        raise ValueError("epsilon must be positive")

    eps_len = 1e-6
    eps_miter = 1e-6
    miter_limit = 4.0

    pm_full = _as_point_list(Pm)
    n_full = len(pm_full)
    if n_full == 0:
        return [], []

    pm_core, closed_effective, has_duplicate_last = _resolve_closed_core(pm_full, closed, eps_len=eps_len)
    n_core = len(pm_core)
    if n_core == 0:
        return [], []

    def compute_joins(points: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n_pts = len(points)
        if n_pts == 0:
            return [], []

        def compute_join_point(i: int, side: str) -> np.ndarray:
            p = points[i]
            prev_idx = _find_prev_distinct(points, i, closed_effective, eps_len)
            next_idx = _find_next_distinct(points, i, closed_effective, eps_len)

            t_prev = _unit(p - points[prev_idx], eps=eps_len) if prev_idx is not None else np.zeros(2, dtype=float)
            t_next = _unit(points[next_idx] - p, eps=eps_len) if next_idx is not None else np.zeros(2, dtype=float)
            if np.linalg.norm(t_prev) < eps_len and np.linalg.norm(t_next) < eps_len:
                return p.copy()
            if np.linalg.norm(t_prev) < eps_len:
                t_prev = t_next
            if np.linalg.norm(t_next) < eps_len:
                t_next = t_prev

            is_endpoint = (not closed_effective) and (i == 0 or i == n_pts - 1)
            if is_endpoint:
                tangent = t_next if i == 0 else t_prev
                n = left_normal(tangent) if side == "left" else right_normal(tangent)
                return p + offset * n

            cross_val = cross2(t_prev, t_next)
            n_prev = left_normal(t_prev) if side == "left" else right_normal(t_prev)
            n_next = left_normal(t_next) if side == "left" else right_normal(t_next)

            m = _unit(n_prev + n_next, eps=eps_len)
            if np.linalg.norm(m) < eps_len:
                m = _unit(n_prev, eps=eps_len)
            denom = float(np.dot(m, n_prev))
            if abs(denom) < eps_miter:
                denom = eps_miter if denom >= 0.0 else -eps_miter
            miter_len = offset / denom
            limit = miter_limit * offset
            if abs(miter_len) > limit:
                miter_len = limit if miter_len >= 0.0 else -limit
            joined = p + m * miter_len

            # 凹角/翻折保护：若 join 落在“错误一侧”，退化为 bevel（用前一段法向）。
            if float(np.dot(joined - p, n_prev)) <= 0.0 or float(np.dot(joined - p, n_next)) <= 0.0:
                return p + offset * n_prev
            if abs(cross_val) > 1e-12 and miter_len <= 0.0:
                return p + offset * n_prev
            return joined

        left = [compute_join_point(i, side="left") for i in range(n_pts)]
        right = [compute_join_point(i, side="right") for i in range(n_pts)]
        return left, right

    def build_kept_indices(points: List[np.ndarray]) -> List[int]:
        n_pts = len(points)
        if n_pts <= 2:
            return list(range(n_pts))
        kept = [0]
        angle_eps = 1e-12
        if closed_effective:
            for i in range(1, n_pts):
                prev = points[(i - 1) % n_pts]
                cur = points[i]
                nxt = points[(i + 1) % n_pts]
                v1 = cur - prev
                v2 = nxt - cur
                if np.linalg.norm(v1) < eps_len or np.linalg.norm(v2) < eps_len:
                    kept.append(i)
                    continue
                t1 = v1 / float(np.linalg.norm(v1))
                t2 = v2 / float(np.linalg.norm(v2))
                if abs(cross2(t1, t2)) > angle_eps or float(np.dot(t1, t2)) < 0.0:
                    kept.append(i)
            kept = sorted(set(kept))
            if len(kept) < 3:
                kept = list(range(n_pts))
            return kept

        for i in range(1, n_pts - 1):
            prev = points[i - 1]
            cur = points[i]
            nxt = points[i + 1]
            v1 = cur - prev
            v2 = nxt - cur
            if np.linalg.norm(v1) < eps_len or np.linalg.norm(v2) < eps_len:
                kept.append(i)
                continue
            t1 = v1 / float(np.linalg.norm(v1))
            t2 = v2 / float(np.linalg.norm(v2))
            if abs(cross2(t1, t2)) > angle_eps or float(np.dot(t1, t2)) < 0.0:
                kept.append(i)
        kept.append(n_pts - 1)
        return kept

    kept = build_kept_indices(pm_core)
    pm_simplified = [pm_core[i] for i in kept]
    pl_s, pr_s = compute_joins(pm_simplified)

    def fill_from_simplified(boundary_s: List[np.ndarray]) -> List[np.ndarray]:
        full = [np.zeros(2, dtype=float) for _ in range(n_core)]

        def fill_run(run_indices: List[int], b0: np.ndarray, b1: np.ndarray) -> None:
            if not run_indices:
                return
            if len(run_indices) == 1:
                full[run_indices[0]] = b0.copy()
                return
            dists = [0.0]
            for idx in range(1, len(run_indices)):
                a = pm_core[run_indices[idx - 1]]
                b = pm_core[run_indices[idx]]
                dists.append(dists[-1] + float(np.linalg.norm(b - a)))
            total = dists[-1]
            if total < eps_len:
                for k in run_indices:
                    full[k] = b0.copy()
                return
            for idx, k in enumerate(run_indices):
                alpha = dists[idx] / total
                full[k] = (1.0 - alpha) * b0 + alpha * b1

        for s_idx in range(len(kept)):
            start = kept[s_idx]
            end = kept[(s_idx + 1) % len(kept)] if closed_effective else kept[s_idx + 1] if s_idx + 1 < len(kept) else None
            if end is None:
                break
            b0 = boundary_s[s_idx]
            b1 = boundary_s[(s_idx + 1) % len(boundary_s)] if closed_effective else boundary_s[s_idx + 1]

            if not closed_effective and start > end:
                continue
            if closed_effective:
                if start <= end:
                    run = list(range(start, end + 1))
                else:
                    run = list(range(start, n_core)) + list(range(0, end + 1))
            else:
                run = list(range(start, end + 1))
            fill_run(run, b0, b1)
        return full

    pl_core = fill_from_simplified(pl_s)
    pr_core = fill_from_simplified(pr_s)

    if has_duplicate_last:
        return pl_core + [pl_core[0]], pr_core + [pr_core[0]]
    return pl_core, pr_core


def segment_intersects(
    a1: Sequence[float],
    a2: Sequence[float],
    b1: Sequence[float],
    b2: Sequence[float],
    eps: float = 1e-9,
) -> bool:
    """判断两线段是否相交（包含端点），用于自检与防御。"""
    a1p = np.asarray(a1, dtype=float)
    a2p = np.asarray(a2, dtype=float)
    b1p = np.asarray(b1, dtype=float)
    b2p = np.asarray(b2, dtype=float)

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return cross2(q - p, r - p)

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orient(a1p, a2p, b1p)
    o2 = orient(a1p, a2p, b2p)
    o3 = orient(b1p, b2p, a1p)
    o4 = orient(b1p, b2p, a2p)

    if (o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps):
        if (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps):
            return True

    if abs(o1) <= eps and on_segment(a1p, b1p, a2p):
        return True
    if abs(o2) <= eps and on_segment(a1p, b2p, a2p):
        return True
    if abs(o3) <= eps and on_segment(b1p, a1p, b2p):
        return True
    if abs(o4) <= eps and on_segment(b1p, a2p, b2p):
        return True
    return False


def count_polyline_self_intersections(points: Sequence[Sequence[float]], closed: bool, eps: float = 1e-9) -> int:
    """统计折线（可闭合）的自交次数（忽略相邻段与共享端点）。"""
    pts = _as_point_list(points)
    if len(pts) < 4:
        return 0
    segments = _iter_segments(pts, closed=closed)
    count = 0
    for i, (a1, a2) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if abs(i - j) <= 1:
                continue
            if closed and i == 0 and j == len(segments) - 1:
                continue
            b1, b2 = segments[j]
            if segment_intersects(a1, a2, b1, b2, eps=eps):
                shared = (
                    np.linalg.norm(a1 - b1) <= eps
                    or np.linalg.norm(a1 - b2) <= eps
                    or np.linalg.norm(a2 - b1) <= eps
                    or np.linalg.norm(a2 - b2) <= eps
                )
                if shared:
                    continue
                count += 1
    return count


def quad_is_degenerate(quad: Sequence[Sequence[float]], eps_len: float = 1e-9, eps_area: float = 1e-12) -> bool:
    if len(quad) != 4:
        return True
    pts = _as_point_list(quad)
    for i in range(4):
        if np.linalg.norm(pts[(i + 1) % 4] - pts[i]) <= eps_len:
            return True
    area2 = 0.0
    for i in range(4):
        x1, y1 = float(pts[i][0]), float(pts[i][1])
        x2, y2 = float(pts[(i + 1) % 4][0]), float(pts[(i + 1) % 4][1])
        area2 += x1 * y2 - y1 * x2
    return abs(area2) * 0.5 <= eps_area


def quad_self_intersects(quad: Sequence[Sequence[float]], eps: float = 1e-9) -> bool:
    """四边形是否自交（蝴蝶形）。"""
    if len(quad) != 4:
        return True
    p = _as_point_list(quad)
    return segment_intersects(p[0], p[1], p[2], p[3], eps=eps) or segment_intersects(p[1], p[2], p[3], p[0], eps=eps)


def is_point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """射线法判断点是否在多边形内，先做包围盒快速过滤。"""
    if not polygon:
        return False

    x, y = point
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False

    inside = False
    p1x, p1y = polygon[0]
    n = len(polygon)
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_to_line_distance(pt: Sequence[float], A: Sequence[float], B: Sequence[float]) -> float:
    """计算点到直线的垂直距离，使用叉积避免除零。"""
    AB = np.asarray(B, dtype=float) - np.asarray(A, dtype=float)
    AP = np.asarray(pt, dtype=float) - np.asarray(A, dtype=float)
    cross_abs = abs(AB[0] * AP[1] - AB[1] * AP[0])
    length_AB = np.linalg.norm(AB)
    if length_AB < 1e-6:
        return float(np.linalg.norm(AP))
    return float(cross_abs / length_AB)


def project_point_to_segment(pt: Sequence[float], p1: Sequence[float], p2: Sequence[float]) -> np.ndarray:
    """将点投影到线段上的最近点，投影落在延长线时会超出原段。"""
    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    vec_seg = p2_arr - p1_arr
    vec_pt = np.asarray(pt, dtype=float) - p1_arr
    denom = float(np.dot(vec_seg, vec_seg))
    if denom < 1e-6:
        return p1_arr.copy()
    t = np.dot(vec_pt, vec_seg) / denom
    return p1_arr + t * vec_seg


def compute_path_segments_length(waypoints: List[np.ndarray], closed: bool = False) -> List[float]:
    """计算路径各段的长度。"""
    n = len(waypoints)
    if n < 2:
        return []
    return [float(np.linalg.norm(waypoints[i + 1] - waypoints[i])) for i in range(n - 1)]


def compute_path_angles(waypoints: List[np.ndarray], closed: bool = False) -> List[float]:
    """计算路径拐点处的转角，逆时针为正。"""
    n = len(waypoints)
    if n < 3:
        return []
    angles: List[float] = []
    n_angles = n - 1 if closed else n
    for i in range(n_angles):
        if not closed and (i == 0 or i == n - 1):
            continue
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        p0 = waypoints[prev_idx]
        p1 = waypoints[i]
        p2 = waypoints[next_idx]
        angles.append(angle_between_vectors(p1 - p0, p2 - p1))
    return angles


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-π, π]。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi
