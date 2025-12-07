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
    pm = [np.array(p, dtype=float) for p in Pm]
    n = len(pm)
    if n == 0:
        return [], []

    if closed is None:
        closed = n > 2 and np.allclose(pm[0], pm[-1], atol=1e-6)

    pl: List[Optional[np.ndarray]] = [None] * n
    pr: List[Optional[np.ndarray]] = [None] * n
    offset = float(epsilon)

    def get_parallel_lines(p1: np.ndarray, p2: np.ndarray, offset_distance: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        normal_vector = np.array([-dy, dx], dtype=float)
        unit_normal = normalize_vector(normal_vector)
        A, B = unit_normal

        def line_equation(distance: float, point: np.ndarray = p1) -> Tuple[float, float, float]:
            C = -(A * point[0] + B * point[1]) + distance
            return A, B, C

        return line_equation(offset_distance), line_equation(-offset_distance)

    def offset_point(p: np.ndarray, direction: np.ndarray, distance: float) -> np.ndarray:
        return np.array([p[0] + direction[1] * distance, p[1] - direction[0] * distance], dtype=float)

    for i in range(n):
        if i == 0:
            if not closed:
                p1, p2 = pm[i], pm[i + 1]
                direction = normalize_vector(p2 - p1)
                pl[i] = offset_point(p1, direction, offset)
                pr[i] = offset_point(p1, direction, -offset)
            else:
                prev_point = pm[-2] if n >= 2 else pm[0]
                next_point = pm[i + 1]
                l1, r1 = get_parallel_lines(prev_point, pm[i], offset)
                l2, r2 = get_parallel_lines(pm[i], next_point, offset)
                pl[i] = find_intersection(l1, l2)
                pr[i] = find_intersection(r1, r2)
        elif i == n - 1:
            if not closed:
                p1, p2 = pm[i - 1], pm[i]
                direction = normalize_vector(p2 - p1)
                pl[i] = offset_point(p2, direction, offset)
                pr[i] = offset_point(p2, direction, -offset)
            else:
                pl[i] = pl[0]
                pr[i] = pr[0]
        else:
            prev_point = pm[i - 1]
            current_point = pm[i]
            next_point = pm[(i + 1) % n] if closed else pm[i + 1]
            l1, r1 = get_parallel_lines(prev_point, current_point, offset)
            l2, r2 = get_parallel_lines(current_point, next_point, offset)

            left = find_intersection(l1, l2)
            right = find_intersection(r1, r2)

            if left is None:
                direction = normalize_vector(next_point - current_point)
                left = offset_point(current_point, direction, offset)
            if right is None:
                direction = normalize_vector(next_point - current_point)
                right = offset_point(current_point, direction, -offset)

            pl[i] = left
            pr[i] = right

    return pl, pr


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
