"""
路径生成器模块
聚焦论文核心实验场景的路径生成实现。
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep


def generate_line_path(
    length: float = 10.0,
    num_points: int = 200,
    angle: float = 0.0,
) -> List[np.ndarray]:
    """
    生成从原点出发的直线路径。

    Args:
        length: 直线长度。
        num_points: 采样点数量。
        angle: 与x轴的夹角（弧度），逆时针为正。
    """
    distances = np.linspace(0.0, length, num_points)
    direction = np.array([np.cos(angle), np.sin(angle)])
    return [dist * direction for dist in distances]


def generate_square_path(
    side_length: float = 10.0,
    num_points: int = 200,
) -> List[np.ndarray]:
    """
    生成从(0,0)出发、逆时针的正方形路径，确保四边点均匀分布且闭合。

    Args:
        side_length: 正方形边长。
        num_points: 总采样点数量（包含起点重复以闭合路径），需≥5。
    """
    if num_points < 5:
        raise ValueError("num_points must be at least 5 to form a closed square path.")

    points_to_distribute = num_points - 1  # 预留一个点用于闭合
    base = points_to_distribute // 4
    remainder = points_to_distribute % 4
    counts = [base + (1 if i < remainder else 0) for i in range(4)]

    vertices = [
        np.array([0.0, 0.0]),
        np.array([side_length, 0.0]),
        np.array([side_length, side_length]),
        np.array([0.0, side_length]),
        np.array([0.0, 0.0]),
    ]

    path_points: List[np.ndarray] = [vertices[0]]
    for edge_idx in range(4):
        start = vertices[edge_idx]
        end = vertices[edge_idx + 1]
        count = counts[edge_idx]
        if count <= 0:
            continue

        # 在边上均匀插值，跳过首点避免重复
        for t in np.linspace(0.0, 1.0, count + 1)[1:]:
            point = start + t * (end - start)
            path_points.append(point)

    # 确保闭合：最后一个点即为起点
    if not np.allclose(path_points[-1], vertices[0]):
        path_points.append(vertices[0].copy())

    return [np.array(p) for p in path_points]


def generate_parallelogram_path(
    base_length: float = 10.0,
    side_length: float = 6.0,
    angle_deg: float = 60.0,
    num_points: int = 200,
) -> List[np.ndarray]:
    """
    生成包含锐角的平行四边形闭环路径（起点重复以闭合）。

    Args:
        base_length: 底边长度。
        side_length: 侧边长度。
        angle_deg: 底边与侧边夹角（度）。<90 为锐角平行四边形。
        num_points: 总采样点数量（包含起点重复以闭合路径），需≥5。
    """
    if num_points < 5:
        raise ValueError("num_points must be at least 5 to form a closed parallelogram path.")
    if base_length <= 0 or side_length <= 0:
        raise ValueError("base_length and side_length must be positive.")
    if not (0.0 < angle_deg < 180.0):
        raise ValueError("angle_deg must be in (0, 180).")

    theta = float(angle_deg) * math.pi / 180.0
    u = np.array([side_length * math.cos(theta), side_length * math.sin(theta)], dtype=float)

    p0 = np.array([0.0, 0.0], dtype=float)
    p1 = np.array([float(base_length), 0.0], dtype=float)
    p2 = p1 + u
    p3 = p0 + u

    vertices = [p0, p1, p2, p3, p0]

    points_to_distribute = num_points - 1
    edge_lengths = np.array(
        [
            float(np.linalg.norm(p1 - p0)),
            float(np.linalg.norm(p2 - p1)),
            float(np.linalg.norm(p3 - p2)),
            float(np.linalg.norm(p0 - p3)),
        ],
        dtype=float,
    )
    total = float(edge_lengths.sum())
    if total <= 1e-12:
        raise ValueError("Degenerate parallelogram.")

    raw = edge_lengths / total * points_to_distribute
    counts = np.floor(raw).astype(int)
    remainder = points_to_distribute - int(counts.sum())
    if remainder > 0:
        frac_order = np.argsort(-(raw - counts))
        for k in range(remainder):
            counts[int(frac_order[k % 4])] += 1

    path_points: List[np.ndarray] = [vertices[0]]
    for edge_idx in range(4):
        start = vertices[edge_idx]
        end = vertices[edge_idx + 1]
        count = int(counts[edge_idx])
        if count <= 0:
            continue
        for t in np.linspace(0.0, 1.0, count + 1)[1:]:
            path_points.append(start + t * (end - start))

    if not np.allclose(path_points[-1], vertices[0]):
        path_points.append(vertices[0].copy())
    return [np.array(p) for p in path_points]


def generate_s_shape_path(
    scale: float = 10.0,
    num_points: int = 200,
    amplitude: float = 5.0,
    periods: float = 2.0,
) -> List[np.ndarray]:
    """
    生成S形路径（基于正弦函数的平滑曲线）。
    """
    t = np.linspace(0.0, 1.0, num_points)
    x = scale * t
    y = amplitude * np.sin(2 * np.pi * periods * t)
    return [np.array([x[i], y[i]]) for i in range(num_points)]


def generate_s_shape_bspline(
    scale: float = 10.0,
    num_points: int = 200,
    control_points: Optional[List[Tuple[float, float]]] = None,
    smoothing: float = 0.0,
) -> List[np.ndarray]:
    """
    生成基于B样条的S形平滑曲线。
    """
    if control_points is None:
        control_points = [
            (0.0, 0.0),
            (2.0, 3.0),
            (5.0, 5.0),
            (8.0, 3.0),
            (10.0, 0.0),
            (12.0, -3.0),
            (15.0, -5.0),
            (18.0, -3.0),
            (20.0, 0.0),
        ]

    scaled_points = [(p[0] * scale / 20.0, p[1] * scale / 20.0) for p in control_points]
    x_ctrl = [p[0] for p in scaled_points]
    y_ctrl = [p[1] for p in scaled_points]

    tck, _ = splprep([x_ctrl, y_ctrl], s=smoothing, k=3)
    u_new = np.linspace(0.0, 1.0, num_points)
    x_new, y_new = splev(u_new, tck)

    return [np.array([x_new[i], y_new[i]]) for i in range(num_points)]


def generate_sharp_angle_path(
    segment_length: float = 10.0,
    turn_angle_deg: float = 30.0,
    num_points: int = 200,
) -> List[np.ndarray]:
    """
    生成包含锐角拐点的折线路径（open）。

    说明：
    - 起点在 (-L, 0)，拐点在 (0, 0)，终点沿 turn_angle_deg 方向延伸 L。
    - 拐点夹角约为 turn_angle_deg（越小越“尖”）。
    """
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    L = float(segment_length)
    theta = float(turn_angle_deg) * math.pi / 180.0
    p0 = np.array([-L, 0.0], dtype=float)
    p1 = np.array([0.0, 0.0], dtype=float)
    p2 = np.array([L * math.cos(theta), L * math.sin(theta)], dtype=float)

    n1 = max(2, num_points // 2)
    n2 = max(2, num_points - n1 + 1)  # +1 是为了让总点数接近 num_points，同时避免丢尾点

    pts: List[np.ndarray] = []
    for t in np.linspace(0.0, 1.0, n1, endpoint=False):
        pts.append(p0 + t * (p1 - p0))
    for t in np.linspace(0.0, 1.0, n2, endpoint=True):
        pts.append(p1 + t * (p2 - p1))
    return pts[:num_points]


def get_path_by_name(
    path_name: str,
    scale: float = 10.0,
    num_points: int = 200,
    **kwargs,
) -> List[np.ndarray]:
    """
    根据名称获取路径。

    支持: 'line', 'square', 's_shape', 's_shape_bspline'
    """
    path_generators = {
        "line": generate_line_path,
        "square": generate_square_path,
        "parallelogram": generate_parallelogram_path,
        "s_shape": generate_s_shape_path,
        "s_shape_bspline": generate_s_shape_bspline,
        "sharp_angle": generate_sharp_angle_path,
    }

    if path_name not in path_generators:
        raise ValueError(f"未知路径类型: {path_name}. 可用类型: {list(path_generators.keys())}")

    generator = path_generators[path_name]

    if path_name == "line":
        return generator(length=scale, num_points=num_points, angle=kwargs.get("angle", 0.0))
    if path_name == "square":
        return generator(side_length=scale, num_points=num_points)
    if path_name == "parallelogram":
        return generator(
            base_length=kwargs.get("base_length", scale),
            side_length=kwargs.get("side_length", scale * 0.6),
            angle_deg=kwargs.get("angle_deg", 60.0),
            num_points=num_points,
        )
    if path_name == "s_shape":
        return generator(
            scale=scale,
            num_points=num_points,
            amplitude=kwargs.get("amplitude", scale / 2),
            periods=kwargs.get("periods", 2.0),
        )
    if path_name == "s_shape_bspline":
        return generator(
            scale=scale,
            num_points=num_points,
            control_points=kwargs.get("control_points"),
            smoothing=kwargs.get("smoothing", 0.0),
        )
    if path_name == "sharp_angle":
        return generator(
            segment_length=kwargs.get("segment_length", scale),
            turn_angle_deg=kwargs.get("turn_angle_deg", 30.0),
            num_points=num_points,
        )

    return generator(scale=scale, num_points=num_points)


def compute_path_length(path_points: List[np.ndarray]) -> float:
    """计算路径总长度。"""
    total_length = 0.0
    for i in range(len(path_points) - 1):
        total_length += np.linalg.norm(path_points[i + 1] - path_points[i])
    return total_length


def compute_path_curvature(path_points: List[np.ndarray]) -> List[float]:
    """计算路径各点的曲率。"""
    curvatures = [0.0]
    for i in range(1, len(path_points) - 1):
        p1 = path_points[i - 1]
        p2 = path_points[i]
        p3 = path_points[i + 1]

        v1 = p2 - p1
        v2 = p3 - p2
        cross_val = np.cross(v1, v2)

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        len3 = np.linalg.norm(p3 - p1)

        if len1 * len2 * len3 > 1e-10:
            curvature = 2 * abs(cross_val) / (len1 * len2 * len3)
        else:
            curvature = 0.0

        curvatures.append(curvature)

    curvatures.append(0.0)
    return curvatures


__all__ = [
    "generate_line_path",
    "generate_square_path",
    "generate_parallelogram_path",
    "generate_s_shape_path",
    "generate_s_shape_bspline",
    "generate_sharp_angle_path",
    "get_path_by_name",
    "compute_path_length",
    "compute_path_curvature",
]
