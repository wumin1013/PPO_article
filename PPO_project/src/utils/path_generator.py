"""
路径生成器模块
聚焦论文核心实验场景的路径生成实现。
"""

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
        "s_shape": generate_s_shape_path,
        "s_shape_bspline": generate_s_shape_bspline,
    }

    if path_name not in path_generators:
        raise ValueError(f"未知路径类型: {path_name}. 可用类型: {list(path_generators.keys())}")

    generator = path_generators[path_name]

    if path_name == "line":
        return generator(length=scale, num_points=num_points, angle=kwargs.get("angle", 0.0))
    if path_name == "square":
        return generator(side_length=scale, num_points=num_points)
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
    "generate_s_shape_path",
    "generate_s_shape_bspline",
    "get_path_by_name",
    "compute_path_length",
    "compute_path_curvature",
]
