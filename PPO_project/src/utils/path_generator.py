"""
路径生成器模块
实现论文实验所需的参数化路径生成函数（对应论文图7）

包含:
1. S-Shape Path: 基于正弦函数的平滑曲线
2. Butterfly Path: 具有尖锐转角的复杂闭合曲线
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev


def generate_s_shape_path(
    scale: float = 10.0,
    num_points: int = 200,
    amplitude: float = 5.0,
    periods: float = 2.0
) -> List[np.ndarray]:
    """
    生成S形路径（基于正弦函数的平滑曲线）
    
    路径方程:
        x(t) = scale * t
        y(t) = amplitude * sin(2π * periods * t)
    其中 t ∈ [0, 1]
    
    Args:
        scale: X方向的缩放比例（路径总长度）
        num_points: 采样点数量（采样密度）
        amplitude: Y方向振幅
        periods: 正弦波周期数
    
    Returns:
        密集点云数组 Pm，每个元素为 np.array([x, y])
    """
    t = np.linspace(0, 1, num_points)
    
    x = scale * t
    y = amplitude * np.sin(2 * np.pi * periods * t)
    
    # 构建点云数组
    path_points = [np.array([x[i], y[i]]) for i in range(num_points)]
    
    return path_points


def generate_s_shape_bspline(
    scale: float = 10.0,
    num_points: int = 200,
    control_points: Optional[List[Tuple[float, float]]] = None,
    smoothing: float = 0.0
) -> List[np.ndarray]:
    """
    生成基于B样条的S形平滑曲线
    
    Args:
        scale: 缩放比例
        num_points: 输出采样点数量
        control_points: 控制点列表，默认生成标准S形
        smoothing: B样条平滑因子
    
    Returns:
        密集点云数组 Pm
    """
    if control_points is None:
        # 默认S形控制点
        control_points = [
            (0.0, 0.0),
            (2.0, 3.0),
            (5.0, 5.0),
            (8.0, 3.0),
            (10.0, 0.0),
            (12.0, -3.0),
            (15.0, -5.0),
            (18.0, -3.0),
            (20.0, 0.0)
        ]
    
    # 应用缩放
    scaled_points = [(p[0] * scale / 20.0, p[1] * scale / 20.0) for p in control_points]
    
    # 分离x和y坐标
    x_ctrl = [p[0] for p in scaled_points]
    y_ctrl = [p[1] for p in scaled_points]
    
    # B样条拟合
    tck, u = splprep([x_ctrl, y_ctrl], s=smoothing, k=3)
    
    # 生成平滑曲线点
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    path_points = [np.array([x_new[i], y_new[i]]) for i in range(num_points)]
    
    return path_points


def generate_butterfly_path(
    scale: float = 5.0,
    num_points: int = 500,
    center: Tuple[float, float] = (0.0, 0.0)
) -> List[np.ndarray]:
    """
    生成蝶形路径（具有尖锐转角的复杂闭合曲线）
    
    使用蝶形曲线参数方程:
        r(t) = e^sin(t) - 2*cos(4t) + sin^5((2t-π)/24)
        x(t) = r(t) * cos(t)
        y(t) = r(t) * sin(t)
    其中 t ∈ [0, 2π]
    
    该曲线具有:
    - 闭合特性：起点和终点重合
    - 尖锐转角：在某些位置曲率变化剧烈
    - 复杂形态：类似蝴蝶翅膀的对称结构
    
    Args:
        scale: 缩放比例
        num_points: 采样点数量（采样密度）
        center: 曲线中心点坐标
    
    Returns:
        密集点云数组 Pm
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # 蝶形曲线极坐标方程
    r = np.exp(np.sin(t)) - 2 * np.cos(4 * t) + np.sin((2 * t - np.pi) / 24) ** 5
    
    # 转换为笛卡尔坐标
    x = scale * r * np.cos(t) + center[0]
    y = scale * r * np.sin(t) + center[1]
    
    # 构建点云数组
    path_points = [np.array([x[i], y[i]]) for i in range(num_points)]
    
    return path_points


def generate_butterfly_sharp(
    scale: float = 10.0,
    num_points: int = 400,
    center: Tuple[float, float] = (0.0, 0.0),
    sharpness: float = 1.0
) -> List[np.ndarray]:
    """
    生成带有更尖锐转角的蝶形路径变体
    
    使用修改的蝶形曲线方程，增强转角的尖锐程度:
        x(t) = sin(t) * (e^cos(t) - 2*cos(4t) - sin^5(t/12))
        y(t) = cos(t) * (e^cos(t) - 2*cos(4t) - sin^5(t/12))
    
    Args:
        scale: 缩放比例
        num_points: 采样点数量
        center: 曲线中心点坐标
        sharpness: 尖锐度参数（>1增强转角，<1平滑转角）
    
    Returns:
        密集点云数组 Pm
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    
    # 修改的蝶形曲线方程
    common_term = np.exp(np.cos(t)) - 2 * np.cos(4 * t) - np.sin(t / 12) ** 5
    
    # 应用尖锐度调整
    common_term = np.sign(common_term) * np.abs(common_term) ** sharpness
    
    x = scale * np.sin(t) * common_term + center[0]
    y = scale * np.cos(t) * common_term + center[1]
    
    # 构建点云数组
    path_points = [np.array([x[i], y[i]]) for i in range(num_points)]
    
    return path_points


def generate_lemniscate_path(
    scale: float = 10.0,
    num_points: int = 300,
    center: Tuple[float, float] = (0.0, 0.0)
) -> List[np.ndarray]:
    """
    生成双纽线（∞形）路径
    
    使用伯努利双纽线方程:
        x(t) = scale * cos(t) / (1 + sin²(t))
        y(t) = scale * sin(t) * cos(t) / (1 + sin²(t))
    
    特点：具有中心交叉点，是测试尖锐转角的理想路径
    
    Args:
        scale: 缩放比例
        num_points: 采样点数量
        center: 曲线中心点坐标
    
    Returns:
        密集点云数组 Pm
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    
    denom = 1 + np.sin(t) ** 2
    x = scale * np.cos(t) / denom + center[0]
    y = scale * np.sin(t) * np.cos(t) / denom + center[1]
    
    path_points = [np.array([x[i], y[i]]) for i in range(num_points)]
    
    return path_points


def generate_u_shape_path(
    width: float = 10.0,
    height: float = 10.0,
    num_points: int = 100
) -> List[np.ndarray]:
    """
    生成U形路径（原始测试路径）
    
    Args:
        width: U形宽度
        height: U形高度
        num_points: 总采样点数
    
    Returns:
        密集点云数组 Pm
    """
    points_per_segment = num_points // 4
    
    # 四个段：底边、右边、顶边、左边
    segment1 = [np.array([i * width / points_per_segment, 0.0]) 
                for i in range(points_per_segment)]
    segment2 = [np.array([width, i * height / points_per_segment]) 
                for i in range(points_per_segment)]
    segment3 = [np.array([width - i * width / points_per_segment, height]) 
                for i in range(points_per_segment)]
    segment4 = [np.array([0.0, height - i * height / points_per_segment]) 
                for i in range(points_per_segment)]
    
    path_points = segment1 + segment2 + segment3 + segment4
    path_points.append(np.array([0.0, 0.0]))  # 闭合路径
    
    return path_points


def generate_star_path(
    outer_radius: float = 10.0,
    inner_radius: float = 4.0,
    num_points: int = 5,
    samples_per_segment: int = 20,
    center: Tuple[float, float] = (0.0, 0.0)
) -> List[np.ndarray]:
    """
    生成星形路径（具有尖锐顶点）
    
    Args:
        outer_radius: 外圆半径（星尖位置）
        inner_radius: 内圆半径（星谷位置）
        num_points: 星形顶点数量
        samples_per_segment: 每段的采样点数
        center: 中心点坐标
    
    Returns:
        密集点云数组 Pm
    """
    path_points = []
    total_points = num_points * 2  # 顶点和谷点交替
    
    for i in range(total_points):
        angle = i * np.pi / num_points - np.pi / 2  # 从顶部开始
        
        if i % 2 == 0:
            # 外圆顶点
            x = outer_radius * np.cos(angle) + center[0]
            y = outer_radius * np.sin(angle) + center[1]
        else:
            # 内圆谷点
            x = inner_radius * np.cos(angle) + center[0]
            y = inner_radius * np.sin(angle) + center[1]
        
        path_points.append(np.array([x, y]))
        
        # 在顶点之间插值
        if i < total_points - 1:
            next_angle = (i + 1) * np.pi / num_points - np.pi / 2
            if (i + 1) % 2 == 0:
                next_x = outer_radius * np.cos(next_angle) + center[0]
                next_y = outer_radius * np.sin(next_angle) + center[1]
            else:
                next_x = inner_radius * np.cos(next_angle) + center[0]
                next_y = inner_radius * np.sin(next_angle) + center[1]
            
            for j in range(1, samples_per_segment):
                t = j / samples_per_segment
                interp_x = x + t * (next_x - x)
                interp_y = y + t * (next_y - y)
                path_points.append(np.array([interp_x, interp_y]))
    
    # 闭合路径
    path_points.append(path_points[0].copy())
    
    return path_points


def get_path_by_name(
    path_name: str,
    scale: float = 10.0,
    num_points: int = 200,
    **kwargs
) -> List[np.ndarray]:
    """
    根据名称获取路径
    
    Args:
        path_name: 路径名称 ('s_shape', 'butterfly', 'lemniscate', 'u_shape', 'star')
        scale: 缩放比例
        num_points: 采样点数
        **kwargs: 其他路径特定参数
    
    Returns:
        密集点云数组 Pm
    """
    path_generators = {
        's_shape': generate_s_shape_path,
        's_shape_bspline': generate_s_shape_bspline,
        'butterfly': generate_butterfly_path,
        'butterfly_sharp': generate_butterfly_sharp,
        'lemniscate': generate_lemniscate_path,
        'u_shape': generate_u_shape_path,
        'star': generate_star_path
    }
    
    if path_name not in path_generators:
        raise ValueError(f"未知路径类型: {path_name}. 可用类型: {list(path_generators.keys())}")
    
    generator = path_generators[path_name]
    
    # 根据不同路径类型传递参数
    if path_name == 's_shape':
        return generator(scale=scale, num_points=num_points, 
                        amplitude=kwargs.get('amplitude', scale/2),
                        periods=kwargs.get('periods', 2.0))
    elif path_name == 's_shape_bspline':
        return generator(scale=scale, num_points=num_points,
                        control_points=kwargs.get('control_points'),
                        smoothing=kwargs.get('smoothing', 0.0))
    elif path_name == 'butterfly':
        return generator(scale=scale, num_points=num_points,
                        center=kwargs.get('center', (0.0, 0.0)))
    elif path_name == 'butterfly_sharp':
        return generator(scale=scale, num_points=num_points,
                        center=kwargs.get('center', (0.0, 0.0)),
                        sharpness=kwargs.get('sharpness', 1.0))
    elif path_name == 'lemniscate':
        return generator(scale=scale, num_points=num_points,
                        center=kwargs.get('center', (0.0, 0.0)))
    elif path_name == 'u_shape':
        return generator(width=scale, height=scale, num_points=num_points)
    elif path_name == 'star':
        return generator(outer_radius=scale, 
                        inner_radius=kwargs.get('inner_radius', scale * 0.4),
                        num_points=kwargs.get('star_points', 5),
                        samples_per_segment=num_points // 10,
                        center=kwargs.get('center', (0.0, 0.0)))
    
    return generator(scale=scale, num_points=num_points)


def compute_path_length(path_points: List[np.ndarray]) -> float:
    """
    计算路径总长度
    
    Args:
        path_points: 路径点列表
    
    Returns:
        路径总长度
    """
    total_length = 0.0
    for i in range(len(path_points) - 1):
        total_length += np.linalg.norm(path_points[i+1] - path_points[i])
    return total_length


def compute_path_curvature(path_points: List[np.ndarray]) -> List[float]:
    """
    计算路径各点的曲率
    
    使用三点法估计曲率:
        κ = 2 * |cross(p2-p1, p3-p2)| / (|p2-p1| * |p3-p2| * |p3-p1|)
    
    Args:
        path_points: 路径点列表
    
    Returns:
        各点的曲率列表
    """
    curvatures = [0.0]  # 第一个点曲率设为0
    
    for i in range(1, len(path_points) - 1):
        p1 = path_points[i - 1]
        p2 = path_points[i]
        p3 = path_points[i + 1]
        
        v1 = p2 - p1
        v2 = p3 - p2
        
        cross = np.cross(v1, v2)
        
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        len3 = np.linalg.norm(p3 - p1)
        
        if len1 * len2 * len3 > 1e-10:
            curvature = 2 * abs(cross) / (len1 * len2 * len3)
        else:
            curvature = 0.0
        
        curvatures.append(curvature)
    
    curvatures.append(0.0)  # 最后一个点曲率设为0
    
    return curvatures
