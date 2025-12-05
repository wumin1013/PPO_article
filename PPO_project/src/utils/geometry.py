"""
几何计算工具模块
提取路径规划中常用的几何计算函数
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from numpy.linalg import norm


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """归一化向量"""
    length = norm(v)
    if length < 1e-6:
        return np.zeros_like(v)
    return v / length


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的夹角 (弧度)
    使用atan2确保正确的角度方向(逆时针为正)
    """
    len1 = norm(v1)
    len2 = norm(v2)
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 0.0
    
    dot_product = np.dot(v1, v2) / (len1 * len2)
    cross_product = np.cross(v1, v2) / (len1 * len2)
    
    return math.atan2(cross_product, dot_product)


def point_to_line_distance(point: np.ndarray, 
                          line_start: np.ndarray, 
                          line_end: np.ndarray) -> float:
    """计算点到线段的最短距离"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = norm(line_vec)
    
    if line_len < 1e-6:
        return norm(point_vec)
    
    line_unitvec = line_vec / line_len
    projection_length = np.dot(point_vec, line_unitvec)
    
    if projection_length < 0:
        return norm(point_vec)
    elif projection_length > line_len:
        return norm(point - line_end)
    else:
        projection = line_start + projection_length * line_unitvec
        return norm(point - projection)


def project_point_to_segment(point: np.ndarray,
                            line_start: np.ndarray,
                            line_end: np.ndarray) -> np.ndarray:
    """将点投影到线段上"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len_sq = np.dot(line_vec, line_vec)
    
    if line_len_sq < 1e-6:
        return line_start.copy()
    
    t = np.dot(point_vec, line_vec) / line_len_sq
    t = np.clip(t, 0, 1)
    
    return line_start + t * line_vec


def compute_path_segments_length(waypoints: List[np.ndarray], 
                                 closed: bool = False) -> List[float]:
    """计算路径各段的长度"""
    n = len(waypoints)
    if n < 2:
        return []
    
    lengths = []
    for i in range(n - 1):
        length = norm(waypoints[i + 1] - waypoints[i])
        lengths.append(length)
    
    # 闭合路径不需要额外计算(最后一点与第一点重合)
    return lengths


def compute_path_angles(waypoints: List[np.ndarray], 
                       closed: bool = False) -> List[float]:
    """
    计算路径拐点处的转角
    逆时针为正，顺时针为负
    """
    n = len(waypoints)
    if n < 3:
        return []
    
    angles = []
    
    # 确定需要计算角度的点范围
    if closed:
        # 闭合路径：所有点都计算角度(排除重复的最后一点)
        n_angles = n - 1
    else:
        # 非闭合路径：只计算中间点的角度
        n_angles = n
    
    for i in range(n_angles):
        # 跳过首尾点(非闭合路径)
        if not closed and (i == 0 or i == n - 1):
            continue
        
        # 获取三个连续点
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        
        p0 = waypoints[prev_idx]
        p1 = waypoints[i]
        p2 = waypoints[next_idx]
        
        # 计算向量和角度
        vec1 = p1 - p0
        vec2 = p2 - p1
        
        angle = angle_between_vectors(vec1, vec2)
        angles.append(angle)
    
    return angles


def generate_offset_paths(waypoints: List[np.ndarray],
                         offset_distance: float,
                         closed: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    生成路径的左右偏移边界
    
    Args:
        waypoints: 中心路径点列表
        offset_distance: 单侧偏移距离
        closed: 是否为闭合路径
    
    Returns:
        (left_boundary, right_boundary): 左右边界点列表
    """
    n = len(waypoints)
    if n < 2:
        return [], []
    
    left_boundary = [None] * n
    right_boundary = [None] * n
    
    def get_parallel_lines(p1, p2, dist):
        """计算平行线方程"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # 法向量(逆时针旋转90度)
        normal = np.array([-dy, dx])
        unit_normal = normalize_vector(normal)
        
        A, B = unit_normal
        
        def line_eq(distance, point=p1):
            C = -(A * point[0] + B * point[1]) + distance
            return A, B, C
        
        return line_eq(dist), line_eq(-dist)
    
    def find_intersection(line1, line2):
        """求两直线交点"""
        A1, B1, C1 = line1
        A2, B2, C2 = line2
        
        det = A1 * B2 - A2 * B1
        if abs(det) < 1e-6:
            return None
        
        x = (B1 * C2 - B2 * C1) / det
        y = (C1 * A2 - C2 * A1) / det
        
        return np.array([x, y])
    
    def offset_point(p, direction, dist):
        """简单偏移点(用于端点)"""
        return p + np.array([direction[1] * dist, -direction[0] * dist])
    
    # 处理每个点
    for i in range(n):
        if i == 0 and not closed:
            # 非闭合路径的起点
            p1, p2 = waypoints[i], waypoints[i + 1]
            direction = normalize_vector(p2 - p1)
            left_boundary[i] = offset_point(p1, direction, offset_distance)
            right_boundary[i] = offset_point(p1, direction, -offset_distance)
            
        elif i == n - 1 and not closed:
            # 非闭合路径的终点
            p1, p2 = waypoints[i - 1], waypoints[i]
            direction = normalize_vector(p2 - p1)
            left_boundary[i] = offset_point(p2, direction, offset_distance)
            right_boundary[i] = offset_point(p2, direction, -offset_distance)
            
        else:
            # 中间点或闭合路径
            if i == n - 1 and closed:
                # 闭合路径最后一点 = 第一点
                left_boundary[i] = left_boundary[0]
                right_boundary[i] = right_boundary[0]
            else:
                # 使用三点法计算交点
                prev_idx = (i - 1) % n
                next_idx = (i + 1) % n
                
                prev_p = waypoints[prev_idx]
                curr_p = waypoints[i]
                next_p = waypoints[next_idx]
                
                # 前一段和后一段的平行线
                l1, r1 = get_parallel_lines(prev_p, curr_p, offset_distance)
                l2, r2 = get_parallel_lines(curr_p, next_p, offset_distance)
                
                # 求交点
                left_int = find_intersection(l1, l2)
                right_int = find_intersection(r1, r2)
                
                # 处理平行情况
                if left_int is None:
                    direction = normalize_vector(next_p - curr_p)
                    left_int = offset_point(curr_p, direction, offset_distance)
                
                if right_int is None:
                    direction = normalize_vector(next_p - curr_p)
                    right_int = offset_point(curr_p, direction, -offset_distance)
                
                left_boundary[i] = left_int
                right_boundary[i] = right_int
    
    return left_boundary, right_boundary


def point_in_polygon(point: np.ndarray, polygon: List[np.ndarray]) -> bool:
    """判断点是否在多边形内(射线法)"""
    if not polygon or len(polygon) < 3:
        return False
    
    x, y = point
    n = len(polygon)
    inside = False
    
    p1 = polygon[0]
    for i in range(1, n + 1):
        p2 = polygon[i % n]
        
        if min(p1[1], p2[1]) < y <= max(p1[1], p2[1]):
            if x <= max(p1[0], p2[0]):
                if p1[1] != p2[1]:
                    xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or x <= xinters:
                    inside = not inside
        
        p1 = p2
    
    return inside


def wrap_angle(angle: float) -> float:
    """将角度归一化到 [-π, π]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi
