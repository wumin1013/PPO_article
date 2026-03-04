"""
路径生成器模块
聚焦论文核心实验场景的路径生成实现。
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import splev, splprep


def _resample_path_by_arclength(
    points: List[np.ndarray],
    num_points: int,
    closed: bool = False,
) -> List[np.ndarray]:
    """按弧长对离散路径重采样为指定点数。"""
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    if not points:
        raise ValueError("points must not be empty.")

    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must be 2D coordinates.")

    if closed and not np.allclose(arr[0], arr[-1], atol=1e-9):
        arr = np.vstack([arr, arr[0]])

    seg = np.diff(arr, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return [arr[0].copy() for _ in range(num_points)]

    targets = np.linspace(0.0, total, num_points)
    out: List[np.ndarray] = []
    for s in targets:
        idx = int(np.searchsorted(cumulative, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(seg_len) - 1))
        length = float(seg_len[idx])
        if length <= 1e-12:
            out.append(arr[idx].copy())
            continue
        t = float((s - cumulative[idx]) / length)
        out.append(arr[idx] + t * (arr[idx + 1] - arr[idx]))

    if closed:
        out[-1] = out[0].copy()
    return [np.asarray(p, dtype=float) for p in out]


def _sample_polyline_keep_vertices(
    nodes: List[np.ndarray],
    num_points: int,
    closed: bool = True,
) -> List[np.ndarray]:
    """按段长分配采样并保留折点，避免角点在重采样时被抹平。"""
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")
    if len(nodes) < 2:
        raise ValueError("nodes must contain at least 2 points.")

    pts = [np.asarray(p, dtype=float) for p in nodes]
    if closed and not np.allclose(pts[0], pts[-1], atol=1e-9):
        pts.append(pts[0].copy())

    edges = len(pts) - 1
    if edges <= 0:
        return [pts[0].copy() for _ in range(num_points)]

    seg_lengths = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(edges)]
    total = max(1e-12, float(sum(seg_lengths)))
    target = int(num_points) - 1
    counts = [max(1, int(round(target * l / total))) for l in seg_lengths]
    delta = target - int(sum(counts))
    if delta != 0:
        order = list(np.argsort(seg_lengths))[::-1]
        k = 0
        while delta != 0 and order:
            idx = int(order[k % len(order)])
            if delta > 0:
                counts[idx] += 1
                delta -= 1
            elif counts[idx] > 1:
                counts[idx] -= 1
                delta += 1
            k += 1

    out: List[np.ndarray] = [pts[0].copy()]
    for i in range(edges):
        n = counts[i]
        p1 = pts[i]
        p2 = pts[i + 1]
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            out.append(p1 + float(t) * (p2 - p1))

    if closed:
        out[-1] = out[0].copy()
    return [np.asarray(p, dtype=float) for p in out]


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
    closed: bool = True,
    start_offset_ratio: float = 0.0,
) -> List[np.ndarray]:
    """
    生成从(0,0)出发、逆时针的正方形路径。

    - closed=True：四边闭合（回到起点）
    - closed=False：open square（仅 3 条边，不回到起点），用于“有拐角且有终点”的训练/验收场景

    Args:
        side_length: 正方形边长。
        num_points: 总采样点数量。
            - closed=True：包含起点重复以闭合路径，需≥5。
            - closed=False：不包含闭合点，需≥4。
        closed: 是否闭合。
        start_offset_ratio: 闭合路径起点沿首边的偏移比例（0~1），0表示顶点起点。
    """
    if closed and num_points < 5:
        raise ValueError("num_points must be at least 5 to form a closed square path.")
    if not closed and num_points < 4:
        raise ValueError("num_points must be at least 4 to form an open square path.")

    edges = 4 if closed else 3
    points_to_distribute = num_points - 1  # 预留首点
    if closed:
        points_to_distribute = num_points - 1  # 预留一个点用于闭合（重复起点）
    base = points_to_distribute // edges
    remainder = points_to_distribute % edges
    counts = [base + (1 if i < remainder else 0) for i in range(edges)]

    vertices = [
        np.array([0.0, 0.0]),
        np.array([side_length, 0.0]),
        np.array([side_length, side_length]),
        np.array([0.0, side_length]),
    ]
    if closed:
        vertices.append(np.array([0.0, 0.0]))

    if closed:
        ratio = float(np.clip(start_offset_ratio, 0.0, 1.0))
        offset = ratio * side_length
        if 1e-9 < offset < (side_length - 1e-9):
            start = vertices[0] + (offset / side_length) * (vertices[1] - vertices[0])
            nodes = [start, vertices[1], vertices[2], vertices[3], vertices[0], start]
            return _sample_polyline_keep_vertices(nodes, num_points=int(num_points), closed=True)

    path_points: List[np.ndarray] = [vertices[0]]
    for edge_idx in range(edges):
        start = vertices[edge_idx]
        end = vertices[edge_idx + 1]
        count = counts[edge_idx]
        if count <= 0:
            continue

        # 在边上均匀插值，跳过首点避免重复
        for t in np.linspace(0.0, 1.0, count + 1)[1:]:
            point = start + t * (end - start)
            path_points.append(point)

    if closed:
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


def generate_circle_path(
    scale: float = 10.0,
    num_points: int = 240,
    closed: bool = True,
) -> List[np.ndarray]:
    """
    生成圆形路径（连续曲率）。
    - scale: 直径
    """
    if num_points < (4 if closed else 3):
        raise ValueError("num_points too small for circle path.")
    radius = float(scale) / 2.0
    sample_count = int(num_points) - 1 if closed else int(num_points)
    t = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    path = [np.array([radius * math.cos(tt), radius * math.sin(tt)], dtype=float) for tt in t]
    if closed:
        path.append(path[0].copy())
    return path


def generate_trapezoid_path(
    scale: float = 10.0,
    num_points: int = 220,
    top_ratio: float = 0.5,
    height_ratio: float = 0.75,
    start_offset_ratio: float = 0.0,
    closed: bool = True,
) -> List[np.ndarray]:
    """
    生成对称梯形路径。
    - scale: 底边长度
    - top_ratio: 顶边与底边的比例
    - height_ratio: 高度与底边的比例
    """
    min_points = 5 if closed else 4
    if num_points < min_points:
        raise ValueError(f"num_points must be at least {min_points}.")
    top_ratio = float(np.clip(top_ratio, 0.1, 1.0))
    height_ratio = float(np.clip(height_ratio, 0.1, 2.0))
    start_offset_ratio = float(np.clip(start_offset_ratio, 0.0, 1.0))
    bottom = float(scale)
    top = bottom * top_ratio
    height = bottom * height_ratio

    bottom_left = np.array([-bottom / 2.0, 0.0], dtype=float)
    bottom_right = np.array([bottom / 2.0, 0.0], dtype=float)
    top_right = np.array([top / 2.0, height], dtype=float)
    top_left = np.array([-top / 2.0, height], dtype=float)

    # 闭环梯形支持底边偏移起点；ratio=0.5 即底边中点。
    if closed:
        start = bottom_left + start_offset_ratio * (bottom_right - bottom_left)
        nodes = [start, bottom_right, top_right, top_left, bottom_left, start]
        return _sample_polyline_keep_vertices(nodes, num_points=int(num_points), closed=True)

    v = [bottom_left, bottom_right, top_right, top_left]
    if closed:
        v.append(v[0].copy())

    edges = len(v) - 1
    edge_lengths = [float(np.linalg.norm(v[i + 1] - v[i])) for i in range(edges)]
    total = max(1e-12, sum(edge_lengths))
    counts = [max(1, int(round((num_points - 1) * l / total))) for l in edge_lengths]
    # 修正总点数，使闭合后恰好 num_points
    delta = (num_points - 1) - sum(counts)
    if delta != 0:
        order = np.argsort(edge_lengths)[::-1]
        k = 0
        while delta != 0 and len(order) > 0:
            idx = int(order[k % len(order)])
            if delta > 0:
                counts[idx] += 1
                delta -= 1
            elif counts[idx] > 1:
                counts[idx] -= 1
                delta += 1
            k += 1

    path: List[np.ndarray] = [v[0].copy()]
    for i in range(edges):
        n = counts[i]
        p1, p2 = v[i], v[i + 1]
        for t in np.linspace(0.0, 1.0, n + 1)[1:]:
            path.append(p1 + float(t) * (p2 - p1))

    if closed:
        path[-1] = path[0].copy()
    return [np.asarray(p, dtype=float) for p in path]


def _generate_butterfly_lemniscate(
    scale: float,
    num_points: int,
    wing_ratio: float,
    phase: float,
    closed: bool,
) -> List[np.ndarray]:
    """简单 8 字蝴蝶（历史实现，保留兼容）。"""
    min_points = 4 if closed else 3
    if num_points < min_points:
        raise ValueError(f"num_points must be at least {min_points}.")

    wing_ratio = float(np.clip(wing_ratio, 0.1, 2.0))
    a = float(scale) / 2.0
    b = float(scale) * wing_ratio / 2.0
    phase = float(phase)

    sample_count = int(num_points) - 1 if closed else int(num_points)
    t = np.linspace(0.0, 2.0 * math.pi, sample_count, endpoint=False)
    tt = t + phase

    x = a * np.sin(tt)
    y = b * np.sin(tt) * np.cos(tt)
    path = [np.array([x[i], y[i]], dtype=float) for i in range(sample_count)]
    if closed:
        path.append(path[0].copy())
    return path


def _generate_butterfly_academic(
    scale: float,
    num_points: int,
    wing_ratio: float,
    long_ratio: float,
    cross_ratio: float,
    closed: bool,
) -> List[np.ndarray]:
    """
    参考 Boon 图形风格的蝴蝶闭环（无尖角、无自交）。
    采用“长短直线趋势 + 圆弧过渡”的控制点模板，再用周期 B 样条平滑，
    保证中心线 Pm 连续光滑，避免偏移时出现尖点放大问题。
    """
    min_points = 24 if closed else 16
    if num_points < min_points:
        raise ValueError(f"num_points must be at least {min_points}.")

    wing_ratio = float(np.clip(wing_ratio, 0.7, 1.4))
    long_ratio = float(np.clip(long_ratio, 0.9, 1.9))
    cross_ratio = float(np.clip(cross_ratio, 0.05, 0.20))

    # 以 scale=40 为模板尺寸；long_ratio 控制横向长度，wing_ratio 控制纵向展开。
    sx = (float(scale) / 40.0) * long_ratio
    sy = (float(scale) / 40.0) * (0.90 + 0.25 * wing_ratio)
    waist_delta = (cross_ratio - 0.10) * 10.0  # 温和调节腰部开口

    # 顺时针控制点（无自交），外形匹配“长短直线+圆弧组合”的参考图。
    ctrl = np.array(
        [
            [-39.0, 17.0],
            [-31.0, 17.6],
            [-22.0, 17.2],
            [-8.0, 13.0],
            [2.0, 11.0],
            [18.0, 15.5],
            [30.0, 15.8],
            [42.0, 8.0],
            [50.0, -3.0],
            [45.0, -7.5],
            [34.0, -8.5],
            [30.0, -16.8],
            [23.0, -17.8],
            [8.0, -8.2],
            [0.0, -6.0],
            [-8.0, -8.2],
            [-23.0, -17.8],
            [-30.0, -16.8],
            [-34.0, -8.5],
            [-45.0, -7.5],
            [-50.0, 2.0],
            [-44.0, 9.5],
        ],
        dtype=float,
    )

    # 腰部调节：top-middle 下压、bottom-middle 上抬。
    top_ids = [3, 4]
    bot_ids = [13, 14, 15]
    ctrl[top_ids, 1] -= waist_delta
    ctrl[bot_ids, 1] += waist_delta

    ctrl[:, 0] *= sx
    ctrl[:, 1] *= sy
    ctrl = np.vstack([ctrl, ctrl[0]])

    # 用周期 B 样条得到连续光滑闭环基线。
    smooth_s = 4.0 * (float(scale) / 40.0) ** 2
    tck, _ = splprep([ctrl[:, 0], ctrl[:, 1]], s=smooth_s, k=3, per=True)
    sample_count = int(num_points) - 1 if closed else int(num_points)
    u = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    x, y = splev(u, tck)
    pm = np.column_stack([x, y]).astype(float)

    def _nearest_idx(poly: np.ndarray, pt: np.ndarray) -> int:
        d2 = np.sum((poly - pt) ** 2, axis=1)
        return int(np.argmin(d2))

    def _replace_section_with_polyline(
        poly: np.ndarray,
        start_idx: int,
        end_idx: int,
        anchors: np.ndarray,
    ) -> None:
        n = int(poly.shape[0])
        s = int(start_idx)
        e = int(end_idx)
        if e < s:
            e += n
        idxs = np.arange(s, e + 1, dtype=int)
        m = int(len(idxs))
        if m <= 1:
            return

        seg = np.diff(anchors, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        total = float(np.sum(seg_len))
        if total <= 1e-12:
            return
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])

        targets = np.linspace(0.0, total, m)
        repl = np.zeros((m, 2), dtype=float)
        for k, dist in enumerate(targets):
            j = int(np.searchsorted(cum, dist, side="right") - 1)
            j = int(np.clip(j, 0, len(seg_len) - 1))
            l = float(seg_len[j])
            if l <= 1e-12:
                repl[k] = anchors[j]
                continue
            t = float((dist - cum[j]) / l)
            repl[k] = anchors[j] + t * (anchors[j + 1] - anchors[j])

        for k, ii in enumerate(idxs):
            poly[ii % n] = repl[k]

    # 在中部嵌入“长短直线穿插”段（参考用户给图）：
    # - 上中段：短水平 + 长下斜 + 长上斜 + 短水平
    # - 下中段：短斜线 + 长斜线 + 短线 + 长斜线 + 短斜线
    top_anchors = np.array(
        [
            [-30.0, 17.0],
            [-18.0, 17.0],
            [-4.0, 12.0 - 0.5 * waist_delta],
            [18.0, 16.0],
            [30.0, 16.0],
        ],
        dtype=float,
    )
    bot_anchors = np.array(
        [
            [30.0, -16.2],
            [22.0, -17.0],
            [8.0, -8.2 + waist_delta],
            [0.0, -6.0 + 0.6 * waist_delta],
            [-8.0, -8.2 + waist_delta],
            [-22.0, -17.0],
            [-30.0, -16.2],
        ],
        dtype=float,
    )
    top_anchors[:, 0] *= sx
    top_anchors[:, 1] *= sy
    bot_anchors[:, 0] *= sx
    bot_anchors[:, 1] *= sy

    top_start = _nearest_idx(pm, top_anchors[0])
    top_end = _nearest_idx(pm, top_anchors[-1])
    _replace_section_with_polyline(pm, top_start, top_end, top_anchors)

    bot_start = _nearest_idx(pm, bot_anchors[0])
    bot_end = _nearest_idx(pm, bot_anchors[-1])
    _replace_section_with_polyline(pm, bot_start, bot_end, bot_anchors)

    raw = [np.array([pm[i, 0], pm[i, 1]], dtype=float) for i in range(pm.shape[0])]

    if closed:
        raw.append(raw[0].copy())

    return _resample_path_by_arclength(raw, num_points=int(num_points), closed=bool(closed))


def generate_butterfly_path(
    scale: float = 10.0,
    num_points: int = 300,
    wing_ratio: float = 0.6,
    phase: float = math.pi / 2.0,
    long_ratio: float = 1.2,
    cross_ratio: float = 0.08,
    style: str = "academic",
    closed: bool = True,
) -> List[np.ndarray]:
    """
    生成蝴蝶路径。
    - style='academic'：长短直线交叉 + 弧线（默认）
    - style='lemniscate'：简单8字（兼容旧配置）
    """
    style_norm = str(style).strip().lower()
    if style_norm in {"8", "simple", "lemniscate"}:
        return _generate_butterfly_lemniscate(
            scale=scale,
            num_points=num_points,
            wing_ratio=wing_ratio,
            phase=phase,
            closed=closed,
        )
    return _generate_butterfly_academic(
        scale=scale,
        num_points=num_points,
        wing_ratio=wing_ratio,
        long_ratio=long_ratio,
        cross_ratio=cross_ratio,
        closed=closed,
    )


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

    支持: 'line', 'square', 's_shape', 's_shape_bspline', 'butterfly', 'trapezoid', 'circle'
    """
    path_generators = {
        "line": generate_line_path,
        "square": generate_square_path,
        "s_shape": generate_s_shape_path,
        "s_shape_bspline": generate_s_shape_bspline,
        "butterfly": generate_butterfly_path,
        "trapezoid": generate_trapezoid_path,
        "circle": generate_circle_path,
        "sharp_angle": generate_sharp_angle_path,
    }

    if path_name not in path_generators:
        raise ValueError(f"未知路径类型: {path_name}. 可用类型: {list(path_generators.keys())}")

    generator = path_generators[path_name]

    if path_name == "line":
        return generator(length=scale, num_points=num_points, angle=kwargs.get("angle", 0.0))
    if path_name == "square":
        return generator(
            side_length=scale,
            num_points=num_points,
            closed=bool(kwargs.get("closed", True)),
            start_offset_ratio=float(kwargs.get("start_offset_ratio", 0.0)),
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
    if path_name == "butterfly":
        return generator(
            scale=scale,
            num_points=num_points,
            wing_ratio=kwargs.get("wing_ratio", 0.6),
            phase=kwargs.get("phase", math.pi / 2.0),
            long_ratio=kwargs.get("long_ratio", 1.2),
            cross_ratio=kwargs.get("cross_ratio", 0.08),
            style=kwargs.get("style", "academic"),
            closed=bool(kwargs.get("closed", True)),
        )
    if path_name == "trapezoid":
        return generator(
            scale=scale,
            num_points=num_points,
            top_ratio=kwargs.get("top_ratio", 0.5),
            height_ratio=kwargs.get("height_ratio", 0.75),
            start_offset_ratio=kwargs.get("start_offset_ratio", 0.0),
            closed=bool(kwargs.get("closed", True)),
        )
    if path_name == "circle":
        return generator(
            scale=scale,
            num_points=num_points,
            closed=bool(kwargs.get("closed", True)),
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
    "generate_s_shape_path",
    "generate_s_shape_bspline",
    "generate_trapezoid_path",
    "generate_circle_path",
    "generate_butterfly_path",
    "generate_sharp_angle_path",
    "get_path_by_name",
    "compute_path_length",
    "compute_path_curvature",
]
