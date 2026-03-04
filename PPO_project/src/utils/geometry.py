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

    eps_len = 1e-9
    eps_miter = 1e-9
    miter_limit = 8.0

    pm_full = _as_point_list(Pm)
    if not pm_full:
        return [], []

    pm_core, closed_effective, has_duplicate_last = _resolve_closed_core(pm_full, closed, eps_len=1e-6)
    n = len(pm_core)
    if n == 0:
        return [], []
    if n == 1:
        return [pm_core[0].copy()], [pm_core[0].copy()]

    def _endpoint_offset(p: np.ndarray, tangent: np.ndarray, side: str) -> np.ndarray:
        n_vec = left_normal(tangent) if side == "left" else right_normal(tangent)
        n_vec = _unit(n_vec, eps=eps_len)
        return p + offset * n_vec

    def _build_offset_for_side(core: List[np.ndarray], side: str) -> List[np.ndarray]:
        m = len(core)
        if m == 1:
            return [core[0].copy()]

        out: List[np.ndarray] = []
        for i in range(m):
            p = core[i]

            if closed_effective:
                p_prev = core[(i - 1) % m]
                p_next = core[(i + 1) % m]
                t_prev = _unit(p - p_prev, eps=eps_len)
                t_next = _unit(p_next - p, eps=eps_len)
            else:
                if i == 0:
                    t = _unit(core[1] - core[0], eps=eps_len)
                    out.append(_endpoint_offset(p, t, side))
                    continue
                if i == m - 1:
                    t = _unit(core[-1] - core[-2], eps=eps_len)
                    out.append(_endpoint_offset(p, t, side))
                    continue
                p_prev = core[i - 1]
                p_next = core[i + 1]
                t_prev = _unit(p - p_prev, eps=eps_len)
                t_next = _unit(p_next - p, eps=eps_len)

            if np.linalg.norm(t_prev) < eps_len and np.linalg.norm(t_next) < eps_len:
                out.append(p.copy())
                continue
            if np.linalg.norm(t_prev) < eps_len:
                t_prev = t_next
            if np.linalg.norm(t_next) < eps_len:
                t_next = t_prev

            n_prev = left_normal(t_prev) if side == "left" else right_normal(t_prev)
            n_next = left_normal(t_next) if side == "left" else right_normal(t_next)
            n_prev = _unit(n_prev, eps=eps_len)
            n_next = _unit(n_next, eps=eps_len)

            # 近似共线：直接平移
            if abs(cross2(t_prev, t_next)) < 1e-10 and float(np.dot(t_prev, t_next)) > 0.0:
                out.append(p + offset * n_prev)
                continue

            bis = n_prev + n_next
            if np.linalg.norm(bis) < eps_len:
                out.append(p + offset * n_prev)
                continue

            miter_dir = _unit(bis, eps=eps_len)
            denom = float(np.dot(miter_dir, n_prev))
            if abs(denom) < eps_miter:
                out.append(p + offset * n_prev)
                continue

            miter_len = float(np.clip(offset / denom, -miter_limit * offset, miter_limit * offset))
            out.append(p + miter_dir * miter_len)
        return out

    def _build_offset_pair(core: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return _build_offset_for_side(core, "left"), _build_offset_for_side(core, "right")

    def _resample_polyline(
        points: List[np.ndarray],
        target_n: int,
        closed_flag: bool,
        s_targets: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        if target_n <= 1 or not points:
            return [points[0].copy()] if points else []

        arr = np.asarray(points, dtype=float)
        if arr.shape[0] == 1:
            return [arr[0].copy() for _ in range(target_n)]

        if closed_flag and not np.allclose(arr[0], arr[-1], atol=1e-9):
            arr = np.vstack([arr, arr[0]])

        seg = np.diff(arr, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = float(cumulative[-1])
        if total <= 1e-12:
            return [arr[0].copy() for _ in range(target_n)]

        if s_targets is not None and len(s_targets) == target_n:
            s_clipped = np.asarray(s_targets, dtype=float)
            if closed_flag:
                s_clipped = np.mod(s_clipped, 1.0)
                targets = s_clipped * total
            else:
                s_clipped = np.clip(s_clipped, 0.0, 1.0)
                targets = s_clipped * total
        elif closed_flag:
            targets = np.linspace(0.0, total, target_n, endpoint=False)
        else:
            targets = np.linspace(0.0, total, target_n)

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
        return [np.asarray(p, dtype=float) for p in out]

    def _progress_params(core: List[np.ndarray], closed_flag: bool) -> np.ndarray:
        m = len(core)
        if m <= 1:
            return np.zeros((m,), dtype=float)

        if closed_flag:
            seg_len = np.array(
                [float(np.linalg.norm(core[(i + 1) % m] - core[i])) for i in range(m)],
                dtype=float,
            )
            total = float(np.sum(seg_len))
            if total <= 1e-12:
                return np.zeros((m,), dtype=float)
            cum = np.zeros((m,), dtype=float)
            running = 0.0
            for i in range(1, m):
                running += float(seg_len[i - 1])
                cum[i] = running / total
            return cum

        seg_len = np.array(
            [float(np.linalg.norm(core[i + 1] - core[i])) for i in range(m - 1)],
            dtype=float,
        )
        total = float(np.sum(seg_len))
        if total <= 1e-12:
            return np.zeros((m,), dtype=float)
        cum = np.zeros((m,), dtype=float)
        running = 0.0
        for i in range(1, m):
            running += float(seg_len[i - 1])
            cum[i] = running / total
        return cum

    def _simplify_collinear(core: List[np.ndarray], closed_flag: bool) -> List[np.ndarray]:
        pts = [p.copy() for p in core]
        if len(pts) <= (3 if closed_flag else 2):
            return pts

        col_tol = 1e-8
        dot_tol = 1.0 - 1e-9
        max_iter = max(1, 2 * len(pts))
        for _ in range(max_iter):
            changed = False
            if closed_flag:
                m = len(pts)
                if m <= 3:
                    break
                keep: List[np.ndarray] = []
                for i in range(m):
                    prev_p = pts[(i - 1) % m]
                    cur_p = pts[i]
                    next_p = pts[(i + 1) % m]
                    if i == 0:
                        keep.append(cur_p.copy())
                        continue
                    v1 = cur_p - prev_p
                    v2 = next_p - cur_p
                    l1 = float(np.linalg.norm(v1))
                    l2 = float(np.linalg.norm(v2))
                    if l1 < eps_len or l2 < eps_len:
                        changed = True
                        continue
                    u1 = v1 / l1
                    u2 = v2 / l2
                    if abs(cross2(u1, u2)) < col_tol and float(np.dot(u1, u2)) > dot_tol:
                        changed = True
                        continue
                    keep.append(cur_p)
                if len(keep) < 3:
                    break
                pts = [p.copy() for p in keep]
            else:
                m = len(pts)
                if m <= 2:
                    break
                keep = [pts[0].copy()]
                for i in range(1, m - 1):
                    prev_p = pts[i - 1]
                    cur_p = pts[i]
                    next_p = pts[i + 1]
                    v1 = cur_p - prev_p
                    v2 = next_p - cur_p
                    l1 = float(np.linalg.norm(v1))
                    l2 = float(np.linalg.norm(v2))
                    if l1 < eps_len or l2 < eps_len:
                        changed = True
                        continue
                    u1 = v1 / l1
                    u2 = v2 / l2
                    if abs(cross2(u1, u2)) < col_tol and float(np.dot(u1, u2)) > dot_tol:
                        changed = True
                        continue
                    keep.append(cur_p.copy())
                keep.append(pts[-1].copy())
                pts = keep
            if not changed:
                break
        return pts

    pl_core, pr_core = _build_offset_pair(pm_core)

    # 对明显自交场景做保守回退：仅当出现“严格跨段交叉”时触发，
    # 避免把正常的首尾拼接/共线接触误判为自交，从而在角点产生伪尖刺。
    def _segment_proper_intersection(
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        eps: float = 1e-9,
    ) -> bool:
        r = a2 - a1
        s = b2 - b1
        rxs = cross2(r, s)
        if abs(rxs) <= eps:
            return False
        qmp = b1 - a1
        t = cross2(qmp, s) / rxs
        u = cross2(qmp, r) / rxs
        return (eps < t < (1.0 - eps)) and (eps < u < (1.0 - eps))

    def _segment_proper_intersection_point(
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
        eps: float = 1e-9,
    ) -> Optional[np.ndarray]:
        r = a2 - a1
        s = b2 - b1
        rxs = cross2(r, s)
        if abs(rxs) <= eps:
            return None
        qmp = b1 - a1
        t = cross2(qmp, s) / rxs
        u = cross2(qmp, r) / rxs
        if not ((eps < t < (1.0 - eps)) and (eps < u < (1.0 - eps))):
            return None
        return a1 + t * r

    def _count_proper_self_crossings(points: List[np.ndarray], closed_flag: bool) -> int:
        if len(points) < 4:
            return 0
        segments = _iter_segments(points, closed=closed_flag)
        total = 0
        for i, (a1, a2) in enumerate(segments):
            for j in range(i + 1, len(segments)):
                if abs(i - j) <= 1:
                    continue
                if closed_flag and i == 0 and j == len(segments) - 1:
                    continue
                b1, b2 = segments[j]
                if _segment_proper_intersection(a1, a2, b1, b2, eps=1e-9):
                    total += 1
        return total

    def _cyclic_vertices(points: List[np.ndarray], start: int, end: int) -> List[np.ndarray]:
        """取闭环顶点序列 [start ... end]（含端点），沿正向索引。"""
        m = len(points)
        if m == 0:
            return []
        if start <= end:
            return [points[k].copy() for k in range(start, end + 1)]
        return [points[k].copy() for k in range(start, m)] + [points[k].copy() for k in range(0, end + 1)]

    def _polyline_length(points: List[np.ndarray]) -> float:
        if len(points) < 2:
            return 0.0
        return float(sum(np.linalg.norm(points[k + 1] - points[k]) for k in range(len(points) - 1)))

    def _remove_neighbor_duplicates(points: List[np.ndarray], tol: float = 1e-8) -> List[np.ndarray]:
        if not points:
            return []
        out = [points[0].copy()]
        for p in points[1:]:
            if np.linalg.norm(p - out[-1]) > tol:
                out.append(p.copy())
        if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
            out.pop()
        return out

    def _rotate_closed_to_anchor(points: List[np.ndarray], anchor: np.ndarray) -> List[np.ndarray]:
        if not points:
            return []
        d2 = [float(np.sum((p - anchor) ** 2)) for p in points]
        idx = int(np.argmin(d2))
        return [points[(idx + k) % len(points)].copy() for k in range(len(points))]

    def _align_and_resample_clipped(
        clipped: List[np.ndarray],
        original: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        if len(clipped) < 3 or len(original) < 3:
            return None
        s_targets = _progress_params(pm_core, closed_flag=True)
        orig_arr = np.asarray(original, dtype=float)

        candidates: List[Tuple[float, List[np.ndarray]]] = []
        for seq in (clipped, list(reversed(clipped))):
            rotated = _rotate_closed_to_anchor(seq, anchor=original[0])
            sampled = _resample_polyline(
                rotated,
                target_n=n,
                closed_flag=True,
                s_targets=s_targets,
            )
            sample_arr = np.asarray(sampled, dtype=float)
            err = float(np.mean(np.linalg.norm(sample_arr - orig_arr, axis=1)))
            candidates.append((err, sampled))
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _clip_closed_self_loops(points: List[np.ndarray], max_iter: int = 32) -> List[np.ndarray]:
        """闭合折线 loop clipping：发现自交后删除较短回路。"""
        pts = _remove_neighbor_duplicates(points)
        if len(pts) < 4:
            return pts

        for _ in range(max_iter):
            m = len(pts)
            if m < 4:
                break
            hit = None
            for i in range(m):
                i2 = (i + 1) % m
                a1 = pts[i]
                a2 = pts[i2]
                for j in range(i + 1, m):
                    j2 = (j + 1) % m
                    if abs(i - j) <= 1:
                        continue
                    if i == 0 and j == m - 1:
                        continue
                    b1 = pts[j]
                    b2 = pts[j2]
                    x = _segment_proper_intersection_point(a1, a2, b1, b2, eps=1e-9)
                    if x is not None:
                        hit = (i, j, x)
                        break
                if hit is not None:
                    break
            if hit is None:
                break

            i, j, x = hit
            i2 = (i + 1) % m
            j2 = (j + 1) % m
            branch_a = _cyclic_vertices(pts, i2, j)  # x -> ... -> x
            branch_b = _cyclic_vertices(pts, j2, i)  # x -> ... -> x

            len_a = _polyline_length([x.copy()] + branch_a + [x.copy()])
            len_b = _polyline_length([x.copy()] + branch_b + [x.copy()])

            if len_a <= len_b:
                kept = [x.copy()] + branch_b
            else:
                kept = [x.copy()] + branch_a
            pts = _remove_neighbor_duplicates(kept)

        return pts

    def _fallback_centerline_normals(side: str) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(n):
            p = pm_core[i]
            if closed_effective:
                p_prev = pm_core[(i - 1) % n]
                p_next = pm_core[(i + 1) % n]
            else:
                p_prev = pm_core[max(i - 1, 0)]
                p_next = pm_core[min(i + 1, n - 1)]
            t = _unit(p_next - p_prev, eps=eps_len)
            if np.linalg.norm(t) < eps_len:
                t = _unit(p_next - p, eps=eps_len)
            if np.linalg.norm(t) < eps_len:
                t = _unit(p - p_prev, eps=eps_len)
            n_vec = left_normal(t) if side == "left" else right_normal(t)
            n_vec = _unit(n_vec, eps=eps_len)
            out.append(p + offset * n_vec)
        return out

    pl_cross = _count_proper_self_crossings(pl_core, closed_flag=closed_effective)
    pr_cross = _count_proper_self_crossings(pr_core, closed_flag=closed_effective)
    if pl_cross > 0 or pr_cross > 0:
        pm_simplified = _simplify_collinear(pm_core, closed_flag=closed_effective)
        if len(pm_simplified) >= (3 if closed_effective else 2):
            pl_simple, pr_simple = _build_offset_pair(pm_simplified)
            s_targets = _progress_params(pm_core, closed_flag=closed_effective)
            pl_core = _resample_polyline(pl_simple, target_n=n, closed_flag=closed_effective, s_targets=s_targets)
            pr_core = _resample_polyline(pr_simple, target_n=n, closed_flag=closed_effective, s_targets=s_targets)

    # 闭合路径若仍有自交，则做 loop clipping；随后做方向/锚点对齐后再重采样，
    # 尽量保持与 Pm 的索引相位一致。
    if closed_effective:
        if _count_proper_self_crossings(pl_core, closed_flag=True) > 0:
            clipped = _clip_closed_self_loops(pl_core)
            aligned = _align_and_resample_clipped(clipped, pl_core)
            if aligned is not None:
                pl_core = aligned
        if _count_proper_self_crossings(pr_core, closed_flag=True) > 0:
            clipped = _clip_closed_self_loops(pr_core)
            aligned = _align_and_resample_clipped(clipped, pr_core)
            if aligned is not None:
                pr_core = aligned

    if _count_proper_self_crossings(pl_core, closed_flag=closed_effective) > 0:
        pl_core = _fallback_centerline_normals("left")
    if _count_proper_self_crossings(pr_core, closed_flag=closed_effective) > 0:
        pr_core = _fallback_centerline_normals("right")

    def _enforce_seam_collinear(side_points: List[np.ndarray], side: str) -> List[np.ndarray]:
        if not closed_effective or len(side_points) < 3 or len(pm_core) < 3:
            return side_points
        t_prev = _unit(pm_core[0] - pm_core[-1], eps=eps_len)
        t_next = _unit(pm_core[1] - pm_core[0], eps=eps_len)
        if np.linalg.norm(t_prev) < eps_len or np.linalg.norm(t_next) < eps_len:
            return side_points
        if abs(cross2(t_prev, t_next)) < 1e-10 and float(np.dot(t_prev, t_next)) > 0.0:
            t = _unit(t_prev + t_next, eps=eps_len)
            if np.linalg.norm(t) < eps_len:
                t = t_prev
            n_vec = left_normal(t) if side == "left" else right_normal(t)
            n_vec = _unit(n_vec, eps=eps_len)
            side_points[0] = pm_core[0] + offset * n_vec
        return side_points

    pl_core = _enforce_seam_collinear(pl_core, "left")
    pr_core = _enforce_seam_collinear(pr_core, "right")

    if has_duplicate_last:
        return pl_core + [pl_core[0].copy()], pr_core + [pr_core[0].copy()]
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
