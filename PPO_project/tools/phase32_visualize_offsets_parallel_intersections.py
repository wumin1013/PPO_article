from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.geometry import count_polyline_self_intersections
from src.utils.path_generator import get_path_by_name


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def _unit(v: Sequence[float], eps: float = 1e-12) -> np.ndarray:
    vec = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(vec))
    if n < eps:
        return np.zeros(2, dtype=float)
    return vec / n


def _cross2(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _left_normal(t: Sequence[float]) -> np.ndarray:
    return np.array([-float(t[1]), float(t[0])], dtype=float)


def _line_intersection_point_dir(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    den = _cross2(d1, d2)
    if abs(den) < eps:
        return None
    t = _cross2((p2 - p1), d2) / den
    return p1 + t * d1


def _resolve_closed_core(
    points: List[np.ndarray],
    closed: bool | None,
) -> Tuple[List[np.ndarray], bool, bool]:
    if not points:
        return [], False, False
    inferred_closed = len(points) > 2 and np.allclose(points[0], points[-1], atol=1e-6)
    closed_effective = bool(closed) if closed is not None else inferred_closed
    has_dup_last = closed_effective and inferred_closed
    if has_dup_last:
        return points[:-1], closed_effective, True
    return points, closed_effective, False


def _simplify_strict_collinear(points: List[np.ndarray], closed: bool) -> List[np.ndarray]:
    """
    仅删除“严格共线且同向”的冗余点，保留真实折点与曲线点。
    这样可避免方形/梯形在高密采样下出现求交噪声。
    """
    if len(points) <= (3 if closed else 2):
        return [p.copy() for p in points]

    pts = [p.copy() for p in points]
    eps = 1e-10
    max_iter = max(1, 2 * len(pts))

    for _ in range(max_iter):
        changed = False
        if closed:
            m = len(pts)
            if m <= 3:
                break
            keep: List[np.ndarray] = []
            for i in range(m):
                prev_p = pts[(i - 1) % m]
                cur_p = pts[i]
                next_p = pts[(i + 1) % m]
                v1 = cur_p - prev_p
                v2 = next_p - cur_p
                l1 = float(np.linalg.norm(v1))
                l2 = float(np.linalg.norm(v2))
                if l1 < eps or l2 < eps:
                    changed = True
                    continue
                u1 = v1 / l1
                u2 = v2 / l2
                if abs(_cross2(u1, u2)) < eps and float(np.dot(u1, u2)) > 1.0 - 1e-12:
                    changed = True
                    continue
                keep.append(cur_p)
            if len(keep) < 3:
                break
            pts = keep
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
                if l1 < eps or l2 < eps:
                    changed = True
                    continue
                u1 = v1 / l1
                u2 = v2 / l2
                if abs(_cross2(u1, u2)) < eps and float(np.dot(u1, u2)) > 1.0 - 1e-12:
                    changed = True
                    continue
                keep.append(cur_p.copy())
            keep.append(pts[-1].copy())
            pts = keep
        if not changed:
            break

    return pts


def _progress_params(core: List[np.ndarray], closed: bool) -> np.ndarray:
    m = len(core)
    if m <= 1:
        return np.zeros((m,), dtype=float)
    if closed:
        seg_len = np.array([float(np.linalg.norm(core[(i + 1) % m] - core[i])) for i in range(m)], dtype=float)
        total = float(np.sum(seg_len))
        if total <= 1e-12:
            return np.zeros((m,), dtype=float)
        out = np.zeros((m,), dtype=float)
        run = 0.0
        for i in range(1, m):
            run += float(seg_len[i - 1])
            out[i] = run / total
        return out
    seg_len = np.array([float(np.linalg.norm(core[i + 1] - core[i])) for i in range(m - 1)], dtype=float)
    total = float(np.sum(seg_len))
    if total <= 1e-12:
        return np.zeros((m,), dtype=float)
    out = np.zeros((m,), dtype=float)
    run = 0.0
    for i in range(1, m):
        run += float(seg_len[i - 1])
        out[i] = run / total
    return out


def _resample_polyline(points: List[np.ndarray], target_n: int, closed: bool, s_targets: Optional[np.ndarray] = None) -> List[np.ndarray]:
    if target_n <= 1 or not points:
        return [points[0].copy()] if points else []
    arr = np.asarray(points, dtype=float)
    if closed and not np.allclose(arr[0], arr[-1], atol=1e-9):
        arr = np.vstack([arr, arr[0]])
    seg = np.diff(arr, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    if total <= 1e-12:
        return [arr[0].copy() for _ in range(target_n)]

    if s_targets is not None and len(s_targets) == target_n:
        s = np.asarray(s_targets, dtype=float)
        if closed:
            s = np.mod(s, 1.0)
        else:
            s = np.clip(s, 0.0, 1.0)
        targets = s * total
    elif closed:
        targets = np.linspace(0.0, total, target_n, endpoint=False)
    else:
        targets = np.linspace(0.0, total, target_n)

    out: List[np.ndarray] = []
    for dist in targets:
        idx = int(np.searchsorted(cum, dist, side="right") - 1)
        idx = int(np.clip(idx, 0, len(seg_len) - 1))
        l = float(seg_len[idx])
        if l <= 1e-12:
            out.append(arr[idx].copy())
            continue
        t = float((dist - cum[idx]) / l)
        out.append(arr[idx] + t * (arr[idx + 1] - arr[idx]))
    return [np.asarray(p, dtype=float) for p in out]


def _resample_polyline_keep_vertices(points: List[np.ndarray], target_n: int, closed: bool) -> List[np.ndarray]:
    """
    重采样时保留每个折点，避免方形/梯形角点被插值抹平为斜边。
    """
    if target_n <= 1 or not points:
        return [points[0].copy()] if points else []

    core = [np.asarray(p, dtype=float).copy() for p in points]
    if closed and len(core) > 1 and np.allclose(core[0], core[-1], atol=1e-9):
        core = core[:-1]
    m = len(core)
    if m <= 1:
        return [core[0].copy() for _ in range(target_n)]

    edges = m if closed else (m - 1)
    if edges <= 0:
        return [core[0].copy() for _ in range(target_n)]

    lengths = []
    for i in range(edges):
        j = (i + 1) % m if closed else i + 1
        lengths.append(float(np.linalg.norm(core[j] - core[i])))
    total = max(1e-12, float(sum(lengths)))

    if closed:
        target = int(target_n)
        counts = [max(1, int(round(target * l / total))) for l in lengths]
    else:
        target = int(target_n) - 1
        counts = [max(1, int(round(target * l / total))) for l in lengths]

    delta = target - int(sum(counts))
    if delta != 0:
        order = list(np.argsort(lengths))[::-1]
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

    out: List[np.ndarray] = []
    if closed:
        # 每条边含起点不含终点，保证角点由“下一边起点”精确保留。
        for i in range(edges):
            j = (i + 1) % m
            p1 = core[i]
            p2 = core[j]
            n = counts[i]
            for t in np.linspace(0.0, 1.0, n, endpoint=False):
                out.append(p1 + float(t) * (p2 - p1))
    else:
        out.append(core[0].copy())
        for i in range(edges):
            p1 = core[i]
            p2 = core[i + 1]
            n = counts[i]
            for t in np.linspace(0.0, 1.0, n + 1)[1:]:
                out.append(p1 + float(t) * (p2 - p1))

    if len(out) > target_n:
        out = out[:target_n]
    elif len(out) < target_n:
        pad = out[-1].copy() if out else core[0].copy()
        out.extend([pad.copy() for _ in range(target_n - len(out))])
    return [np.asarray(p, dtype=float) for p in out]


def _build_path(path_cfg: Dict[str, Any]) -> List[np.ndarray]:
    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    kwargs = copy.deepcopy(path_cfg.get(path_type, {}))
    if not isinstance(kwargs, dict):
        kwargs = {}
    # 按用户要求：蝴蝶采用学术组合形态
    if path_type == "butterfly":
        kwargs["style"] = "academic"
    if path_type in {"square", "trapezoid"} and "closed" in path_cfg and "closed" not in kwargs:
        kwargs["closed"] = bool(path_cfg["closed"])
    return get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)


def generate_offsets_parallel_intersections(
    Pm: Sequence[Sequence[float]],
    half_epsilon: float,
    closed: bool | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    平行线求交截断法（按用户指定思路）：
    1) 对每一段中心线构造单侧平行“无限长直线”
    2) 每个顶点使用相邻两条平行线求交点
    3) 用交点序列作为该侧偏移路径（即在交点处截断）
    """
    pm_full = [np.asarray(p, dtype=float) for p in Pm]
    if not pm_full:
        return [], []

    pm_core_orig, closed_effective, has_dup_last = _resolve_closed_core(pm_full, closed)
    n = len(pm_core_orig)
    if n == 1:
        return [pm_core_orig[0].copy()], [pm_core_orig[0].copy()]

    # 先做严格共线简化，再在“真实段”上做平行线求交。
    pm_core = _simplify_strict_collinear(pm_core_orig, closed=closed_effective)
    n_core = len(pm_core)
    if n_core < 2:
        pm_core = [p.copy() for p in pm_core_orig]
        n_core = len(pm_core)

    # 构造每段的切向和单侧平移后的直线基点
    seg_count = n_core if closed_effective else (n_core - 1)
    seg_dir: List[np.ndarray] = []
    seg_left_base: List[np.ndarray] = []
    for i in range(seg_count):
        j = (i + 1) % n_core if closed_effective else i + 1
        p1 = pm_core[i]
        p2 = pm_core[j]
        t = _unit(p2 - p1)
        if float(np.linalg.norm(t)) < 1e-12:
            # 零长度段降级：方向置零，后续顶点回退
            seg_dir.append(np.zeros(2, dtype=float))
            seg_left_base.append(p1.copy())
            continue
        n_left = _unit(_left_normal(t))
        seg_dir.append(t)
        seg_left_base.append(p1 + half_epsilon * n_left)

    def _fallback_vertex(i: int, side_sign: float) -> np.ndarray:
        # side_sign: +1 => left, -1 => right
        p = pm_core[i]
        if closed_effective:
            i_prev = (i - 1) % seg_count
            i_next = i % seg_count
            t_prev = seg_dir[i_prev]
            t_next = seg_dir[i_next]
            n_prev = _left_normal(t_prev)
            n_next = _left_normal(t_next)
            n_mix = _unit(n_prev + n_next)
            if float(np.linalg.norm(n_mix)) < 1e-12:
                n_mix = _unit(n_prev)
            if float(np.linalg.norm(n_mix)) < 1e-12:
                n_mix = _unit(n_next)
            if float(np.linalg.norm(n_mix)) < 1e-12:
                return p.copy()
            return p + side_sign * half_epsilon * n_mix
        # open endpoints
        if i == 0:
            t0 = seg_dir[0]
            n0 = _unit(_left_normal(t0))
            return p + side_sign * half_epsilon * n0
        if i == n_core - 1:
            tk = seg_dir[-1]
            nk = _unit(_left_normal(tk))
            return p + side_sign * half_epsilon * nk
        t_prev = seg_dir[i - 1]
        t_next = seg_dir[i]
        n_prev = _left_normal(t_prev)
        n_next = _left_normal(t_next)
        n_mix = _unit(n_prev + n_next)
        if float(np.linalg.norm(n_mix)) < 1e-12:
            n_mix = _unit(n_prev)
        if float(np.linalg.norm(n_mix)) < 1e-12:
            n_mix = _unit(n_next)
        if float(np.linalg.norm(n_mix)) < 1e-12:
            return p.copy()
        return p + side_sign * half_epsilon * n_mix

    miter_limit = 8.0

    def _build_side(side_sign: float) -> List[np.ndarray]:
        # side_sign: +1 -> left, -1 -> right
        out: List[np.ndarray] = []

        def _line_of_seg(k: int) -> Tuple[np.ndarray, np.ndarray]:
            base_l = seg_left_base[k]
            t = seg_dir[k]
            if side_sign > 0:
                return base_l, t
            # 右侧平行线 = 左侧基点沿左法向反向移动 2*half_epsilon
            n_left = _unit(_left_normal(t))
            base_r = base_l - 2.0 * half_epsilon * n_left
            return base_r, t

        if not closed_effective:
            # 首点/末点用端点偏移，中间点用相邻段平行线求交
            out.append(_fallback_vertex(0, side_sign))
            for i in range(1, n_core - 1):
                k_prev = i - 1
                k_next = i
                p1, d1 = _line_of_seg(k_prev)
                p2, d2 = _line_of_seg(k_next)
                x = _line_intersection_point_dir(p1, d1, p2, d2)
                fb = _fallback_vertex(i, side_sign)
                if x is None:
                    out.append(fb)
                    continue
                # 限制近乎平行段导致的超长交点（尖刺）
                if float(np.linalg.norm(x - fb)) > (miter_limit * half_epsilon):
                    out.append(fb)
                else:
                    out.append(x)
            out.append(_fallback_vertex(n_core - 1, side_sign))
            return out

        # 闭合：每个点都由相邻段平行线求交
        for i in range(n_core):
            k_prev = (i - 1) % seg_count
            k_next = i % seg_count
            p1, d1 = _line_of_seg(k_prev)
            p2, d2 = _line_of_seg(k_next)
            x = _line_intersection_point_dir(p1, d1, p2, d2)
            fb = _fallback_vertex(i, side_sign)
            if x is None:
                out.append(fb)
                continue
            if float(np.linalg.norm(x - fb)) > (miter_limit * half_epsilon):
                out.append(fb)
            else:
                out.append(x)
        return out

    pl_core = _build_side(+1.0)
    pr_core = _build_side(-1.0)

    # 若简化改变了点数，则重采样回原点数；重采样时强制保留折点，避免角点被抹平。
    if len(pm_core) != len(pm_core_orig):
        pl_core = _resample_polyline_keep_vertices(pl_core, target_n=len(pm_core_orig), closed=closed_effective)
        pr_core = _resample_polyline_keep_vertices(pr_core, target_n=len(pm_core_orig), closed=closed_effective)

    if has_dup_last:
        return pl_core + [pl_core[0].copy()], pr_core + [pr_core[0].copy()]
    return pl_core, pr_core


def _save_plot(name: str, pm: np.ndarray, pl: np.ndarray, pr: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=140)
    ax.plot(pm[:, 0], pm[:, 1], color="black", linestyle="--", linewidth=1.2, label="Pm (Reference)")
    ax.plot(pl[:, 0], pl[:, 1], color="#1f77b4", linewidth=1.5, label="Pl (Parallel-Intersection)")
    ax.plot(pr[:, 0], pr[:, 1], color="#d62728", linewidth=1.5, label="Pr (Parallel-Intersection)")
    ax.scatter([pm[0, 0]], [pm[0, 1]], c="green", s=42, marker="o", label="Pm Start")
    ax.scatter([pm[-1, 0]], [pm[-1, 1]], c="purple", s=42, marker="x", label="Pm End")
    ax.set_title(f"{name} | Pm / Pl / Pr (parallel-intersection)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize offsets by parallel-line intersections.")
    parser.add_argument("--config", required=True, type=str, help="YAML config path")
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    out_dir = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    half_epsilon = float(cfg["environment"]["epsilon"]) / 2.0

    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), dict) else {}
    curriculum = training_cfg.get("path_curriculum", {}) if isinstance(training_cfg.get("path_curriculum", {}), dict) else {}
    paths = curriculum.get("paths", []) if isinstance(curriculum.get("paths", []), list) and curriculum.get("paths") else [cfg.get("path", {})]

    normalized: List[Dict[str, Any]] = []
    seen = set()
    for i, p in enumerate(paths):
        if not isinstance(p, dict):
            continue
        item = copy.deepcopy(p)
        name = str(item.get("name") or item.get("type") or f"path_{i}")
        if name in seen:
            continue
        seen.add(name)
        item["name"] = name
        normalized.append(item)

    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "half_epsilon": half_epsilon,
        "paths": {},
    }

    for path_cfg in normalized:
        name = str(path_cfg["name"])
        pm_list = _build_path(path_cfg)
        pm = np.asarray(pm_list, dtype=float)
        closed = bool(np.allclose(pm[0], pm[-1], atol=1e-6))

        pl_list, pr_list = generate_offsets_parallel_intersections(pm_list, half_epsilon=half_epsilon, closed=closed)
        pl = np.asarray(pl_list, dtype=float)
        pr = np.asarray(pr_list, dtype=float)

        out_png = out_dir / f"{name}_pm_pl_pr_parallel_intersection.png"
        _save_plot(name, pm, pl, pr, out_png)

        summary["paths"][name] = {
            "closed": closed,
            "num_points": int(len(pm)),
            "self_cross_pl": int(count_polyline_self_intersections(pl.tolist(), closed=closed)),
            "self_cross_pr": int(count_polyline_self_intersections(pr.tolist(), closed=closed)),
            "png": str(out_png),
        }
        print(
            f"[{name}] closed={closed} "
            f"self_cross(Pl/Pr)={summary['paths'][name]['self_cross_pl']}/{summary['paths'][name]['self_cross_pr']}"
        )

    summary_path = out_dir / "parallel_intersection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
