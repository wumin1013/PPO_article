from __future__ import annotations

import argparse
import copy
import json
import math
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

from src.utils.geometry import count_polyline_self_intersections, generate_offset_paths
from src.utils.path_generator import get_path_by_name


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def _normalize(v: Sequence[float]) -> Tuple[float, float]:
    length = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2)
    if length == 0.0:
        return (0.0, 0.0)
    return (float(v[0]) / length, float(v[1]) / length)


def _get_parallel_lines(
    p1: Sequence[float],
    p2: Sequence[float],
    offset_distance: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    normal_vector = (-dy, dx)
    unit_normal_vector = _normalize(normal_vector)
    A, B = unit_normal_vector

    def line_equation(distance: float, point: Sequence[float] = p1) -> Tuple[float, float, float]:
        C = -(A * float(point[0]) + B * float(point[1])) + float(distance)
        return (A, B, C)

    return line_equation(offset_distance), line_equation(-offset_distance)


def _extended_intersection(
    line1: Tuple[float, float, float],
    line2: Tuple[float, float, float],
) -> Optional[Tuple[float, float]]:
    A1, B1, C1 = line1
    A2, B2, C2 = line2
    denominator = A1 * B2 - A2 * B1
    if abs(denominator) < 1e-6:
        return None
    x = (B1 * C2 - B2 * C1) / denominator
    y = (C1 * A2 - C2 * A1) / denominator
    return (float(x), float(y))


def _calculate_intersections(pm3: Sequence[Sequence[float]], offset_distance: float) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    p1, p2, p3 = pm3
    l1, r1 = _get_parallel_lines(p1, p2, offset_distance)
    l2, r2 = _get_parallel_lines(p2, p3, offset_distance)
    pl = _extended_intersection(l1, l2)
    pr = _extended_intersection(r1, r2)
    return pl, pr


def _get_offset_point(
    p: Sequence[float],
    direction: Sequence[float],
    distance: float,
) -> Tuple[float, float]:
    return (
        float(p[0]) + float(direction[1]) * float(distance),
        float(p[1]) - float(direction[0]) * float(distance),
    )


def generate_offset_paths_legacy(
    Pm: Sequence[Sequence[float]],
    half_epsilon: float,
    closed: bool,
) -> Tuple[List[Optional[Tuple[float, float]]], List[Optional[Tuple[float, float]]]]:
    pm = [np.asarray(p, dtype=float) for p in Pm]
    n = len(pm)
    pl: List[Optional[Tuple[float, float]]] = [None] * n
    pr: List[Optional[Tuple[float, float]]] = [None] * n

    for i in range(n):
        if i == 0:
            if not closed:
                p1, p2 = pm[i], pm[i + 1]
                direction_vector = _normalize((p2[0] - p1[0], p2[1] - p1[1]))
                pl[i] = _get_offset_point(p1, direction_vector, half_epsilon)
                pr[i] = _get_offset_point(p1, direction_vector, -half_epsilon)
            else:
                prev_point = pm[-2] if n >= 2 else pm[0]
                next_point = pm[i + 1]
                pl[i], pr[i] = _calculate_intersections([prev_point, pm[i], next_point], half_epsilon)
        elif i == n - 1:
            if not closed:
                p1, p2 = pm[i - 1], pm[i]
                direction_vector = _normalize((p2[0] - p1[0], p2[1] - p1[1]))
                pl[i] = _get_offset_point(p2, direction_vector, half_epsilon)
                pr[i] = _get_offset_point(p2, direction_vector, -half_epsilon)
            else:
                pl[i] = pl[0]
                pr[i] = pr[0]
        else:
            prev_point = pm[i - 1]
            current_point = pm[i]
            next_point = pm[(i + 1) % n] if closed else pm[i + 1]
            pl_val, pr_val = _calculate_intersections([prev_point, current_point, next_point], half_epsilon)
            if pl_val is None:
                direction = _normalize((next_point[0] - current_point[0], next_point[1] - current_point[1]))
                pl_val = _get_offset_point(current_point, direction, half_epsilon)
            if pr_val is None:
                direction = _normalize((next_point[0] - current_point[0], next_point[1] - current_point[1]))
                pr_val = _get_offset_point(current_point, direction, -half_epsilon)
            pl[i] = pl_val
            pr[i] = pr_val

    return pl, pr


def _build_path(path_cfg: Dict[str, Any]) -> List[np.ndarray]:
    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    kwargs = copy.deepcopy(path_cfg.get(path_type, {}))
    if not isinstance(kwargs, dict):
        kwargs = {}

    # 按用户要求：蝴蝶回到原形态（非平滑斜接版）
    if path_type == "butterfly":
        kwargs["style"] = "academic"

    if path_type in {"square", "trapezoid"} and "closed" in path_cfg and "closed" not in kwargs:
        kwargs["closed"] = bool(path_cfg.get("closed"))

    return get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)


def _to_array(points: Sequence[Optional[Tuple[float, float]]]) -> np.ndarray:
    clean: List[np.ndarray] = []
    for p in points:
        if p is None:
            continue
        clean.append(np.array([float(p[0]), float(p[1])], dtype=float))
    return np.asarray(clean, dtype=float)


def _save_plot(name: str, pm: np.ndarray, pl: np.ndarray, pr: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=140)
    ax.plot(pm[:, 0], pm[:, 1], color="black", linestyle="--", linewidth=1.2, label="Pm (Reference)")
    ax.plot(pl[:, 0], pl[:, 1], color="#1f77b4", linewidth=1.4, label="Pl (Legacy Offset)")
    ax.plot(pr[:, 0], pr[:, 1], color="#d62728", linewidth=1.4, label="Pr (Legacy Offset)")

    ax.scatter([pm[0, 0]], [pm[0, 1]], c="green", s=42, marker="o", label="Pm Start")
    ax.scatter([pm[-1, 0]], [pm[-1, 1]], c="purple", s=42, marker="x", label="Pm End")

    ax.set_title(f"{name} | Pm / Pl / Pr (legacy)")
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
    parser = argparse.ArgumentParser(description="Visualize Pm/Pl/Pr using legacy method from PPO最终版_改进.py")
    parser.add_argument("--config", required=True, type=str, help="YAML config path")
    parser.add_argument("--out", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    out_dir = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    epsilon = float(cfg["environment"]["epsilon"])
    half_epsilon = epsilon / 2.0

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

        pl_legacy_raw, pr_legacy_raw = generate_offset_paths_legacy(pm_list, half_epsilon=half_epsilon, closed=closed)
        pl = _to_array(pl_legacy_raw)
        pr = _to_array(pr_legacy_raw)

        # 当前方法仅用于对比统计，不参与绘图
        pl_cur_raw, pr_cur_raw = generate_offset_paths(pm_list, half_epsilon, closed=closed)
        pl_cur = np.asarray(pl_cur_raw, dtype=float)
        pr_cur = np.asarray(pr_cur_raw, dtype=float)

        out_png = out_dir / f"{name}_pm_pl_pr_legacy.png"
        _save_plot(name, pm, pl, pr, out_png)

        summary["paths"][name] = {
            "closed": closed,
            "num_points_pm": int(len(pm)),
            "num_points_pl_legacy": int(len(pl)),
            "num_points_pr_legacy": int(len(pr)),
            "legacy_self_cross_pl": int(count_polyline_self_intersections(pl.tolist(), closed=closed)),
            "legacy_self_cross_pr": int(count_polyline_self_intersections(pr.tolist(), closed=closed)),
            "current_self_cross_pl": int(count_polyline_self_intersections(pl_cur.tolist(), closed=closed)),
            "current_self_cross_pr": int(count_polyline_self_intersections(pr_cur.tolist(), closed=closed)),
            "png": str(out_png),
        }
        print(
            f"[{name}] legacy_cross(Pl/Pr)={summary['paths'][name]['legacy_self_cross_pl']}/{summary['paths'][name]['legacy_self_cross_pr']} "
            f"current_cross(Pl/Pr)={summary['paths'][name]['current_self_cross_pl']}/{summary['paths'][name]['current_self_cross_pr']}"
        )

    summary_path = out_dir / "legacy_offset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
