from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.geometry import generate_offset_paths
from src.utils.path_generator import get_path_by_name


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {path}")
    return cfg


def _build_path(path_cfg: Dict[str, Any]) -> List[np.ndarray]:
    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    kwargs = path_cfg.get(path_type, {})
    if not isinstance(kwargs, dict):
        kwargs = {}
    if path_type == "square" and "closed" in path_cfg and "closed" not in kwargs:
        kwargs["closed"] = bool(path_cfg.get("closed"))
    if path_type == "trapezoid" and "closed" in path_cfg and "closed" not in kwargs:
        kwargs["closed"] = bool(path_cfg.get("closed"))
    return get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)


def _save_plot(name: str, pm: np.ndarray, pl: np.ndarray, pr: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=140)
    ax.plot(pm[:, 0], pm[:, 1], color="black", linestyle="--", linewidth=1.2, label="Pm (Reference)")
    ax.plot(pl[:, 0], pl[:, 1], color="#1f77b4", linewidth=1.4, label="Pl (Left Offset)")
    ax.plot(pr[:, 0], pr[:, 1], color="#d62728", linewidth=1.4, label="Pr (Right Offset)")

    ax.scatter([pm[0, 0]], [pm[0, 1]], c="green", s=42, marker="o", label="Pm Start")
    ax.scatter([pm[-1, 0]], [pm[-1, 1]], c="purple", s=42, marker="x", label="Pm End")

    ax.set_title(f"{name} | Pm / Pl / Pr")
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
    parser = argparse.ArgumentParser(description="Validate and visualize phase32 Pm/Pl/Pr offsets.")
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
        pl_list, pr_list = generate_offset_paths(pm_list, half_epsilon, closed=closed)
        pl = np.asarray(pl_list, dtype=float)
        pr = np.asarray(pr_list, dtype=float)

        left_dist = np.linalg.norm(pl - pm, axis=1)
        right_dist = np.linalg.norm(pr - pm, axis=1)
        left_abs_err = np.abs(left_dist - half_epsilon)
        right_abs_err = np.abs(right_dist - half_epsilon)

        out_png = out_dir / f"{name}_pm_pl_pr.png"
        _save_plot(name, pm, pl, pr, out_png)

        summary["paths"][name] = {
            "closed": closed,
            "num_points": int(len(pm)),
            "left_distance_mean": float(np.mean(left_dist)),
            "left_distance_max_err": float(np.max(left_abs_err)),
            "right_distance_mean": float(np.mean(right_dist)),
            "right_distance_max_err": float(np.max(right_abs_err)),
            "png": str(out_png),
        }
        print(
            f"[{name}] closed={closed} n={len(pm)} "
            f"left_mean={np.mean(left_dist):.6f} left_max_err={np.max(left_abs_err):.6f} "
            f"right_mean={np.mean(right_dist):.6f} right_max_err={np.max(right_abs_err):.6f}"
        )

    summary_path = out_dir / "offset_validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

