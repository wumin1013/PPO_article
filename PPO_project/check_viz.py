"""快速自检脚本，验证实时绘图与路径解析是否正常。"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from src.utils.plotter import TrajectoryPlotter

PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# 避免 OpenMP 重复初始化报错（numpy/mkl 与其他依赖并存时常见）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def test_viz(num_points: int = 100) -> None:
    """生成随机轨迹并调用绘图器，确保窗口可实时刷新。"""
    ref_x = np.linspace(0, 10, num_points)
    ref_y = np.sin(ref_x)
    ref_path = np.stack([ref_x, ref_y], axis=1)

    rng = np.random.default_rng(42)
    traj_points = []
    plotter = TrajectoryPlotter(title="check_viz Demo")
    for idx in range(num_points):
        noise = rng.normal(scale=0.05, size=2)
        point = np.array([ref_x[idx], ref_y[idx]]) + noise
        traj_points.append(point)
        plotter.update(reference_path=ref_path, trajectory=traj_points)
    time.sleep(0.5)
    plotter.close()


def test_paths() -> None:
    """打印根目录与 saved_models 列表，验证路径解析。"""
    print(f"[check_viz] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[check_viz] SAVED_MODELS_DIR = {SAVED_MODELS_DIR}")
    if not SAVED_MODELS_DIR.exists():
        print("[check_viz] saved_models 不存在，跳过目录遍历。")
        return
    experiments = sorted(SAVED_MODELS_DIR.glob("*"))
    if not experiments:
        print("[check_viz] saved_models 目录为空。")
        return
    print("[check_viz] saved_models 内容:")
    for exp in experiments:
        print(f"  - {exp}")


if __name__ == "__main__":
    print("[check_viz] Test 1: 实时绘图自检")
    test_viz()
    print("[check_viz] Test 2: 路径解析自检")
    test_paths()
