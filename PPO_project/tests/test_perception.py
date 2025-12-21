from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.environment import Env


def test_lookahead_features_left_turn() -> None:
    """前瞻特征自检：短距离左转时，预瞄点应在车体坐标系 Y 正方向偏转。"""
    path_points = [
        np.array([0.0, 0.0]),
        np.array([0.01, 0.0]),  # 10mm 直线
        np.array([0.01, 0.02]),  # 左转上行
    ]
    env = Env(
        device="cpu",
        epsilon=0.5,
        interpolation_period=0.1,
        MAX_VEL=1.0,
        MAX_ACC=2.0,
        MAX_JERK=3.0,
        MAX_ANG_VEL=1.5,
        MAX_ANG_ACC=3.0,
        MAX_ANG_JERK=5.0,
        Pm=path_points,
        max_steps=50,
        lookahead_points=3,
        return_normalized_obs=False,
    )
    _ = env.reset()

    raw_features = env._compute_lookahead_features()
    lookahead = np.asarray(raw_features, dtype=float).reshape(-1, 3)

    assert np.any(lookahead[:, 1] > 0), "左转预瞄点未在Y正方向偏转"
    assert float(lookahead[-1, 1]) > 0, "末端预瞄点未体现左转偏移"

