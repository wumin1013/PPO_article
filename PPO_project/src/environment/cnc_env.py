"""
CNC轨迹环境入口，复用优化后的 `legacy_env.Env` 并暴露统一的构造接口。
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from src.environment.legacy_env import Env
from src.environment.kinematics import apply_kinematic_constraints


def create_environment_from_config(config: Dict, path_points: Iterable, device=None) -> Env:
    """根据配置字典构建 Env 实例。"""
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    Pm = [np.array(pt) for pt in path_points]
    reward_weights = config.get("reward_weights", {})

    return Env(
        device=device or env_cfg.get("device"),
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=Pm,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=reward_weights,
    )


__all__ = ["Env", "create_environment_from_config", "apply_kinematic_constraints"]
