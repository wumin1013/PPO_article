"""
Streamlit仿真后端：封装不同模型/参数的单次轨迹仿真
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# 确保本项目与上级目录可被导入
for p in (BASE_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 导入原有训练代码中的核心组件
from src.environment.legacy_env import Env  # type: ignore
from src.utils.path_generator import get_path_by_name  # type: ignore
from src.algorithms.baselines import SCurvePlanner  # type: ignore


@dataclass
class SimulationConfig:
    model_name: str
    path_name: str
    disable_jerk_reward: bool
    disable_kcm: bool
    max_velocity: float
    max_jerk: float
    epsilon: float


@dataclass
class SimulationResult:
    trajectory: List[Tuple[float, float]]
    pm: List[Tuple[float, float]]
    pl: List[Tuple[float, float]]
    pr: List[Tuple[float, float]]
    time: List[float]
    velocity: List[float]
    acceleration: List[float]
    jerk: List[float]
    contour_error: List[float]
    progress: List[float]
    kcm_intervention: List[float]
    metrics: Dict[str, float]
    jerk_limit: float
    epsilon: float


def _load_base_config() -> Dict:
    config_path = BASE_DIR / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_path(path_name: str, base_cfg: Dict) -> List[np.ndarray]:
    path_cfg = base_cfg.get("path", {})
    scale = path_cfg.get("scale", 10.0)
    num_points = path_cfg.get("num_points", 200)
    lower = path_name.lower()
    if lower.startswith("s"):
        return get_path_by_name("s_shape", scale=scale, num_points=num_points, **path_cfg.get("s_shape", {}))
    return get_path_by_name("butterfly", scale=scale, num_points=num_points, **path_cfg.get("butterfly", {}))


def _inflate_constraints(base: Dict, factor: float = 50.0) -> Dict:
    inflated = base.copy()
    for key in [
        "MAX_VEL",
        "MAX_ACC",
        "MAX_JERK",
        "MAX_ANG_VEL",
        "MAX_ANG_ACC",
        "MAX_ANG_JERK",
    ]:
        inflated[key] = base[key] * factor
    return inflated


def _clamp(val: float, low: float, high: float) -> float:
    return float(max(low, min(high, val)))


def _curvature_aware_speed(env: Env, max_vel: float) -> float:
    seg_idx, dist_next = env._update_segment_info()  # type: ignore
    next_angle = env._get_next_angle(seg_idx)  # type: ignore
    curvature = abs(next_angle) / np.pi
    distance_factor = _clamp(dist_next / (env.half_epsilon * 6 + 1e-6), 0.2, 1.0)
    slow_factor = _clamp(1.0 - 0.5 * curvature, 0.25, 1.0)
    return max_vel * distance_factor * slow_factor


def _clean_path(path_list: List) -> List[Tuple[float, float]]:
    return [tuple(map(float, p)) for p in path_list if p is not None and not np.isnan(p).any()]


def run_simulation(config: SimulationConfig) -> SimulationResult:
    base_cfg = _load_base_config()
    env_cfg = base_cfg.get("environment", {})
    kcm_cfg = base_cfg.get("kinematic_constraints", {})

    dt = env_cfg.get("interpolation_period", 0.1)
    path_points = _build_path(config.path_name, base_cfg)

    constraints = kcm_cfg.copy()
    constraints["MAX_VEL"] = float(config.max_velocity)
    constraints["MAX_JERK"] = float(config.max_jerk)
    if config.disable_kcm:
        constraints = _inflate_constraints(constraints, factor=200.0)

    env = Env(
        device=torch.device("cpu"),
        epsilon=config.epsilon,
        interpolation_period=dt,
        MAX_VEL=constraints["MAX_VEL"],
        MAX_ACC=constraints["MAX_ACC"],
        MAX_JERK=constraints["MAX_JERK"],
        MAX_ANG_VEL=constraints["MAX_ANG_VEL"],
        MAX_ANG_ACC=constraints["MAX_ANG_ACC"],
        MAX_ANG_JERK=constraints["MAX_ANG_JERK"],
        Pm=path_points,
        max_steps=env_cfg.get("max_steps", 3000),
        lookahead_points=env_cfg.get("lookahead_points", 5),
    )

    # 兼容可视化所需的 Pl/Pr
    if not hasattr(env, "Pl"):
        env.Pl = env.cache.get("Pl", [])  # type: ignore
    if not hasattr(env, "Pr"):
        env.Pr = env.cache.get("Pr", [])  # type: ignore

    state = env.reset()

    planner = None
    if config.model_name.lower().startswith("traditional"):
        planner = SCurvePlanner(
            max_vel=constraints["MAX_VEL"],
            max_acc=constraints["MAX_ACC"],
            max_jerk=constraints["MAX_JERK"],
            dt=dt,
        )

    intent_speed = 0.0

    time_s: List[float] = []
    vel: List[float] = []
    acc: List[float] = []
    jerks: List[float] = []
    errors: List[float] = []
    progresses: List[float] = []
    kcm_list: List[float] = []
    traj: List[Tuple[float, float]] = []

    step_idx = 0
    done = False
    while not done and step_idx < env.max_steps:
        if config.model_name.lower().startswith("j-nnc"):
            target_speed = _curvature_aware_speed(env, config.max_velocity)
            smoothing = 0.25 if not config.disable_jerk_reward else 0.6
            intent_speed += (target_speed - intent_speed) * smoothing
            theta_prime = 0.0
        elif config.model_name.lower().startswith("nnc"):
            target_speed = config.max_velocity * 0.95
            smoothing = 0.9
            intent_speed += (target_speed - intent_speed) * smoothing
            theta_prime = 0.0
        else:
            # Traditional planner
            seg_idx = getattr(env, "current_segment_idx", 0)
            next_idx = min(seg_idx + 1, len(env.Pm) - 1)
            target_pos = np.array(env.Pm[next_idx])
            path_angle = env._get_path_direction(env.current_position)  # type: ignore
            path_direction = np.array([np.cos(path_angle), np.sin(path_angle)])
            ang_vel, lin_vel = planner.plan_velocity(env.current_position, target_pos, path_direction)  # type: ignore
            intent_speed = lin_vel
            theta_prime = ang_vel

        action = [float(theta_prime), float(max(intent_speed, 0.0))]
        next_state, _, done, info = env.step(action)
        state = next_state

        # 记录时间序列
        time_s.append(step_idx * dt)
        vel.append(float(env.velocity))
        acc.append(float(env.acceleration))
        jerks.append(float(env.jerk))
        errors.append(float(info.get("contour_error", 0.0)))
        progresses.append(float(info.get("progress", 0.0)))
        kcm_list.append(float(info.get("kcm_intervention", 0.0)))
        traj.append((float(env.current_position[0]), float(env.current_position[1])))

        step_idx += 1

    total_time = step_idx * dt
    max_error = float(max(errors)) if errors else 0.0
    mean_error = float(np.mean(errors)) if errors else 0.0
    max_jerk = float(max(np.abs(jerks))) if jerks else 0.0
    mean_jerk = float(np.mean(np.abs(jerks))) if jerks else 0.0

    result = SimulationResult(
        trajectory=traj,
        pm=_clean_path(env.Pm),
        pl=_clean_path(env.Pl),
        pr=_clean_path(env.Pr),
        time=time_s,
        velocity=vel,
        acceleration=acc,
        jerk=jerks,
        contour_error=errors,
        progress=progresses,
        kcm_intervention=kcm_list,
        metrics={
            "total_time": total_time,
            "max_error": max_error,
            "mean_error": mean_error,
            "max_jerk": max_jerk,
            "mean_jerk": mean_jerk,
        },
        jerk_limit=config.max_jerk,
        epsilon=config.epsilon,
    )
    return result


def build_default_config() -> SimulationConfig:
    base_cfg = _load_base_config()
    kcm_cfg = base_cfg.get("kinematic_constraints", {})
    env_cfg = base_cfg.get("environment", {})
    return SimulationConfig(
        model_name="J-NNC (Ours)",
        path_name="S-Shape",
        disable_jerk_reward=False,
        disable_kcm=False,
        max_velocity=float(kcm_cfg.get("MAX_VEL", 1.0)),
        max_jerk=float(kcm_cfg.get("MAX_JERK", 3.0)),
        epsilon=float(env_cfg.get("epsilon", 0.5)),
    )

