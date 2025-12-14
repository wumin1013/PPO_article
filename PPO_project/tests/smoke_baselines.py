"""Baseline 冒烟测试：验证 baseline_nnc / baseline_s_curve 可以正常启动并运行若干步。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    _build_path,
    _extract_baseline_type,
    _log_run_hyperparams,
    _set_global_seed,
    load_config,
)
from src.algorithms.baselines import create_baseline_agent  # noqa: E402
from src.environment import Env  # noqa: E402


def _build_env(config: dict, device: torch.device) -> Env:
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    reward_weights = config.get("reward_weights", {})

    Pm = _build_path(path_cfg)
    env = Env(
        device=device,
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
    return env


def run_smoke(config_path: str, mode: str, max_steps: int, seed: int) -> None:
    if mode not in {"baseline_nnc", "baseline_s_curve"}:
        raise ValueError("仅支持 baseline_nnc / baseline_s_curve 冒烟")

    # 先将传入路径解析为绝对路径，避免相对路径被重复拼接（如 "PPO_project/configs/xxx.yaml"）。
    resolved_config_path = str(Path(config_path).resolve())
    config, resolved_path = load_config(resolved_config_path)
    experiment_cfg = config.setdefault("experiment", {})
    experiment_cfg["mode"] = mode
    config["seed"] = seed
    experiment_cfg["seed"] = seed
    _set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _build_env(config, device)
    gamma_for_log = config.get("ppo", {}).get("gamma")
    _log_run_hyperparams(seed, env, gamma_for_log, mode)

    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    config["state_dim"] = obs_space.shape[0] if obs_space is not None else env.observation_dim
    config["action_dim"] = act_space.shape[0] if act_space is not None else env.action_space_dim
    config["observation_space"] = obs_space
    config["action_space"] = act_space

    baseline_type = _extract_baseline_type(mode)
    agent = create_baseline_agent(baseline_type, config, device)
    print(f"[SMOKE] baseline={baseline_type} config={resolved_path}")

    state = env.reset()
    done = False
    steps = 0
    total_reward = 0.0

    while steps < max_steps and not done:
        action = agent.take_action(state)
        state, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward

    print(
        f"[SMOKE] finished baseline={baseline_type} steps={steps} "
        f"done={done} total_reward={total_reward:.3f}"
    )
    print(f"[SMOKE] last info: step={info.get('step')} contour_error={info.get('contour_error')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline 冒烟测试")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline_nnc",
        choices=["baseline_nnc", "baseline_s_curve"],
        help="基线模式",
    )
    parser.add_argument("--steps", type=int, default=120, help="运行步数上限")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(args.config, args.mode, args.steps, args.seed)
