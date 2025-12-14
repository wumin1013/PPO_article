"""新旧环境行为一致性冒烟测试。

固定随机种子、同一路径与动作序列，对比新版 Env 与 legacy Env 的位置/进度/奖励差异。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from legacy_env import LegacyEnv
from main import _build_path, load_config
from src.environment import Env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _resolve_config_path(path_str: str) -> str:
    """解析配置路径，优先使用仓库内 configs 目录。"""
    raw = Path(path_str)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(ROOT / raw)
    if not raw.is_absolute():
        candidates.append((ROOT / "configs" / raw.name))
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    # 最后兜底返回原始绝对路径（留给上层报错）
    return str(candidates[0].resolve())


def _init_envs(config_path: str) -> Tuple[Env, LegacyEnv, dict]:
    resolved_cfg_path = _resolve_config_path(config_path)
    config, resolved = load_config(resolved_cfg_path)
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    reward_weights = config.get("reward_weights", {})

    Pm = _build_path(path_cfg)
    env_new = Env(
        device="cpu",
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

    env_old = LegacyEnv(
        device="cpu",
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
        reward_weights=reward_weights,
    )
    return env_new, env_old, {"config_path": resolved, "path_type": path_cfg.get("type", "unknown")}


def summarize_err(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"[PARITY] {name}: no data")
        return
    print(
        f"[PARITY] {name}: mean={arr.mean():.6f} max={arr.max():.6f} "
        f"p50={np.quantile(arr,0.5):.6f} p90={np.quantile(arr,0.9):.6f} p99={np.quantile(arr,0.99):.6f}"
    )


def run_parity_check(config_path: str, steps: int, seed: int) -> None:
    set_seed(seed)
    env_new, env_old, meta = _init_envs(config_path)
    state_new = env_new.reset()
    state_old = env_old.reset()

    rng = np.random.default_rng(seed)
    pos_errs = []
    progress_errs = []
    reward_errs = []
    action_gap_policy_exec = []

    for step_idx in range(steps):
        raw_action = np.array([rng.normal(loc=0.0, scale=0.3), rng.uniform(0.0, 1.0)], dtype=float)
        clipped_action = np.array([np.clip(raw_action[0], -1.0, 1.0), np.clip(raw_action[1], 0.0, 1.0)], dtype=float)

        state_new, r_new, done_new, info_new = env_new.step(clipped_action)
        state_old, r_old, done_old, info_old = env_old.step(clipped_action)

        pos_errs.append(float(np.linalg.norm(info_new["position"] - info_old["position"])))
        progress_errs.append(abs(float(info_new.get("progress", 0.0)) - float(info_old.get("progress", 0.0))))
        reward_errs.append(abs(float(r_new) - float(r_old)))

        gap_new = info_new.get("action_gap_abs")
        if gap_new is not None:
            action_gap_policy_exec.append(np.linalg.norm(np.asarray(gap_new)))

        if done_new and done_old:
            break

    print(f"[PARITY] config={meta['config_path']} path={meta['path_type']} seed={seed} steps_run={len(pos_errs)}")
    summarize_err("position_diff", np.asarray(pos_errs))
    summarize_err("progress_diff", np.asarray(progress_errs))
    summarize_err("reward_diff", np.asarray(reward_errs))
    summarize_err("action_exec_vs_policy", np.asarray(action_gap_policy_exec) if action_gap_policy_exec else np.array([]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新旧环境行为一致性回归测试")
    parser.add_argument("--config", type=str, default="configs/train_line.yaml", help="YAML 配置文件路径")
    parser.add_argument("--steps", type=int, default=200, help="运行步数上限")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = _resolve_config_path(args.config)
    run_parity_check(config_path, args.steps, args.seed)
