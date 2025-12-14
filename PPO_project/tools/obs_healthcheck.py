"""观测健康检查：验证 lookahead 特征无 NaN/Inf 且归一化合理。"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.environment import Env  # noqa: E402
    from src.utils.path_generator import get_path_by_name  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"[ERROR] 依赖缺失：{exc}. 请先安装 gymnasium 等依赖，示例: python -m pip install -r PPO_project/requirements.txt")
    raise


def check_path(config_path: Path, steps: int = 100, seed: int = 42) -> None:
    np.random.seed(seed)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    if path_cfg["type"] == "waypoints":
        Pm = [np.array(wp) for wp in path_cfg["waypoints"]]
    else:
        path_type = path_cfg["type"]
        scale = path_cfg.get("scale", 10.0)
        num_points = path_cfg.get("num_points", 200)
        kwargs = path_cfg.get(path_type, {})
        Pm = get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)
    env = Env(
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
        reward_weights=config.get("reward_weights", {}),
    )

    state = env.reset()
    base_len = len(env.base_state_keys)
    fs = env.lookahead_feature_size
    s_vals = []
    d_vals = []
    nan_flags = 0
    for _ in range(steps):
        action = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(0.0, 1.0)], dtype=float)
        state, _, done, _ = env.step(action)
        if np.any(~np.isfinite(state)):
            nan_flags += 1
        s_chunk = state[base_len :: fs]
        d_chunk = state[base_len + 1 :: fs]
        s_vals.append(s_chunk)
        d_vals.append(d_chunk)
        if done:
            state = env.reset()
    s_all = np.concatenate(s_vals) if s_vals else np.array([])
    d_all = np.concatenate(d_vals) if d_vals else np.array([])
    print(f"[CHECK] path={path_cfg.get('type','?')} steps={steps} nan_steps={nan_flags}")
    if s_all.size:
        print(f"  s_norm min={s_all.min():.4f} mean={s_all.mean():.4f} max={s_all.max():.4f}")
    if d_all.size:
        sat = float(np.mean(np.abs(d_all) > 0.95))
        print(f"  d_norm min={d_all.min():.4f} mean={d_all.mean():.4f} max={d_all.max():.4f} |sat|>0.95 ratio={sat:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lookahead 观测健康检查")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config_dir = ROOT / "configs"
    for name in ["train_line.yaml", "train_square.yaml", "train_s_shape.yaml"]:
        cfg_path = config_dir / name
        if cfg_path.exists():
            check_path(cfg_path, steps=args.steps, seed=args.seed)
        else:
            print(f"[WARN] missing config {cfg_path}")


if __name__ == "__main__":
    main()
