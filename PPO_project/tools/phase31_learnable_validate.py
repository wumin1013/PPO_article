from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import Env, create_env_compatible
from src.utils.path_generator import get_path_by_name


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid config: {path}")
    return data


def _build_env(config: dict, *, device: torch.device) -> Env:
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    path_cfg = config["path"]
    reward_weights = config.get("reward_weights", {})

    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    extra_kwargs = {k: v for k, v in path_cfg.items() if k not in {"type", "scale", "num_points"}}
    path_points = get_path_by_name(str(path_cfg["type"]), scale=scale, num_points=num_points, **extra_kwargs)

    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    use_obs_normalizer = bool(training_cfg.get("use_obs_normalizer", False))

    return create_env_compatible(
        device=device,
        epsilon=env_cfg["epsilon"],
        interpolation_period=env_cfg["interpolation_period"],
        MAX_VEL=kcm_cfg["MAX_VEL"],
        MAX_ACC=kcm_cfg["MAX_ACC"],
        MAX_JERK=kcm_cfg["MAX_JERK"],
        MAX_ANG_VEL=kcm_cfg["MAX_ANG_VEL"],
        MAX_ANG_ACC=kcm_cfg["MAX_ANG_ACC"],
        MAX_ANG_JERK=kcm_cfg["MAX_ANG_JERK"],
        Pm=path_points,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        lookahead_obs_enabled=env_cfg.get("lookahead_obs_enabled", True),
        lookahead_obs_scales=env_cfg.get("lookahead_obs_scales", [1.0]),
        reward_weights=reward_weights,
        curvature_observation=env_cfg.get("curvature_observation"),
        return_normalized_obs=not use_obs_normalizer,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _action_effect_check(env: Env) -> Dict[str, Any]:
    _ = env.reset()
    action_low = np.array([0.0, 0.70, 0.10], dtype=float)
    action_high = np.array([0.0, 0.70, 0.90], dtype=float)

    _ = env.step(action_low)
    low_dist = float(getattr(env, "_lookahead_dist_active", getattr(env, "lookahead_dist", 0.0)))

    _ = env.reset()
    _ = env.step(action_high)
    high_dist = float(getattr(env, "_lookahead_dist_active", getattr(env, "lookahead_dist", 0.0)))

    passed = bool(high_dist > low_dist + 1e-6)
    return {
        "passed": passed,
        "dist_low_u": low_dist,
        "dist_high_u": high_dist,
        "delta": float(high_dist - low_dist),
    }


def _region_split_check(env: Env, *, episodes: int, max_steps: int, seed: int) -> Dict[str, Any]:
    straight_dists: List[float] = []
    corner_dists: List[float] = []

    for ep in range(int(episodes)):
        _set_seed(seed + ep)
        _ = env.reset()
        done = False
        step = 0
        while not done and step < int(max_steps):
            action = np.array([0.0, 0.75, 0.50], dtype=float)
            _obs, _reward, done, info = env.step(action)

            p4_status = info.get("p4_status", {}) if isinstance(info, dict) else {}
            if not isinstance(p4_status, dict):
                p4_status = {}
            corridor_status = info.get("corridor_status", {}) if isinstance(info, dict) else {}
            if not isinstance(corridor_status, dict):
                corridor_status = {}

            dist_active = float(
                p4_status.get(
                    "lookahead_dist_active",
                    getattr(env, "_lookahead_dist_active", getattr(env, "lookahead_dist", 0.0)),
                )
            )
            is_corner = bool(corridor_status.get("corner_phase", False))
            if is_corner:
                corner_dists.append(dist_active)
            else:
                straight_dists.append(dist_active)
            step += 1

    straight_mean = float(np.mean(straight_dists)) if straight_dists else float("nan")
    corner_mean = float(np.mean(corner_dists)) if corner_dists else float("nan")

    passed = bool(
        np.isfinite(straight_mean)
        and np.isfinite(corner_mean)
        and len(straight_dists) > 10
        and len(corner_dists) > 10
        and corner_mean > straight_mean + 1e-3
    )

    return {
        "passed": passed,
        "straight_mean": straight_mean,
        "corner_mean": corner_mean,
        "straight_samples": int(len(straight_dists)),
        "corner_samples": int(len(corner_dists)),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase31 learnable lookahead validation")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes for region split check")
    parser.add_argument("--max_steps", type=int, default=1200, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(cfg_path)
    _set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = _build_env(config, device=device)

    action_dim = int(getattr(env, "action_space_dim", 0))
    policy_action_enabled = bool(getattr(env, "lookahead_control_policy_action", False))
    action_dim_check = {
        "passed": bool(policy_action_enabled and action_dim == 3),
        "policy_action_enabled": policy_action_enabled,
        "action_space_dim": action_dim,
    }

    effect_check = _action_effect_check(env)
    split_check = _region_split_check(
        env,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )

    passed = bool(action_dim_check["passed"] and effect_check["passed"] and split_check["passed"])

    summary = {
        "phase": "phase31_learnable_validate",
        "passed": passed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(cfg_path),
        "seed": int(args.seed),
        "checks": {
            "action_dim": action_dim_check,
            "policy_action_effect": effect_check,
            "region_split": split_check,
        },
    }

    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
