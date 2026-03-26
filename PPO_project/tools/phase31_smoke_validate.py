from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _write_trace_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("step,e_n,v,omega,cornerness,corner_phase,corridor_enabled,dist_to_turn,progress\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _check_obs_dim(env: Env) -> Dict[str, Any]:
    expected = int(len(getattr(env, "base_state_keys", [])) + int(getattr(env, "lookahead_obs_dim", 0)))
    obs_dim = int(getattr(env, "observation_space").shape[0])
    return {
        "passed": bool(obs_dim == expected),
        "obs_dim": obs_dim,
        "expected_dim": expected,
        "base_dim": int(len(getattr(env, "base_state_keys", []))),
        "lookahead_obs_dim": int(getattr(env, "lookahead_obs_dim", 0)),
        "n_scales": int(len(getattr(env, "lookahead_obs_scales", []))),
    }


def _check_obs_off_zero(config: dict, *, device: torch.device) -> Dict[str, Any]:
    cfg_on = copy.deepcopy(config)
    cfg_off = copy.deepcopy(config)
    cfg_on.setdefault("environment", {})
    cfg_off.setdefault("environment", {})
    cfg_on["environment"]["lookahead_obs_enabled"] = True
    cfg_off["environment"]["lookahead_obs_enabled"] = False

    env_on = _build_env(cfg_on, device=device)
    env_off = _build_env(cfg_off, device=device)
    obs_on = env_on.reset()
    obs_off = env_off.reset()
    dim_on = int(obs_on.shape[0])
    dim_off = int(obs_off.shape[0])
    lookahead_dim = int(getattr(env_off, "lookahead_obs_dim", 0))
    if lookahead_dim > 0:
        off_slice = np.asarray(obs_off[-lookahead_dim:], dtype=float)
        on_slice = np.asarray(obs_on[-lookahead_dim:], dtype=float)
        off_zero = bool(np.allclose(off_slice, 0.0, atol=1e-6))
        on_non_zero = bool(np.any(np.abs(on_slice) > 1e-8))
    else:
        off_zero = True
        on_non_zero = True

    passed = bool(dim_on == dim_off and off_zero)
    return {
        "passed": passed,
        "obs_dim_on": dim_on,
        "obs_dim_off": dim_off,
        "lookahead_dim": lookahead_dim,
        "off_slice_all_zero": off_zero,
        "on_slice_has_signal": on_non_zero,
    }


def _check_theta_ref_independent(config: dict, *, device: torch.device) -> Dict[str, Any]:
    env = _build_env(config, device=device)
    env.reset()
    theta_ref_a = float(env.get_tangent_direction(env.current_position, record=False))
    base = float(getattr(env, "lookahead_dist", 1.0))
    env.lookahead_dist = float(base * 3.0 + 1.0)
    theta_ref_b = float(env.get_tangent_direction(env.current_position, record=False))
    delta = float(abs(env._wrap_angle(theta_ref_b - theta_ref_a)))
    passed = bool(delta <= 1e-9)
    return {
        "passed": passed,
        "theta_ref_delta_abs_rad": delta,
        "lookahead_dist_a": base,
        "lookahead_dist_b": float(env.lookahead_dist),
    }


def _run_cornerness_rollout(
    config: dict,
    *,
    device: torch.device,
    episodes: int,
    max_steps: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    env = _build_env(config, device=device)
    _set_seed(seed)

    cornerness_vals: List[float] = []
    cornerness_corner: List[float] = []
    cornerness_straight: List[float] = []
    trace_rows: List[Dict[str, float]] = []

    for ep in range(int(episodes)):
        _set_seed(seed + ep)
        _ = env.reset()
        done = False
        step = 0
        while not done and step < int(max_steps):
            action = np.array([0.0, 0.7], dtype=float)
            _obs, _reward, done, info = env.step(action)
            p4_status = info.get("p4_status", {})
            if not isinstance(p4_status, dict):
                p4_status = {}
            corridor_status = info.get("corridor_status", {})
            if not isinstance(corridor_status, dict):
                corridor_status = {}

            cornerness = float(info.get("cornerness", p4_status.get("cornerness", 0.0)))
            corner_phase = bool(corridor_status.get("corner_phase", env.turn_info.get("corner_phase", False)))
            e_n = corridor_status.get("e_n", None)
            if e_n is None:
                e_n = float(env.state[1]) * float(env.half_epsilon) if len(env.state) > 1 else 0.0

            cornerness_vals.append(cornerness)
            if corner_phase:
                cornerness_corner.append(cornerness)
            else:
                cornerness_straight.append(cornerness)

            if ep == 0:
                trace_rows.append(
                    {
                        "step": float(step),
                        "e_n": float(e_n),
                        "v": float(env.velocity),
                        "omega": float(env.angular_vel),
                        "cornerness": float(cornerness),
                        "corner_phase": 1.0 if corner_phase else 0.0,
                        "corridor_enabled": 1.0 if bool(corridor_status.get("enabled", False)) else 0.0,
                        "dist_to_turn": float(corridor_status.get("dist_to_turn", float("inf"))),
                        "progress": float(info.get("progress", 0.0)),
                    }
                )
            step += 1

    if cornerness_vals:
        c_diff = np.abs(np.diff(np.asarray(cornerness_vals, dtype=float)))
        c_diff_p95 = float(np.percentile(c_diff, 95)) if c_diff.size > 0 else 0.0
    else:
        c_diff_p95 = float("inf")

    straight_mean = float(np.mean(cornerness_straight)) if cornerness_straight else float("nan")
    corner_mean = float(np.mean(cornerness_corner)) if cornerness_corner else float("nan")

    pass_straight_near_zero = bool(math.isfinite(straight_mean) and straight_mean <= 0.35)
    pass_corner_rise = bool(
        math.isfinite(corner_mean)
        and (
            (math.isfinite(straight_mean) and corner_mean >= straight_mean + 0.05)
            or corner_mean >= 0.40
        )
    )
    pass_smooth = bool(math.isfinite(c_diff_p95) and c_diff_p95 <= 0.25)
    passed = bool(pass_straight_near_zero and pass_corner_rise and pass_smooth)

    summary = {
        "passed": passed,
        "straight_mean": straight_mean,
        "corner_mean": corner_mean,
        "corner_samples": int(len(cornerness_corner)),
        "straight_samples": int(len(cornerness_straight)),
        "cornerness_diff_p95": c_diff_p95,
        "checks": {
            "straight_near_zero": pass_straight_near_zero,
            "corner_rise": pass_corner_rise,
            "smooth_p95": pass_smooth,
        },
    }
    return summary, trace_rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase31(v3.2) smoke validation tool.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--episodes", type=int, default=2, help="Smoke episodes for cornerness rollout")
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
    obs_dim_check = _check_obs_dim(env)
    obs_off_check = _check_obs_off_zero(config, device=device)
    theta_ref_check = _check_theta_ref_independent(config, device=device)
    cornerness_check, trace_rows = _run_cornerness_rollout(
        config,
        device=device,
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        seed=int(args.seed),
    )

    reward_weights = config.get("reward_weights", {}) if isinstance(config.get("reward_weights", {}), dict) else {}
    cornerness_cfg = reward_weights.get("cornerness", {}) if isinstance(reward_weights.get("cornerness", {}), dict) else {}

    passed = bool(
        obs_dim_check.get("passed")
        and obs_off_check.get("passed")
        and theta_ref_check.get("passed")
        and cornerness_check.get("passed")
    )

    summary = {
        "phase": "phase31_smoke_v3_2",
        "passed": passed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(cfg_path),
        "seed": int(args.seed),
        "epsilon": float(getattr(env, "epsilon", float("nan"))),
        "half_epsilon": float(getattr(env, "half_epsilon", float("nan"))),
        "lookahead_obs_enabled": bool(getattr(env, "lookahead_obs_enabled", False)),
        "lookahead_obs_scales": [float(v) for v in getattr(env, "lookahead_obs_scales", [])],
        "lookahead_obs_dim": int(getattr(env, "lookahead_obs_dim", 0)),
        "cornerness_config": cornerness_cfg,
        "checks": {
            "obs_dim": obs_dim_check,
            "obs_off_zero": obs_off_check,
            "theta_ref_independence": theta_ref_check,
            "cornerness_behavior": cornerness_check,
        },
    }

    _write_json(out_dir / "summary.json", summary)
    _write_trace_csv(out_dir / "trace.csv", trace_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
