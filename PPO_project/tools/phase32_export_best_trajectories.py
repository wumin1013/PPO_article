from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.ppo import PPOContinuous
from src.environment import Env, create_env_compatible
from src.utils.geometry import generate_offset_paths
from src.utils.path_generator import get_path_by_name


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid config: {path}")
    return data


def _build_path(path_cfg: Dict[str, Any]) -> List[np.ndarray]:
    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    extra = path_cfg.get(path_type, {})
    if not isinstance(extra, dict):
        extra = {}
    if path_type == "square" and "closed" in path_cfg and "closed" not in extra:
        extra["closed"] = bool(path_cfg.get("closed"))
    return get_path_by_name(path_type, scale=scale, num_points=num_points, **extra)


def _make_env(config: dict, path_cfg: Dict[str, Any], device: torch.device) -> Env:
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    reward_weights = config.get("reward_weights", {})
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    use_obs_normalizer = bool(training_cfg.get("use_obs_normalizer", False))
    path_points = _build_path(path_cfg)
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
        path_name=str(path_cfg.get("name") or path_cfg.get("type") or ""),
        disable_kcm=bool(config.get("experiment", {}).get("enable_kcm", True) is False),
    )


def _make_agent(config: dict, env: Env, device: torch.device) -> PPOContinuous:
    ppo_cfg = config["ppo"]
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    return PPOContinuous(
        state_dim=None,
        hidden_dim=ppo_cfg["hidden_dim"],
        action_dim=None,
        actor_lr=ppo_cfg["actor_lr"],
        critic_lr=ppo_cfg["critic_lr"],
        lmbda=ppo_cfg["lmbda"],
        epochs=ppo_cfg["epochs"],
        eps=ppo_cfg["eps"],
        gamma=ppo_cfg["gamma"],
        ent_coef=ppo_cfg.get("ent_coef", 0.0),
        device=device,
        observation_space=obs_space,
        action_space=act_space,
    )


def _rollout_with_model(config: dict, path_cfg: Dict[str, Any], model_path: Path, device: torch.device) -> dict:
    env = _make_env(config, path_cfg, device)
    agent = _make_agent(config, env, device)

    ckpt = torch.load(model_path, map_location=device)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()
    agent.critic.eval()

    state = env.reset(random_start=False)
    done = False
    info: dict = {}
    total_reward = 0.0
    trajectory = [np.asarray(env.current_position, dtype=float).copy()]
    velocities = [float(getattr(env, "velocity", 0.0))]
    rows: List[Dict[str, float]] = [
        {
            "step": 0,
            "x": float(trajectory[-1][0]),
            "y": float(trajectory[-1][1]),
            "velocity": velocities[-1],
            "acceleration": float(getattr(env, "acceleration", 0.0)),
            "jerk": float(getattr(env, "jerk", 0.0)),
            "contour_error": float(env.get_contour_error(env.current_position)),
            "kcm_intervention": float(getattr(env, "kcm_intervention", 0.0)),
            "raw_linear_jerk_demand": 0.0,
            "disable_kcm": float(bool(getattr(env, "disable_kcm", False))),
        }
    ]

    step = 0
    with torch.no_grad():
        while not done:
            action = np.asarray(agent.take_action(state), dtype=float).flatten()
            if getattr(env, "action_space", None) is not None:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                action = np.array([np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)], dtype=float)

            state, reward, done, info = env.step(action)
            step += 1
            total_reward += float(reward)

            pos = np.asarray(env.current_position, dtype=float).copy()
            vel = float(getattr(env, "velocity", 0.0))
            trajectory.append(pos)
            velocities.append(vel)
            rows.append(
                {
                    "step": step,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "velocity": vel,
                    "acceleration": float(getattr(env, "acceleration", 0.0)),
                    "jerk": float(info.get("jerk", getattr(env, "jerk", 0.0))),
                    "contour_error": float(info.get("contour_error", 0.0)),
                    "kcm_intervention": float(info.get("kcm_intervention", 0.0)),
                    "raw_linear_jerk_demand": float(info.get("raw_linear_jerk_demand", 0.0)),
                    "disable_kcm": float(bool(info.get("disable_kcm", getattr(env, "disable_kcm", False)))),
                }
            )

            if step > int(config["environment"]["max_steps"]) + 5:
                break

    progress = float(info.get("progress", 0.0))
    score = progress * 1_000_000.0 + total_reward
    return {
        "model": str(model_path),
        "score": float(score),
        "reward": float(total_reward),
        "progress": float(progress),
        "steps": int(step),
        "done_reason": str(info.get("done_reason", "unknown")),
        "trajectory": np.asarray(trajectory, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "rows": rows,
        "reference": np.asarray(env.Pm, dtype=float),
        "half_epsilon": float(env.half_epsilon),
    }


def _save_plot(path_name: str, result: dict, out_png: Path) -> None:
    traj = np.asarray(result["trajectory"], dtype=float)
    vel = np.asarray(result["velocities"], dtype=float)
    ref = np.asarray(result["reference"], dtype=float)
    half_epsilon = float(result.get("half_epsilon", 1.0))
    closed = bool(np.allclose(ref[0], ref[-1], atol=1e-6))
    pl_list, pr_list = generate_offset_paths(ref, half_epsilon, closed=closed)
    pl = np.asarray(pl_list, dtype=float)
    pr = np.asarray(pr_list, dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 6.4), dpi=150)
    ax.plot(pl[:, 0], pl[:, 1], color="#1f77b4", linewidth=1.1, alpha=0.9, label="Pl")
    ax.plot(pr[:, 0], pr[:, 1], color="#d62728", linewidth=1.1, alpha=0.9, label="Pr")
    ax.plot(ref[:, 0], ref[:, 1], "k--", linewidth=1.0, label="Pm (Reference)")

    if traj.shape[0] >= 2:
        segments = np.stack([traj[:-1], traj[1:]], axis=1)
        seg_vel = vel[1:] if vel.shape[0] == traj.shape[0] else vel[: segments.shape[0]]
        if seg_vel.size == 0:
            seg_vel = np.zeros((segments.shape[0],), dtype=float)
        vmin = float(np.min(seg_vel))
        vmax = float(np.max(seg_vel))
        if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) < 1e-6:
            vmin = float(np.nan_to_num(vmin, nan=0.0))
            vmax = vmin + 1.0
        line = LineCollection(segments, cmap="turbo", linewidths=2.0, alpha=0.95)
        line.set_array(seg_vel)
        line.set_clim(vmin, vmax)
        ax.add_collection(line)
        cb = fig.colorbar(line, ax=ax)
    else:
        scatter = ax.scatter(traj[:, 0], traj[:, 1], c=vel, cmap="turbo", s=16, alpha=0.95, label="Trajectory")
        cb = fig.colorbar(scatter, ax=ax)

    ax.scatter([ref[0, 0]], [ref[0, 1]], c="green", marker="o", s=55, label="Ref Start")
    ax.scatter([ref[-1, 0]], [ref[-1, 1]], c="red", marker="x", s=55, label="Ref End")
    ax.scatter([traj[-1, 0]], [traj[-1, 1]], c="black", marker="*", s=45, label="Trajectory End")
    cb.set_label("Velocity (mm/s)")
    ax.set_title(
        f"{path_name} | best model rollout\n"
        f"steps={result['steps']} progress={result['progress']:.4f} reward={result['reward']:.2f} "
        f"done={result.get('done_reason', 'unknown')}"
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _save_csv(rows: List[Dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "x",
        "y",
        "velocity",
        "acceleration",
        "jerk",
        "contour_error",
        "kcm_intervention",
        "raw_linear_jerk_demand",
        "disable_kcm",
    ]
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(fieldnames) + "\n")
        for row in rows:
            f.write(",".join(f"{float(row.get(name, 0.0)):.10f}" for name in fieldnames) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export phase32 best trajectory plots for each path.")
    parser.add_argument("--config", required=True, type=str, help="YAML config path")
    parser.add_argument("--run_dir", required=True, type=str, help="training run dir")
    parser.add_argument("--out", required=True, type=str, help="output directory")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out_dir = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(config_path)
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    curriculum = training_cfg.get("path_curriculum", {}) if isinstance(training_cfg.get("path_curriculum", {}), dict) else {}
    paths = curriculum.get("paths", []) if isinstance(curriculum.get("paths", []), list) else []
    if not paths:
        paths = [config.get("path", {})]

    normalized_paths: List[Dict[str, Any]] = []
    seen = set()
    for item in paths:
        if not isinstance(item, dict):
            continue
        cfg = copy.deepcopy(item)
        name = str(cfg.get("name") or cfg.get("type") or f"path_{len(normalized_paths)}")
        if name in seen:
            continue
        seen.add(name)
        cfg["name"] = name
        normalized_paths.append(cfg)

    ckpt_dir = run_dir / "checkpoints"
    model_candidates: List[Path] = []
    best_model = ckpt_dir / "best_model.pth"
    if best_model.exists():
        model_candidates.append(best_model)
    all_tracking = sorted(ckpt_dir.glob("tracking_model*.pth"))
    model_candidates.extend(all_tracking)
    # 去重并保序
    model_candidates = list(dict.fromkeys([p for p in model_candidates if p.exists()]))
    if not model_candidates:
        raise FileNotFoundError(f"No model checkpoints found under: {ckpt_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "models": [str(p) for p in model_candidates],
        "paths": {},
    }

    for path_cfg in normalized_paths:
        path_name = str(path_cfg["name"])
        best_result = None
        for model_path in model_candidates:
            result = _rollout_with_model(config, path_cfg, model_path, device)
            if (best_result is None) or (float(result["score"]) > float(best_result["score"])):
                best_result = result

        assert best_result is not None
        png_path = out_dir / f"{path_name}_best.png"
        csv_path = out_dir / f"{path_name}_best.csv"
        _save_plot(path_name, best_result, png_path)
        _save_csv(best_result["rows"], csv_path)

        summary["paths"][path_name] = {
            "model": best_result["model"],
            "reward": best_result["reward"],
            "progress": best_result["progress"],
            "steps": best_result["steps"],
            "done_reason": best_result.get("done_reason", "unknown"),
            "png": str(png_path),
            "csv": str(csv_path),
        }
        print(
            f"[{path_name}] model={Path(best_result['model']).name} "
            f"progress={best_result['progress']:.4f} reward={best_result['reward']:.2f} "
            f"done={best_result.get('done_reason', 'unknown')}"
        )

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
