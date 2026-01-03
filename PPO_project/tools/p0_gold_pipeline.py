from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
ACCEPTANCE_SCRIPT = PROJECT_ROOT / "tools" / "acceptance_suite.py"
ARCHIVE_ROOT = REPO_ROOT / "PPO_FINAL_OPTIMIZATION_PLAN" / "00_Archive" / "P0_gold"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.ppo import PPOContinuous
from src.environment import Env
from src.utils.logger import DataLogger


@dataclass
class SeedResult:
    seed: int
    config_path: Path
    run_dir: Path
    model_path: Path
    eval_out_dir: Path
    summary: Dict[str, object]
    eval_seed: int
    episode_set: Optional[str]
    eval_command: List[str]


def _run(cmd: Sequence[str], *, cwd: Path) -> None:
    print(f"[run] {' '.join(map(str, cmd))}")
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _parse_seeds(value: str) -> List[int]:
    seeds: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("seeds is empty")
    return seeds


def _write_seed_config(base_config: Path, seed: int, out_path: Path) -> None:
    text = base_config.read_text(encoding="utf-8")
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if line.lstrip().startswith("seed:"):
            prefix = line[: len(line) - len(line.lstrip())]
            lines[idx] = f"{prefix}seed: {seed}"
            replaced = True
            break
    if not replaced:
        lines.insert(0, f"seed: {seed}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    extra_kwargs = {k: v for k, v in path_cfg.items() if k not in {"type", "scale", "num_points"}}
    from src.utils.path_generator import get_path_by_name

    path_points = get_path_by_name(path_type, scale=scale, num_points=num_points, **extra_kwargs)

    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    use_obs_normalizer = bool(training_cfg.get("use_obs_normalizer", False))

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
        Pm=path_points,
        max_steps=env_cfg["max_steps"],
        lookahead_points=env_cfg.get("lookahead_points", 5),
        reward_weights=reward_weights,
        return_normalized_obs=not use_obs_normalizer,
    )
    return env


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_agent(config: dict, env: Env, *, device: torch.device) -> PPOContinuous:
    ppo_cfg = config["ppo"]
    agent = PPOContinuous(
        state_dim=None,
        hidden_dim=int(ppo_cfg["hidden_dim"]),
        action_dim=None,
        actor_lr=float(ppo_cfg["actor_lr"]),
        critic_lr=float(ppo_cfg["critic_lr"]),
        lmbda=float(ppo_cfg["lmbda"]),
        epochs=int(ppo_cfg["epochs"]),
        eps=float(ppo_cfg["eps"]),
        gamma=float(ppo_cfg["gamma"]),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        device=device,
        observation_space=getattr(env, "observation_space", None),
        action_space=getattr(env, "action_space", None),
    )
    return agent


def _take_action(agent: PPOContinuous, state: np.ndarray, *, deterministic: bool) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
    state_tensor = torch.from_numpy(state_arr).to(agent.device)
    with torch.no_grad():
        mu, std = agent.actor(state_tensor)
        if deterministic:
            action = mu
        else:
            action = torch.distributions.Normal(mu, std).sample()
    return action.squeeze(0).detach().cpu().numpy()


def _rollout_trace(config: dict, model_path: Path, *, deterministic: bool) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _build_env(config, device=device)
    agent = _load_agent(config, env, device=device)

    with model_path.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor.eval()
    agent.critic.eval()

    trace_rows: List[Dict[str, float]] = []
    obs = env.reset()
    done = False
    step_idx = 0
    dt = float(env.interpolation_period)

    with torch.no_grad():
        while not done:
            action = _take_action(agent, obs, deterministic=deterministic)
            if getattr(env, "action_space", None) is not None:
                low, high = env.action_space.low, env.action_space.high
                action = np.clip(action, low, high)
            else:
                action = np.array([np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)], dtype=float)

            obs, _reward, done, info = env.step(action)
            corridor_status = info.get("corridor_status", {})
            if not isinstance(corridor_status, dict):
                corridor_status = {}
            p4_status = info.get("p4_status", {})
            if not isinstance(p4_status, dict):
                p4_status = {}
            ref_point = DataLogger.project_to_path(
                position=env.current_position,
                path_points=env.Pm,
                segment_index=info.get("segment_idx", getattr(env, "current_segment_idx", 0)),
            )
            corner_mode = float(p4_status.get("corner_mode", 0.0))
            corner_phase = bool(corridor_status.get("corner_phase", False))
            corner_mask = 1.0 if (corner_mode >= 0.5 or corner_phase) else 0.0
            dist_to_corner = p4_status.get("dist_to_turn", corridor_status.get("dist_to_turn", float("inf")))
            if dist_to_corner is None:
                dist_to_corner = float("inf")
            recovery_active = bool(float(p4_status.get("recovery_cap_active", 0.0)) >= 0.5)
            mode_proxy = 2.0 if recovery_active else (1.0 if corner_mask >= 0.5 else 0.0)
            mode_label = "recovery" if recovery_active else ("corner" if corner_mask >= 0.5 else "normal")
            trace_rows.append(
                {
                    "timestamp": float(step_idx * dt),
                    "position_x": float(env.current_position[0]),
                    "position_y": float(env.current_position[1]),
                    "reference_x": float(ref_point[0]),
                    "reference_y": float(ref_point[1]),
                    "velocity": float(env.velocity),
                    "acceleration": float(env.acceleration),
                    "jerk": float(env.jerk),
                    "omega": float(getattr(env, "angular_vel", 0.0)),
                    "domega": float(getattr(env, "angular_acc", 0.0)),
                    "jerk_proxy": float(getattr(env, "angular_jerk", 0.0)),
                    "contour_error": float(info.get("contour_error", 0.0)),
                    "kcm_intervention": float(info.get("kcm_intervention", 0.0)),
                    "corner_mask": float(corner_mask),
                    "dist_to_corner": float(dist_to_corner),
                    "mode": mode_label,
                    "mode_proxy": float(mode_proxy),
                }
            )
            step_idx += 1

    return trace_rows


def _write_trace_csv(rows: List[Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_overlay(rows: List[Dict[str, float]], env: Env, out_path: Path) -> None:
    if not rows:
        return
    ref_points = env.Pm
    traj_x = [row["position_x"] for row in rows]
    traj_y = [row["position_y"] for row in rows]
    ref_x = [float(p[0]) for p in ref_points]
    ref_y = [float(p[1]) for p in ref_points]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref_x, ref_y, "--", color="#1f77b4", linewidth=1.5, label="Reference")
    ax.plot(traj_x, traj_y, color="#e03131", linewidth=2.0, label="Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Overlay: Trajectory vs Reference")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_timeseries(rows: List[Dict[str, float]], *, key: str, ylabel: str, title: str, out_path: Path) -> None:
    if not rows:
        return
    t = [row["timestamp"] for row in rows]
    y = [row[key] for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(t, y, color="#1f77b4", linewidth=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _read_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", {})


def _select_best(results: List[SeedResult]) -> SeedResult:
    def key_fn(result: SeedResult) -> Tuple[float, float, float, float, float]:
        summary = result.summary
        return (
            1.0 if summary.get("passed") else 0.0,
            float(summary.get("success_rate", 0.0)),
            -float(summary.get("stall_rate", 1.0)),
            float(summary.get("mean_progress_final", 0.0)),
            -float(summary.get("max_abs_contour_error", float("inf"))),
        )

    return sorted(results, key=key_fn, reverse=True)[0]


def _hash_env_params(config: dict) -> str:
    payload = {
        "environment": config.get("environment", {}),
        "kinematic_constraints": config.get("kinematic_constraints", {}),
        "path": config.get("path", {}),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_info(repo_root: Path) -> Dict[str, Optional[object]]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root), text=True).strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"commit": None, "dirty": None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("DISABLE_FINAL_PLOT", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser(description="P0 gold pipeline runner.")
    parser.add_argument("--config", type=str, required=True, help="Base config path")
    parser.add_argument("--seeds", type=str, default="42,43", help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--eval_episodes", type=int, default=50, help="Episodes for p0_eval")
    parser.add_argument("--seed_eval", type=int, default=None, help="Override eval seed for acceptance_suite")
    parser.add_argument("--episode_set", type=str, default=None, help="Episode set label or seed list file")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions for eval/trace")
    parser.add_argument("--archive_root", type=str, default=str(ARCHIVE_ROOT), help="Archive root dir")
    parser.add_argument("--run_id", type=str, default=None, help="Reuse an existing run_id folder")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_checkpoint if available")
    args = parser.parse_args(argv)

    base_config = Path(args.config)
    if not base_config.is_absolute():
        base_config = (PROJECT_ROOT / base_config).resolve()
    seeds = _parse_seeds(args.seeds)
    deterministic = bool(args.deterministic)

    run_id = args.run_id or datetime.now().strftime("P0_gold_%Y%m%d_%H%M%S")
    results: List[SeedResult] = []

    for seed in seeds:
        seed_config = PROJECT_ROOT / "configs" / f"{base_config.stem}_seed{seed}.yaml"
        _write_seed_config(base_config, seed, seed_config)

        run_dir = PROJECT_ROOT / "saved_models" / f"p0_seed{seed}" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            str(MAIN_SCRIPT),
            "--mode",
            "train",
            "--config",
            str(seed_config),
            "--experiment_name",
            f"p0_seed{seed}",
            "--experiment_dir",
            str(run_dir),
        ]
        if args.resume:
            resume_path = run_dir / "checkpoints" / "latest_checkpoint.pth"
            if resume_path.exists():
                train_cmd.extend(["--resume", str(resume_path)])
        _run(train_cmd, cwd=PROJECT_ROOT)

        model_path = run_dir / "checkpoints" / "best_model.pth"
        if not model_path.exists():
            fallback = run_dir / "checkpoints" / "tracking_model_final.pth"
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f"model not found for seed {seed}: {run_dir}")

        eval_out = PROJECT_ROOT / "artifacts" / f"p0_eval_seed{seed}_{run_id}"
        eval_seed = int(args.seed_eval) if args.seed_eval is not None else int(seed)
        eval_cmd = [
            sys.executable,
            str(ACCEPTANCE_SCRIPT),
            "--phase",
            "p0_eval",
            "--config",
            str(seed_config),
            "--model",
            str(model_path),
            "--episodes",
            str(int(args.eval_episodes)),
            "--out",
            str(eval_out),
        ]
        eval_cmd.extend(["--seed", str(eval_seed)])
        if args.episode_set:
            eval_cmd.extend(["--episode_set", str(args.episode_set)])
        if deterministic:
            eval_cmd.append("--deterministic")
        summary_path = eval_out / "summary.json"
        try:
            _run(eval_cmd, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as exc:
            if summary_path.exists():
                print(
                    "[warn] p0_eval exited non-zero (likely failed thresholds). "
                    f"Using existing summary.json at {summary_path}."
                )
            else:
                raise

        summary = _read_summary(summary_path)
        results.append(
            SeedResult(
                seed=seed,
                config_path=seed_config,
                run_dir=run_dir,
                model_path=model_path,
                eval_out_dir=eval_out,
                summary=summary,
                eval_seed=eval_seed,
                episode_set=str(args.episode_set) if args.episode_set else None,
                eval_command=list(eval_cmd),
            )
        )

    best = _select_best(results)

    config_snapshot = best.run_dir / "config.yaml"
    config_for_trace = _load_yaml(config_snapshot if config_snapshot.exists() else best.config_path)
    seed_for_trace = int(config_for_trace.get("seed", config_for_trace.get("experiment", {}).get("seed", 42)))
    _set_seed(seed_for_trace)

    trace_rows = _rollout_trace(config_for_trace, best.model_path, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_for_plot = _build_env(config_for_trace, device=device)

    archive_root = Path(args.archive_root)
    archive_dir = archive_root / run_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = archive_dir / "checkpoint.pth"
    config_out = archive_dir / "config.yaml"
    summary_out = archive_dir / "summary.json"
    trace_out = archive_dir / "trace.csv"
    overlay_out = archive_dir / "overlay.png"
    v_out = archive_dir / "v_t.png"
    e_out = archive_dir / "e_n_t.png"

    summary_path = best.eval_out_dir / "summary.json"
    config_source = config_snapshot if config_snapshot.exists() else best.config_path

    checkpoint_out.write_bytes(best.model_path.read_bytes())
    config_out.write_bytes(config_source.read_bytes())
    summary_out.write_bytes(summary_path.read_bytes())

    _write_trace_csv(trace_rows, trace_out)
    _plot_overlay(trace_rows, env_for_plot, overlay_out)
    _plot_timeseries(trace_rows, key="velocity", ylabel="Velocity", title="v(t)", out_path=v_out)
    _plot_timeseries(trace_rows, key="contour_error", ylabel="Contour Error", title="e_n(t)", out_path=e_out)

    git_info = _git_info(REPO_ROOT)
    env_hash = _hash_env_params(config_for_trace)
    config_hash = _hash_file(config_out)

    eval_command_text = " ".join(best.eval_command)
    manifest = {
        "run_id": run_id,
        "selected_seed": best.seed,
        "seeds": [r.seed for r in results],
        "selection_metrics": {str(r.seed): r.summary for r in results},
        "training": {
            "seed": best.seed,
            "num_episodes": config_for_trace.get("training", {}).get("num_episodes"),
            "max_steps": config_for_trace.get("environment", {}).get("max_steps"),
        },
        "evaluation": {
            "seed": int(best.eval_seed),
            "episodes": int(args.eval_episodes),
            "episode_set": best.episode_set,
            "deterministic": deterministic,
        },
        "eval": {
            "command": eval_command_text,
            "seed_eval": int(best.eval_seed),
            "episodes": int(args.eval_episodes),
            "episode_set": best.episode_set,
            "deterministic": deterministic,
        },
        "rollout_det": {
            "command": "internal: p0_gold_pipeline.rollout_trace",
            "seed_eval": int(seed_for_trace),
            "deterministic": deterministic,
            "episode_id": 0,
        },
        "env_params_hash": env_hash,
        "config_sha256": config_hash,
        "git": git_info,
        "files": {
            "checkpoint": str(checkpoint_out.name),
            "config": str(config_out.name),
            "summary": str(summary_out.name),
            "trace": str(trace_out.name),
            "overlay": str(overlay_out.name),
            "v_t": str(v_out.name),
            "e_n_t": str(e_out.name),
        },
        "source": {
            "model_path": str(best.model_path),
            "config_path": str(config_source),
            "eval_summary_path": str(summary_path),
        },
    }
    if trace_out.exists():
        manifest["rollout_det"]["trace_hash"] = _hash_file(trace_out)
    with (archive_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] P0_gold archived at: {archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
