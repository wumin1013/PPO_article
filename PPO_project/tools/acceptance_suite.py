from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.ppo import PPOContinuous
from src.environment import Env
from src.utils.path_generator import get_path_by_name


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path_str: str, *, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    # 1) relative to CWD
    if path.exists():
        return path.resolve()
    # 2) relative to PPO_project root (tools/..)
    candidate = project_root / path
    return candidate.resolve()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return data


def _build_env(config: dict, *, device: torch.device) -> Tuple[Env, float]:
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
    return env, float(env.half_epsilon)


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _is_finite(x: Any) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False


def _is_finite_array(arr: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(arr, dtype=float)).all())
    except Exception:
        return False


def _classify_done(env: Env, last_info: dict) -> str:
    if bool(getattr(env, "reached_target", False)) or bool(getattr(env, "lap_completed", False)):
        return "success"
    if bool(getattr(env, "_p4_stall_triggered", False)):
        return "stall"
    contour_error = float(last_info.get("contour_error", float("nan")))
    oob_boundary = float(getattr(env, "_oob_half_epsilon", getattr(env, "half_epsilon", float("nan"))))
    if math.isfinite(contour_error) and math.isfinite(oob_boundary) and contour_error > oob_boundary:
        return "oob"
    if int(getattr(env, "current_step", 0)) >= int(getattr(env, "max_steps", 0)):
        return "max_steps"
    return "done"


@dataclass
class SmokeSummary:
    phase: str
    passed: bool
    episodes: int
    mean_return: float
    mean_progress_final: float
    has_non_finite: bool
    thresholds: Dict[str, float]
    timestamp: str
    config_path: str


def run_p0_smoke(config: dict, *, episodes: int) -> Tuple[SmokeSummary, List[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, _half_eps = _build_env(config, device=device)

    returns: List[float] = []
    progress_finals: List[float] = []
    has_non_finite = False
    details: List[dict] = []

    zero_action = np.array([0.0, 0.0], dtype=float)
    for ep in range(int(episodes)):
        obs = env.reset()
        if not _is_finite_array(obs):
            has_non_finite = True

        done = False
        ep_return = 0.0
        info: dict = {}
        max_abs_error = 0.0

        while not done:
            obs, reward, done, info = env.step(zero_action)
            ep_return += float(reward)
            if not _is_finite(reward) or not _is_finite_array(obs):
                has_non_finite = True
            contour_error = float(info.get("contour_error", float("nan")))
            if not math.isfinite(contour_error):
                has_non_finite = True
            else:
                max_abs_error = max(max_abs_error, abs(contour_error))

        final_progress = float(info.get("progress", 0.0))
        if not math.isfinite(final_progress):
            has_non_finite = True
            final_progress = float("nan")

        done_reason = _classify_done(env, info)
        returns.append(float(ep_return))
        progress_finals.append(final_progress)
        details.append(
            {
                "episode": ep,
                "return": float(ep_return),
                "progress_final": float(final_progress),
                "done_reason": done_reason,
                "max_abs_contour_error": float(max_abs_error),
                "steps": int(getattr(env, "current_step", 0)),
            }
        )

    mean_return = float(np.nanmean(np.array(returns, dtype=float))) if returns else float("nan")
    mean_progress_final = (
        float(np.nanmean(np.array(progress_finals, dtype=float))) if progress_finals else float("nan")
    )

    thresholds = {"mean_return_lt": -20.0, "mean_progress_final_lt": 0.02}
    passed = (
        (not has_non_finite)
        and math.isfinite(mean_return)
        and math.isfinite(mean_progress_final)
        and mean_return < thresholds["mean_return_lt"]
        and mean_progress_final < thresholds["mean_progress_final_lt"]
    )

    summary = SmokeSummary(
        phase="p0_smoke",
        passed=bool(passed),
        episodes=int(episodes),
        mean_return=float(mean_return),
        mean_progress_final=float(mean_progress_final),
        has_non_finite=bool(has_non_finite),
        thresholds=thresholds,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        config_path=str(config.get("__resolved_config_path__", "")),
    )
    return summary, details


@dataclass
class EvalSummary:
    phase: str
    passed: bool
    episodes: int
    model_path: str
    deterministic: bool
    success_rate: float
    stall_rate: float
    mean_progress_final: float
    max_abs_contour_error: float
    has_non_finite: bool
    thresholds: Dict[str, float]
    half_epsilon: float
    timestamp: str
    config_path: str


def _take_action(agent: PPOContinuous, state: np.ndarray, *, deterministic: bool) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
    state_tensor = torch.from_numpy(state_arr).to(agent.device)
    with torch.no_grad():
        mu, std = agent.actor(state_tensor)
        if deterministic:
            action = mu
        else:
            action_dist = torch.distributions.Normal(mu, std)
            action = action_dist.sample()
    return action.squeeze(0).detach().cpu().numpy()


def run_p0_eval(
    config: dict,
    *,
    model_path: Path,
    episodes: int,
    deterministic: bool,
) -> Tuple[EvalSummary, List[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, half_epsilon = _build_env(config, device=device)

    ppo_config = config.get("ppo", {})
    if not isinstance(ppo_config, dict):
        raise ValueError("配置缺少 ppo 字段")

    agent = PPOContinuous(
        state_dim=None,
        hidden_dim=int(ppo_config["hidden_dim"]),
        action_dim=None,
        actor_lr=float(ppo_config["actor_lr"]),
        critic_lr=float(ppo_config["critic_lr"]),
        lmbda=float(ppo_config["lmbda"]),
        epochs=int(ppo_config["epochs"]),
        eps=float(ppo_config["eps"]),
        gamma=float(ppo_config["gamma"]),
        ent_coef=float(ppo_config.get("ent_coef", 0.0)),
        device=device,
        observation_space=getattr(env, "observation_space", None),
        action_space=getattr(env, "action_space", None),
    )

    with model_path.open("rb") as f:
        checkpoint = torch.load(f, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.actor.eval()
    agent.critic.eval()

    successes = 0
    stalls = 0
    progress_finals: List[float] = []
    max_abs_error_global = 0.0
    has_non_finite = False
    details: List[dict] = []

    start_wall = time.perf_counter()
    report_every = max(1, int(episodes) // 10)
    for ep in range(int(episodes)):
        obs = env.reset()
        if not _is_finite_array(obs):
            has_non_finite = True

        done = False
        ep_return = 0.0
        info: dict = {}
        max_abs_error = 0.0

        while not done:
            action = _take_action(agent, obs, deterministic=deterministic)
            if getattr(env, "action_space", None) is not None:
                low = env.action_space.low
                high = env.action_space.high
                action = np.clip(action, low, high)
            else:
                action = np.array([np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)], dtype=float)

            obs, reward, done, info = env.step(action)
            ep_return += float(reward)

            if not _is_finite(reward) or not _is_finite_array(obs):
                has_non_finite = True

            contour_error = float(info.get("contour_error", float("nan")))
            if not math.isfinite(contour_error):
                has_non_finite = True
            else:
                max_abs_error = max(max_abs_error, abs(contour_error))

        final_progress = float(info.get("progress", 0.0))
        if not math.isfinite(final_progress):
            has_non_finite = True
            final_progress = float("nan")

        done_reason = _classify_done(env, info)
        success = done_reason == "success"
        stall = done_reason == "stall"
        successes += int(success)
        stalls += int(stall)
        progress_finals.append(final_progress)
        max_abs_error_global = max(max_abs_error_global, max_abs_error)

        details.append(
            {
                "episode": ep,
                "return": float(ep_return),
                "progress_final": float(final_progress),
                "done_reason": done_reason,
                "max_abs_contour_error": float(max_abs_error),
                "steps": int(getattr(env, "current_step", 0)),
            }
        )
        if (ep + 1) % report_every == 0 or ep == 0 or (ep + 1) == int(episodes):
            elapsed = time.perf_counter() - start_wall
            print(
                f"[p0_eval] {ep+1}/{int(episodes)} done_reason={done_reason} "
                f"progress_final={final_progress:.4f} steps={int(getattr(env, 'current_step', 0))} "
                f"elapsed={elapsed:.1f}s"
            )

    success_rate = float(successes / max(1, int(episodes)))
    stall_rate = float(stalls / max(1, int(episodes)))
    mean_progress_final = (
        float(np.nanmean(np.array(progress_finals, dtype=float))) if progress_finals else float("nan")
    )

    thresholds = {
        "success_rate_ge": 0.80,
        "stall_rate_le": 0.05,
        "mean_progress_final_ge": 0.95,
        "max_abs_contour_error_le": float(half_epsilon),
    }
    passed = (
        (not has_non_finite)
        and success_rate >= thresholds["success_rate_ge"]
        and stall_rate <= thresholds["stall_rate_le"]
        and math.isfinite(mean_progress_final)
        and mean_progress_final >= thresholds["mean_progress_final_ge"]
        and max_abs_error_global <= thresholds["max_abs_contour_error_le"]
    )

    summary = EvalSummary(
        phase="p0_eval",
        passed=bool(passed),
        episodes=int(episodes),
        model_path=str(model_path),
        deterministic=bool(deterministic),
        success_rate=float(success_rate),
        stall_rate=float(stall_rate),
        mean_progress_final=float(mean_progress_final),
        max_abs_contour_error=float(max_abs_error_global),
        has_non_finite=bool(has_non_finite),
        thresholds=thresholds,
        half_epsilon=float(half_epsilon),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        config_path=str(config.get("__resolved_config_path__", "")),
    )
    return summary, details


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unified acceptance suite for P0/P1/P2 workflows.")
    parser.add_argument("--phase", type=str, required=True, help="p0_smoke | p0_eval (P1/P2 phases may be added later)")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path (required for *_eval)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--out", type=str, required=True, help="Output directory for artifacts (summary.json)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mu) for eval")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    resolved_config_path = _resolve_path(args.config, project_root=project_root)
    config = _load_yaml(resolved_config_path)
    config["__resolved_config_path__"] = str(resolved_config_path)

    seed = int(config.get("seed", config.get("experiment", {}).get("seed", 42)))
    _set_seed(seed)

    out_dir = _resolve_path(args.out, project_root=project_root)
    _ensure_out_dir(out_dir)

    phase = str(args.phase).strip()
    if phase == "p0_smoke":
        summary, details = run_p0_smoke(config, episodes=args.episodes)
        payload = {"summary": asdict(summary), "episodes": details}
        _write_json(out_dir / "summary.json", payload)
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0 if summary.passed else 1

    if phase == "p0_eval":
        if not args.model:
            raise SystemExit("--model is required for p0_eval")
        resolved_model_path = _resolve_path(args.model, project_root=project_root)
        if not resolved_model_path.exists():
            raise SystemExit(f"model file not found: {resolved_model_path}")
        summary, details = run_p0_eval(
            config,
            model_path=resolved_model_path,
            episodes=args.episodes,
            deterministic=args.deterministic,
        )
        payload = {"summary": asdict(summary), "episodes": details}
        _write_json(out_dir / "summary.json", payload)
        print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
        return 0 if summary.passed else 1

    raise SystemExit(f"unknown phase: {phase}")


if __name__ == "__main__":
    raise SystemExit(main())
