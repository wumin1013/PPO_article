"""
Checkpoint helpers for dual-track saving and resume.

- latest_checkpoint.pth: full training state for resume (model + optimizer + schedulers + counters)
- best_model.pth: lightweight weights for paper/inference (model_state_dict only)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class ResumeState:
    """Metadata restored from a latest checkpoint."""

    episode: int
    global_step: int
    best_eval_reward: float
    last_smoothed_reward: Optional[float]
    experiment_dir: Optional[Path]
    obs_normalizer_stats: Optional[Dict[str, Any]]


class CheckpointManager:
    """Manage saving/loading dual checkpoints in a single experiment directory."""

    def __init__(
        self,
        models_dir: Path | str,
        latest_filename: str = "latest_checkpoint.pth",
        best_filename: str = "best_model.pth",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.models_dir / latest_filename
        self.best_path = self.models_dir / best_filename

    def save_latest(
        self,
        agent: Any,
        episode_idx: int,
        global_step: int,
        best_eval_reward: float,
        config: Dict[str, Any],
        *,
        last_smoothed_reward: Optional[float] = None,
        obs_normalizer_stats: Optional[Dict[str, Any]] = None,
        experiment_dir: Optional[Path] = None,
    ) -> Path:
        """Persist full training state for resume."""
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "actor": _safe_state_dict(getattr(agent, "actor", None)),
            "critic": _safe_state_dict(getattr(agent, "critic", None)),
            "actor_optimizer": _safe_state_dict(getattr(agent, "actor_optimizer", None)),
            "critic_optimizer": _safe_state_dict(getattr(agent, "critic_optimizer", None)),
            "actor_scheduler": _safe_state_dict(getattr(agent, "actor_scheduler", None)),
            "critic_scheduler": _safe_state_dict(getattr(agent, "critic_scheduler", None)),
            "episode": int(episode_idx),
            "global_step": int(global_step),
            "best_eval_reward": float(best_eval_reward),
            "last_smoothed_reward": float(last_smoothed_reward) if last_smoothed_reward is not None else None,
            "obs_normalizer_stats": obs_normalizer_stats,
            "config": config,
            "experiment_dir": str(experiment_dir) if experiment_dir else None,
        }
        # Work around PyTorch Windows path issues with non-ASCII absolute paths by using a file object.
        with self.latest_path.open("wb") as f:
            torch.save(payload, f)
        return self.latest_path

    def save_best(
        self,
        agent: Any,
        eval_reward: float,
        episode_idx: int,
        global_step: int,
        config: Dict[str, Any],
    ) -> Path:
        """Persist lightweight best model weights for paper/inference."""
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "actor": _safe_state_dict(getattr(agent, "actor", None)),
            "critic": _safe_state_dict(getattr(agent, "critic", None)),
            "config": config,
            "eval_reward": float(eval_reward),
            "episode": int(episode_idx),
            "global_step": int(global_step),
        }
        # Work around PyTorch Windows path issues with non-ASCII absolute paths by using a file object.
        with self.best_path.open("wb") as f:
            torch.save(payload, f)
        return self.best_path


def load_for_resume(checkpoint_path: Path | str, agent: Any, device: torch.device) -> ResumeState:
    """Load training state from latest_checkpoint.pth into the agent."""
    checkpoint_path = Path(checkpoint_path)
    # weights_only=False is required to load optimizer/scheduler state_dict safely for trusted checkpoints.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    _load_module(getattr(agent, "actor", None), checkpoint.get("actor"))
    _load_module(getattr(agent, "critic", None), checkpoint.get("critic"))
    _load_module(getattr(agent, "actor_optimizer", None), checkpoint.get("actor_optimizer"))
    _load_module(getattr(agent, "critic_optimizer", None), checkpoint.get("critic_optimizer"))
    _load_module(getattr(agent, "actor_scheduler", None), checkpoint.get("actor_scheduler"))
    _load_module(getattr(agent, "critic_scheduler", None), checkpoint.get("critic_scheduler"))

    experiment_dir = _resolve_experiment_dir(checkpoint, checkpoint_path)

    return ResumeState(
        episode=int(checkpoint.get("episode", -1)),
        global_step=int(checkpoint.get("global_step", 0)),
        best_eval_reward=float(checkpoint.get("best_eval_reward", float("-inf"))),
        last_smoothed_reward=checkpoint.get("last_smoothed_reward"),
        experiment_dir=experiment_dir,
        obs_normalizer_stats=checkpoint.get("obs_normalizer_stats"),
    )


def _safe_state_dict(obj: Any) -> Optional[Dict[str, Any]]:
    return obj.state_dict() if hasattr(obj, "state_dict") and obj is not None else None


def _load_module(target: Any, state: Optional[Dict[str, Any]]) -> None:
    if target is None or state is None or not hasattr(target, "load_state_dict"):
        return
    target.load_state_dict(state)


def _resolve_experiment_dir(checkpoint: Dict[str, Any], checkpoint_path: Path) -> Optional[Path]:
    raw_path = checkpoint.get("experiment_dir")
    if isinstance(raw_path, str) and raw_path:
        return Path(raw_path)
    # Fallback: assume <timestamp>/checkpoints/<file>
    try:
        return checkpoint_path.parent.parent
    except Exception:
        return None


__all__ = ["CheckpointManager", "load_for_resume", "ResumeState"]
