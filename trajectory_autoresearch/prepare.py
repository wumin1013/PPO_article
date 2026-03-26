from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


RESEARCH_ROOT = Path(__file__).resolve().parent
PPO_ROOT = RESEARCH_ROOT.parent / "PPO_project"
WORKSPACE_DIR = RESEARCH_ROOT / "workspace"
RUNS_DIR = RESEARCH_ROOT / "runs"
ARCHIVES_DIR = RESEARCH_ROOT / "archives"
RESULTS_TSV = RESEARCH_ROOT / "results.tsv"

MAIN_SCRIPT = PPO_ROOT / "main.py"
ACCEPTANCE_SCRIPT = PPO_ROOT / "tools" / "acceptance_suite.py"
EXPORT_SCRIPT = PPO_ROOT / "tools" / "phase32_export_best_trajectories.py"
DEFAULT_BASE_CONFIG_SOURCE = PPO_ROOT / "configs" / "default.yaml"

CURRENT_BEST_CONFIG = WORKSPACE_DIR / "current_best.yaml"
CURRENT_BEST_STATE = WORKSPACE_DIR / "current_best.json"
BASE_CONFIG_COPY = WORKSPACE_DIR / "base_config.yaml"

DEFAULT_PATH_NAMES = ("square", "s_shape", "butterfly", "trapezoid", "circle")
RESULTS_HEADER = [
    "experiment_id",
    "candidate",
    "parent_experiment_id",
    "status",
    "keep",
    "score",
    "pass_count",
    "mean_success_rate",
    "mean_progress_final",
    "mean_stall_rate",
    "mean_error_ratio",
    "max_error_ratio",
    "git_head",
    "description",
    "config_path",
    "run_dir",
    "model_path",
    "latest_checkpoint",
    "eval_summary_path",
    "rollouts_summary_path",
    "started_at",
    "finished_at",
]


PATH_LIBRARY: Dict[str, Dict[str, Any]] = {
    "square": {
        "name": "square",
        "type": "square",
        "closed": True,
        "scale": 22.0,
        "num_points": 260,
        "square": {"start_offset_ratio": 0.20},
    },
    "s_shape": {
        "name": "s_shape",
        "type": "s_shape",
        "scale": 26.0,
        "num_points": 280,
        "s_shape": {"amplitude": 6.0, "periods": 1.5},
    },
    "butterfly": {
        "name": "butterfly",
        "type": "butterfly",
        "scale": 40.0,
        "num_points": 420,
        "butterfly": {
            "style": "academic",
            "wing_ratio": 1.00,
            "long_ratio": 1.45,
            "cross_ratio": 0.10,
            "closed": True,
        },
    },
    "trapezoid": {
        "name": "trapezoid",
        "type": "trapezoid",
        "scale": 24.0,
        "num_points": 260,
        "trapezoid": {
            "top_ratio": 0.45,
            "height_ratio": 0.80,
            "start_offset_ratio": 0.50,
            "closed": True,
        },
    },
    "circle": {
        "name": "circle",
        "type": "circle",
        "scale": 24.0,
        "num_points": 280,
        "circle": {"closed": True},
    },
}


@dataclass
class ExperimentResult:
    experiment_id: str
    candidate: str
    parent_experiment_id: str
    status: str
    keep: bool
    score: float
    pass_count: int
    mean_success_rate: float
    mean_progress_final: float
    mean_stall_rate: float
    mean_error_ratio: float
    max_error_ratio: float
    git_head: str
    description: str
    config_path: str
    run_dir: str
    model_path: str
    latest_checkpoint: str
    eval_summary_path: str
    rollouts_summary_path: str
    started_at: str
    finished_at: str


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid yaml: {path}")
    return data


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, allow_unicode=True, sort_keys=False)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2)


def ensure_results_tsv() -> None:
    if RESULTS_TSV.exists() and RESULTS_TSV.stat().st_size > 0:
        return
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def append_result_row(result: ExperimentResult) -> None:
    ensure_results_tsv()
    with RESULTS_TSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(asdict(result))


def ensure_workspace(base_config_source: Path = DEFAULT_BASE_CONFIG_SOURCE) -> dict:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    ensure_results_tsv()

    base_cfg = load_yaml(base_config_source)
    if not BASE_CONFIG_COPY.exists():
        write_yaml(BASE_CONFIG_COPY, base_cfg)
    if not CURRENT_BEST_CONFIG.exists():
        write_yaml(CURRENT_BEST_CONFIG, base_cfg)
    if not CURRENT_BEST_STATE.exists():
        write_json(
            CURRENT_BEST_STATE,
            {
                "experiment_id": "",
                "score": float("-inf"),
                "pass_count": -1,
                "mean_success_rate": 0.0,
                "mean_progress_final": 0.0,
                "mean_stall_rate": 1.0,
                "mean_error_ratio": 999.0,
                "max_error_ratio": 999.0,
                "latest_checkpoint": "",
                "run_dir": "",
                "config_path": str(CURRENT_BEST_CONFIG),
                "git_head": get_git_head(),
                "updated_at": now_text(),
            },
        )

    return {
        "workspace_dir": str(WORKSPACE_DIR),
        "base_config": str(BASE_CONFIG_COPY),
        "current_best_config": str(CURRENT_BEST_CONFIG),
        "current_best_state": str(CURRENT_BEST_STATE),
        "results_tsv": str(RESULTS_TSV),
    }


def load_current_best_state() -> dict:
    ensure_workspace()
    if not CURRENT_BEST_STATE.exists():
        return {}
    with CURRENT_BEST_STATE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def save_current_best_state(result: ExperimentResult) -> None:
    payload = asdict(result)
    payload["updated_at"] = now_text()
    write_json(CURRENT_BEST_STATE, payload)


def promote_candidate(config_path: Path, result: ExperimentResult) -> None:
    cfg = load_yaml(config_path)
    write_yaml(CURRENT_BEST_CONFIG, cfg)
    save_current_best_state(result)


def get_nested(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def set_nested(cfg: dict, dotted_key: str, value: Any) -> None:
    cursor = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, float(value))))


def mul_nested(
    cfg: dict,
    dotted_key: str,
    factor: float,
    *,
    low: float | None = None,
    high: float | None = None,
) -> None:
    value = get_nested(cfg, dotted_key)
    if value is None:
        return
    new_value = float(value) * float(factor)
    if low is not None:
        new_value = max(float(low), new_value)
    if high is not None:
        new_value = min(float(high), new_value)
    set_nested(cfg, dotted_key, new_value)


def build_selected_paths(path_names: Sequence[str]) -> list[dict]:
    selected: list[dict] = []
    for name in path_names:
        key = str(name).strip()
        spec = PATH_LIBRARY.get(key)
        if spec is None:
            available = ", ".join(sorted(PATH_LIBRARY))
            raise ValueError(f"unknown path '{key}', available: {available}")
        selected.append(copy.deepcopy(spec))
    if not selected:
        raise ValueError("path list is empty")
    return selected


def resolve_python_command(conda_env: str = "PPO") -> list[str]:
    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env.lower() == conda_env.lower():
        return [sys.executable]

    candidates = []
    env_override = os.environ.get("CONDA_EXE", "").strip()
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path(r"D:\Anaconda\Scripts\conda.exe"))

    for candidate in candidates:
        if candidate.exists():
            return [str(candidate), "run", "-n", conda_env, "python"]
    return [sys.executable]


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Optional[Path] = None,
    timeout_seconds: float | None = None,
    check: bool = True,
) -> int:
    if log_path is None:
        completed = subprocess.run(list(cmd), cwd=str(cwd), check=False, timeout=timeout_seconds)
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        return int(completed.returncode)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n\n")
        f.flush()
        completed = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            check=False,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
        )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return int(completed.returncode)


def load_checkpoint_episode(checkpoint_path: Path) -> int:
    import torch

    with checkpoint_path.open("rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    return int(checkpoint.get("episode", -1))


def build_train_config(
    base_config: Mapping[str, Any],
    *,
    experiment_name: str,
    path_specs: Sequence[Mapping[str, Any]],
    total_episodes: int,
    time_budget_seconds: float | None,
    seed: int,
) -> dict:
    cfg = copy.deepcopy(dict(base_config))
    if not isinstance(cfg.get("training"), dict):
        cfg["training"] = {}
    if not isinstance(cfg.get("experiment"), dict):
        cfg["experiment"] = {}

    training_cfg = cfg["training"]
    experiment_cfg = cfg["experiment"]

    training_cfg["num_episodes"] = int(max(1, total_episodes))
    training_cfg["enable_final_visualization"] = False
    training_cfg["enable_latest_trajectory"] = False
    training_cfg["traj_write_interval_steps"] = 0
    training_cfg["step_log_interval_steps"] = int(max(10, int(training_cfg.get("step_log_interval_steps", 10) or 10)))
    training_cfg["time_budget_seconds"] = float(time_budget_seconds) if time_budget_seconds else None
    training_cfg["path_curriculum"] = {
        "enabled": True,
        "mode": "round_robin",
        "episodes_per_path": 2,
        "seed": int(seed),
        "paths": copy.deepcopy(list(path_specs)),
    }

    cfg["seed"] = int(seed)
    cfg["path"] = copy.deepcopy(dict(path_specs[0]))
    experiment_cfg["category"] = "trajectory_autoresearch"
    experiment_cfg["name"] = experiment_name
    return cfg


def train_candidate(
    *,
    config_path: Path,
    run_dir: Path,
    conda_env: str,
    resume_path: Optional[Path],
    timeout_seconds: float | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = resolve_python_command(conda_env) + [
        str(MAIN_SCRIPT),
        "--mode",
        "train",
        "--config",
        str(config_path),
        "--experiment_dir",
        str(run_dir),
    ]
    if resume_path is not None:
        cmd.extend(["--resume", str(resume_path)])
    log_path = run_dir / "train.log"
    run_command(cmd, cwd=PPO_ROOT, log_path=log_path, timeout_seconds=timeout_seconds)
    return log_path


def find_model_checkpoint(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_model.pth",
        run_dir / "checkpoints" / "tracking_model_final.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no model checkpoint found under {run_dir / 'checkpoints'}")


def find_latest_checkpoint(run_dir: Path) -> Path:
    candidate = run_dir / "checkpoints" / "latest_checkpoint.pth"
    if not candidate.exists():
        raise FileNotFoundError(f"latest checkpoint missing: {candidate}")
    return candidate


def build_eval_config(*, trained_config_path: Path, out_path: Path, path_cfg: Mapping[str, Any]) -> Path:
    cfg = load_yaml(trained_config_path)
    cfg["path"] = copy.deepcopy(dict(path_cfg))
    write_yaml(out_path, cfg)
    return out_path


def read_eval_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError(f"invalid summary payload: {summary_path}")
    return summary


def aggregate_eval_results(path_results: Dict[str, dict]) -> dict:
    if not path_results:
        return {
            "path_results": {},
            "aggregated": {
                "path_count": 0,
                "pass_count": 0,
                "mean_success_rate": 0.0,
                "mean_stall_rate": 1.0,
                "mean_progress_final": 0.0,
                "mean_error_ratio": 999.0,
                "max_error_ratio": 999.0,
                "score": float("-inf"),
            },
        }

    success_values: list[float] = []
    stall_values: list[float] = []
    progress_values: list[float] = []
    error_ratios: list[float] = []
    pass_count = 0

    for summary in path_results.values():
        success_values.append(float(summary.get("success_rate", 0.0)))
        stall_values.append(float(summary.get("stall_rate", 1.0)))
        progress_values.append(float(summary.get("mean_progress_final", 0.0)))
        pass_count += int(bool(summary.get("passed", False)))
        max_err = float(summary.get("max_abs_contour_error", 1e9))
        half_eps = max(float(summary.get("half_epsilon", 1e-6)), 1e-6)
        error_ratios.append(max_err / half_eps)

    mean_success = float(mean(success_values))
    mean_stall = float(mean(stall_values))
    mean_progress = float(mean(progress_values))
    mean_error_ratio = float(mean(error_ratios))
    max_error_ratio = float(max(error_ratios))
    score = (
        1000.0 * float(pass_count)
        + 120.0 * mean_success
        + 25.0 * mean_progress
        - 80.0 * mean_stall
        - 20.0 * mean_error_ratio
    )

    return {
        "path_results": path_results,
        "aggregated": {
            "path_count": len(path_results),
            "pass_count": int(pass_count),
            "mean_success_rate": mean_success,
            "mean_stall_rate": mean_stall,
            "mean_progress_final": mean_progress,
            "mean_error_ratio": mean_error_ratio,
            "max_error_ratio": max_error_ratio,
            "score": score,
        },
    }


def evaluate_model_across_paths(
    *,
    trained_config_path: Path,
    model_path: Path,
    out_dir: Path,
    path_specs: Sequence[Mapping[str, Any]],
    episodes: int,
    deterministic: bool,
    seed: int,
    conda_env: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_results: Dict[str, dict] = {}

    for path_cfg in path_specs:
        path_name = str(path_cfg.get("name") or path_cfg.get("type") or f"path_{len(path_results)}")
        eval_cfg_path = build_eval_config(
            trained_config_path=trained_config_path,
            out_path=out_dir / "configs" / f"{path_name}.yaml",
            path_cfg=path_cfg,
        )
        eval_out_dir = out_dir / "eval" / path_name
        log_path = eval_out_dir / "acceptance.log"
        cmd = resolve_python_command(conda_env) + [
            str(ACCEPTANCE_SCRIPT),
            "--phase",
            "p0_eval",
            "--config",
            str(eval_cfg_path),
            "--model",
            str(model_path),
            "--episodes",
            str(int(episodes)),
            "--out",
            str(eval_out_dir),
            "--seed",
            str(int(seed)),
        ]
        if deterministic:
            cmd.append("--deterministic")
        run_command(cmd, cwd=PPO_ROOT, log_path=log_path, check=False)
        path_results[path_name] = read_eval_summary(eval_out_dir / "summary.json")

    payload = aggregate_eval_results(path_results)
    write_json(out_dir / "summary.json", payload)
    return payload


def export_best_rollouts(
    *,
    config_path: Path,
    run_dir: Path,
    out_dir: Path,
    conda_env: str,
) -> Path:
    cmd = resolve_python_command(conda_env) + [
        str(EXPORT_SCRIPT),
        "--config",
        str(config_path),
        "--run_dir",
        str(run_dir),
        "--out",
        str(out_dir),
    ]
    run_command(cmd, cwd=PPO_ROOT, log_path=out_dir / "export.log")
    return out_dir / "summary.json"


def get_git_head() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(RESEARCH_ROOT.parent),
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def next_experiment_id(iteration: int, candidate_name: str) -> str:
    safe_name = str(candidate_name).strip().replace(" ", "_")
    return f"{now_tag()}_{iteration:04d}_{safe_name}"


def latest_checkpoint_from_state(state: Mapping[str, Any]) -> Optional[Path]:
    raw_path = str(state.get("latest_checkpoint", "")).strip()
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def copy_tree_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
