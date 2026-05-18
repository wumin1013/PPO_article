from __future__ import annotations

import copy
import csv
import json
import math
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
LOCAL_RESULTS_DIR = RESEARCH_ROOT / "_local_results"
WORKSPACE_DIR = LOCAL_RESULTS_DIR / "workspace"
RUNS_DIR = LOCAL_RESULTS_DIR / "runs"
ARCHIVES_DIR = LOCAL_RESULTS_DIR / "archives"
PAPER_RUNS_DIR = LOCAL_RESULTS_DIR / "paper_runs"
LONG_RUNS_DIR = LOCAL_RESULTS_DIR / "long_runs"
RESULTS_TSV = LOCAL_RESULTS_DIR / "results.tsv"
RESULTS_LOCK = LOCAL_RESULTS_DIR / "results.tsv.lock"
LEADERBOARD_JSON = WORKSPACE_DIR / "leaderboard.json"
LEADERBOARD_MD = WORKSPACE_DIR / "leaderboard.md"
CURRENT_BEST_ARCHIVE = ARCHIVES_DIR / "current_best.json"
PROMOTED_ARCHIVE_DIR = ARCHIVES_DIR / "promoted"

MAIN_SCRIPT = PPO_ROOT / "main.py"
ACCEPTANCE_SCRIPT = PPO_ROOT / "tools" / "acceptance_suite.py"
EXPORT_SCRIPT = PPO_ROOT / "tools" / "phase32_export_best_trajectories.py"
DEFAULT_BASE_CONFIG_SOURCE = PPO_ROOT / "configs" / "default.yaml"

CURRENT_BEST_CONFIG = WORKSPACE_DIR / "current_best.yaml"
CURRENT_BEST_STATE = WORKSPACE_DIR / "current_best.json"
BASE_CONFIG_COPY = WORKSPACE_DIR / "base_config.yaml"
LOCAL_ARTIFACT_ENTRY_NAMES = {
    "workspace",
    "runs",
    "archives",
    "paper_runs",
    "long_runs",
    "results.tsv",
    "results.tsv.lock",
}

DEFAULT_PATH_NAMES = ("square", "s_shape", "butterfly", "trapezoid", "circle")
PATH_MAX_STEPS_HINTS: Dict[str, int] = {
    "square": 60000,
    "s_shape": 36000,
    "trapezoid": 36000,
    "circle": 45000,
    "butterfly": 36000,
}
COMPLETION_PROGRESS_THRESHOLD = 0.99
SCORE_PROFILES: Dict[str, Dict[str, float]] = {
    "stage1": {
        "pass_count": 320.0,
        "strict_pass_count": 160.0,
        "pass_rate": 120.0,
        "strict_pass_rate": 75.0,
        "success": 180.0,
        "progress": 110.0,
        "stall": -130.0,
        "mean_error": -26.0,
        "max_error": -12.0,
        "completion_time": -18.0,
        "max_completion_time": -6.0,
    },
    "stage2": {
        "pass_count": 1350.0,
        "strict_pass_count": 520.0,
        "pass_rate": 120.0,
        "strict_pass_rate": 150.0,
        "success": 95.0,
        "progress": 30.0,
        "stall": -80.0,
        "mean_error": -20.0,
        "max_error": -12.0,
        "completion_time": -55.0,
        "max_completion_time": -10.0,
    },
}
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
    "mean_completion_time_seconds",
    "max_completion_time_seconds",
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
        "scale": 100.0,
        "num_points": 1200,
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
        "scale": 28.0,
        "num_points": 720,
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
        "scale": 100.0,
        "num_points": 960,
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
    mean_completion_time_seconds: float
    max_completion_time_seconds: float
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


def resolve_artifact_path(path_like: Any) -> Path:
    """Resolve old in-repo artifact paths after moving run outputs to _local_results."""
    raw = str(path_like or "").strip()
    if not raw:
        return Path(raw)

    path = Path(raw)
    if path.exists():
        return path

    normalized = raw.replace("/", "\\")
    lower = normalized.lower()
    marker = "\\trajectory_autoresearch\\"
    suffix = ""
    marker_index = lower.find(marker)
    if marker_index >= 0:
        suffix = normalized[marker_index + len(marker) :]
    elif lower.startswith("trajectory_autoresearch\\"):
        suffix = normalized[len("trajectory_autoresearch\\") :]

    if suffix:
        parts = Path(suffix).parts
        if parts and str(parts[0]).lower() in LOCAL_ARTIFACT_ENTRY_NAMES:
            candidate = LOCAL_RESULTS_DIR.joinpath(*parts)
            if candidate.exists():
                return candidate

    return path


class _ResultsLock:
    def __init__(self, timeout_seconds: float = 120.0, poll_seconds: float = 0.1) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.fd: Optional[int] = None

    def __enter__(self) -> "_ResultsLock":
        RESULTS_LOCK.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + max(1.0, self.timeout_seconds)
        while True:
            try:
                self.fd = os.open(str(RESULTS_LOCK), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"timed out waiting for results lock: {RESULTS_LOCK}")
                time.sleep(max(0.05, self.poll_seconds))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        try:
            RESULTS_LOCK.unlink()
        except FileNotFoundError:
            pass


def _ensure_results_tsv_unlocked() -> None:
    if RESULTS_TSV.exists() and RESULTS_TSV.stat().st_size > 0:
        return
    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def ensure_results_tsv() -> None:
    with _ResultsLock():
        _ensure_results_tsv_unlocked()


def append_result_row(result: ExperimentResult) -> None:
    with _ResultsLock():
        _ensure_results_tsv_unlocked()
        with RESULTS_TSV.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
            writer.writerow(asdict(result))


def upsert_result_row(result: ExperimentResult) -> None:
    with _ResultsLock():
        _ensure_results_tsv_unlocked()
        rows: list[dict[str, Any]] = []
        replaced = False
        with RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if str(row.get("experiment_id", "")).strip() == str(result.experiment_id):
                    if not replaced:
                        rows.append(asdict(result))
                        replaced = True
                    continue
                rows.append({name: row.get(name, "") for name in RESULTS_HEADER})
        if not replaced:
            rows.append(asdict(result))

        temp_path = RESULTS_TSV.parent / f"{RESULTS_TSV.name}.{os.getpid()}.tmp"
        try:
            with temp_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
                writer.writeheader()
                for row in rows:
                    writer.writerow({name: row.get(name, "") for name in RESULTS_HEADER})
            os.replace(temp_path, RESULTS_TSV)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _parse_bool(raw: Any) -> bool:
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _parse_int(raw: Any, default: int = 0) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _parse_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def infer_evaluation_stage(eval_summary_path: Any) -> str:
    raw = str(eval_summary_path or "").replace("\\", "/").lower()
    if "/stage1/" in raw:
        return "stage1"
    if "/stage2/" in raw:
        return "stage2"
    if raw:
        return "full"
    return "unknown"


def read_results_history() -> list[dict]:
    ensure_results_tsv()
    rows: list[dict] = []
    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiment_id = str(row.get("experiment_id", "")).strip()
            if not experiment_id:
                continue
            rows.append(
                {
                    **row,
                    "experiment_id": experiment_id,
                    "candidate": str(row.get("candidate", "")).strip(),
                    "parent_experiment_id": str(row.get("parent_experiment_id", "")).strip(),
                    "status": str(row.get("status", "")).strip(),
                    "evaluation_stage": infer_evaluation_stage(row.get("eval_summary_path", "")),
                    "keep": _parse_bool(row.get("keep", False)),
                    "score": _parse_float(row.get("score", float("-inf")), float("-inf")),
                    "pass_count": _parse_int(row.get("pass_count", 0), 0),
                    "mean_success_rate": _parse_float(row.get("mean_success_rate", 0.0), 0.0),
                    "mean_progress_final": _parse_float(row.get("mean_progress_final", 0.0), 0.0),
                    "mean_stall_rate": _parse_float(row.get("mean_stall_rate", 1.0), 1.0),
                    "mean_error_ratio": _parse_float(row.get("mean_error_ratio", 999.0), 999.0),
                    "max_error_ratio": _parse_float(row.get("max_error_ratio", 999.0), 999.0),
                    "mean_completion_time_seconds": _parse_float(
                        row.get("mean_completion_time_seconds", 999999.0), 999999.0
                    ),
                    "max_completion_time_seconds": _parse_float(
                        row.get("max_completion_time_seconds", 999999.0), 999999.0
                    ),
                }
            )
    return rows


def summarize_candidate_history(history: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for row in history:
        candidate = str(row.get("candidate", "")).strip()
        if not candidate:
            continue
        item = stats.setdefault(
            candidate,
            {
                "tries": 0,
                "ok_runs": 0,
                "keep_count": 0,
                "failed_runs": 0,
                "best_score": float("-inf"),
                "last_score": float("-inf"),
                "last_status": "",
                "last_experiment_id": "",
                "last_finished_at": "",
                "_rows": [],
            },
        )
        item["tries"] += 1
        if str(row.get("status", "")) == "ok":
            item["ok_runs"] += 1
        else:
            item["failed_runs"] += 1
        if bool(row.get("keep", False)):
            item["keep_count"] += 1

        score = _parse_float(row.get("score", float("-inf")), float("-inf"))
        item["best_score"] = max(float(item["best_score"]), score)
        item["last_score"] = score
        item["last_status"] = str(row.get("status", ""))
        item["last_experiment_id"] = str(row.get("experiment_id", ""))
        item["last_finished_at"] = str(row.get("finished_at", ""))
        item["_rows"].append(dict(row))

    for candidate, item in stats.items():
        rows = list(item.pop("_rows", []))
        recent_non_keep_streak = 0
        for row in reversed(rows):
            if bool(row.get("keep", False)):
                break
            recent_non_keep_streak += 1
        tries = max(1, int(item["tries"]))
        item["candidate"] = candidate
        item["keep_rate"] = float(item["keep_count"]) / tries
        item["ok_rate"] = float(item["ok_runs"]) / tries
        item["recent_non_keep_streak"] = int(recent_non_keep_streak)
    return stats


def refresh_workspace_reports(history: Sequence[Mapping[str, Any]], best_state: Mapping[str, Any]) -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        history,
        key=lambda row: (
            0 if str(row.get("status", "")) == "ok" else 1,
            -float(row.get("score", float("-inf"))),
            str(row.get("finished_at", "")),
        ),
    )

    payload = {
        "updated_at": now_text(),
        "current_best": dict(best_state),
        "candidate_stats": summarize_candidate_history(history),
        "results": [dict(row) for row in ranked],
    }
    write_json(LEADERBOARD_JSON, payload)

    lines = ["# trajectory_autoresearch leaderboard", ""]
    if best_state:
        lines.append(
            "Current best: `{experiment_id}` | candidate=`{candidate}` | score=`{score:.3f}` | pass=`{pass_count}` | mean_time=`{mean_time:.3f}s`".format(
                experiment_id=str(best_state.get("experiment_id", "")),
                candidate=str(best_state.get("candidate", "")),
                score=float(best_state.get("score", float("-inf"))),
                pass_count=int(best_state.get("pass_count", 0)),
                mean_time=float(best_state.get("mean_completion_time_seconds", 999999.0)),
            )
        )
        lines.append("")
    lines.extend(
        [
            "| rank | experiment | candidate | status | keep | score | pass | time(s) | success | progress | stall |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(ranked[:50], start=1):
        lines.append(
            "| {rank} | {experiment} | {candidate} | {status} | {keep} | {score:.3f} | {pass_count} | "
            "{mean_time:.3f} | {success:.4f} | {progress:.4f} | {stall:.4f} |".format(
                rank=idx,
                experiment=str(row.get("experiment_id", "")),
                candidate=str(row.get("candidate", "")),
                status=str(row.get("status", "")),
                keep="Y" if bool(row.get("keep", False)) else "N",
                score=float(row.get("score", float("-inf"))),
                pass_count=int(row.get("pass_count", 0)),
                mean_time=float(row.get("mean_completion_time_seconds", 999999.0)),
                success=float(row.get("mean_success_rate", 0.0)),
                progress=float(row.get("mean_progress_final", 0.0)),
                stall=float(row.get("mean_stall_rate", 1.0)),
            )
        )
    LEADERBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                "mean_completion_time_seconds": 999999.0,
                "max_completion_time_seconds": 999999.0,
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


def recommended_path_max_steps(path_cfg: Mapping[str, Any], default_max_steps: int) -> int:
    base_steps = max(1, int(default_max_steps))
    name_key = str(path_cfg.get("name") or "").strip().lower()
    type_key = str(path_cfg.get("type") or "").strip().lower()
    hinted = PATH_MAX_STEPS_HINTS.get(name_key)
    if hinted is None:
        hinted = PATH_MAX_STEPS_HINTS.get(type_key)
    if hinted is None:
        return base_steps
    return max(base_steps, int(hinted))


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
    environment_cfg = cfg.setdefault("environment", {})
    base_max_steps = int(environment_cfg.get("max_steps", 4000) or 4000)
    selected_max_steps = max(recommended_path_max_steps(path_cfg, base_max_steps) for path_cfg in path_specs)
    environment_cfg["max_steps"] = int(selected_max_steps)

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
    mode: str = "train",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = resolve_python_command(conda_env) + [
        str(MAIN_SCRIPT),
        "--mode",
        str(mode or "train"),
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
    environment_cfg = cfg.setdefault("environment", {})
    base_max_steps = int(environment_cfg.get("max_steps", 4000) or 4000)
    environment_cfg["max_steps"] = int(recommended_path_max_steps(path_cfg, base_max_steps))
    write_yaml(out_path, cfg)
    return out_path


def read_eval_payload(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError(f"invalid summary payload: {summary_path}")
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []
    return {
        "summary": summary,
        "episodes": episodes,
    }


def _normalize_eval_path_result(path_result: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    if not isinstance(path_result, Mapping):
        return {}, []
    summary = path_result.get("summary", path_result)
    if not isinstance(summary, dict):
        summary = {}
    episodes_raw = path_result.get("episodes", [])
    episodes = [dict(item) for item in episodes_raw if isinstance(item, Mapping)]
    return summary, episodes


def _completed_episode(episode: Mapping[str, Any]) -> bool:
    done_reason = str(episode.get("done_reason", "")).strip().lower()
    progress = _parse_float(episode.get("progress_final", 0.0), 0.0)
    return bool(done_reason == "success" or progress >= float(COMPLETION_PROGRESS_THRESHOLD))


def _summary_dt_seconds(summary: Mapping[str, Any]) -> float | None:
    config_raw = str(summary.get("config_path", "")).strip()
    if not config_raw:
        return None
    config_path = Path(config_raw)
    if not config_path.exists():
        return None
    try:
        cfg = load_yaml(config_path)
    except Exception:
        return None
    dt = _parse_float(cfg.get("environment", {}).get("interpolation_period", 0.0), 0.0)
    if not math.isfinite(dt) or dt <= 0.0:
        return None
    return float(dt)


def aggregate_eval_results(path_results: Dict[str, dict], *, score_profile: str = "stage2") -> dict:
    weights = SCORE_PROFILES.get(str(score_profile), SCORE_PROFILES["stage2"])
    if not path_results:
        return {
            "path_results": {},
            "aggregated": {
                "path_count": 0,
                "pass_count": 0,
                "strict_pass_count": 0,
                "mean_success_rate": 0.0,
                "mean_stall_rate": 1.0,
                "mean_progress_final": 0.0,
                "mean_error_ratio": 999.0,
                "max_error_ratio": 999.0,
                "mean_completion_time_seconds": 999999.0,
                "max_completion_time_seconds": 999999.0,
                "pass_rate": 0.0,
                "strict_pass_rate": 0.0,
                "score_profile": str(score_profile),
                "score": float("-inf"),
            },
        }

    success_values: list[float] = []
    stall_values: list[float] = []
    progress_values: list[float] = []
    error_ratios: list[float] = []
    completion_time_values: list[float] = []
    pass_count = 0
    strict_pass_count = 0

    for path_result in path_results.values():
        summary, episodes = _normalize_eval_path_result(path_result)
        success = float(summary.get("success_rate", 0.0))
        progress = float(summary.get("mean_progress_final", 0.0))
        strict_passed = bool(summary.get("passed", False))
        completed = strict_passed or progress >= float(COMPLETION_PROGRESS_THRESHOLD)

        success_values.append(success)
        stall_values.append(float(summary.get("stall_rate", 1.0)))
        progress_values.append(progress)
        pass_count += int(completed)
        strict_pass_count += int(strict_passed)
        max_err = float(summary.get("max_abs_contour_error", 1e9))
        half_eps = max(float(summary.get("half_epsilon", 1e-6)), 1e-6)
        error_ratios.append(max_err / half_eps)

        dt_seconds = _summary_dt_seconds(summary)
        completion_times_for_path: list[float] = []
        if dt_seconds is not None:
            for episode in episodes:
                if not _completed_episode(episode):
                    continue
                steps = _parse_float(episode.get("steps", float("nan")), float("nan"))
                if not math.isfinite(steps) or steps <= 0.0:
                    continue
                completion_times_for_path.append(float(steps) * float(dt_seconds))
        if completion_times_for_path:
            completion_time_values.append(float(mean(completion_times_for_path)))

    mean_success = float(mean(success_values))
    mean_stall = float(mean(stall_values))
    mean_progress = float(mean(progress_values))
    mean_error_ratio = float(mean(error_ratios))
    max_error_ratio = float(max(error_ratios))
    mean_completion_time_seconds = (
        float(mean(completion_time_values)) if completion_time_values else 999999.0
    )
    max_completion_time_seconds = (
        float(max(completion_time_values)) if completion_time_values else 999999.0
    )
    path_count = len(path_results)
    pass_rate = float(pass_count / max(1, path_count))
    strict_pass_rate = float(strict_pass_count / max(1, path_count))
    score = (
        float(weights.get("pass_count", 0.0)) * float(pass_count)
        + float(weights.get("strict_pass_count", 0.0)) * float(strict_pass_count)
        + float(weights.get("pass_rate", 0.0)) * pass_rate
        + float(weights.get("strict_pass_rate", 0.0)) * strict_pass_rate
        + float(weights.get("success", 0.0)) * mean_success
        + float(weights.get("progress", 0.0)) * mean_progress
        + float(weights.get("stall", 0.0)) * mean_stall
        + float(weights.get("mean_error", 0.0)) * mean_error_ratio
        + float(weights.get("max_error", 0.0)) * max_error_ratio
    )
    if completion_time_values:
        score += float(weights.get("completion_time", 0.0)) * mean_completion_time_seconds
        score += float(weights.get("max_completion_time", 0.0)) * max_completion_time_seconds

    return {
        "path_results": path_results,
        "aggregated": {
            "path_count": path_count,
            "pass_count": int(pass_count),
            "strict_pass_count": int(strict_pass_count),
            "pass_rate": pass_rate,
            "strict_pass_rate": strict_pass_rate,
            "mean_success_rate": mean_success,
            "mean_stall_rate": mean_stall,
            "mean_progress_final": mean_progress,
            "mean_error_ratio": mean_error_ratio,
            "max_error_ratio": max_error_ratio,
            "mean_completion_time_seconds": mean_completion_time_seconds,
            "max_completion_time_seconds": max_completion_time_seconds,
            "score_profile": str(score_profile),
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
    score_profile: str = "stage2",
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
        path_results[path_name] = read_eval_payload(eval_out_dir / "summary.json")

    payload = aggregate_eval_results(path_results, score_profile=score_profile)
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
    path = resolve_artifact_path(raw_path)
    return path if path.exists() else None


def copy_tree_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def copy_file_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def archive_promoted_result(result: ExperimentResult) -> Path:
    archive_dir = PROMOTED_ARCHIVE_DIR / result.experiment_id
    archive_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(result.run_dir)
    copy_file_if_exists(Path(result.config_path), archive_dir / "config.yaml")
    copy_file_if_exists(run_dir / "train.log", archive_dir / "train.log")
    copy_file_if_exists(run_dir / "experiment_summary.json", archive_dir / "experiment_summary.json")
    copy_tree_if_exists(run_dir / "evaluation", archive_dir / "evaluation")
    copy_tree_if_exists(run_dir / "best_rollouts", archive_dir / "best_rollouts")
    copy_tree_if_exists(run_dir / "checkpoints", archive_dir / "checkpoints")

    payload = asdict(result)
    payload["archived_at"] = now_text()
    payload["archive_dir"] = str(archive_dir)
    write_json(archive_dir / "result.json", payload)
    write_json(CURRENT_BEST_ARCHIVE, payload)
    return archive_dir
