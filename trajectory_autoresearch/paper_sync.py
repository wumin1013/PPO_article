from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection

from prepare import (
    RESEARCH_ROOT,
    build_selected_paths,
    load_current_best_state,
    load_yaml,
    recommended_path_max_steps,
)

PPO_ROOT = RESEARCH_ROOT.parent / "PPO_project"
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))

from src.algorithms.baselines import create_baseline_agent
from src.algorithms.ppo import PPOContinuous
from src.environment import create_env_compatible
from src.utils.geometry import generate_offset_paths
from src.utils.path_generator import get_path_by_name


PAPER_ROOT = RESEARCH_ROOT.parent / "论文项目"
PAPER_GENERATED_DIR = PAPER_ROOT / "generated"
PAPER_FIGURES_DIR = PAPER_ROOT / "figures" / "generated"
PAPER_RUNS_DIR = RESEARCH_ROOT / "paper_runs"
LONG_RUNS_DIR = RESEARCH_ROOT / "long_runs"
RESULTS_TSV = RESEARCH_ROOT / "results.tsv"

MAIN_RESULTS_TEX = PAPER_GENERATED_DIR / "main_results_table.tex"
ABLATION_TEX = PAPER_GENERATED_DIR / "ablation_table.tex"
APPENDIX_TEX = PAPER_GENERATED_DIR / "appendix_autosearch.tex"
SUMMARY_JSON = PAPER_GENERATED_DIR / "paper_bridge_summary.json"
QUAL_FIG = PAPER_FIGURES_DIR / "qualitative_results.png"
SQUARE_CORNER_ZOOM_FIG = PAPER_FIGURES_DIR / "square_corner_zoom.png"
KCM_FIG = PAPER_FIGURES_DIR / "kcm_analysis.png"
JERK_COMPARE_FIG = PAPER_FIGURES_DIR / "jerk_constraint_comparison.png"
FULL_TRACE_SUMMARY_JSON = PAPER_GENERATED_DIR / "full_method_trace_summary.json"
BASELINE_TRACE_SUMMARY_JSON = PAPER_GENERATED_DIR / "baseline_trace_summary.json"
ABL_NO_KCM_TRACE_SUMMARY_JSON = PAPER_GENERATED_DIR / "abl_no_kcm_trace_summary.json"

TRACE_PATH_NAMES = ("square", "circle", "butterfly")
TRACE_SCHEMA_VERSION = 3
TRACE_ROLLOUT_RETRIES = 1

VARIANT_LABELS = {
    "full_method_snapshot": "本文最终方法",
    "baseline_policy": "NNC 基线",
    "abl_fixed_lookahead": "固定前瞻",
    "abl_no_lookahead_obs": "无前瞻观测",
    "abl_no_dual_reward": "无直线/拐角双奖励",
    "abl_no_kcm": "无KCM",
}

PLOT_DPI = 400
PLOT_FACE_COLOR = "#fcfcfe"
GRID_STYLE = dict(linestyle=":", alpha=0.28, linewidth=0.8)
RAW_LINE_ALPHA = 0.24
SMOOTH_LINE_WIDTH = 1.9
TITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 9
PAPER_COMPLETION_PROGRESS_THRESHOLD = 0.90

matplotlib.rcParams.update(
    {
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "font.family": "DejaVu Sans",
        "font.size": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync running experiments into paper-ready tables and figures.")
    parser.add_argument("--once", action="store_true", help="执行一次同步")
    parser.add_argument("--watch", action="store_true", help="持续同步")
    parser.add_argument("--interval-seconds", type=int, default=600, help="watch 模式的同步周期")
    parser.add_argument("--max-iterations", type=int, default=0, help="watch 模式最大轮数；0 表示无限")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_figure_outputs(fig: plt.Figure, png_path: Path, *, dpi: int = PLOT_DPI, facecolor: str | None = None) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=facecolor)
    fig.savefig(svg_path, bbox_inches="tight", facecolor=facecolor)


def _load_csv_rows(path: Path, *, delimiter: str = ",") -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return float(parsed)


def _fmt(value: Any, digits: int = 3, empty: str = "待完成") -> str:
    parsed = _safe_float(value, None)
    if parsed is None:
        return empty
    return f"{parsed:.{digits}f}"


def _smooth_series(values: list[float] | np.ndarray, *, window: int = 31) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 3:
        return arr
    win = int(max(3, min(window, arr.size)))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return arr
    kernel = np.ones((win,), dtype=float) / float(win)
    return np.convolve(arr, kernel, mode="same")


def _corner_segments(x: np.ndarray, cornerness: np.ndarray, *, threshold: float = 0.55) -> list[tuple[float, float]]:
    if x.size == 0 or cornerness.size == 0:
        return []
    mask = np.asarray(cornerness >= threshold, dtype=bool)
    segments: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, active in enumerate(mask):
        if active and start_idx is None:
            start_idx = idx
        if (not active) and start_idx is not None:
            segments.append((float(x[start_idx]), float(x[max(start_idx, idx - 1)])))
            start_idx = None
    if start_idx is not None:
        segments.append((float(x[start_idx]), float(x[-1])))
    return segments


def _decorate_time_axis(ax: plt.Axes, x: np.ndarray, cornerness: np.ndarray) -> None:
    for start, end in _corner_segments(x, cornerness):
        ax.axvspan(start, end, color="#ffd8a8", alpha=0.16, linewidth=0.0)
    ax.grid(True, **GRID_STYLE)


def _latex_escape(text: str) -> str:
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    result = str(text)
    for raw, escaped in mapping.items():
        result = result.replace(raw, escaped)
    return result


def _latex_mono(text: str) -> str:
    return r"\texttt{" + _latex_escape(text) + r"}"


def _list_suite_dirs() -> list[Path]:
    candidates = sorted(PAPER_RUNS_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return [candidate for candidate in candidates if (candidate / "suite_manifest.json").exists()]


def _find_latest_suite() -> Path | None:
    candidates = _list_suite_dirs()
    return candidates[0] if candidates else None


def _find_latest_long_run_status() -> Path | None:
    candidates = sorted(LONG_RUNS_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        status = candidate / "status.json"
        if status.exists():
            return status
    return None


def _load_eval_payload(summary_path: str | Path) -> dict:
    raw_path = str(summary_path).strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if (not path.exists()) or path.is_dir():
        return {}
    return _read_json(path)


def _eval_path_entry(eval_payload: dict, path_name: str) -> dict:
    path_results = eval_payload.get("path_results", {})
    if not isinstance(path_results, dict):
        return {}
    row = path_results.get(path_name)
    return row if isinstance(row, dict) else {}


def _eval_path_summary(eval_payload: dict, path_name: str) -> dict:
    row = _eval_path_entry(eval_payload, path_name)
    summary = row.get("summary", row)
    return summary if isinstance(summary, dict) else {}


def _completed_eval_episode(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    done_reason = str(row.get("done_reason", "")).strip().lower()
    progress = _safe_float(row.get("progress_final"), 0.0) or 0.0
    return bool(done_reason == "success" or progress >= float(PAPER_COMPLETION_PROGRESS_THRESHOLD))


def _path_completion_rate(eval_payload: dict, path_name: str) -> float | None:
    path_entry = _eval_path_entry(eval_payload, path_name)
    episodes = path_entry.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        return None
    completed = sum(1 for row in episodes if _completed_eval_episode(row))
    return float(completed) / float(len(episodes))


def _select_best_training_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    def _key(row: dict) -> tuple:
        return (
            _safe_float(row.get("progress"), -1.0),
            -_safe_float(row.get("rmse_error"), 1e9),
            -_safe_float(row.get("mean_velocity"), -1.0),
            -_safe_float(row.get("steps"), -1.0),
        )

    return max(rows, key=_key)


def _load_training_summary(run_dir: Path) -> dict:
    rows = _load_csv_rows(run_dir / "logs" / "paper_metrics_train_multi_path.csv")
    best_row = _select_best_training_row(rows)
    if best_row is None:
        return {}
    return {
        "episode_idx": int(float(best_row.get("episode_idx", 0))),
        "rmse_error": _safe_float(best_row.get("rmse_error")),
        "mean_jerk": _safe_float(best_row.get("mean_jerk")),
        "roughness_proxy": _safe_float(best_row.get("roughness_proxy")),
        "mean_velocity": _safe_float(best_row.get("mean_velocity")),
        "max_error": _safe_float(best_row.get("max_error")),
        "mean_kcm_intervention": _safe_float(best_row.get("mean_kcm_intervention")),
        "steps": _safe_float(best_row.get("steps")),
        "progress": _safe_float(best_row.get("progress")),
    }


def _load_step_series(run_dir: Path, episode_idx: int) -> dict[str, list[float]]:
    rows = _load_csv_rows(run_dir / "logs" / "step_metrics_train_multi_path.csv")
    series = {
        "env_step": [],
        "velocity": [],
        "contour_error": [],
        "kcm_intervention": [],
        "lookahead_dist_active": [],
        "cornerness": [],
    }
    for row in rows:
        row_episode = _safe_float(row.get("episode_idx"))
        if row_episode is None or int(row_episode) != int(episode_idx):
            continue
        for key in list(series):
            series[key].append(float(_safe_float(row.get(key), 0.0) or 0.0))
    return series


def _build_path_points(path_cfg: dict[str, Any]) -> list[np.ndarray]:
    path_type = str(path_cfg["type"])
    scale = float(path_cfg.get("scale", 10.0))
    num_points = int(path_cfg.get("num_points", 200))
    extra = path_cfg.get(path_type, {})
    if not isinstance(extra, dict):
        extra = {}
    if path_type == "square" and "closed" in path_cfg and "closed" not in extra:
        extra["closed"] = bool(path_cfg.get("closed"))
    return get_path_by_name(path_type, scale=scale, num_points=num_points, **extra)


def _make_env(config: dict[str, Any], path_cfg: dict[str, Any], device: torch.device):
    env_cfg = config["environment"]
    kcm_cfg = config["kinematic_constraints"]
    reward_weights = config.get("reward_weights", {})
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), dict) else {}
    use_obs_normalizer = bool(training_cfg.get("use_obs_normalizer", False))
    path_points = _build_path_points(path_cfg)
    max_steps = recommended_path_max_steps(path_cfg, int(env_cfg.get("max_steps", 4000) or 4000))
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
        max_steps=max_steps,
        lookahead_points=env_cfg.get("lookahead_points", 5),
        lookahead_obs_enabled=env_cfg.get("lookahead_obs_enabled", True),
        lookahead_obs_scales=env_cfg.get("lookahead_obs_scales", [1.0]),
        reward_weights=reward_weights,
        curvature_observation=env_cfg.get("curvature_observation"),
        return_normalized_obs=not use_obs_normalizer,
        disable_kcm=bool(config.get("experiment", {}).get("enable_kcm", True) is False),
    )


def _make_agent(config: dict[str, Any], env, device: torch.device):
    ppo_cfg = config["ppo"]
    experiment_cfg = config.get("experiment", {}) if isinstance(config.get("experiment", {}), dict) else {}
    paper_variant = str(experiment_cfg.get("paper_variant", "")).strip()
    experiment_mode = str(experiment_cfg.get("mode", "")).strip().lower()
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    if (
        paper_variant in {"baseline_policy", "abl_no_kcm"}
        or experiment_mode in {"baseline_nnc", "ablation_no_kcm"}
        or experiment_cfg.get("enable_kcm") is False
    ):
        baseline_cfg = dict(config)
        baseline_cfg["state_dim"] = obs_space.shape[0] if obs_space is not None else env.observation_dim
        baseline_cfg["action_dim"] = act_space.shape[0] if act_space is not None else env.action_space_dim
        baseline_cfg["observation_space"] = obs_space
        baseline_cfg["action_space"] = act_space
        return create_baseline_agent("nnc", baseline_cfg, device)
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


def _take_deterministic_action(agent: PPOContinuous, state: Any) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
    state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        mu, _ = agent.actor(state_tensor)
    return mu.squeeze(0).detach().cpu().numpy().astype(float)


def _take_policy_action(agent: Any, state: Any, *, deterministic: bool) -> np.ndarray:
    if deterministic and hasattr(agent, "actor") and hasattr(agent, "device"):
        return _take_deterministic_action(agent, state)
    action = np.asarray(agent.take_action(state), dtype=float).flatten()
    return action


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path_name",
        "env_step",
        "time_s",
        "progress",
        "reward",
        "x",
        "y",
        "contour_error",
        "velocity",
        "acceleration",
        "jerk",
        "omega",
        "angular_acc",
        "angular_jerk",
        "raw_linear_jerk_demand",
        "raw_angular_jerk_demand",
        "kcm_intervention",
        "disable_kcm",
        "cornerness",
        "lookahead_dist_active",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rollout_trace(config: dict[str, Any], path_cfg: dict[str, Any], model_path: Path, *, retries: int = 1) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    best_payload: dict[str, Any] | None = None

    for attempt_idx in range(max(1, int(retries))):
        env = _make_env(config, path_cfg, device)
        agent = _make_agent(config, env, device)
        agent.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            agent.critic.load_state_dict(ckpt["critic"])
        agent.actor.eval()
        agent.critic.eval()

        dt = float(config.get("environment", {}).get("interpolation_period", 1.0))
        state = env.reset(random_start=False)
        total_reward = 0.0
        done = False
        info: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        threshold_completed = False

        while not done:
            action = _take_policy_action(agent, state, deterministic=False)
            if getattr(env, "action_space", None) is not None:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                action = np.array([np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)], dtype=float)

            next_state, reward, done, info = env.step(action)
            p4_status = info.get("p4_status", {})
            if not isinstance(p4_status, dict):
                p4_status = {}

            env_step = int(info.get("step", len(rows) + 1))
            progress_now = float(info.get("progress", 0.0))
            current_position = np.asarray(getattr(env, "current_position", np.zeros((2,), dtype=float)), dtype=float).reshape(-1)
            rows.append(
                {
                    "path_name": str(path_cfg.get("name") or path_cfg.get("type")),
                    "env_step": env_step,
                    "time_s": env_step * dt,
                    "progress": progress_now,
                    "reward": float(reward),
                    "x": float(current_position[0]) if current_position.size >= 1 else 0.0,
                    "y": float(current_position[1]) if current_position.size >= 2 else 0.0,
                    "contour_error": float(info.get("contour_error", 0.0)),
                    "velocity": float(getattr(env, "velocity", 0.0)),
                    "acceleration": float(getattr(env, "acceleration", 0.0)),
                    "jerk": float(info.get("jerk", getattr(env, "jerk", 0.0))),
                    "omega": float(getattr(env, "angular_vel", 0.0)),
                    "angular_acc": float(getattr(env, "angular_acc", 0.0)),
                    "angular_jerk": float(getattr(env, "angular_jerk", 0.0)),
                    "raw_linear_jerk_demand": float(info.get("raw_linear_jerk_demand", 0.0)),
                    "raw_angular_jerk_demand": float(info.get("raw_angular_jerk_demand", 0.0)),
                    "kcm_intervention": float(info.get("kcm_intervention", 0.0)),
                    "disable_kcm": bool(info.get("disable_kcm", False)),
                    "cornerness": float(info.get("cornerness", 0.0)),
                    "lookahead_dist_active": float(p4_status.get("lookahead_dist_active", getattr(env, "_lookahead_dist_active", 0.0))),
                }
            )
            total_reward += float(reward)
            state = next_state
            if progress_now >= float(PAPER_COMPLETION_PROGRESS_THRESHOLD):
                threshold_completed = True
                done = True

            if env_step > int(getattr(env, "max_steps", config["environment"]["max_steps"])) + 5:
                break

        progress = float(info.get("progress", 0.0))
        max_abs_contour_error = max((abs(float(row["contour_error"])) for row in rows), default=0.0)
        max_abs_linear_jerk = max((abs(float(row["jerk"])) for row in rows), default=0.0)
        max_abs_angular_jerk = max((abs(float(row["angular_jerk"])) for row in rows), default=0.0)
        payload = {
            "path_name": str(path_cfg.get("name") or path_cfg.get("type")),
            "score": progress * 1_000_000.0 + total_reward,
            "reward": float(total_reward),
            "progress": progress,
            "steps": int(len(rows)),
            "done_reason": "success" if threshold_completed else str(info.get("done_reason", "unknown")),
            "csv_rows": rows,
            "linear_jerk_limit": float(getattr(env, "MAX_JERK", 0.0)),
            "angular_jerk_limit": float(getattr(env, "MAX_ANG_JERK", 0.0)),
            "max_abs_linear_jerk": float(max_abs_linear_jerk),
            "max_abs_angular_jerk": float(max_abs_angular_jerk),
            "max_abs_contour_error": float(max_abs_contour_error),
            "half_epsilon": float(getattr(env, "half_epsilon", 0.0)),
            "audit": {
                "disable_kcm": bool(getattr(env, "disable_kcm", False)),
                "lookahead_obs_enabled": bool(getattr(env, "lookahead_obs_enabled", True)),
                "lookahead_control_enabled": bool(getattr(env, "lookahead_control_enabled", False)),
                "lookahead_policy_action": bool(getattr(env, "lookahead_control_policy_action", False)),
                "cornerness_enabled": bool(config.get("reward_weights", {}).get("cornerness", {}).get("enabled", False)),
                "lookahead_reward_enabled": bool(config.get("reward_weights", {}).get("lookahead_reward", {}).get("enabled", False)),
            },
        }
        if best_payload is None or (
            float(payload.get("progress", 0.0)),
            float(payload.get("score", float("-inf"))),
            -float(payload.get("max_abs_contour_error", 1e9)),
            -float(payload.get("steps", 1e9)),
        ) > (
            float(best_payload.get("progress", 0.0)),
            float(best_payload.get("score", float("-inf"))),
            -float(best_payload.get("max_abs_contour_error", 1e9)),
            -float(best_payload.get("steps", 1e9)),
        ):
            best_payload = payload

    return best_payload or {}


def _variant_model_path(variant: dict[str, Any], path_name: str) -> Path | None:
    rollouts_paths = variant.get("rollouts_summary", {}).get("paths", {})
    if isinstance(rollouts_paths, dict):
        item = rollouts_paths.get(path_name)
        if isinstance(item, dict):
            model_raw = str(item.get("model", "")).strip()
            if model_raw:
                model_path = Path(model_raw)
                if model_path.exists():
                    return model_path

    direct_model = str(variant.get("model_path", "")).strip()
    if direct_model:
        direct_model_path = Path(direct_model)
        if direct_model_path.exists():
            return direct_model_path

    state = variant.get("state", {}) if isinstance(variant.get("state", {}), dict) else {}
    state_model = str(state.get("model_path", "")).strip()
    if state_model:
        state_model_path = Path(state_model)
        if state_model_path.exists():
            return state_model_path
    return None


def _variant_signature(variant: dict[str, Any], trace_tag: str) -> dict[str, Any]:
    signature = {
        "trace_tag": trace_tag,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "config_path": str(variant.get("config_path", "")),
        "variant_name": str(variant.get("name", "")),
        "status": str(variant.get("status", "")),
        "models": {},
        "trace_paths": [],
    }
    for path_name in TRACE_PATH_NAMES:
        model_path = _variant_model_path(variant, path_name)
        signature["models"][path_name] = str(model_path) if model_path is not None else ""
    return signature


def _build_variant_trace(variant: dict[str, Any], trace_tag: str, *, path_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    summary_path = PAPER_GENERATED_DIR / f"{trace_tag}_trace_summary.json"
    config_raw = str(variant.get("config_path", "")).strip()
    if not config_raw:
        return {}
    config_path = Path(config_raw)
    if not config_path.exists():
        return {}

    signature = _variant_signature(variant, trace_tag)
    selected_paths = tuple(path_names or TRACE_PATH_NAMES)
    signature["trace_paths"] = list(selected_paths)
    if summary_path.exists():
        cached = _read_json(summary_path)
        if cached.get("signature") == signature:
            return cached

    config = load_yaml(config_path)
    path_specs = build_selected_paths(list(selected_paths))
    candidates: list[dict[str, Any]] = []

    for path_cfg in path_specs:
        path_name = str(path_cfg.get("name") or path_cfg.get("type"))
        model_path = _variant_model_path(variant, path_name)
        if model_path is None:
            continue
        expected_row = _rollout_path_entry(variant.get("rollouts_summary", {}), path_name)
        config_for_trace = copy.deepcopy(config)
        expected_steps = int(_safe_float(expected_row.get("steps"), 0.0) or 0.0)
        if expected_steps > 0:
            env_cfg = config_for_trace.setdefault("environment", {})
            current_max_steps = int(env_cfg.get("max_steps", expected_steps) or expected_steps)
            env_cfg["max_steps"] = min(current_max_steps, int(max(expected_steps + 2000, expected_steps * 1.20)))
        if _rollout_completed(expected_row):
            if trace_tag == "full_method":
                retries = TRACE_ROLLOUT_RETRIES
            elif trace_tag == "abl_no_kcm":
                retries = 2
            else:
                retries = 1
        else:
            retries = 1
        trace_payload = _rollout_trace(config_for_trace, path_cfg, model_path, retries=retries)
        csv_path = PAPER_GENERATED_DIR / f"{trace_tag}_{path_name}_trace.csv"
        _write_trace_csv(csv_path, trace_payload["csv_rows"])
        candidates.append(
            {
                "path_name": path_name,
                "model_path": str(model_path),
                "csv_path": str(csv_path),
                "score": float(trace_payload["score"]),
                "reward": float(trace_payload["reward"]),
                "progress": float(trace_payload["progress"]),
                "steps": int(trace_payload["steps"]),
                "done_reason": str(trace_payload["done_reason"]),
                "linear_jerk_limit": float(trace_payload["linear_jerk_limit"]),
                "angular_jerk_limit": float(trace_payload["angular_jerk_limit"]),
                "max_abs_linear_jerk": float(trace_payload["max_abs_linear_jerk"]),
                "max_abs_angular_jerk": float(trace_payload["max_abs_angular_jerk"]),
                "max_abs_contour_error": float(trace_payload["max_abs_contour_error"]),
                "half_epsilon": float(trace_payload["half_epsilon"]),
                "audit": dict(trace_payload.get("audit", {})),
            }
        )

    if not candidates:
        return {}

    best_candidate = max(
        candidates,
        key=lambda row: (
            float(row.get("score", float("-inf"))),
            float(row.get("progress", -1.0)),
            -float(row.get("max_abs_contour_error", 1e9)),
        ),
    )
    payload = {
        "variant_name": str(variant.get("name", trace_tag)),
        "variant_label": str(variant.get("label", variant.get("name", trace_tag))),
        "signature": signature,
        "selected_path": str(best_candidate.get("path_name", "")),
        "selected_csv_path": str(best_candidate.get("csv_path", "")),
        "selected_steps": int(best_candidate.get("steps", 0)),
        "selected_done_reason": str(best_candidate.get("done_reason", "")),
        "selected_score": float(best_candidate.get("score", float("-inf"))),
        "candidates": candidates,
        "audit": dict(best_candidate.get("audit", {})),
    }
    _write_json(summary_path, payload)
    return payload


def _rows_from_trace_summary(summary: dict[str, Any], path_name: str | None = None) -> list[dict[str, Any]]:
    if not summary:
        return []
    csv_path = ""
    if path_name:
        for candidate in summary.get("candidates", []):
            if str(candidate.get("path_name", "")) == str(path_name):
                csv_path = str(candidate.get("csv_path", ""))
                break
    if not csv_path:
        csv_path = str(summary.get("selected_csv_path", ""))
    if not csv_path:
        return []
    return _load_csv_rows(Path(csv_path))


def _candidate_by_path(summary: dict[str, Any], path_name: str) -> dict[str, Any]:
    for candidate in summary.get("candidates", []):
        if str(candidate.get("path_name", "")) == str(path_name):
            return candidate
    return {}


def _load_cached_trace_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json(path)
    return payload if isinstance(payload, dict) else {}


def _choose_jerk_compare_path(full_trace: dict[str, Any], abl_trace: dict[str, Any]) -> str:
    common_paths = sorted(
        set(str(item.get("path_name", "")) for item in full_trace.get("candidates", []))
        & set(str(item.get("path_name", "")) for item in abl_trace.get("candidates", []))
    )
    if not common_paths:
        return ""

    def _key(path_name: str) -> tuple[float, float, float, float]:
        full_row = _candidate_by_path(full_trace, path_name)
        abl_row = _candidate_by_path(abl_trace, path_name)
        full_limit = max(float(full_row.get("linear_jerk_limit", 0.0)), 1e-6)
        abl_limit = max(float(abl_row.get("linear_jerk_limit", 0.0)), 1e-6)
        full_ratio = float(full_row.get("max_abs_linear_jerk", 0.0)) / full_limit
        abl_ratio = float(abl_row.get("max_abs_linear_jerk", 0.0)) / abl_limit
        return (
            1.0 if full_ratio <= 1.0 else 0.0,
            1.0 if abl_ratio > 1.0 else 0.0,
            abl_ratio - full_ratio,
            float(full_row.get("score", float("-inf"))),
        )

    return max(common_paths, key=_key)


def _max_path_error(eval_payload: dict) -> float | None:
    path_results = eval_payload.get("path_results", {})
    if not isinstance(path_results, dict) or not path_results:
        return None
    values = []
    for path_name in path_results:
        err = _safe_float(_eval_path_summary(eval_payload, str(path_name)).get("max_abs_contour_error"))
        if err is not None:
            values.append(err)
    return max(values) if values else None


def _variant_payload_from_manifest(manifest: dict, *, suite_dir: Path | None = None, suite_mtime: float = 0.0) -> dict:
    run_dir = Path(str(manifest.get("run_dir", "")).strip())
    eval_payload = _load_eval_payload(manifest.get("eval_summary_path", ""))
    rollouts_summary = {}
    rollouts_path_raw = str(manifest.get("rollouts_summary_path", "")).strip()
    rollouts_path = Path(rollouts_path_raw) if rollouts_path_raw else None
    if rollouts_path is not None and rollouts_path.exists() and not rollouts_path.is_dir():
        rollouts_summary = _read_json(rollouts_path)
    training_summary = _load_training_summary(run_dir) if run_dir.exists() else {}
    return {
        "name": str(manifest.get("name", "")),
        "label": str(manifest.get("label", manifest.get("name", ""))),
        "status": str(manifest.get("status", "")),
        "run_dir": str(run_dir),
        "eval_payload": eval_payload,
        "rollouts_summary": rollouts_summary,
        "training_summary": training_summary,
        "config_path": str(manifest.get("config_path", "")),
        "source_experiment_id": str(manifest.get("source_experiment_id", "")),
        "started_at": str(manifest.get("started_at", "")),
        "finished_at": str(manifest.get("finished_at", "")),
        "ablation_audit": dict(manifest.get("ablation_audit", {})) if isinstance(manifest.get("ablation_audit", {}), dict) else {},
        "suite_dir": str(suite_dir or ""),
        "suite_mtime": float(suite_mtime),
    }


def _load_suite_bundle(suite_dir: Path) -> dict:
    suite_manifest = _read_json(suite_dir / "suite_manifest.json")
    variants: dict[str, dict] = {}
    suite_mtime = float(suite_dir.stat().st_mtime)
    for variant_name, manifest in dict(suite_manifest.get("variants", {})).items():
        manifest_path = suite_dir / variant_name / "variant_manifest.json"
        payload = _read_json(manifest_path) if manifest_path.exists() else manifest
        variants[str(variant_name)] = _variant_payload_from_manifest(payload, suite_dir=suite_dir, suite_mtime=suite_mtime)
    return {"suite_dir": str(suite_dir), "suite_manifest": suite_manifest, "variants": variants, "suite_mtime": suite_mtime}


def _preferred_variant(existing: dict, candidate: dict) -> dict:
    if not existing:
        return candidate
    existing_completed = str(existing.get("status", "")).lower() == "completed"
    candidate_completed = str(candidate.get("status", "")).lower() == "completed"
    if candidate_completed and not existing_completed:
        return candidate
    if existing_completed and not candidate_completed:
        return existing
    existing_time = str(existing.get("finished_at") or existing.get("started_at") or "")
    candidate_time = str(candidate.get("finished_at") or candidate.get("started_at") or "")
    if candidate_time and existing_time:
        if candidate_time >= existing_time:
            return candidate
        return existing
    if float(candidate.get("suite_mtime", 0.0)) >= float(existing.get("suite_mtime", 0.0)):
        return candidate
    return existing


def _load_latest_suite_bundle() -> dict:
    suite_dirs = _list_suite_dirs()
    if not suite_dirs:
        return {"suite_dir": "", "suite_dirs": [], "suite_manifest": {}, "variants": {}}
    bundles = [_load_suite_bundle(suite_dir) for suite_dir in suite_dirs]
    variants: dict[str, dict] = {}
    for bundle in bundles:
        for variant_name, payload in bundle.get("variants", {}).items():
            key = str(variant_name)
            variants[key] = _preferred_variant(variants.get(key, {}), payload)
    primary_bundle = bundles[0]
    return {
        "suite_dir": str(primary_bundle.get("suite_dir", "")),
        "suite_dirs": [str(bundle.get("suite_dir", "")) for bundle in bundles],
        "suite_manifest": dict(primary_bundle.get("suite_manifest", {})),
        "variants": variants,
    }


def _load_current_best_variant() -> dict:
    state = load_current_best_state()
    if not state:
        return {}
    eval_payload = _load_eval_payload(state.get("eval_summary_path", ""))
    rollouts_summary = {}
    rollouts_path_raw = str(state.get("rollouts_summary_path", "")).strip()
    rollouts_path = Path(rollouts_path_raw) if rollouts_path_raw else None
    if rollouts_path is not None and rollouts_path.exists() and not rollouts_path.is_dir():
        rollouts_summary = _read_json(rollouts_path)
    run_dir = Path(str(state.get("run_dir", "")).strip())
    training_summary = _load_training_summary(run_dir) if run_dir.exists() else {}
    return {
        "name": "current_best_search",
        "label": "当前搜索最优",
        "status": str(state.get("status", "")),
        "run_dir": str(run_dir),
        "eval_payload": eval_payload,
        "rollouts_summary": rollouts_summary,
        "training_summary": training_summary,
        "config_path": str(state.get("config_path", "")),
        "model_path": str(state.get("model_path", "")),
        "latest_checkpoint": str(state.get("latest_checkpoint", "")),
        "source_experiment_id": str(state.get("experiment_id", "")),
        "state": state,
    }


def _path_metric(eval_payload: dict, path_name: str, key: str) -> float | None:
    return _safe_float(_eval_path_summary(eval_payload, path_name).get(key))


def _jerk_overlimit(trace_summary: dict[str, Any], path_name: str) -> float | None:
    row = _candidate_by_path(trace_summary, path_name)
    if not row:
        return None
    max_abs = _safe_float(row.get("max_abs_linear_jerk"))
    limit = _safe_float(row.get("linear_jerk_limit"))
    if max_abs is None or limit is None:
        return None
    return max(0.0, float(max_abs) / max(float(limit), 1e-6) - 1.0)


def _display_jerk_overlimit(variant: dict, trace_summary: dict[str, Any], path_name: str) -> float | None:
    value = _jerk_overlimit(trace_summary, path_name)
    if value is not None:
        return value
    audit = variant.get("ablation_audit", {})
    if isinstance(audit, dict) and audit.get("enable_kcm") is True:
        return 0.0
    return None


def _rollout_path_entry(rollouts_summary: dict, path_name: str) -> dict:
    paths = rollouts_summary.get("paths", {})
    if not isinstance(paths, dict):
        return {}
    row = paths.get(path_name)
    return row if isinstance(row, dict) else {}


def _rollout_completed(row: dict) -> bool:
    if not isinstance(row, dict):
        return False
    progress = _safe_float(row.get("progress"), 0.0) or 0.0
    done_reason = str(row.get("done_reason", "")).strip().lower()
    return bool(progress >= float(PAPER_COMPLETION_PROGRESS_THRESHOLD) or done_reason == "success")


def _variant_dt_seconds(variant: dict) -> float | None:
    config_raw = str(variant.get("config_path", "")).strip()
    if not config_raw:
        return None
    config_path = Path(config_raw)
    if not config_path.exists():
        return None
    cfg = load_yaml(config_path)
    return _safe_float(cfg.get("environment", {}).get("interpolation_period"))


def _completion_time_seconds_from_eval(variant: dict, path_name: str) -> float | None:
    eval_payload = variant.get("eval_payload", {})
    if not isinstance(eval_payload, dict):
        return None
    path_entry = _eval_path_entry(eval_payload, path_name)
    if not path_entry:
        return None
    episodes = path_entry.get("episodes", [])
    if not isinstance(episodes, list):
        return None
    dt = _variant_dt_seconds(variant)
    if dt is None:
        return None
    values = []
    for episode in episodes:
        if not _completed_eval_episode(episode):
            continue
        steps = _safe_float(episode.get("steps"))
        if steps is None or steps <= 0.0:
            continue
        values.append(float(steps) * float(dt))
    return float(np.mean(values)) if values else None


def _completion_time_seconds_from_trace_summary(
    summary: dict[str, Any],
    variant: dict,
    path_name: str,
) -> float | None:
    candidate = _candidate_by_path(summary, path_name)
    if not candidate:
        return None
    steps = _safe_float(candidate.get("steps"))
    dt = _variant_dt_seconds(variant)
    if steps is None or dt is None or steps <= 0.0:
        return None
    return float(steps) * float(dt)


def _completion_time_seconds(
    variant: dict,
    path_name: str,
    trace_summary: dict[str, Any] | None = None,
) -> float | None:
    if trace_summary:
        traced = _completion_time_seconds_from_trace_summary(trace_summary, variant, path_name)
        if traced is not None:
            return traced
    row = _rollout_path_entry(variant.get("rollouts_summary", {}), path_name)
    if not _rollout_completed(row):
        return _completion_time_seconds_from_eval(variant, path_name)
    steps = _safe_float(row.get("steps"))
    dt = _variant_dt_seconds(variant)
    if steps is None or dt is None:
        return _completion_time_seconds_from_eval(variant, path_name)
    return float(steps) * float(dt)


def _mean_completion_seconds(
    variant: dict,
    trace_summary: dict[str, Any] | None = None,
) -> float | None:
    values = []
    for path_name in TRACE_PATH_NAMES:
        value = _completion_time_seconds(variant, path_name, trace_summary=trace_summary)
        if value is not None:
            values.append(value)
    if values:
        return float(np.mean(values))
    eval_payload = variant.get("eval_payload", {})
    if isinstance(eval_payload, dict):
        aggregated = eval_payload.get("aggregated", {})
        if isinstance(aggregated, dict):
            value = _safe_float(aggregated.get("mean_completion_time_seconds"))
            if value is not None and value < 999998.0:
                return value
    return None


def _completed_path_count(variant: dict) -> int:
    eval_payload = variant.get("eval_payload", {})
    if isinstance(eval_payload, dict):
        path_results = eval_payload.get("path_results", {})
        if isinstance(path_results, dict) and path_results:
            count = 0
            for path_name, row in path_results.items():
                summary = _eval_path_summary(eval_payload, path_name)
                progress = _safe_float(summary.get("mean_progress_final"), 0.0) or 0.0
                done_like = bool(progress >= float(PAPER_COMPLETION_PROGRESS_THRESHOLD))
                if bool(summary.get("passed", False)) or done_like:
                    count += 1
            return count
    count = 0
    for path_name in TRACE_PATH_NAMES:
        if _rollout_completed(_rollout_path_entry(variant.get("rollouts_summary", {}), path_name)):
            count += 1
    return count


def _completed_variant_or_empty(payload: dict) -> dict:
    if str(payload.get("status", "")).lower() == "completed":
        return payload
    return {}


def _select_main_full_variant(suite_variants: dict[str, dict], current_best_variant: dict) -> tuple[dict, str]:
    current_best_state = current_best_variant.get("state", {}) if isinstance(current_best_variant, dict) else {}
    current_best_experiment_id = str(current_best_state.get("experiment_id", "")).strip()
    suite_full = _completed_variant_or_empty(suite_variants.get("full_method_snapshot", {}))
    suite_source_experiment_id = str(suite_full.get("source_experiment_id", "")).strip()

    if not suite_full:
        return current_best_variant, "current_best"
    if current_best_experiment_id and suite_source_experiment_id and suite_source_experiment_id != current_best_experiment_id:
        return current_best_variant, "current_best"
    return suite_full, "paper_suite"


def _build_main_results_tex(
    full_method: dict,
    baseline: dict,
    full_trace_summary: dict[str, Any],
    baseline_trace_summary: dict[str, Any],
) -> str:
    targets = [("square", "square"), ("circle", "circle"), ("butterfly", "butterfly")]
    baseline_label = _latex_escape(str(baseline.get("label", "NNC 基线")) or "NNC 基线")
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{本文最终方法与基线策略在代表性路径上的对比结果。}",
        r"\label{tab:main_results}",
        r"\resizebox{0.92\textwidth}{!}{",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        rf"\textbf{{路径}} & \textbf{{指标}} & \textbf{{本文最终方法}} & \textbf{{{baseline_label}}}\\",
        r"\midrule",
    ]
    for idx, (path_key, path_label) in enumerate(targets):
        if idx > 0:
            lines.append(r"\midrule")
        lines.append(
            rf"\multirow{{5}}{{*}}{{{_latex_escape(path_label)}}}"
            rf" & 完成率 & {_fmt(_path_completion_rate(full_method.get('eval_payload', {}), path_key))}"
            rf" & {_fmt(_path_completion_rate(baseline.get('eval_payload', {}), path_key))}\\"
        )
        lines.append(
            rf"& 最终进度 & {_fmt(_path_metric(full_method.get('eval_payload', {}), path_key, 'mean_progress_final'))}"
            rf" & {_fmt(_path_metric(baseline.get('eval_payload', {}), path_key, 'mean_progress_final'))}\\"
        )
        lines.append(
            rf"& 最大轮廓误差 & {_fmt(_path_metric(full_method.get('eval_payload', {}), path_key, 'max_abs_contour_error'))}"
            rf" & {_fmt(_path_metric(baseline.get('eval_payload', {}), path_key, 'max_abs_contour_error'))}\\"
        )
        lines.append(
            rf"& 线捷度最大相对超限 & {_fmt(_display_jerk_overlimit(full_method, full_trace_summary, path_key))}"
            rf" & {_fmt(_display_jerk_overlimit(baseline, baseline_trace_summary, path_key))}\\"
        )
        lines.append(
            rf"& 完成时间 (s) & {_fmt(_completion_time_seconds(full_method, path_key, trace_summary=full_trace_summary))}"
            rf" & {_fmt(_completion_time_seconds(baseline, path_key, trace_summary=baseline_trace_summary))}\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _build_ablation_tex(rows: list[dict], trace_summaries: dict[str, dict[str, Any]] | None = None) -> str:
    trace_summaries = trace_summaries or {}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{结构化消融结果汇总。}",
        r"\label{tab:ablation}",
        r"\resizebox{0.98\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{模型配置} & \textbf{完成路径数} & \textbf{均值末进度} & \textbf{全局最大轮廓误差} & \textbf{平均完成时间 (s)} & \textbf{最佳回合平均KCM干预度}\\",
        r"\midrule",
    ]
    for row in rows:
        eval_payload = row.get("eval_payload", {})
        aggregated = eval_payload.get("aggregated", {}) if isinstance(eval_payload, dict) else {}
        kcm_metric = row.get("training_summary", {}).get("mean_kcm_intervention")
        if row.get("name") == "abl_no_kcm":
            kcm_text = "-"
        else:
            kcm_text = _fmt(kcm_metric)
        mean_completion = _mean_completion_seconds(
            row,
            trace_summary=trace_summaries.get(str(row.get("name", "")), {}),
        )
        lines.append(
            rf"{_latex_escape(str(row.get('label', row.get('name', ''))))}"
            rf" & {_fmt(_completed_path_count(row), 0)}"
            rf" & {_fmt(aggregated.get('mean_progress_final'))}"
            rf" & {_fmt(_max_path_error(eval_payload))}"
            rf" & {_fmt(mean_completion)}"
            rf" & {kcm_text}\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_stats() -> tuple[int, int, list[dict]]:
    rows = _load_csv_rows(RESULTS_TSV, delimiter="\t")
    keep_rows = []
    for row in rows:
        keep = str(row.get("keep", "")).strip().lower() in {"true", "1", "yes"}
        if keep:
            keep_rows.append(row)
    keep_rows.sort(key=lambda row: _safe_float(row.get("score"), float("-inf")) or float("-inf"), reverse=True)
    return len(rows), len(keep_rows), keep_rows[:5]


def _build_appendix_tex(current_best: dict, suite_variants: dict[str, dict]) -> str:
    total_runs, promoted_runs, _ = _candidate_stats()
    best_state = current_best.get("state", {})

    lines = [
        f"截至 {_latex_escape(time.strftime('%Y-%m-%d %H:%M:%S'))}，autoresearch 共记录 {total_runs} 个实验，"
        f"其中共有 {promoted_runs} 个配置被晋升为阶段最优。当前搜索最优配置为 "
        f"{_latex_mono(str(best_state.get('experiment_id', '待更新')))}，"
        f"候选族为 {_latex_mono(str(best_state.get('candidate', '待更新')))}，"
        f"综合得分为 {_latex_mono(_fmt(best_state.get('score')))}。",
        "",
        r"\subsection{搜索空间}",
        "当前 autoresearch 并不直接修改 PPO 主体结构，而是在固定评测协议下围绕有限且可解释的候选配置族进行探索。搜索空间主要包括：进度/时间压强类、转角释放与平滑权衡类、学习型前瞻恢复类以及训练动态调节类。各候选仅对已有奖励项、前瞻控制项和 PPO 超参数做小范围增量调整，以保证搜索过程可复现且便于分析。",
        "",
        r"\subsection{选择准则}",
        "当前 autoresearch 流程采用两阶段筛选。第一阶段使用较低成本的路径子集与 episode 数进行粗筛，重点关注完成路径数、成功率与稳定性；第二阶段仅对入围候选进行全路径复评估，并将平均完成时间显式纳入综合得分。候选的晋升规则为：先比较完成路径数，再比较包含完成时间项的综合得分；若得分接近，则继续比较平均完成时间、成功率、平均进度与停滞率。",
        "",
    ]

    if suite_variants:
        completed = [
            _latex_escape(VARIANT_LABELS.get(name, name))
            for name, payload in suite_variants.items()
            if str(payload.get("status", "")).lower() == "completed"
        ]
        if completed:
            lines.append("论文专用复现实验已完成的配置包括：" + "、".join(completed) + "。")
    else:
        lines.append("论文专用复现实验结果尚未生成。")

    lines.append("")
    return "\n".join(lines)


def _placeholder_png(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.38, body, ha="center", va="center", fontsize=12, wrap=True)
    fig.tight_layout()
    _save_figure_outputs(fig, path, dpi=160)
    plt.close(fig)


def _read_rollout_png(rollouts_summary: dict, path_name: str) -> Path | None:
    paths = rollouts_summary.get("paths", {})
    if not isinstance(paths, dict):
        return None
    row = paths.get(path_name)
    if not isinstance(row, dict):
        return None
    png = Path(str(row.get("png", "")).strip())
    return png if png.exists() else None


def _rollout_path_csv(variant: dict[str, Any], path_name: str) -> Path | None:
    row = _rollout_path_entry(variant.get("rollouts_summary", {}), path_name)
    csv_raw = str(row.get("csv", "")).strip()
    if not csv_raw:
        return None
    csv_path = Path(csv_raw)
    return csv_path if csv_path.exists() else None


def _variant_config(variant: dict[str, Any]) -> dict[str, Any]:
    config_raw = str(variant.get("config_path", "")).strip()
    if not config_raw:
        return {}
    config_path = Path(config_raw)
    if not config_path.exists():
        return {}
    return load_yaml(config_path)


def _variant_path_cfg(variant: dict[str, Any], path_name: str) -> dict[str, Any]:
    config = _variant_config(variant)
    selected_paths = config.get("path_curriculum", {}).get("selected_paths", [])
    if isinstance(selected_paths, list):
        for item in selected_paths:
            if not isinstance(item, dict):
                continue
            item_name = str(item.get("name") or item.get("type") or "").strip()
            if item_name == path_name:
                return dict(item)
    fallback = build_selected_paths([path_name])
    return dict(fallback[0]) if fallback else {}


def _points_to_array(points: list[Any]) -> np.ndarray:
    coords: list[np.ndarray] = []
    for point in points:
        if point is None:
            continue
        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
            continue
        coords.append(arr[:2])
    if not coords:
        return np.empty((0, 2), dtype=float)
    return np.vstack(coords)


def _reference_geometry(variant: dict[str, Any], path_name: str) -> dict[str, Any]:
    path_cfg = _variant_path_cfg(variant, path_name)
    if not path_cfg:
        return {}
    path_type = str(path_cfg.get("type") or path_name).strip()
    nested_cfg = path_cfg.get(path_type, {})
    kwargs = copy.deepcopy(nested_cfg) if isinstance(nested_cfg, dict) else {}
    if "closed" in path_cfg:
        kwargs.setdefault("closed", bool(path_cfg.get("closed")))
    scale = float(path_cfg.get("scale", 10.0) or 10.0)
    num_points = int(path_cfg.get("num_points", 600) or 600)
    ref_points = get_path_by_name(path_type, scale=scale, num_points=num_points, **kwargs)
    ref_arr = _points_to_array(list(ref_points))
    config = _variant_config(variant)
    epsilon = float(config.get("environment", {}).get("epsilon", 0.0) or 0.0)
    half_eps = max(epsilon / 2.0, 1e-6)
    closed_flag = bool(kwargs.get("closed", path_cfg.get("closed", True)))
    left_path, right_path = generate_offset_paths(ref_arr, half_eps, closed=closed_flag)
    return {
        "center": ref_arr,
        "left": _points_to_array(list(left_path)),
        "right": _points_to_array(list(right_path)),
        "half_epsilon": half_eps,
    }


def _load_rollout_series(variant: dict[str, Any], path_name: str) -> dict[str, Any]:
    csv_path = _rollout_path_csv(variant, path_name)
    row = _rollout_path_entry(variant.get("rollouts_summary", {}), path_name)
    if csv_path is None:
        return {}
    rows = _load_csv_rows(csv_path)
    if not rows:
        return {}
    x = np.asarray([float(_safe_float(item.get("x"), 0.0) or 0.0) for item in rows], dtype=float)
    y = np.asarray([float(_safe_float(item.get("y"), 0.0) or 0.0) for item in rows], dtype=float)
    velocity = np.asarray([float(_safe_float(item.get("velocity"), 0.0) or 0.0) for item in rows], dtype=float)
    step_values = np.asarray([float(_safe_float(item.get("step"), idx) or idx) for idx, item in enumerate(rows)], dtype=float)
    return {
        "csv_path": str(csv_path),
        "x": x,
        "y": y,
        "velocity": velocity,
        "step": step_values,
        "steps": int(_safe_float(row.get("steps"), len(rows)) or len(rows)),
        "display_steps": int(_safe_float(row.get("steps"), len(rows)) or len(rows)),
        "progress": float(_safe_float(row.get("progress"), 0.0) or 0.0),
        "reward": _safe_float(row.get("reward")),
        "done_reason": str(row.get("done_reason", "unknown")),
    }


def _trim_series_for_display(series: dict[str, Any], geometry: dict[str, Any]) -> dict[str, Any]:
    if not series:
        return series
    x = np.asarray(series.get("x", []), dtype=float)
    y = np.asarray(series.get("y", []), dtype=float)
    step = np.asarray(series.get("step", []), dtype=float)
    if x.size < 32 or y.size < 32 or step.size != x.size:
        return series

    center = np.asarray(geometry.get("center", np.empty((0, 2))), dtype=float)
    if center.shape[0] < 8:
        return series
    ref_deltas = np.diff(center, axis=0)
    ref_length = float(np.sum(np.linalg.norm(ref_deltas, axis=1)))
    if ref_length <= 1e-6:
        return series

    half_eps = float(geometry.get("half_epsilon", 0.0) or 0.0)
    close_tol = max(half_eps * 1.5, ref_length * 0.005)
    travel = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.column_stack([x, y]), axis=0), axis=1))])
    start_xy = np.array([x[0], y[0]], dtype=float)
    dist_to_start = np.linalg.norm(np.column_stack([x, y]) - start_xy, axis=1)

    clip_idx = None
    min_travel = ref_length * 0.85
    for idx in range(16, len(x)):
        if travel[idx] >= min_travel and dist_to_start[idx] <= close_tol:
            clip_idx = idx
            break
    if clip_idx is None:
        return series

    clipped = dict(series)
    clipped["x"] = x[: clip_idx + 1]
    clipped["y"] = y[: clip_idx + 1]
    clipped["velocity"] = np.asarray(series.get("velocity", []), dtype=float)[: clip_idx + 1]
    clipped["step"] = step[: clip_idx + 1]
    clipped["display_steps"] = int(step[clip_idx])
    return clipped


def _plot_reference_geometry(ax: plt.Axes, geometry: dict[str, Any]) -> None:
    center = np.asarray(geometry.get("center", np.empty((0, 2))), dtype=float)
    left = np.asarray(geometry.get("left", np.empty((0, 2))), dtype=float)
    right = np.asarray(geometry.get("right", np.empty((0, 2))), dtype=float)
    if left.size:
        ax.plot(left[:, 0], left[:, 1], color="#4c6ef5", linewidth=1.2, linestyle="--", alpha=0.95, label="Pl")
    if right.size:
        ax.plot(right[:, 0], right[:, 1], color="#ff6b6b", linewidth=1.2, linestyle="--", alpha=0.95, label="Pr")
    if center.size:
        ax.plot(center[:, 0], center[:, 1], color="#495057", linewidth=1.0, linestyle="-.", alpha=0.95, label="Pm (Reference)")


def _plot_velocity_rollout(ax: plt.Axes, series: dict[str, Any], *, vmax: float) -> LineCollection | None:
    x = np.asarray(series.get("x", []), dtype=float)
    y = np.asarray(series.get("y", []), dtype=float)
    velocity = np.asarray(series.get("velocity", []), dtype=float)
    if x.size < 2 or y.size < 2:
        return None
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    segment_velocity = 0.5 * (velocity[:-1] + velocity[1:]) if velocity.size >= 2 else np.zeros((segments.shape[0],), dtype=float)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    collection = LineCollection(segments, cmap="turbo", norm=norm)
    collection.set_array(segment_velocity)
    collection.set_linewidth(2.1)
    collection.set_capstyle("round")
    collection.set_joinstyle("round")
    ax.add_collection(collection)
    ax.scatter([x[0]], [y[0]], s=24, color="#2b8a3e", marker="o", zorder=5, label="Ref Start")
    ax.scatter([x[-1]], [y[-1]], s=28, color="#c92a2a", marker="x", zorder=5, label="Trajectory End")
    ax.autoscale_view()
    return collection


def _set_plain_y_ticks(ax: plt.Axes) -> None:
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _jerk_display_limit(y: np.ndarray, y_s: np.ndarray, limit: float) -> float:
    abs_y = np.abs(y) if y.size else np.zeros((0,), dtype=float)
    abs_y_s = np.abs(y_s) if y_s.size else np.zeros((0,), dtype=float)
    q99 = float(np.percentile(abs_y, 99.0)) if abs_y.size else 0.0
    smooth_peak = float(np.max(abs_y_s)) if abs_y_s.size else 0.0
    baseline = max(float(limit) * 1.15, 1.0)
    return max(baseline, q99 * 1.20, smooth_peak * 1.35)


def _choose_qualitative_paths(full_method: dict, baseline: dict) -> list[str]:
    preferred = ["square", "circle", "butterfly"]
    full_paths = full_method.get("rollouts_summary", {}).get("paths", {})
    baseline_paths = baseline.get("rollouts_summary", {}).get("paths", {})

    union_names: set[str] = set()
    if isinstance(full_paths, dict):
        union_names.update(str(name) for name in full_paths.keys())
    if isinstance(baseline_paths, dict):
        union_names.update(str(name) for name in baseline_paths.keys())

    selected = [name for name in preferred if name in union_names]
    if selected:
        return selected
    return preferred


def _first_significant_corner(center: np.ndarray) -> int | None:
    if center.shape[0] < 3:
        return None
    deltas = np.diff(center, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = norms > 1e-8
    dirs = np.zeros_like(deltas)
    dirs[valid] = deltas[valid] / norms[valid][:, None]
    for idx in range(1, dirs.shape[0]):
        if not valid[idx - 1] or not valid[idx]:
            continue
        cos_theta = float(np.clip(np.dot(dirs[idx - 1], dirs[idx]), -1.0, 1.0))
        turn_deg = math.degrees(math.acos(cos_theta))
        if turn_deg >= 25.0:
            return idx
    return None


def _square_corner_zoom_window(geometry: dict[str, Any]) -> tuple[float, float, float, float, np.ndarray] | None:
    center = np.asarray(geometry.get("center", np.empty((0, 2))), dtype=float)
    if center.shape[0] < 3:
        return None
    corner_idx = _first_significant_corner(center)
    if corner_idx is None:
        return None
    corner = center[corner_idx]
    path_span = max(float(np.ptp(center[:, 0])), float(np.ptp(center[:, 1])), 1.0)
    half_eps = float(geometry.get("half_epsilon", 0.0) or 0.0)
    window_half = max(path_span * 0.18, half_eps * 8.0, 1.2)
    return (
        float(corner[0] - window_half),
        float(corner[0] + window_half),
        float(corner[1] - window_half),
        float(corner[1] + window_half),
        corner,
    )


def _build_square_corner_zoom_figure(full_method: dict, baseline: dict) -> None:
    if not full_method and not baseline:
        _placeholder_png(SQUARE_CORNER_ZOOM_FIG, "Square Corner Zoom Pending", "Waiting for full method and baseline square rollouts.")
        return

    geometry_variant = full_method if full_method else baseline
    geometry = _reference_geometry(geometry_variant, "square")
    zoom_window = _square_corner_zoom_window(geometry)
    if not geometry or zoom_window is None:
        _placeholder_png(SQUARE_CORNER_ZOOM_FIG, "Square Corner Zoom Pending", "Reference geometry for the square path is not ready yet.")
        return

    full_series = _load_rollout_series(full_method, "square") if full_method else {}
    baseline_series = _load_rollout_series(baseline, "square") if baseline else {}
    if not full_series and not baseline_series:
        _placeholder_png(SQUARE_CORNER_ZOOM_FIG, "Square Corner Zoom Pending", "Waiting for vector rollout traces from the full method and baseline.")
        return

    x_min, x_max, y_min, y_max, corner = zoom_window
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    panel_specs = [
        ("Full Method", full_series, "#0b7285"),
        ("NNC Baseline", baseline_series, "#d9480f"),
    ]
    for ax, (title, series, traj_color) in zip(np.atleast_1d(axes), panel_specs):
        ax.set_facecolor(PLOT_FACE_COLOR)
        ax.grid(True, **GRID_STYLE)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        _plot_reference_geometry(ax, geometry)
        ax.scatter([corner[0]], [corner[1]], s=30, color="#212529", marker="s", zorder=5, label="Corner")
        if series:
            display_series = _trim_series_for_display(series, geometry)
            x = np.asarray(display_series.get("x", []), dtype=float)
            y = np.asarray(display_series.get("y", []), dtype=float)
            if x.size and y.size:
                ax.plot(x, y, color=traj_color, linewidth=2.8, alpha=0.98, zorder=4, label=title)
        else:
            ax.text(0.5, 0.5, "Pending", ha="center", va="center", fontsize=13, transform=ax.transAxes)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="best", fontsize=8, framealpha=0.92)

    fig.tight_layout(pad=1.0)
    SQUARE_CORNER_ZOOM_FIG.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, SQUARE_CORNER_ZOOM_FIG, dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    plt.close(fig)


def _build_qualitative_figure(full_method: dict, baseline: dict) -> None:
    path_plan = [(name, name) for name in _choose_qualitative_paths(full_method, baseline)]
    panels = []
    for path_key, label in path_plan:
        panels.append((label, path_key, _load_rollout_series(full_method, path_key), _load_rollout_series(baseline, path_key)))

    if not any(item[2] or item[3] for item in panels):
        _placeholder_png(QUAL_FIG, "Qualitative Figure Pending", "Waiting for vector rollout traces from the full method and baseline.")
        return

    fig, axes = plt.subplots(len(panels), 2, figsize=(12.4, 4.7 * len(panels)), dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    axes = np.atleast_2d(axes)
    for row_axes, (label, path_key, full_series, base_series) in zip(axes, panels):
        vmax = max(
            float(np.max(full_series.get("velocity", [0.0]))) if full_series else 0.0,
            float(np.max(base_series.get("velocity", [0.0]))) if base_series else 0.0,
            1.0,
        )
        titles_and_series = [
            (f"{label} | Full Method", full_method, full_series),
            (f"{label} | Baseline", baseline, base_series),
        ]
        for ax, (title, variant, series) in zip(row_axes, titles_and_series):
            ax.set_facecolor(PLOT_FACE_COLOR)
            ax.grid(True, **GRID_STYLE)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            if series:
                geometry = _reference_geometry(variant, path_key)
                display_series = _trim_series_for_display(series, geometry)
                _plot_reference_geometry(ax, geometry)
                collection = _plot_velocity_rollout(ax, display_series, vmax=vmax)
                if collection is not None:
                    colorbar = fig.colorbar(collection, ax=ax, fraction=0.046, pad=0.02)
                    colorbar.set_label("Velocity (mm/s)", fontsize=LABEL_FONT_SIZE)
                    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
                shown_steps = int(display_series.get("display_steps", display_series.get("steps", 0)) or 0)
                total_steps = int(series.get("steps", shown_steps) or shown_steps)
                step_text = str(shown_steps) if shown_steps == total_steps else f"{shown_steps}/{total_steps}"
                ax.set_title(
                    f"{title}\nsteps={step_text} progress={display_series.get('progress', 0.0):.3f} "
                    f"done={display_series.get('done_reason', 'unknown')}",
                    fontsize=11,
                )
                ax.legend(loc="best", fontsize=8, framealpha=0.90)
            else:
                ax.text(0.5, 0.5, "Pending", ha="center", va="center", fontsize=13, transform=ax.transAxes)
                ax.set_title(title, fontsize=11)
    fig.tight_layout(pad=1.0)
    QUAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, QUAL_FIG, dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    plt.close(fig)


def _build_kcm_figure(trace_summary: dict[str, Any]) -> None:
    example_path = "square"
    rows = _rows_from_trace_summary(trace_summary, path_name=example_path)
    if not rows:
        _placeholder_png(KCM_FIG, "Behavior Figure Pending", "Waiting for the full-step deterministic square trace from the full method.")
        return

    selected_meta = _candidate_by_path(trace_summary, example_path)
    selected_steps = int(selected_meta.get("steps", len(rows)) or len(rows))
    done_reason = str(selected_meta.get("done_reason", "unknown"))
    x = np.asarray([float(_safe_float(row.get("env_step"), idx + 1) or (idx + 1)) for idx, row in enumerate(rows)], dtype=float)
    contour_error = np.asarray([float(_safe_float(row.get("contour_error"), 0.0) or 0.0) for row in rows], dtype=float)
    jerk = np.asarray([float(_safe_float(row.get("jerk"), 0.0) or 0.0) for row in rows], dtype=float)
    kcm = np.asarray([float(_safe_float(row.get("kcm_intervention"), 0.0) or 0.0) for row in rows], dtype=float)
    cornerness = np.asarray([float(_safe_float(row.get("cornerness"), 0.0) or 0.0) for row in rows], dtype=float)

    err_s = _smooth_series(contour_error, window=61)
    jerk_s = _smooth_series(jerk, window=61)
    kcm_s = _smooth_series(kcm, window=61)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.2, 8.8),
        dpi=PLOT_DPI,
        sharex=True,
        facecolor=PLOT_FACE_COLOR,
        gridspec_kw={"height_ratios": [1.15, 1.15, 0.95]},
    )
    fig.suptitle(f"Representative behavior trace | path={example_path} | steps={selected_steps} | done={done_reason}", fontsize=14, y=0.995)

    axes[0].plot(x, contour_error, color="#f08c8c", linewidth=0.8, alpha=RAW_LINE_ALPHA)
    axes[0].plot(x, err_s, color="#d62728", linewidth=SMOOTH_LINE_WIDTH, label="contour error")
    half_epsilon = float(selected_meta.get("half_epsilon", 0.0) or 0.0)
    if half_epsilon > 0.0:
        axes[0].axhline(half_epsilon, color="#7048e8", linewidth=1.1, linestyle="--", label=r"$\epsilon/2$")
    axes[0].set_ylabel(r"$e_c$ (mm)")
    axes[0].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    _decorate_time_axis(axes[0], x, cornerness)

    axes[1].plot(x, jerk, color="#f3a683", linewidth=0.8, alpha=RAW_LINE_ALPHA)
    axes[1].plot(x, jerk_s, color="#e8590c", linewidth=SMOOTH_LINE_WIDTH, label="jerk")
    jerk_limit = float(selected_meta.get("linear_jerk_limit", 0.0) or 0.0)
    if jerk_limit > 0.0:
        axes[1].axhspan(-jerk_limit, jerk_limit, color="#d3f9d8", alpha=0.22)
        axes[1].axhline(jerk_limit, color="#2f9e44", linewidth=1.0, linestyle="--", label=r"$+J_{max}$")
        axes[1].axhline(-jerk_limit, color="#2f9e44", linewidth=1.0, linestyle=":", label=r"$-J_{max}$")
    axes[1].set_ylabel(r"$j$ (mm/s$^3$)")
    axes[1].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    _decorate_time_axis(axes[1], x, cornerness)

    axes[2].plot(x, kcm, color="#ffd8a8", linewidth=0.8, alpha=RAW_LINE_ALPHA)
    axes[2].plot(x, kcm_s, color="#f08c00", linewidth=SMOOTH_LINE_WIDTH, label="KCM intervention")
    axes[2].set_ylabel(r"$\eta$ (-)")
    axes[2].set_xlabel("Step")
    axes[2].set_ylim(bottom=min(-0.02, float(np.min(kcm)) - 0.02), top=max(1.02, float(np.max(kcm)) + 0.02))
    axes[2].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    _decorate_time_axis(axes[2], x, cornerness)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.982))
    KCM_FIG.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, KCM_FIG, dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    plt.close(fig)


def _build_jerk_constraint_figure(full_trace: dict[str, Any], abl_no_kcm_trace: dict[str, Any]) -> dict[str, Any]:
    if not full_trace or not abl_no_kcm_trace:
        _placeholder_png(JERK_COMPARE_FIG, "Jerk Comparison Pending", "Waiting for both the full method and no-KCM ablation traces.")
        return {}

    compare_path = "square"

    full_rows = _rows_from_trace_summary(full_trace, path_name=compare_path)
    abl_rows = _rows_from_trace_summary(abl_no_kcm_trace, path_name=compare_path)
    full_meta = _candidate_by_path(full_trace, compare_path)
    abl_meta = _candidate_by_path(abl_no_kcm_trace, compare_path)
    if (not full_rows) or (not abl_rows):
        _placeholder_png(JERK_COMPARE_FIG, "Jerk Comparison Pending", "Trace rows for the comparison path are not ready yet.")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=PLOT_DPI, sharey=False, facecolor=PLOT_FACE_COLOR)
    compare_specs = [
        ("Full method", full_rows, full_meta, "#1f77b4"),
        ("No KCM", abl_rows, abl_meta, "#d62728"),
    ]

    for ax, (title, rows, meta, color) in zip(axes, compare_specs):
        x = np.asarray([float(_safe_float(row.get("env_step"), idx + 1) or (idx + 1)) for idx, row in enumerate(rows)], dtype=float)
        y = np.asarray([float(_safe_float(row.get("jerk"), 0.0) or 0.0) for row in rows], dtype=float)
        y_s = _smooth_series(y, window=61)
        limit = float(meta.get("linear_jerk_limit", 0.0))
        max_abs = float(meta.get("max_abs_linear_jerk", 0.0))
        ratio = max_abs / max(limit, 1e-6)
        ax.axhspan(-limit, limit, color="#d3f9d8", alpha=0.22, label=r"$|j|\leq J_{max}$")
        ax.plot(x, y, color=color, linewidth=0.8, alpha=RAW_LINE_ALPHA)
        ax.plot(x, y_s, color=color, linewidth=SMOOTH_LINE_WIDTH, label=title)
        if limit > 0.0:
            exceed_mask = np.abs(y) > limit
            if np.any(exceed_mask):
                ax.fill_between(x, np.sign(y) * limit, y, where=exceed_mask, color="#ff6b6b", alpha=0.20, interpolate=True)
        ax.axhline(limit, color="#2f9e44", linestyle="--", linewidth=1.0, label=r"$+J_{max}$")
        ax.axhline(-limit, color="#2f9e44", linestyle=":", linewidth=1.0, label=r"$-J_{max}$")
        peak_abs = float(np.max(np.abs(y))) if y.size else 0.0
        display_limit = _jerk_display_limit(y, y_s, limit)
        ax.set_ylim(-display_limit, display_limit)
        ax.set_title(f"{title} | path={compare_path}\npeak|j|/Jmax={ratio:.2f}", fontsize=11)
        ax.set_xlabel("Step")
        ax.grid(True, **GRID_STYLE)
        _set_plain_y_ticks(ax)
        ax.text(
            0.02,
            0.96,
            f"peak|j|={max_abs:.1f} mm/s$^3$\nJmax={limit:.1f} mm/s$^3$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.72, "edgecolor": "#dee2e6"},
        )
        if peak_abs > display_limit * 1.05:
            ax.text(
                0.02,
                0.08,
                "display clipped for readability",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#868e96",
            )
    axes[0].set_ylabel(r"j (mm/s$^3$)")
    axes[0].legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    JERK_COMPARE_FIG.parent.mkdir(parents=True, exist_ok=True)
    _save_figure_outputs(fig, JERK_COMPARE_FIG, dpi=PLOT_DPI, facecolor=PLOT_FACE_COLOR)
    plt.close(fig)
    return {
        "compare_path": compare_path,
        "full_method_ratio": float(full_meta.get("max_abs_linear_jerk", 0.0)) / max(float(full_meta.get("linear_jerk_limit", 0.0)), 1e-6),
        "abl_no_kcm_ratio": float(abl_meta.get("max_abs_linear_jerk", 0.0)) / max(float(abl_meta.get("linear_jerk_limit", 0.0)), 1e-6),
    }


def sync_once() -> dict:
    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    suite_bundle = _load_latest_suite_bundle()
    suite_variants = dict(suite_bundle.get("variants", {}))
    current_best_variant = _load_current_best_variant()
    full_variant, full_variant_source = _select_main_full_variant(suite_variants, current_best_variant)
    baseline_variant = _completed_variant_or_empty(suite_variants.get("baseline_policy", {}))
    abl_no_kcm_variant = _completed_variant_or_empty(suite_variants.get("abl_no_kcm", {}))

    ablation_rows = [full_variant]
    for key in ["abl_fixed_lookahead", "abl_no_kcm", "abl_no_lookahead_obs", "abl_no_dual_reward"]:
        if key in suite_variants:
            ablation_rows.append(suite_variants[key])

    full_trace_summary = _build_variant_trace(full_variant, "full_method")
    baseline_trace_summary = _build_variant_trace(baseline_variant, "baseline") if baseline_variant else {}
    abl_no_kcm_trace_summary = _build_variant_trace(abl_no_kcm_variant, "abl_no_kcm") if abl_no_kcm_variant else {}
    jerk_compare_summary = _build_jerk_constraint_figure(full_trace_summary, abl_no_kcm_trace_summary)

    _write_text(MAIN_RESULTS_TEX, _build_main_results_tex(full_variant, baseline_variant, full_trace_summary, baseline_trace_summary))
    _write_text(
        ABLATION_TEX,
        _build_ablation_tex(
            [row for row in ablation_rows if row],
            trace_summaries={
                "full_method_snapshot": full_trace_summary,
                "abl_no_kcm": abl_no_kcm_trace_summary,
            },
        ),
    )
    _write_text(APPENDIX_TEX, _build_appendix_tex(current_best_variant, suite_variants))
    _build_qualitative_figure(full_variant, baseline_variant)
    _build_square_corner_zoom_figure(full_variant, baseline_variant)
    _build_kcm_figure(full_trace_summary)

    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_suite_dir": str(suite_bundle.get("suite_dir", "")),
        "suite_dirs": list(suite_bundle.get("suite_dirs", [])),
        "latest_long_run_status": str(_find_latest_long_run_status() or ""),
        "main_full_method_source": full_variant_source,
        "variant_audits": {
            key: dict(value.get("ablation_audit", {}))
            for key, value in suite_variants.items()
            if isinstance(value, dict)
        },
        "full_method_trace": full_trace_summary,
        "baseline_trace": baseline_trace_summary,
        "jerk_compare": jerk_compare_summary,
        "files": {
            "main_results_tex": str(MAIN_RESULTS_TEX),
            "ablation_tex": str(ABLATION_TEX),
            "appendix_tex": str(APPENDIX_TEX),
            "qualitative_figure": str(QUAL_FIG),
            "square_corner_zoom_figure": str(SQUARE_CORNER_ZOOM_FIG),
            "kcm_figure": str(KCM_FIG),
            "jerk_compare_figure": str(JERK_COMPARE_FIG),
            "full_trace_summary": str(FULL_TRACE_SUMMARY_JSON),
            "baseline_trace_summary": str(BASELINE_TRACE_SUMMARY_JSON),
            "abl_no_kcm_trace_summary": str(ABL_NO_KCM_TRACE_SUMMARY_JSON),
        },
        "suite_variants": {key: {"label": value.get("label"), "status": value.get("status")} for key, value in suite_variants.items()},
    }
    _write_json(SUMMARY_JSON, summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.watch:
        iteration = 0
        while True:
            sync_once()
            iteration += 1
            if args.max_iterations > 0 and iteration >= int(args.max_iterations):
                break
            time.sleep(max(30, int(args.interval_seconds)))
        return 0

    sync_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
