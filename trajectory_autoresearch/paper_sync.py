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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from prepare import RESEARCH_ROOT, build_selected_paths, load_current_best_state, load_yaml

PPO_ROOT = RESEARCH_ROOT.parent / "PPO_project"
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))

from src.algorithms.ppo import PPOContinuous
from src.environment import create_env_compatible
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
KCM_FIG = PAPER_FIGURES_DIR / "kcm_analysis.png"
JERK_COMPARE_FIG = PAPER_FIGURES_DIR / "jerk_constraint_comparison.png"
FULL_TRACE_SUMMARY_JSON = PAPER_GENERATED_DIR / "full_method_trace_summary.json"
ABL_NO_KCM_TRACE_SUMMARY_JSON = PAPER_GENERATED_DIR / "abl_no_kcm_trace_summary.json"

TRACE_PATH_NAMES = ("square", "circle", "butterfly")

VARIANT_LABELS = {
    "full_method_snapshot": "本文最终方法",
    "baseline_policy": "基线策略",
    "abl_fixed_lookahead": "固定前瞻",
    "abl_no_kcm": "无KCM",
    "abl_no_cornerness": "无拐角感知平滑",
}


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


def _latex_escape(text: str) -> str:
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    result = str(text)
    for raw, escaped in mapping.items():
        result = result.replace(raw, escaped)
    return result


def _find_latest_suite() -> Path | None:
    candidates = sorted(PAPER_RUNS_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "suite_manifest.json").exists():
            return candidate
    return None


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


def _make_agent(config: dict[str, Any], env, device: torch.device) -> PPOContinuous:
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


def _take_deterministic_action(agent: PPOContinuous, state: Any) -> np.ndarray:
    state_arr = np.asarray(state, dtype=np.float32).reshape(1, -1)
    state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        mu, _ = agent.actor(state_tensor)
    return mu.squeeze(0).detach().cpu().numpy().astype(float)


def _write_trace_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path_name",
        "env_step",
        "time_s",
        "reward",
        "contour_error",
        "velocity",
        "acceleration",
        "jerk",
        "omega",
        "angular_acc",
        "angular_jerk",
        "kcm_intervention",
        "cornerness",
        "lookahead_dist_active",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rollout_trace(config: dict[str, Any], path_cfg: dict[str, Any], model_path: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = _make_env(config, path_cfg, device)
    agent = _make_agent(config, env, device)

    ckpt = torch.load(model_path, map_location=device)
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

    while not done:
        action = _take_deterministic_action(agent, state)
        if getattr(env, "action_space", None) is not None:
            action = np.clip(action, env.action_space.low, env.action_space.high)
        else:
            action = np.array([np.clip(action[0], -1.0, 1.0), np.clip(action[1], 0.0, 1.0)], dtype=float)

        next_state, reward, done, info = env.step(action)
        p4_status = info.get("p4_status", {})
        if not isinstance(p4_status, dict):
            p4_status = {}

        env_step = int(info.get("step", len(rows) + 1))
        rows.append(
            {
                "path_name": str(path_cfg.get("name") or path_cfg.get("type")),
                "env_step": env_step,
                "time_s": env_step * dt,
                "reward": float(reward),
                "contour_error": float(info.get("contour_error", 0.0)),
                "velocity": float(getattr(env, "velocity", 0.0)),
                "acceleration": float(getattr(env, "acceleration", 0.0)),
                "jerk": float(info.get("jerk", getattr(env, "jerk", 0.0))),
                "omega": float(getattr(env, "angular_vel", 0.0)),
                "angular_acc": float(getattr(env, "angular_acc", 0.0)),
                "angular_jerk": float(getattr(env, "angular_jerk", 0.0)),
                "kcm_intervention": float(info.get("kcm_intervention", 0.0)),
                "cornerness": float(info.get("cornerness", 0.0)),
                "lookahead_dist_active": float(p4_status.get("lookahead_dist_active", getattr(env, "_lookahead_dist_active", 0.0))),
            }
        )
        total_reward += float(reward)
        state = next_state

        if env_step > int(config["environment"]["max_steps"]) + 5:
            break

    progress = float(info.get("progress", 0.0))
    max_abs_contour_error = max((abs(float(row["contour_error"])) for row in rows), default=0.0)
    max_abs_linear_jerk = max((abs(float(row["jerk"])) for row in rows), default=0.0)
    max_abs_angular_jerk = max((abs(float(row["angular_jerk"])) for row in rows), default=0.0)
    return {
        "path_name": str(path_cfg.get("name") or path_cfg.get("type")),
        "score": progress * 1_000_000.0 + total_reward,
        "reward": float(total_reward),
        "progress": progress,
        "steps": int(len(rows)),
        "done_reason": str(info.get("done_reason", "unknown")),
        "csv_rows": rows,
        "linear_jerk_limit": float(getattr(env, "MAX_JERK", 0.0)),
        "angular_jerk_limit": float(getattr(env, "MAX_ANG_JERK", 0.0)),
        "max_abs_linear_jerk": float(max_abs_linear_jerk),
        "max_abs_angular_jerk": float(max_abs_angular_jerk),
        "max_abs_contour_error": float(max_abs_contour_error),
    }


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
        "config_path": str(variant.get("config_path", "")),
        "variant_name": str(variant.get("name", "")),
        "status": str(variant.get("status", "")),
        "models": {},
    }
    for path_name in TRACE_PATH_NAMES:
        model_path = _variant_model_path(variant, path_name)
        signature["models"][path_name] = str(model_path) if model_path is not None else ""
    return signature


def _build_variant_trace(variant: dict[str, Any], trace_tag: str) -> dict[str, Any]:
    summary_path = PAPER_GENERATED_DIR / f"{trace_tag}_trace_summary.json"
    config_raw = str(variant.get("config_path", "")).strip()
    if not config_raw:
        return {}
    config_path = Path(config_raw)
    if not config_path.exists():
        return {}

    signature = _variant_signature(variant, trace_tag)
    if summary_path.exists():
        cached = _read_json(summary_path)
        if cached.get("signature") == signature:
            return cached

    config = load_yaml(config_path)
    path_specs = build_selected_paths(list(TRACE_PATH_NAMES))
    candidates: list[dict[str, Any]] = []

    for path_cfg in path_specs:
        path_name = str(path_cfg.get("name") or path_cfg.get("type"))
        model_path = _variant_model_path(variant, path_name)
        if model_path is None:
            continue
        trace_payload = _rollout_trace(config, path_cfg, model_path)
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
    for row in path_results.values():
        if not isinstance(row, dict):
            continue
        err = _safe_float(row.get("max_abs_contour_error"))
        if err is not None:
            values.append(err)
    return max(values) if values else None


def _variant_payload_from_manifest(manifest: dict) -> dict:
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
    }


def _load_latest_suite_bundle() -> dict:
    suite_dir = _find_latest_suite()
    if suite_dir is None:
        return {"suite_dir": "", "suite_manifest": {}, "variants": {}}
    suite_manifest = _read_json(suite_dir / "suite_manifest.json")
    variants: dict[str, dict] = {}
    for variant_name, manifest in dict(suite_manifest.get("variants", {})).items():
        manifest_path = suite_dir / variant_name / "variant_manifest.json"
        payload = _read_json(manifest_path) if manifest_path.exists() else manifest
        variants[str(variant_name)] = _variant_payload_from_manifest(payload)
    return {"suite_dir": str(suite_dir), "suite_manifest": suite_manifest, "variants": variants}


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
    path_results = eval_payload.get("path_results", {})
    if not isinstance(path_results, dict):
        return None
    row = path_results.get(path_name)
    if not isinstance(row, dict):
        return None
    return _safe_float(row.get(key))


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


def _build_main_results_tex(full_method: dict, baseline: dict) -> str:
    targets = [("square", "square"), ("circle", "circle"), ("butterfly", "butterfly")]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{本文最终方法与基线策略在代表性路径上的对比结果。}",
        r"\label{tab:main_results}",
        r"\resizebox{0.92\textwidth}{!}{",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"\textbf{路径} & \textbf{指标} & \textbf{本文最终方法} & \textbf{基线策略}\\",
        r"\midrule",
    ]
    for idx, (path_key, path_label) in enumerate(targets):
        if idx > 0:
            lines.append(r"\midrule")
        lines.append(
            rf"\multirow{{3}}{{*}}{{{_latex_escape(path_label)}}}"
            rf" & 成功率 & {_fmt(_path_metric(full_method.get('eval_payload', {}), path_key, 'success_rate'))}"
            rf" & {_fmt(_path_metric(baseline.get('eval_payload', {}), path_key, 'success_rate'))}\\"
        )
        lines.append(
            rf"& 最终进度 & {_fmt(_path_metric(full_method.get('eval_payload', {}), path_key, 'mean_progress_final'))}"
            rf" & {_fmt(_path_metric(baseline.get('eval_payload', {}), path_key, 'mean_progress_final'))}\\"
        )
        lines.append(
            rf"& 最大轮廓误差 & {_fmt(_path_metric(full_method.get('eval_payload', {}), path_key, 'max_abs_contour_error'))}"
            rf" & {_fmt(_path_metric(baseline.get('eval_payload', {}), path_key, 'max_abs_contour_error'))}\\"
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


def _build_ablation_tex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{结构化消融结果汇总。}",
        r"\label{tab:ablation}",
        r"\resizebox{0.98\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{模型配置} & \textbf{通过路径数} & \textbf{均值成功率} & \textbf{均值末进度} & \textbf{全局最大轮廓误差} & \textbf{最佳回合平均KCM干预度}\\",
        r"\midrule",
    ]
    for row in rows:
        eval_payload = row.get("eval_payload", {})
        aggregated = eval_payload.get("aggregated", {}) if isinstance(eval_payload, dict) else {}
        lines.append(
            rf"{_latex_escape(str(row.get('label', row.get('name', ''))))}"
            rf" & {_fmt(aggregated.get('pass_count'), 0)}"
            rf" & {_fmt(aggregated.get('mean_success_rate'))}"
            rf" & {_fmt(aggregated.get('mean_progress_final'))}"
            rf" & {_fmt(_max_path_error(eval_payload))}"
            rf" & {_fmt(row.get('training_summary', {}).get('mean_kcm_intervention'))}\\"
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
    total_runs, promoted_runs, top_rows = _candidate_stats()
    latest_status_path = _find_latest_long_run_status()
    latest_status = _read_json(latest_status_path) if latest_status_path is not None else {}
    best_state = current_best.get("state", {})

    lines = [
        f"截至 {_latex_escape(time.strftime('%Y-%m-%d %H:%M:%S'))}，自治搜索共记录 {total_runs} 个实验，"
        f"其中共有 {promoted_runs} 个配置被晋升为阶段最优。当前搜索最优配置为 "
        f"`{_latex_escape(str(best_state.get('experiment_id', '待更新')) )}`，"
        f"候选族为 `{_latex_escape(str(best_state.get('candidate', '待更新')) )}`，"
        f"综合得分为 `{_fmt(best_state.get('score'))}`。",
        "",
        r"\subsection{搜索空间}",
        "当前自动搜索并不直接修改 PPO 主体结构，而是在固定评测协议下围绕有限的候选配置族进行探索。搜索空间包括：前瞻增强类、角区平滑增强类、停滞抑制类、探索强度调节类以及价值网络更新速率调节类。各候选仅对已有配置项做小范围、可解释的增量调整，以保证搜索过程可复现且便于分析。",
        "",
        r"\subsection{选择准则}",
        "当前自治流程采用两阶段筛选。第一阶段使用较低成本的路径子集与 episode 数进行粗筛，重点关注通过率、成功率与稳定性；第二阶段仅对入围候选进行全路径复评估，并以通过路径数优先、综合得分次之的规则决定是否晋升。若候选的通过路径数超过当前最优，则直接保留；若通过路径数相同，则按成功率、平均进度与停滞率等指标顺序比较。",
        "",
        r"\subsection{代表性结果摘要}",
    ]

    if latest_status:
        lines.append(
            "当前仍在运行的长时搜索任务编号为 "
            f"`{_latex_escape(str(latest_status.get('run_id', '')) )}`，"
            f"状态为 `{_latex_escape(str(latest_status.get('status', 'unknown')) )}`，"
            f"预计结束时间为 `{_latex_escape(str(latest_status.get('deadline_time', '待更新')) )}`。"
        )
        lines.append("")

    if top_rows:
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\caption{截至当前的代表性晋升配置摘要。}",
                r"\resizebox{0.92\textwidth}{!}{",
                r"\begin{tabular}{llll}",
                r"\toprule",
                r"\textbf{实验ID} & \textbf{候选族} & \textbf{说明} & \textbf{综合得分}\\",
                r"\midrule",
            ]
        )
        for row in top_rows:
            lines.append(
                rf"`{_latex_escape(str(row.get('experiment_id', '')) )}`"
                rf" & `{_latex_escape(str(row.get('candidate', '')) )}`"
                rf" & {_latex_escape(str(row.get('description', '')))}"
                rf" & {_fmt(row.get('score'))}\\"
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
    else:
        lines.append("代表性晋升配置将在明日结果收敛后补充。")
        lines.append("")

    if suite_variants:
        completed = [
            VARIANT_LABELS.get(name, name)
            for name, payload in suite_variants.items()
            if str(payload.get("status", "")).lower() == "completed"
        ]
        if completed:
            lines.append("论文专用复现实验已完成的配置包括：" + "、".join(completed) + "。")
        else:
            lines.append("论文专用复现实验仍在运行中，结构化消融结果将在明日统一回填。")
    else:
        lines.append("论文专用复现实验结果尚未生成，相关表格将根据后台实验自动刷新。")

    lines.append("")
    return "\n".join(lines)


def _placeholder_png(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=160)
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.38, body, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path)
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


def _build_qualitative_figure(full_method: dict, baseline: dict) -> None:
    path_plan = [(name, name) for name in _choose_qualitative_paths(full_method, baseline)]
    panels = []
    for path_key, label in path_plan:
        panels.append(
            (
                label,
                _read_rollout_png(full_method.get("rollouts_summary", {}), path_key),
                _read_rollout_png(baseline.get("rollouts_summary", {}), path_key),
            )
        )

    if not any(item[1] or item[2] for item in panels):
        _placeholder_png(QUAL_FIG, "Qualitative Figure Pending", "Waiting for rollout images from the full method and baseline.")
        return

    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 4.8 * len(panels)), dpi=160)
    if len(panels) == 1:
        axes = [axes]
    for row_axes, (label, full_png, base_png) in zip(axes, panels):
        titles = [f"{label} | Full Method", f"{label} | Baseline"]
        for ax, png_path, title in zip(row_axes, [full_png, base_png], titles):
            ax.axis("off")
            ax.set_title(title, fontsize=11)
            if png_path is not None:
                ax.imshow(mpimg.imread(png_path))
            else:
                ax.text(0.5, 0.5, "Pending", ha="center", va="center", fontsize=12)
    fig.tight_layout()
    QUAL_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(QUAL_FIG)
    plt.close(fig)


def _build_kcm_figure(trace_summary: dict[str, Any]) -> None:
    rows = _rows_from_trace_summary(trace_summary)
    if not rows:
        _placeholder_png(KCM_FIG, "Behavior Figure Pending", "Waiting for a full-step deterministic trace from the full method.")
        return

    x = [int(float(row.get("env_step", 0))) for row in rows]
    velocity = [float(_safe_float(row.get("velocity"), 0.0) or 0.0) for row in rows]
    contour_error = [float(_safe_float(row.get("contour_error"), 0.0) or 0.0) for row in rows]
    lookahead = [float(_safe_float(row.get("lookahead_dist_active"), 0.0) or 0.0) for row in rows]
    kcm = [float(_safe_float(row.get("kcm_intervention"), 0.0) or 0.0) for row in rows]
    cornerness = [float(_safe_float(row.get("cornerness"), 0.0) or 0.0) for row in rows]

    selected_path = str(trace_summary.get("selected_path", "unknown"))
    selected_steps = int(trace_summary.get("selected_steps", len(rows)))
    done_reason = str(trace_summary.get("selected_done_reason", "unknown"))

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.0), dpi=160, sharex=True)
    fig.suptitle(
        f"Selected full-step rollout | path={selected_path} | steps={selected_steps} | done={done_reason}",
        fontsize=12,
    )

    axes[0].plot(x, velocity, color="#1f77b4", linewidth=1.3)
    axes[0].set_ylabel("v (mm/s)")
    axes[0].grid(True, linestyle=":", alpha=0.4)

    axes[1].plot(x, contour_error, color="#d62728", linewidth=1.3)
    axes[1].set_ylabel("e_c (mm)")
    axes[1].grid(True, linestyle=":", alpha=0.4)

    axes[2].plot(x, lookahead, color="#2ca02c", linewidth=1.3, label="lookahead")
    ax_corner = axes[2].twinx()
    ax_corner.plot(x, cornerness, color="#9467bd", linewidth=1.0, linestyle="--", alpha=0.85, label="cornerness")
    axes[2].set_ylabel("L (mm)")
    ax_corner.set_ylabel("cornerness (-)")
    axes[2].grid(True, linestyle=":", alpha=0.4)
    lines_left, labels_left = axes[2].get_legend_handles_labels()
    lines_right, labels_right = ax_corner.get_legend_handles_labels()
    axes[2].legend(lines_left + lines_right, labels_left + labels_right, loc="upper right", fontsize=8)

    axes[3].plot(x, kcm, color="#ff7f0e", linewidth=1.3)
    axes[3].set_ylabel(r"$\eta$ (-)")
    axes[3].set_xlabel("Step (-)")
    axes[3].grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    KCM_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(KCM_FIG)
    plt.close(fig)


def _build_jerk_constraint_figure(full_trace: dict[str, Any], abl_no_kcm_trace: dict[str, Any]) -> dict[str, Any]:
    if not full_trace or not abl_no_kcm_trace:
        _placeholder_png(JERK_COMPARE_FIG, "Jerk Comparison Pending", "Waiting for both the full method and no-KCM ablation traces.")
        return {}

    compare_path = _choose_jerk_compare_path(full_trace, abl_no_kcm_trace)
    if not compare_path:
        _placeholder_png(JERK_COMPARE_FIG, "Jerk Comparison Pending", "No common path is available for the jerk-constraint comparison.")
        return {}

    full_rows = _rows_from_trace_summary(full_trace, path_name=compare_path)
    abl_rows = _rows_from_trace_summary(abl_no_kcm_trace, path_name=compare_path)
    full_meta = _candidate_by_path(full_trace, compare_path)
    abl_meta = _candidate_by_path(abl_no_kcm_trace, compare_path)
    if (not full_rows) or (not abl_rows):
        _placeholder_png(JERK_COMPARE_FIG, "Jerk Comparison Pending", "Trace rows for the comparison path are not ready yet.")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160, sharey=True)
    compare_specs = [
        ("本文最终方法", full_rows, full_meta, "#1f77b4"),
        ("无KCM", abl_rows, abl_meta, "#d62728"),
    ]

    for ax, (title, rows, meta, color) in zip(axes, compare_specs):
        x = [int(float(row.get("env_step", 0))) for row in rows]
        y = [float(_safe_float(row.get("jerk"), 0.0) or 0.0) for row in rows]
        limit = float(meta.get("linear_jerk_limit", 0.0))
        max_abs = float(meta.get("max_abs_linear_jerk", 0.0))
        ratio = max_abs / max(limit, 1e-6)
        ax.plot(x, y, color=color, linewidth=1.2)
        ax.axhline(limit, color="#2f9e44", linestyle="--", linewidth=1.0, label="+Jmax")
        ax.axhline(-limit, color="#2f9e44", linestyle=":", linewidth=1.0, label="-Jmax")
        ax.set_title(f"{title} | path={compare_path}\nmax|j|/Jmax={ratio:.2f}")
        ax.set_xlabel("Step (-)")
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel(r"j (mm/s$^3$)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    JERK_COMPARE_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(JERK_COMPARE_FIG)
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
    for key in ["abl_fixed_lookahead", "abl_no_kcm", "abl_no_cornerness"]:
        if key in suite_variants:
            ablation_rows.append(suite_variants[key])

    full_trace_summary = _build_variant_trace(full_variant, "full_method")
    abl_no_kcm_trace_summary = _build_variant_trace(abl_no_kcm_variant, "abl_no_kcm") if abl_no_kcm_variant else {}
    jerk_compare_summary = _build_jerk_constraint_figure(full_trace_summary, abl_no_kcm_trace_summary)

    _write_text(MAIN_RESULTS_TEX, _build_main_results_tex(full_variant, baseline_variant))
    _write_text(ABLATION_TEX, _build_ablation_tex([row for row in ablation_rows if row]))
    _write_text(APPENDIX_TEX, _build_appendix_tex(current_best_variant, suite_variants))
    _build_qualitative_figure(full_variant, baseline_variant)
    _build_kcm_figure(full_trace_summary)

    summary = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_suite_dir": str(suite_bundle.get("suite_dir", "")),
        "latest_long_run_status": str(_find_latest_long_run_status() or ""),
        "main_full_method_source": full_variant_source,
        "full_method_trace": full_trace_summary,
        "jerk_compare": jerk_compare_summary,
        "files": {
            "main_results_tex": str(MAIN_RESULTS_TEX),
            "ablation_tex": str(ABLATION_TEX),
            "appendix_tex": str(APPENDIX_TEX),
            "qualitative_figure": str(QUAL_FIG),
            "kcm_figure": str(KCM_FIG),
            "jerk_compare_figure": str(JERK_COMPARE_FIG),
            "full_trace_summary": str(FULL_TRACE_SUMMARY_JSON),
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
