from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from prepare import (
    CURRENT_BEST_CONFIG,
    PAPER_RUNS_DIR,
    RUNS_DIR,
    ExperimentResult,
    archive_promoted_result,
    append_result_row,
    build_selected_paths,
    build_train_config,
    clamp,
    ensure_workspace,
    evaluate_model_across_paths,
    export_best_rollouts,
    find_latest_checkpoint,
    find_model_checkpoint,
    get_git_head,
    get_nested,
    latest_checkpoint_from_state,
    load_checkpoint_episode,
    load_current_best_state,
    load_yaml,
    mul_nested,
    next_experiment_id,
    now_text,
    promote_candidate,
    read_results_history,
    refresh_workspace_reports,
    set_nested,
    summarize_candidate_history,
    train_candidate,
    upsert_result_row,
    write_json,
    write_yaml,
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    apply: Callable[[dict, float], None]


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    amplitude: float
    description: str
    apply: Callable[[dict], None]


AMP_MIN = 0.65
AMP_MAX = 1.60
AMP_SCORE_GAIN_SCALE = 120.0
DEFAULT_TEACHER_TARGET_STEPS = {
    "square": 8643,
    "circle": 5893,
    "butterfly": 6593,
}
_MISSING = object()
_FIXED_LOOKAHEAD_SEED_KEYS = (
    "ppo.actor_lr",
    "ppo.critic_lr",
    "ppo.gamma",
    "ppo.lmbda",
    "ppo.epochs",
    "ppo.ent_coef",
    "reward_weights.w_s",
    "reward_weights.w_e",
    "reward_weights.w_tau",
    "reward_weights.w_smooth",
    "reward_weights.w_ang_acc",
    "reward_weights.corner_w_tau_scale",
    "reward_weights.p4.speed_weight",
    "reward_weights.p4.time_penalty",
    "reward_weights.p4.v_min",
    "reward_weights.p4.exit_progress_mult",
    "reward_weights.p4.exit_speed_target_min",
    "reward_weights.p4.stall_penalty",
    "reward_weights.p4.stall_steps",
    "reward_weights.p4.stall_progress_eps",
    "reward_weights.p8.vcap_rate_up",
    "reward_weights.p8.vcap_rate_down",
    "reward_weights.p8.ang_cap_min_ratio",
    "reward_weights.p8.corner_exit_e_release_ratio",
    "reward_weights.p8.corner_exit_psi_release_deg",
    "reward_weights.p8.recovery_e_release_ratio",
    "reward_weights.p8.use_recovery_cap",
    "reward_weights.cornerness.w_track0",
    "reward_weights.cornerness.w_track_min",
    "reward_weights.cornerness.w_smooth0",
    "reward_weights.cornerness.theta0_deg",
    "reward_weights.cornerness.ema_tau_steps",
    "reward_weights.cornerness.smooth_source",
    "reward_weights.cornerness.smooth_power",
    "reward_weights.lookahead.distance",
    "reward_weights.lookahead_control.min_dist",
    "reward_weights.lookahead_control.max_dist",
    "reward_weights.lookahead_control.straight_dist",
    "reward_weights.lookahead_control.corner_dist",
    "reward_weights.lookahead_control.action_default",
    "reward_weights.lookahead_control.mix_gain",
)


def _scale_up(base_delta: float, amplitude: float) -> float:
    return 1.0 + float(base_delta) * float(amplitude)


def _scale_down(base_delta: float, amplitude: float) -> float:
    return max(0.05, 1.0 - float(base_delta) * float(amplitude))


def _int_delta(base_delta: int, amplitude: float) -> int:
    return max(1, int(round(float(base_delta) * max(0.5, float(amplitude)))))


def _path_names_from_cfg(cfg: Mapping[str, object]) -> list[str]:
    names: list[str] = []
    curriculum_candidates = [
        cfg.get("path_curriculum", {}),
        cfg.get("training", {}).get("path_curriculum", {}) if isinstance(cfg.get("training", {}), Mapping) else {},
    ]
    for curriculum in curriculum_candidates:
        if not isinstance(curriculum, Mapping):
            continue
        raw_paths = curriculum.get("paths", [])
        if not isinstance(raw_paths, Sequence):
            continue
        for item in raw_paths:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("type") or "").strip()
            if name and name not in names:
                names.append(name)
    primary = cfg.get("path", {})
    if isinstance(primary, Mapping):
        name = str(primary.get("name") or primary.get("type") or "").strip()
        if name and name not in names:
            names.insert(0, name)
    return names


def _load_teacher_target_steps(path_names: Sequence[str]) -> dict[str, int]:
    targets = dict(DEFAULT_TEACHER_TARGET_STEPS)
    paper_runs_root = PAPER_RUNS_DIR
    if paper_runs_root.exists():
        summaries = sorted(
            paper_runs_root.glob("*/abl_fixed_lookahead/best_rollouts/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for summary_path in summaries:
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            paths = payload.get("paths", {})
            if not isinstance(paths, Mapping):
                continue
            loaded_any = False
            for name, item in paths.items():
                if not isinstance(item, Mapping):
                    continue
                try:
                    steps = int(item.get("steps", 0))
                except (TypeError, ValueError):
                    steps = 0
                key = str(name).strip()
                if key and steps > 1:
                    targets[key] = steps
                    loaded_any = True
            if loaded_any:
                break

    filtered: dict[str, int] = {}
    if path_names:
        fallback_default = int(round(sum(targets.values()) / max(1, len(targets))))
        for name in path_names:
            key = str(name).strip()
            if not key:
                continue
            filtered[key] = int(targets.get(key, fallback_default))
        return filtered
    return targets


def _load_latest_fixed_lookahead_config() -> dict | None:
    paper_runs_root = PAPER_RUNS_DIR
    if not paper_runs_root.exists():
        return None
    config_paths = sorted(
        paper_runs_root.glob("*/abl_fixed_lookahead/config.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for config_path in config_paths:
        try:
            payload = load_yaml(config_path)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _fixed_prior_norm_from_cfg(cfg: Mapping[str, object]) -> float:
    action_default = get_nested(cfg, "reward_weights.lookahead_control.action_default", _MISSING)
    if action_default is not _MISSING:
        try:
            return clamp(float(action_default), 0.0, 1.0)
        except (TypeError, ValueError):
            pass

    min_dist = float(get_nested(cfg, "reward_weights.lookahead_control.min_dist", 0.8) or 0.8)
    max_dist = float(get_nested(cfg, "reward_weights.lookahead_control.max_dist", 4.0) or 4.0)
    dist = float(get_nested(cfg, "reward_weights.lookahead.distance", min_dist) or min_dist)
    span = max(max_dist - min_dist, 1e-6)
    return clamp((dist - min_dist) / span, 0.0, 1.0)


def _copy_seed_fields(dst_cfg: dict, src_cfg: Mapping[str, object], dotted_keys: Sequence[str]) -> None:
    for dotted_key in dotted_keys:
        value = get_nested(src_cfg, dotted_key, _MISSING)
        if value is _MISSING:
            continue
        set_nested(dst_cfg, dotted_key, copy.deepcopy(value))


def _seed_from_fixed_lookahead(cfg: dict) -> dict | None:
    fixed_cfg = _load_latest_fixed_lookahead_config()
    if fixed_cfg is None:
        return None
    _copy_seed_fields(cfg, fixed_cfg, _FIXED_LOOKAHEAD_SEED_KEYS)
    return fixed_cfg


def candidate_specs() -> list[CandidateSpec]:
    def baseline(cfg: dict, amplitude: float) -> None:
        return

    def _scale_value(cfg: dict, dotted_key: str, factor: float, *, low: float, high: float) -> None:
        current = get_nested(cfg, dotted_key)
        if current is None:
            return
        set_nested(cfg, dotted_key, clamp(float(current) * float(factor), low, high))

    def _enable_learnable_lookahead(cfg: dict) -> None:
        set_nested(cfg, "reward_weights.lookahead_control.enabled", True)
        set_nested(cfg, "reward_weights.lookahead_control.policy_action", True)
        set_nested(cfg, "experiment.enable_kcm", True)
        set_nested(cfg, "environment.lookahead_obs_enabled", True)
        set_nested(cfg, "environment.lookahead_points", max(8, int(get_nested(cfg, "environment.lookahead_points", 8) or 8)))
        set_nested(cfg, "reward_weights.cornerness.enabled", True)
        set_nested(cfg, "reward_weights.lookahead_reward.enabled", True)

    def _set_fixed_lookahead_prior(cfg: dict) -> None:
        min_dist = float(get_nested(cfg, "reward_weights.lookahead_control.min_dist", 0.8) or 0.8)
        max_dist = float(get_nested(cfg, "reward_weights.lookahead_control.max_dist", 4.0) or 4.0)
        straight_dist = float(get_nested(cfg, "reward_weights.lookahead_control.straight_dist", 1.2) or 1.2)
        corner_dist = float(get_nested(cfg, "reward_weights.lookahead_control.corner_dist", 2.4) or 2.4)
        default_dist = float(get_nested(cfg, "reward_weights.lookahead.distance", 2.5) or 2.5)
        fixed_dist = min(max(default_dist, min_dist), max_dist)
        fixed_dist = min(max((fixed_dist + 0.5 * (straight_dist + corner_dist)) / 2.0, min_dist), max_dist)
        set_nested(cfg, "reward_weights.lookahead.distance", fixed_dist)
        dist_span = max(max_dist - min_dist, 1e-6)
        prior_u = clamp((fixed_dist - min_dist) / dist_span, 0.0, 1.0)
        set_nested(cfg, "reward_weights.lookahead_control.action_default", prior_u)

    def _enable_teacher_progress(
        cfg: dict,
        *,
        amplitude: float,
        lag_weight: float,
        lead_bonus: float,
        finish_bonus: float,
        slack: float,
        fresh_start: bool,
    ) -> None:
        _enable_learnable_lookahead(cfg)
        _set_fixed_lookahead_prior(cfg)
        path_names = _path_names_from_cfg(cfg)
        target_steps = _load_teacher_target_steps(path_names)
        default_target_steps = int(round(sum(target_steps.values()) / max(1, len(target_steps))))
        set_nested(cfg, "reward_weights.teacher_progress.enabled", True)
        set_nested(cfg, "reward_weights.teacher_progress.path_target_steps", target_steps)
        set_nested(cfg, "reward_weights.teacher_progress.default_target_steps", default_target_steps)
        set_nested(cfg, "reward_weights.teacher_progress.lag_weight", lag_weight)
        set_nested(cfg, "reward_weights.teacher_progress.lead_bonus", lead_bonus)
        set_nested(cfg, "reward_weights.teacher_progress.finish_bonus", finish_bonus)
        set_nested(cfg, "reward_weights.teacher_progress.slack", slack)
        set_nested(cfg, "reward_weights.teacher_progress.power", 2.0)
        set_nested(cfg, "reward_weights.teacher_progress.lag_clip", 0.30)
        if fresh_start:
            set_nested(cfg, "experiment.force_fresh_start", True)
            training_cfg = cfg.setdefault("training", {})
            training_cfg["num_episodes"] = max(
                int(training_cfg.get("num_episodes", 0) or 0),
                max(120, int(round(150 * max(1.0, amplitude)))),
            )
            current_budget = float(training_cfg.get("time_budget_seconds", 0.0) or 0.0)
            budget_floor = 900.0 * max(1.0, amplitude)
            training_cfg["time_budget_seconds"] = max(budget_floor, current_budget * 1.8 if current_budget > 0.0 else 0.0)
            curriculum_cfg = training_cfg.get("path_curriculum", {})
            if isinstance(curriculum_cfg, dict):
                curriculum_cfg["episodes_per_path"] = max(4, int(curriculum_cfg.get("episodes_per_path", 2) or 2))
            set_nested(cfg, "ppo.actor_lr", max(float(get_nested(cfg, "ppo.actor_lr", 1.2e-4) or 1.2e-4), 1.2e-4))
            set_nested(cfg, "ppo.critic_lr", max(float(get_nested(cfg, "ppo.critic_lr", 1.5e-4) or 1.5e-4), 1.5e-4))
            set_nested(cfg, "ppo.epochs", max(8, int(get_nested(cfg, "ppo.epochs", 8) or 8)))

    def _enable_speed_objective(
        cfg: dict,
        *,
        amplitude: float,
        straight_target_ratio: float,
        corner_target_ratio: float,
        straight_weight: float,
        corner_weight: float,
        floor_weight: float,
        finish_bonus_weight: float,
        late_penalty_weight: float,
        fresh_start: bool,
    ) -> None:
        _enable_teacher_progress(
            cfg,
            amplitude=amplitude,
            lag_weight=30.0,
            lead_bonus=3.2,
            finish_bonus=22.0,
            slack=0.016,
            fresh_start=fresh_start,
        )
        set_nested(cfg, "reward_weights.speed_profile.enabled", True)
        set_nested(cfg, "reward_weights.speed_profile.straight_target_ratio", straight_target_ratio)
        set_nested(cfg, "reward_weights.speed_profile.corner_target_ratio", corner_target_ratio)
        set_nested(cfg, "reward_weights.speed_profile.straight_weight", straight_weight)
        set_nested(cfg, "reward_weights.speed_profile.corner_weight", corner_weight)
        set_nested(cfg, "reward_weights.speed_profile.floor_weight", floor_weight)
        set_nested(cfg, "reward_weights.speed_profile.straight_floor_ratio", max(0.20, straight_target_ratio * 0.78))
        set_nested(cfg, "reward_weights.speed_profile.corner_floor_ratio", max(0.10, corner_target_ratio * 0.72))
        set_nested(cfg, "reward_weights.speed_profile.overspeed_weight", 0.75)
        set_nested(cfg, "reward_weights.speed_profile.power", 2.0)
        set_nested(cfg, "reward_weights.finish_efficiency.enabled", True)
        set_nested(cfg, "reward_weights.finish_efficiency.bonus_weight", finish_bonus_weight)
        set_nested(cfg, "reward_weights.finish_efficiency.late_penalty_weight", late_penalty_weight)
        set_nested(cfg, "reward_weights.finish_efficiency.margin_clip", 0.42)

    def _enable_fixed_speed_distill(
        cfg: dict,
        *,
        amplitude: float,
        straight_target_ratio: float,
        corner_target_ratio: float,
        straight_weight: float,
        corner_weight: float,
        floor_weight: float,
        finish_bonus_weight: float,
        late_penalty_weight: float,
        track_weight: float,
        smooth_weight: float,
        settle_bonus: float,
        corner_target_boost: float,
        fresh_start: bool,
    ) -> None:
        fixed_cfg = _seed_from_fixed_lookahead(cfg) or cfg
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=straight_target_ratio,
            corner_target_ratio=corner_target_ratio,
            straight_weight=straight_weight,
            corner_weight=corner_weight,
            floor_weight=floor_weight,
            finish_bonus_weight=finish_bonus_weight,
            late_penalty_weight=late_penalty_weight,
            fresh_start=fresh_start,
        )
        prior_norm = _fixed_prior_norm_from_cfg(fixed_cfg)
        set_nested(cfg, "reward_weights.teacher_lookahead.enabled", True)
        set_nested(cfg, "reward_weights.teacher_lookahead.fixed_prior_norm", prior_norm)
        set_nested(cfg, "reward_weights.teacher_lookahead.straight_target_norm", prior_norm)
        set_nested(cfg, "reward_weights.teacher_lookahead.corner_target_norm", clamp(prior_norm + corner_target_boost, 0.0, 1.0))
        set_nested(cfg, "reward_weights.teacher_lookahead.track_weight", track_weight)
        set_nested(cfg, "reward_weights.teacher_lookahead.smooth_weight", smooth_weight)
        set_nested(cfg, "reward_weights.teacher_lookahead.settle_bonus", settle_bonus)
        set_nested(cfg, "reward_weights.teacher_lookahead.settle_band", 0.08)
        set_nested(
            cfg,
            "reward_weights.teacher_lookahead.straight_speed_floor_ratio",
            clamp(max(0.35, straight_target_ratio * 0.92), 0.20, 0.98),
        )
        set_nested(
            cfg,
            "reward_weights.teacher_lookahead.corner_speed_floor_ratio",
            clamp(max(0.16, corner_target_ratio * 0.88), 0.08, 0.90),
        )
        set_nested(cfg, "reward_weights.teacher_lookahead.floor_weight", max(0.5, floor_weight * 0.65))
        set_nested(cfg, "reward_weights.teacher_lookahead.power", 2.0)
        set_nested(cfg, "reward_weights.lookahead_control.action_default", prior_norm)

    def _enable_residual_teacher_distill(
        cfg: dict,
        *,
        amplitude: float,
        straight_target_ratio: float,
        corner_target_ratio: float,
        straight_weight: float,
        corner_weight: float,
        floor_weight: float,
        finish_bonus_weight: float,
        late_penalty_weight: float,
        track_weight: float,
        smooth_weight: float,
        settle_bonus: float,
        residual_band_ratio: float,
        residual_corner_only: bool,
        end_lock_progress: float,
        gate_min: float,
        fresh_start: bool,
    ) -> None:
        fixed_cfg = _seed_from_fixed_lookahead(cfg) or cfg
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=straight_target_ratio,
            corner_target_ratio=corner_target_ratio,
            straight_weight=straight_weight,
            corner_weight=corner_weight,
            floor_weight=floor_weight,
            finish_bonus_weight=finish_bonus_weight,
            late_penalty_weight=late_penalty_weight,
            fresh_start=fresh_start,
        )
        _enable_learnable_lookahead(cfg)
        set_nested(cfg, "reward_weights.lookahead_control.residual_mode", True)
        set_nested(cfg, "reward_weights.lookahead_control.residual_band_ratio", residual_band_ratio)
        set_nested(cfg, "reward_weights.lookahead_control.residual_corner_only", residual_corner_only)
        set_nested(cfg, "reward_weights.lookahead_control.residual_gate_min", gate_min)
        set_nested(cfg, "reward_weights.lookahead_control.end_lock_progress", end_lock_progress)
        set_nested(cfg, "reward_weights.lookahead_control.action_default", 0.5)
        set_nested(cfg, "reward_weights.teacher_lookahead.enabled", True)
        set_nested(cfg, "reward_weights.teacher_lookahead.residual_mode", True)
        set_nested(cfg, "reward_weights.teacher_lookahead.fixed_prior_norm", _fixed_prior_norm_from_cfg(fixed_cfg))
        set_nested(cfg, "reward_weights.teacher_lookahead.track_weight", track_weight)
        set_nested(cfg, "reward_weights.teacher_lookahead.straight_hold_weight", track_weight * 1.15)
        set_nested(cfg, "reward_weights.teacher_lookahead.smooth_weight", smooth_weight)
        set_nested(cfg, "reward_weights.teacher_lookahead.settle_bonus", settle_bonus)
        set_nested(cfg, "reward_weights.teacher_lookahead.settle_band", 0.10)
        set_nested(
            cfg,
            "reward_weights.teacher_lookahead.straight_speed_floor_ratio",
            clamp(max(0.35, straight_target_ratio * 0.92), 0.20, 0.98),
        )
        set_nested(
            cfg,
            "reward_weights.teacher_lookahead.corner_speed_floor_ratio",
            clamp(max(0.16, corner_target_ratio * 0.88), 0.08, 0.90),
        )
        set_nested(cfg, "reward_weights.teacher_lookahead.floor_weight", max(0.5, floor_weight * 0.65))
        set_nested(cfg, "reward_weights.teacher_lookahead.power", 2.0)

    def teacher_progress_seed(cfg: dict, amplitude: float) -> None:
        _enable_teacher_progress(
            cfg,
            amplitude=amplitude,
            lag_weight=26.0,
            lead_bonus=2.5,
            finish_bonus=18.0,
            slack=0.018,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.12, amplitude), low=2.0, high=18.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.20, amplitude),
            low=-0.08,
            high=-0.002,
        )
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.08, amplitude), low=0.5, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.08, amplitude), low=0.3, high=10.0)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_down(0.10, amplitude), low=0.20, high=0.80)

    def teacher_progress_aggressive(cfg: dict, amplitude: float) -> None:
        _enable_teacher_progress(
            cfg,
            amplitude=amplitude,
            lag_weight=34.0,
            lead_bonus=3.5,
            finish_bonus=24.0,
            slack=0.015,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.28, amplitude), low=2.0, high=20.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.85, amplitude),
            low=-0.10,
            high=-0.003,
        )
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.22, amplitude), low=0.25, high=0.85)
        mul_nested(cfg, "reward_weights.p4.exit_progress_mult", _scale_up(0.18, amplitude), low=1.0, high=2.2)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.55, amplitude), low=0.01, high=0.30)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.12, amplitude), low=0.08, high=0.35)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.16, amplitude), low=0.4, high=12.0)

    def teacher_progress_corner_release(cfg: dict, amplitude: float) -> None:
        _enable_teacher_progress(
            cfg,
            amplitude=amplitude,
            lag_weight=28.0,
            lead_bonus=2.8,
            finish_bonus=20.0,
            slack=0.018,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.65, amplitude), low=0.01, high=0.30)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_down", _scale_up(0.25, amplitude), low=0.01, high=0.18)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.18, amplitude), low=0.08, high=0.38)
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_down(0.16, amplitude), low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_down(0.16, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.10, amplitude), low=1.5, high=4.8)

    def teacher_progress_stable(cfg: dict, amplitude: float) -> None:
        _enable_teacher_progress(
            cfg,
            amplitude=amplitude,
            lag_weight=22.0,
            lead_bonus=2.0,
            finish_bonus=16.0,
            slack=0.022,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_up(0.16, amplitude), low=0.3, high=10.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_up(0.18, amplitude), low=0.05, high=1.50)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.10, amplitude), low=0.5, high=12.0)
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.10, amplitude), low=2.0, high=18.0)

    def speed_profile_seed(cfg: dict, amplitude: float) -> None:
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.72,
            corner_target_ratio=0.34,
            straight_weight=4.0,
            corner_weight=1.6,
            floor_weight=2.6,
            finish_bonus_weight=18.0,
            late_penalty_weight=7.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.18, amplitude), low=2.0, high=20.0)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.40, amplitude), low=-0.12, high=-0.003)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.12, amplitude), low=0.35, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.08, amplitude), low=0.2, high=10.0)

    def speed_profile_straight_fast(cfg: dict, amplitude: float) -> None:
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.78,
            corner_target_ratio=0.32,
            straight_weight=5.0,
            corner_weight=1.5,
            floor_weight=3.0,
            finish_bonus_weight=22.0,
            late_penalty_weight=9.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.32, amplitude), low=2.0, high=22.0)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.95, amplitude), low=-0.14, high=-0.004)
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.22, amplitude), low=0.25, high=0.88)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.18, amplitude), low=0.25, high=12.0)
        mul_nested(cfg, "reward_weights.cornerness.w_track0", _scale_down(0.15, amplitude), low=0.25, high=10.0)

    def speed_profile_corner_balance(cfg: dict, amplitude: float) -> None:
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.70,
            corner_target_ratio=0.40,
            straight_weight=3.8,
            corner_weight=2.4,
            floor_weight=2.4,
            finish_bonus_weight=18.0,
            late_penalty_weight=7.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_down(0.14, amplitude), low=0.15, high=8.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_down(0.10, amplitude), low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.45, amplitude), low=0.01, high=0.30)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.16, amplitude), low=0.08, high=0.40)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.12, amplitude), low=1.5, high=5.0)

    def speed_profile_finish_biased(cfg: dict, amplitude: float) -> None:
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.74,
            corner_target_ratio=0.34,
            straight_weight=4.4,
            corner_weight=1.6,
            floor_weight=2.8,
            finish_bonus_weight=26.0,
            late_penalty_weight=12.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.exit_progress_mult", _scale_up(0.24, amplitude), low=1.0, high=2.4)
        mul_nested(cfg, "reward_weights.p4.exit_speed_target_min", _scale_up(0.08, amplitude), low=0.75, high=1.30)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.70, amplitude), low=-0.14, high=-0.004)
        mul_nested(cfg, "reward_weights.p4.stall_penalty", _scale_up(0.30, amplitude), low=-45.0, high=-2.0)
        mul_nested(cfg, "reward_weights.p4.stall_steps", _scale_down(0.22, amplitude), low=90.0, high=3000.0)

    def fixed_speed_distill_seed(cfg: dict, amplitude: float) -> None:
        _enable_fixed_speed_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.80,
            corner_target_ratio=0.42,
            straight_weight=5.0,
            corner_weight=2.0,
            floor_weight=3.0,
            finish_bonus_weight=24.0,
            late_penalty_weight=10.0,
            track_weight=1.9,
            smooth_weight=0.45,
            settle_bonus=0.35,
            corner_target_boost=0.10,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.18, amplitude), low=0.25, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.12, amplitude), low=0.15, high=10.0)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.55, amplitude), low=-0.14, high=-0.004)

    def fixed_speed_distill_aggressive(cfg: dict, amplitude: float) -> None:
        _enable_fixed_speed_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.86,
            corner_target_ratio=0.48,
            straight_weight=6.4,
            corner_weight=2.3,
            floor_weight=3.4,
            finish_bonus_weight=30.0,
            late_penalty_weight=14.0,
            track_weight=2.2,
            smooth_weight=0.35,
            settle_bonus=0.28,
            corner_target_boost=0.12,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.34, amplitude), low=2.0, high=24.0)
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.18, amplitude), low=0.25, high=0.90)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.60, amplitude), low=0.01, high=0.35)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.16, amplitude), low=0.10, high=0.42)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.95, amplitude), low=-0.16, high=-0.004)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_down(0.12, amplitude), low=0.20, high=0.80)

    def fixed_speed_distill_corner_safe(cfg: dict, amplitude: float) -> None:
        _enable_fixed_speed_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.78,
            corner_target_ratio=0.46,
            straight_weight=5.2,
            corner_weight=2.6,
            floor_weight=2.8,
            finish_bonus_weight=24.0,
            late_penalty_weight=10.0,
            track_weight=2.0,
            smooth_weight=0.65,
            settle_bonus=0.32,
            corner_target_boost=0.14,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_up(0.18, amplitude), low=0.20, high=10.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_up(0.15, amplitude), low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_up(0.10, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_down", _scale_up(0.22, amplitude), low=0.01, high=0.22)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.08, amplitude), low=1.5, high=5.2)

    def residual_fixed_distill_seed(cfg: dict, amplitude: float) -> None:
        _enable_residual_teacher_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.80,
            corner_target_ratio=0.42,
            straight_weight=5.0,
            corner_weight=2.0,
            floor_weight=3.0,
            finish_bonus_weight=24.0,
            late_penalty_weight=10.0,
            track_weight=1.8,
            smooth_weight=0.35,
            settle_bonus=0.35,
            residual_band_ratio=0.18,
            residual_corner_only=True,
            end_lock_progress=0.975,
            gate_min=0.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.22, amplitude), low=2.0, high=22.0)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.60, amplitude), low=-0.16, high=-0.004)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.16, amplitude), low=0.25, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.10, amplitude), low=0.15, high=10.0)

    def residual_fixed_distill_aggressive(cfg: dict, amplitude: float) -> None:
        _enable_residual_teacher_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.86,
            corner_target_ratio=0.48,
            straight_weight=6.4,
            corner_weight=2.4,
            floor_weight=3.3,
            finish_bonus_weight=30.0,
            late_penalty_weight=14.0,
            track_weight=1.6,
            smooth_weight=0.28,
            settle_bonus=0.25,
            residual_band_ratio=0.22,
            residual_corner_only=False,
            end_lock_progress=0.972,
            gate_min=0.20,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.34, amplitude), low=2.0, high=24.0)
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.18, amplitude), low=0.25, high=0.90)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.55, amplitude), low=0.01, high=0.35)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.14, amplitude), low=0.10, high=0.42)
        _scale_value(cfg, "reward_weights.p4.time_penalty", _scale_up(0.95, amplitude), low=-0.18, high=-0.004)

    def residual_fixed_distill_corner_only(cfg: dict, amplitude: float) -> None:
        _enable_residual_teacher_distill(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=0.78,
            corner_target_ratio=0.46,
            straight_weight=5.4,
            corner_weight=2.7,
            floor_weight=2.8,
            finish_bonus_weight=24.0,
            late_penalty_weight=11.0,
            track_weight=2.1,
            smooth_weight=0.55,
            settle_bonus=0.30,
            residual_band_ratio=0.20,
            residual_corner_only=True,
            end_lock_progress=0.97,
            gate_min=0.0,
            fresh_start=True,
        )
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_up(0.18, amplitude), low=0.20, high=10.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_up(0.15, amplitude), low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_up(0.10, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_down", _scale_up(0.20, amplitude), low=0.01, high=0.22)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.08, amplitude), low=1.5, high=5.2)

    def progress_harder(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.w_s", _scale_up(0.28, amplitude), low=4.0, high=40.0)
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.18, amplitude), low=1.0, high=16.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.60, amplitude),
            low=-0.08,
            high=-0.002,
        )
        mul_nested(cfg, "reward_weights.p4.exit_progress_mult", _scale_up(0.12, amplitude), low=1.0, high=2.0)
        mul_nested(cfg, "reward_weights.p4.exit_speed_target_min", _scale_up(0.05, amplitude), low=0.75, high=1.25)
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.12, amplitude), low=0.20, high=0.75)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.10, amplitude), low=0.08, high=0.35)

    def progress_strict(cfg: dict, amplitude: float) -> None:
        progress_harder(cfg, amplitude)
        _scale_value(
            cfg,
            "reward_weights.p4.stall_penalty",
            _scale_up(0.35, amplitude),
            low=-40.0,
            high=-1.0,
        )
        mul_nested(cfg, "reward_weights.p4.stall_steps", _scale_down(0.25, amplitude), low=120.0, high=5000.0)
        mul_nested(cfg, "reward_weights.p4.stall_progress_eps", _scale_up(0.80, amplitude), low=1e-6, high=5e-3)

    def straight_charge_fast(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.w_s", _scale_up(0.35, amplitude), low=4.0, high=45.0)
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.24, amplitude), low=1.0, high=18.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.75, amplitude),
            low=-0.10,
            high=-0.003,
        )
        mul_nested(cfg, "reward_weights.p4.v_min", _scale_up(0.20, amplitude), low=0.20, high=0.82)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.12, amplitude), low=0.3, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.10, amplitude), low=0.2, high=10.0)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.18, amplitude), low=0.08, high=0.38)

    def corner_release_fast(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.22, amplitude), low=0.08, high=0.35)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.60, amplitude), low=0.005, high=0.25)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_down", _scale_up(0.25, amplitude), low=0.005, high=0.15)
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_down(0.12, amplitude), low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_down(0.20, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.10, amplitude), low=1.5, high=4.5)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_up(0.08, amplitude), low=0.25, high=0.90)

    def track_relaxed_fast(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.20, amplitude), low=0.3, high=12.0)
        mul_nested(cfg, "reward_weights.cornerness.w_track0", _scale_down(0.18, amplitude), low=0.3, high=12.0)
        mul_nested(cfg, "reward_weights.cornerness.w_track_min", _scale_down(0.15, amplitude), low=0.1, high=10.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.15, amplitude), low=0.2, high=10.0)
        mul_nested(cfg, "reward_weights.w_s", _scale_up(0.15, amplitude), low=4.0, high=40.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.30, amplitude),
            low=-0.08,
            high=-0.002,
        )

    def smooth_turn_stable(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_up(0.18, amplitude), low=0.2, high=10.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_up(0.16, amplitude), low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_up(0.25, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.10, amplitude), low=0.08, high=0.35)

    def lookahead_dynamic_soft(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_down(0.18, amplitude), low=0.25, high=0.80)
        mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", _scale_up(0.10, amplitude), low=0.8, high=3.2)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.08, amplitude), low=1.5, high=4.2)
        mul_nested(cfg, "reward_weights.lookahead_control.max_dist", _scale_up(0.08, amplitude), low=2.0, high=5.5)
        mul_nested(cfg, "reward_weights.lookahead_reward.corner_target", _scale_up(0.08, amplitude), low=0.45, high=0.96)
        mul_nested(cfg, "reward_weights.lookahead_reward.w_corner", _scale_down(0.12, amplitude), low=0.2, high=2.0)

    def lookahead_dynamic_fast(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_up(0.14, amplitude), low=0.30, high=0.90)
        mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", _scale_up(0.12, amplitude), low=0.8, high=3.4)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.10, amplitude), low=1.5, high=4.5)
        mul_nested(cfg, "reward_weights.lookahead_control.max_dist", _scale_up(0.10, amplitude), low=2.0, high=5.8)
        mul_nested(cfg, "reward_weights.lookahead_reward.corner_target", _scale_up(0.10, amplitude), low=0.45, high=0.96)
        mul_nested(cfg, "reward_weights.p4.speed_weight", _scale_up(0.10, amplitude), low=1.0, high=18.0)
        _scale_value(
            cfg,
            "reward_weights.p4.time_penalty",
            _scale_up(0.18, amplitude),
            low=-0.08,
            high=-0.002,
        )

    def lookahead_horizon_expanded(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        set_nested(cfg, "environment.lookahead_obs_scales", [0.75, 1.0, 1.5])
        set_nested(
            cfg,
            "environment.lookahead_points",
            max(10, int(round(float(get_nested(cfg, "environment.lookahead_points", 8) or 8) * _scale_up(0.22, amplitude)))),
        )
        mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", _scale_up(0.16, amplitude), low=1.0, high=4.2)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.18, amplitude), low=1.8, high=5.2)
        mul_nested(cfg, "reward_weights.lookahead_control.max_dist", _scale_up(0.18, amplitude), low=2.5, high=6.5)
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_up(0.12, amplitude), low=0.30, high=0.92)
        mul_nested(cfg, "reward_weights.lookahead_reward.w_corner", _scale_up(0.10, amplitude), low=0.25, high=2.50)

    def reward_rebalance_fast(cfg: dict, amplitude: float) -> None:
        _enable_speed_objective(
            cfg,
            amplitude=amplitude,
            straight_target_ratio=clamp(0.88 * _scale_up(0.06, amplitude), 0.72, 0.98),
            corner_target_ratio=clamp(0.58 * _scale_up(0.06, amplitude), 0.34, 0.84),
            straight_weight=clamp(6.8 * _scale_up(0.18, amplitude), 2.0, 16.0),
            corner_weight=clamp(5.4 * _scale_up(0.12, amplitude), 1.0, 14.0),
            floor_weight=clamp(2.8 * _scale_up(0.10, amplitude), 0.5, 8.0),
            finish_bonus_weight=clamp(18.0 * _scale_up(0.20, amplitude), 6.0, 40.0),
            late_penalty_weight=clamp(18.0 * _scale_up(0.20, amplitude), 4.0, 45.0),
            fresh_start=False,
        )
        mul_nested(cfg, "reward_weights.w_e", _scale_down(0.22, amplitude), low=0.20, high=12.0)
        mul_nested(cfg, "reward_weights.w_tau", _scale_down(0.18, amplitude), low=0.15, high=10.0)
        mul_nested(cfg, "reward_weights.w_smooth", _scale_down(0.10, amplitude), low=0.01, high=1.20)
        mul_nested(cfg, "reward_weights.cornerness.w_track0", _scale_down(0.14, amplitude), low=0.15, high=10.0)
        mul_nested(cfg, "reward_weights.cornerness.w_track_min", _scale_down(0.12, amplitude), low=0.08, high=8.0)

    def kcm_margin_release(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.24, amplitude), low=0.10, high=0.42)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_up", _scale_up(0.80, amplitude), low=0.01, high=0.35)
        mul_nested(cfg, "reward_weights.p8.vcap_rate_down", _scale_up(0.35, amplitude), low=0.01, high=0.20)
        mul_nested(cfg, "reward_weights.p8.corner_exit_e_release_ratio", _scale_up(0.20, amplitude), low=0.20, high=0.95)
        mul_nested(cfg, "reward_weights.p8.corner_exit_psi_release_deg", _scale_up(0.18, amplitude), low=8.0, high=75.0)
        mul_nested(cfg, "reward_weights.p8.recovery_e_release_ratio", _scale_up(0.15, amplitude), low=0.20, high=0.95)
        set_nested(cfg, "reward_weights.p8.use_recovery_cap", False)
        mul_nested(cfg, "reward_weights.w_s", _scale_up(0.12, amplitude), low=4.0, high=40.0)

    def ppo_horizon_stable(cfg: dict, amplitude: float) -> None:
        _enable_learnable_lookahead(cfg)
        mul_nested(cfg, "ppo.gamma", _scale_up(0.003, amplitude), low=0.985, high=0.999)
        mul_nested(cfg, "ppo.lmbda", _scale_up(0.01, amplitude), low=0.92, high=0.997)
        epochs = int(cfg.get("ppo", {}).get("epochs", 8) or 8)
        set_nested(cfg, "ppo.epochs", min(16, epochs + _int_delta(2, amplitude)))
        mul_nested(cfg, "ppo.actor_lr", _scale_down(0.10, amplitude), low=5e-5, high=5e-4)
        mul_nested(cfg, "ppo.critic_lr", _scale_down(0.05, amplitude), low=6e-5, high=5e-4)
        training_cfg = cfg.setdefault("training", {})
        training_cfg["num_episodes"] = max(int(training_cfg.get("num_episodes", 0) or 0), 140)
        training_cfg["time_budget_seconds"] = max(float(training_cfg.get("time_budget_seconds", 0.0) or 0.0), 1200.0)

    def critic_balanced(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.critic_lr", _scale_up(0.18, amplitude), low=5e-5, high=5e-4)
        mul_nested(cfg, "ppo.actor_lr", _scale_up(0.08, amplitude), low=1e-5, high=5e-4)
        mul_nested(cfg, "ppo.lmbda", _scale_up(0.01, amplitude), low=0.90, high=0.995)

    def actor_mild(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.actor_lr", _scale_up(0.15, amplitude), low=1e-5, high=5e-4)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", min(14, epochs + _int_delta(1, amplitude)))

    def exploration_refresh(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.ent_coef", _scale_up(0.25, amplitude), low=0.0, high=0.05)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", min(14, epochs + _int_delta(1, amplitude)))
        mul_nested(cfg, "reward_weights.control_authority.tangent_blend", _scale_down(0.10, amplitude), low=0.0, high=0.30)

    return [
        CandidateSpec("baseline", "保留当前最优配置，建立或刷新基线", baseline),
        CandidateSpec("residual_fixed_distill_seed", "以前瞻固定反馈律为 teacher，仅学习残差并在终点附近锁定到 teacher", residual_fixed_distill_seed),
        CandidateSpec("residual_fixed_distill_aggressive", "学习 teacher 残差而非绝对前瞻值，并提高直线快进给与完成时间压强", residual_fixed_distill_aggressive),
        CandidateSpec("residual_fixed_distill_corner_only", "只允许拐角区学习前瞻残差，直线段基本跟随 teacher，同时增加终点收敛锁定", residual_fixed_distill_corner_only),
        CandidateSpec("fixed_speed_distill_seed", "从最近固定前瞻配置迁移速度型参数，再开启可学习前瞻重训完整方法", fixed_speed_distill_seed),
        CandidateSpec("fixed_speed_distill_aggressive", "把固定前瞻的快节奏更强地蒸馏到完整方法，重点压缩直线段和整体完成时间", fixed_speed_distill_aggressive),
        CandidateSpec("fixed_speed_distill_corner_safe", "以固定前瞻速度为起点，同时保留更强角区平滑与KCM安全边界", fixed_speed_distill_corner_safe),
        CandidateSpec("teacher_progress_seed", "以固定前瞻步数节奏为教师目标，fresh-start 重训完整方法", teacher_progress_seed),
        CandidateSpec("teacher_progress_aggressive", "以教师步数节奏为硬目标，强化直线快进给与终点时间", teacher_progress_aggressive),
        CandidateSpec("teacher_progress_corner_release", "沿教师节奏加快出弯释放，同时保持 KCM 约束下的平滑过渡", teacher_progress_corner_release),
        CandidateSpec("teacher_progress_stable", "沿教师节奏重训完整方法，但保留更稳的角区平滑权重", teacher_progress_stable),
        CandidateSpec("speed_profile_seed", "以直线高速度和拐角保速为目标，fresh-start 重训完整方法", speed_profile_seed),
        CandidateSpec("speed_profile_straight_fast", "显式鼓励直线段更高速度并压缩整体完成时间", speed_profile_straight_fast),
        CandidateSpec("speed_profile_corner_balance", "提高拐角保速目标，在平滑与出弯速度之间重新平衡", speed_profile_corner_balance),
        CandidateSpec("speed_profile_finish_biased", "把完成效率和末段时间压强前置，减少整体拖尾", speed_profile_finish_biased),
        CandidateSpec("progress_harder", "提高进度奖励与时间压强，优先更快到终点", progress_harder),
        CandidateSpec("progress_strict", "更强地惩罚停滞与拖慢，避免局部静止最优", progress_strict),
        CandidateSpec("straight_charge_fast", "强化直线快进给与整体时间目标", straight_charge_fast),
        CandidateSpec("corner_release_fast", "保持转角平滑的同时加快出弯释放", corner_release_fast),
        CandidateSpec("track_relaxed_fast", "适度放松误差惩罚，换取更小降速", track_relaxed_fast),
        CandidateSpec("smooth_turn_stable", "增强拐角平滑项，减少急剧速度塌陷", smooth_turn_stable),
        CandidateSpec("lookahead_dynamic_soft", "温和放大学习型前瞻的预瞄范围与转角目标", lookahead_dynamic_soft),
        CandidateSpec("lookahead_dynamic_fast", "增强学习型前瞻的速度导向与预瞄强度", lookahead_dynamic_fast),
        CandidateSpec("lookahead_horizon_expanded", "扩大学习型前瞻视野和观测尺度，探索更远预瞄策略", lookahead_horizon_expanded),
        CandidateSpec("reward_rebalance_fast", "重排速度/误差/平滑奖励权重，尝试更激进但仍受约束的速度分布", reward_rebalance_fast),
        CandidateSpec("kcm_margin_release", "保持KCM约束但放松约束释放节奏，探索更快出弯", kcm_margin_release),
        CandidateSpec("ppo_horizon_stable", "提高时域信用分配稳定性，探索更长视野策略收敛", ppo_horizon_stable),
        CandidateSpec("critic_balanced", "同步抬高 value 拟合与策略步幅", critic_balanced),
        CandidateSpec("actor_mild", "温和加快 actor 更新", actor_mild),
        CandidateSpec("exploration_refresh", "轻度提高探索，尝试跳出保守减速策略", exploration_refresh),
    ]


def _weighted_recent_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    weights = list(range(1, len(values) + 1))
    total = sum(weights)
    return sum(float(value) * weight for value, weight in zip(values, weights)) / max(1, total)


def _is_comparable_eval_row(row: dict) -> bool:
    if str(row.get("status", "")).strip() != "ok":
        return False
    if str(row.get("evaluation_stage", "")).strip().lower() not in {"stage2", "full"}:
        return False
    try:
        score = float(row.get("score", float("-inf")))
    except (TypeError, ValueError):
        return False
    return math.isfinite(score)


def collect_recent_score_gains(candidate_name: str, history: Sequence[dict], lookback: int) -> list[float]:
    lookback = max(1, int(lookback))
    rows_by_id = {
        str(row.get("experiment_id", "")).strip(): row
        for row in history
        if str(row.get("experiment_id", "")).strip()
    }
    gains: list[float] = []
    for row in history:
        if str(row.get("candidate", "")).strip() != candidate_name:
            continue
        if not _is_comparable_eval_row(row):
            continue
        parent_id = str(row.get("parent_experiment_id", "")).strip()
        if not parent_id:
            continue
        parent_row = rows_by_id.get(parent_id)
        if parent_row is None or not _is_comparable_eval_row(parent_row):
            continue
        gains.append(float(row.get("score", 0.0)) - float(parent_row.get("score", 0.0)))
    return gains[-lookback:]


def compute_candidate_amplitude(spec: CandidateSpec, history: Sequence[dict], lookback: int) -> float:
    if spec.name == "baseline":
        return 1.0

    gains = collect_recent_score_gains(spec.name, history, lookback)
    if not gains:
        return 1.0

    gain_signal = 0.65 * _weighted_recent_mean(gains) + 0.35 * float(gains[-1])
    mean_abs_gain = sum(abs(float(value)) for value in gains) / max(1, len(gains))
    gain_scale = max(AMP_SCORE_GAIN_SCALE, mean_abs_gain * 3.0)
    confidence = min(1.0, len(gains) / max(1, int(lookback)))
    amplitude_delta = clamp(gain_signal / gain_scale, -0.45, 0.60) * confidence
    return clamp(1.0 + amplitude_delta, AMP_MIN, AMP_MAX)


def materialize_candidate(spec: CandidateSpec, amplitude: float) -> CandidatePlan:
    amplitude = float(clamp(amplitude, AMP_MIN, AMP_MAX))

    def _apply(cfg: dict) -> None:
        spec.apply(cfg, amplitude)

    return CandidatePlan(
        name=spec.name,
        amplitude=amplitude,
        description=f"{spec.description} | amp={amplitude:.2f}x",
        apply=_apply,
    )


def _enforce_full_method(cfg: dict) -> None:
    set_nested(cfg, "experiment.enable_kcm", True)
    set_nested(cfg, "environment.lookahead_obs_enabled", True)
    set_nested(cfg, "environment.lookahead_points", max(8, int(get_nested(cfg, "environment.lookahead_points", 8) or 8)))
    set_nested(cfg, "reward_weights.lookahead_control.enabled", True)
    set_nested(cfg, "reward_weights.lookahead_control.policy_action", True)
    set_nested(cfg, "reward_weights.cornerness.enabled", True)
    set_nested(cfg, "reward_weights.lookahead_reward.enabled", True)
    set_nested(cfg, "experiment.paper_variant", "")
    set_nested(cfg, "experiment.paper_label", "")
    set_nested(cfg, "experiment.paper_description", "")


def _resume_checkpoint_compatible(parent_cfg: Mapping[str, object], child_cfg: Mapping[str, object]) -> bool:
    parent_policy_action = bool(get_nested(parent_cfg, "reward_weights.lookahead_control.policy_action", False))
    child_policy_action = bool(get_nested(child_cfg, "reward_weights.lookahead_control.policy_action", False))
    if parent_policy_action != child_policy_action:
        return False
    parent_lookahead_enabled = bool(get_nested(parent_cfg, "reward_weights.lookahead_control.enabled", False))
    child_lookahead_enabled = bool(get_nested(child_cfg, "reward_weights.lookahead_control.enabled", False))
    if parent_lookahead_enabled != child_lookahead_enabled:
        return False
    return True


def choose_candidate_batch(
    iteration: int,
    has_best: bool,
    history: Sequence[dict],
    batch_size: int,
    amp_lookback: int,
) -> list[CandidatePlan]:
    specs = candidate_specs()
    batch_size = max(1, int(batch_size))
    if not has_best:
        return [materialize_candidate(specs[0], 1.0)]

    history_stats = summarize_candidate_history(history)
    pool = specs[1:] if len(specs) > 1 else specs
    order = {spec.name: idx for idx, spec in enumerate(pool)}

    def _score_key(spec: CandidateSpec) -> tuple:
        stat = history_stats.get(spec.name, {})
        best_score = float(stat.get("best_score", float("-inf")))
        if best_score == float("-inf"):
            best_score = -1e18
        return (
            int(stat.get("recent_non_keep_streak", 0)),
            -float(stat.get("keep_rate", 0.0)),
            -float(stat.get("ok_rate", 0.0)),
            -best_score,
            int(stat.get("tries", 0)),
            order.get(spec.name, 999),
        )

    untried = [spec for spec in pool if int(history_stats.get(spec.name, {}).get("tries", 0)) == 0]
    tried = sorted([spec for spec in pool if spec not in untried], key=_score_key)

    selected_specs: list[CandidateSpec] = []
    if untried:
        selected_specs.extend(untried[:batch_size])
        remaining_slots = batch_size - len(selected_specs)
        if remaining_slots > 0:
            selected_specs.extend(tried[:remaining_slots])
    else:
        if not tried:
            tried = pool
        top_window = tried[: max(batch_size, min(6, len(tried)))]
        start = int(iteration) % max(1, len(top_window))
        for offset in range(batch_size):
            selected_specs.append(top_window[(start + offset) % len(top_window)])

    unique_specs: list[CandidateSpec] = []
    seen: set[str] = set()
    for spec in selected_specs:
        if spec.name in seen:
            continue
        unique_specs.append(spec)
        seen.add(spec.name)

    plans = [materialize_candidate(spec, compute_candidate_amplitude(spec, history, amp_lookback)) for spec in unique_specs]
    return plans[:batch_size]


def should_keep(result: ExperimentResult, best_state: dict, score_epsilon: float) -> bool:
    if result.status != "ok":
        return False

    best_pass = int(best_state.get("pass_count", -1))
    best_score = float(best_state.get("score", float("-inf")))
    best_success = float(best_state.get("mean_success_rate", 0.0))
    best_progress = float(best_state.get("mean_progress_final", 0.0))
    best_stall = float(best_state.get("mean_stall_rate", 1.0))
    best_time = float(best_state.get("mean_completion_time_seconds", 999999.0))

    if result.pass_count > best_pass:
        return True
    if result.pass_count < best_pass:
        return False
    if result.score > best_score + score_epsilon:
        return True
    if abs(result.score - best_score) <= score_epsilon:
        if result.mean_completion_time_seconds < best_time - 1e-6:
            return True
        if result.mean_success_rate > best_success + 1e-6:
            return True
        if (
            abs(result.mean_success_rate - best_success) <= 1e-6
            and result.mean_progress_final > best_progress + 1e-6
        ):
            return True
        if (
            abs(result.mean_success_rate - best_success) <= 1e-6
            and abs(result.mean_progress_final - best_progress) <= 1e-6
            and result.mean_stall_rate < best_stall - 1e-6
        ):
            return True
    return False


def compute_total_episodes(resume_checkpoint: Path | None, extra_episodes: int) -> int:
    extra = max(1, int(extra_episodes))
    if resume_checkpoint is None:
        return extra
    base_episode = load_checkpoint_episode(resume_checkpoint) + 1
    return max(base_episode + extra, base_episode + 1)


def select_screen_paths(path_specs: Sequence[dict], raw_screen_paths: str | None) -> list[dict]:
    if not path_specs:
        return []

    if raw_screen_paths:
        requested = [item.strip() for item in str(raw_screen_paths).split(",") if item.strip()]
        requested_set = set(requested)
        selected = [dict(path_cfg) for path_cfg in path_specs if str(path_cfg.get("name") or path_cfg.get("type")) in requested_set]
        if selected:
            return selected

    preferred = ["square", "s_shape", "circle", "trapezoid", "butterfly"]
    selected: list[dict] = []
    seen: set[str] = set()
    by_name = {str(path_cfg.get("name") or path_cfg.get("type")): dict(path_cfg) for path_cfg in path_specs}
    for name in preferred:
        if name in by_name and name not in seen:
            selected.append(by_name[name])
            seen.add(name)
        if len(selected) >= min(3, len(path_specs)):
            break

    if not selected:
        selected = [dict(path_cfg) for path_cfg in path_specs[: min(3, len(path_specs))]]
    return selected


def rank_stage1_results(results: Sequence[ExperimentResult]) -> list[ExperimentResult]:
    ok_results = [result for result in results if result.status == "ok"]
    return sorted(
        ok_results,
        key=lambda result: (
            -float(result.score),
            -int(result.pass_count),
            -float(result.mean_success_rate),
            -float(result.mean_progress_final),
            float(result.mean_stall_rate),
        ),
    )


def _write_experiment_summary(
    result: ExperimentResult,
    *,
    metadata: dict,
    evaluation_label: str,
    evaluation_payload: dict | None = None,
    error: str | None = None,
    trace_text: str | None = None,
) -> None:
    payload = {
        "result": result.__dict__,
        "metadata": metadata,
        "evaluation_label": evaluation_label,
    }
    if evaluation_payload is not None:
        payload["evaluation"] = evaluation_payload
    if error is not None:
        payload["error"] = error
    if trace_text is not None:
        payload["traceback"] = trace_text
    write_json(Path(result.run_dir) / "experiment_summary.json", payload)


def run_single_experiment(
    *,
    iteration: int,
    candidate: CandidatePlan,
    parent_config_path: Path,
    parent_state: dict,
    args: argparse.Namespace,
    path_specs: Sequence[dict],
    eval_episodes: int,
    evaluation_label: str,
    export_rollouts: bool,
) -> ExperimentResult:
    experiment_id = next_experiment_id(iteration, candidate.name)
    run_dir = RUNS_DIR / experiment_id
    config_path = run_dir / "config.yaml"
    started_at = now_text()
    git_head = get_git_head()

    resume_checkpoint = latest_checkpoint_from_state(parent_state)
    total_episodes = compute_total_episodes(resume_checkpoint, args.extra_episodes)

    base_config = load_yaml(parent_config_path)
    config = build_train_config(
        base_config,
        experiment_name=experiment_id,
        path_specs=path_specs,
        total_episodes=total_episodes,
        time_budget_seconds=args.time_budget_seconds,
        seed=args.seed,
    )
    candidate.apply(config)
    _enforce_full_method(config)

    if bool(get_nested(config, "experiment.force_fresh_start", False)):
        resume_checkpoint = None

    if resume_checkpoint is not None and not _resume_checkpoint_compatible(base_config, config):
        resume_checkpoint = None

    mix_gain = config.get("reward_weights", {}).get("lookahead_control", {}).get("mix_gain")
    if mix_gain is not None:
        set_nested(config, "reward_weights.lookahead_control.mix_gain", clamp(float(mix_gain), 0.10, 0.95))

    metadata = {
        "experiment_id": experiment_id,
        "candidate": candidate.name,
        "candidate_amplitude": float(candidate.amplitude),
        "description": candidate.description,
        "parent_experiment_id": str(parent_state.get("experiment_id", "")),
        "parent_run_dir": str(parent_state.get("run_dir", "")),
        "git_head": git_head,
        "created_at": started_at,
    }
    config["autoresearch"] = metadata
    write_yaml(config_path, config)

    eval_summary_path = run_dir / "evaluation" / evaluation_label / "summary.json"
    rollouts_summary_path = run_dir / "best_rollouts" / "summary.json"

    result = ExperimentResult(
        experiment_id=experiment_id,
        candidate=candidate.name,
        parent_experiment_id=str(parent_state.get("experiment_id", "")),
        status="failed",
        keep=False,
        score=float("-inf"),
        pass_count=0,
        mean_success_rate=0.0,
        mean_progress_final=0.0,
        mean_stall_rate=1.0,
        mean_error_ratio=999.0,
        max_error_ratio=999.0,
        mean_completion_time_seconds=999999.0,
        max_completion_time_seconds=999999.0,
        git_head=git_head,
        description=candidate.description,
        config_path=str(config_path),
        run_dir=str(run_dir),
        model_path="",
        latest_checkpoint="",
        eval_summary_path=str(eval_summary_path),
        rollouts_summary_path=str(rollouts_summary_path if export_rollouts else ""),
        started_at=started_at,
        finished_at=started_at,
    )

    try:
        train_candidate(
            config_path=config_path,
            run_dir=run_dir,
            conda_env=args.conda_env,
            resume_path=resume_checkpoint,
            timeout_seconds=args.process_timeout_seconds,
        )
        model_path = find_model_checkpoint(run_dir)
        latest_checkpoint = find_latest_checkpoint(run_dir)
        eval_payload = evaluate_model_across_paths(
            trained_config_path=run_dir / "config.yaml",
            model_path=model_path,
            out_dir=run_dir / "evaluation" / evaluation_label,
            path_specs=path_specs,
            episodes=eval_episodes,
            deterministic=args.deterministic_eval,
            seed=args.eval_seed,
            conda_env=args.conda_env,
            score_profile=evaluation_label,
        )
        aggregated = eval_payload["aggregated"]
        result.status = "ok"
        result.score = float(aggregated["score"])
        result.pass_count = int(aggregated["pass_count"])
        result.mean_success_rate = float(aggregated["mean_success_rate"])
        result.mean_progress_final = float(aggregated["mean_progress_final"])
        result.mean_stall_rate = float(aggregated["mean_stall_rate"])
        result.mean_error_ratio = float(aggregated["mean_error_ratio"])
        result.max_error_ratio = float(aggregated["max_error_ratio"])
        result.mean_completion_time_seconds = float(aggregated.get("mean_completion_time_seconds", 999999.0))
        result.max_completion_time_seconds = float(aggregated.get("max_completion_time_seconds", 999999.0))
        result.model_path = str(model_path)
        result.latest_checkpoint = str(latest_checkpoint)
        result.eval_summary_path = str(run_dir / "evaluation" / evaluation_label / "summary.json")
        if export_rollouts:
            export_summary = export_best_rollouts(
                config_path=run_dir / "config.yaml",
                run_dir=run_dir,
                out_dir=run_dir / "best_rollouts",
                conda_env=args.conda_env,
            )
            result.rollouts_summary_path = str(export_summary)

        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label=evaluation_label,
            evaluation_payload=eval_payload,
        )
    except Exception as exc:
        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label=evaluation_label,
            error=str(exc),
            trace_text=traceback.format_exc(),
        )

    result.finished_at = now_text()
    return result


def refine_experiment_result(
    result: ExperimentResult,
    *,
    args: argparse.Namespace,
    path_specs: Sequence[dict],
    eval_episodes: int,
    evaluation_label: str,
    export_rollouts: bool,
) -> ExperimentResult:
    run_dir = Path(result.run_dir)
    metadata = {
        "experiment_id": result.experiment_id,
        "candidate": result.candidate,
        "description": result.description,
        "refined_at": now_text(),
        "evaluation_label": evaluation_label,
    }

    try:
        model_path = Path(result.model_path) if result.model_path else find_model_checkpoint(run_dir)
        latest_checkpoint = Path(result.latest_checkpoint) if result.latest_checkpoint else find_latest_checkpoint(run_dir)
        eval_payload = evaluate_model_across_paths(
            trained_config_path=Path(result.config_path),
            model_path=model_path,
            out_dir=run_dir / "evaluation" / evaluation_label,
            path_specs=path_specs,
            episodes=eval_episodes,
            deterministic=args.deterministic_eval,
            seed=args.eval_seed,
            conda_env=args.conda_env,
            score_profile=evaluation_label,
        )
        aggregated = eval_payload["aggregated"]
        result.status = "ok"
        result.score = float(aggregated["score"])
        result.pass_count = int(aggregated["pass_count"])
        result.mean_success_rate = float(aggregated["mean_success_rate"])
        result.mean_progress_final = float(aggregated["mean_progress_final"])
        result.mean_stall_rate = float(aggregated["mean_stall_rate"])
        result.mean_error_ratio = float(aggregated["mean_error_ratio"])
        result.max_error_ratio = float(aggregated["max_error_ratio"])
        result.mean_completion_time_seconds = float(aggregated.get("mean_completion_time_seconds", 999999.0))
        result.max_completion_time_seconds = float(aggregated.get("max_completion_time_seconds", 999999.0))
        result.model_path = str(model_path)
        result.latest_checkpoint = str(latest_checkpoint)
        result.eval_summary_path = str(run_dir / "evaluation" / evaluation_label / "summary.json")
        if export_rollouts:
            export_summary = export_best_rollouts(
                config_path=Path(result.config_path),
                run_dir=run_dir,
                out_dir=run_dir / "best_rollouts",
                conda_env=args.conda_env,
            )
            result.rollouts_summary_path = str(export_summary)

        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label=evaluation_label,
            evaluation_payload=eval_payload,
        )
    except Exception as exc:
        result.status = "failed"
        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label=evaluation_label,
            error=str(exc),
            trace_text=traceback.format_exc(),
        )

    result.finished_at = now_text()
    return result


def upgrade_promising_experiment(
    result: ExperimentResult,
    *,
    args: argparse.Namespace,
    bonus_episodes: int,
    bonus_time_budget_seconds: float,
    bonus_process_timeout_seconds: float,
) -> ExperimentResult:
    run_dir = Path(result.run_dir)
    config_path = Path(result.config_path)
    config = load_yaml(config_path)
    latest_checkpoint = Path(result.latest_checkpoint) if result.latest_checkpoint else find_latest_checkpoint(run_dir)
    metadata = {
        "experiment_id": result.experiment_id,
        "candidate": result.candidate,
        "description": result.description,
        "upgraded_at": now_text(),
        "bonus_episodes": int(bonus_episodes),
        "bonus_time_budget_seconds": float(bonus_time_budget_seconds),
    }

    training_cfg = config.setdefault("training", {})
    current_total = int(training_cfg.get("num_episodes", 0) or 0)
    training_cfg["num_episodes"] = max(current_total + max(1, int(bonus_episodes)), current_total + 1)
    current_budget = float(training_cfg.get("time_budget_seconds", 0.0) or 0.0)
    training_cfg["time_budget_seconds"] = max(current_budget, float(bonus_time_budget_seconds))
    write_yaml(config_path, config)

    try:
        train_candidate(
            config_path=config_path,
            run_dir=run_dir,
            conda_env=args.conda_env,
            resume_path=latest_checkpoint,
            timeout_seconds=float(bonus_process_timeout_seconds),
        )
        result.model_path = str(find_model_checkpoint(run_dir))
        result.latest_checkpoint = str(find_latest_checkpoint(run_dir))
        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label="upgrade_train",
        )
    except Exception as exc:
        result.status = "failed"
        _write_experiment_summary(
            result,
            metadata=metadata,
            evaluation_label="upgrade_train",
            error=str(exc),
            trace_text=traceback.format_exc(),
        )
    result.finished_at = now_text()
    return result


def should_upgrade_candidate(
    result: ExperimentResult,
    *,
    ranked_index: int,
    args: argparse.Namespace,
    screen_path_count: int,
) -> bool:
    if result.status != "ok":
        return False
    if ranked_index >= max(0, int(args.upgrade_top_k)):
        return False
    if int(args.upgrade_top_k) <= 0:
        return False
    required_pass = max(1, min(screen_path_count, int(args.upgrade_min_pass_count)))
    if result.pass_count >= required_pass:
        return True
    if result.mean_progress_final >= float(args.upgrade_progress_threshold):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous experiment loop for PPO trajectory smoothing.")
    parser.add_argument("--setup-only", action="store_true", help="仅初始化工作区，不运行实验")
    parser.add_argument("--max-experiments", type=int, default=1, help="最多运行多少个候选实验；0 表示无限循环")
    parser.add_argument("--candidate-batch-size", type=int, default=3, help="每一轮先训练多少个候选")
    parser.add_argument("--screen-top-k", type=int, default=1, help="粗筛后进入复评估的 top-k 候选数")
    parser.add_argument("--screen-eval-episodes", type=int, default=1, help="粗筛阶段每条路径的评测 episode 数")
    parser.add_argument("--screen-paths", type=str, default="", help="粗筛阶段使用的路径名列表，逗号分隔；为空则自动选子集")
    parser.add_argument("--amp-lookback", type=int, default=4, help="按最近 N 次完整评测的真实得分增益缩放候选 amp")
    parser.add_argument("--extra-episodes", type=int, default=40, help="每轮在父代基础上追加的 episode 数")
    parser.add_argument("--time-budget-seconds", type=float, default=900.0, help="每轮训练的墙钟时间预算")
    parser.add_argument("--process-timeout-seconds", type=float, default=7200.0, help="单个训练子进程超时")
    parser.add_argument("--eval-episodes", type=int, default=8, help="复评估阶段每条路径的评测 episode 数")
    parser.add_argument("--upgrade-top-k", type=int, default=1, help="对 stage1 前 k 个有潜力的候选追加长训")
    parser.add_argument("--upgrade-min-pass-count", type=int, default=2, help="触发补训的最小通过路径数")
    parser.add_argument("--upgrade-progress-threshold", type=float, default=0.985, help="触发补训的最小均值末进度")
    parser.add_argument("--upgrade-extra-episodes", type=int, default=80, help="补训时额外追加的 episode 数")
    parser.add_argument("--upgrade-time-budget-seconds", type=float, default=1200.0, help="补训阶段的训练时间预算")
    parser.add_argument("--upgrade-process-timeout-seconds", type=float, default=5400.0, help="补训阶段的进程超时")
    parser.add_argument("--eval-seed", type=int, default=43, help="统一评测随机种子")
    parser.add_argument("--seed", type=int, default=42, help="训练配置中写入的基础随机种子")
    parser.add_argument("--paths", type=str, default="square,circle,butterfly", help="逗号分隔路径列表")
    parser.add_argument("--conda-env", type=str, default="PPO", help="训练与评测使用的 conda 环境名")
    parser.add_argument("--deterministic-eval", action="store_true", help="评测时使用确定性策略")
    parser.add_argument("--score-epsilon", type=float, default=1e-6, help="分数比较容差")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = ensure_workspace()
    history = read_results_history()
    refresh_workspace_reports(history, load_current_best_state())
    if args.setup_only:
        print(f"[SETUP] workspace initialized: {workspace}")
        return 0

    path_names = [item.strip() for item in args.paths.split(",") if item.strip()]
    path_specs = build_selected_paths(path_names)
    screen_path_specs = select_screen_paths(path_specs, args.screen_paths)

    remaining_runs = None if int(args.max_experiments) == 0 else int(args.max_experiments)
    experiment_counter = len(history)

    while remaining_runs is None or remaining_runs > 0:
        best_state = load_current_best_state()
        has_best = bool(str(best_state.get("experiment_id", "")).strip())
        parent_config_path = CURRENT_BEST_CONFIG

        batch_size = max(1, int(args.candidate_batch_size))
        if remaining_runs is not None:
            batch_size = min(batch_size, remaining_runs)

        candidate_batch = choose_candidate_batch(
            experiment_counter,
            has_best=has_best,
            history=history,
            batch_size=batch_size,
            amp_lookback=int(args.amp_lookback),
        )

        print(
            "[BATCH] count={count} screen_paths={screen_paths} stage2_top_k={top_k}".format(
                count=len(candidate_batch),
                screen_paths=",".join(str(item.get("name") or item.get("type")) for item in screen_path_specs),
                top_k=int(args.screen_top_k),
            )
        )

        stage1_results: list[ExperimentResult] = []
        for batch_offset, candidate in enumerate(candidate_batch):
            result = run_single_experiment(
                iteration=experiment_counter + batch_offset,
                candidate=candidate,
                parent_config_path=parent_config_path,
                parent_state=best_state,
                args=args,
                path_specs=screen_path_specs,
                eval_episodes=int(args.screen_eval_episodes),
                evaluation_label="stage1",
                export_rollouts=False,
            )
            stage1_results.append(result)

        ranked_stage1 = rank_stage1_results(stage1_results)
        top_k = max(0, min(int(args.screen_top_k), len(ranked_stage1))) if ranked_stage1 else 0
        finalist_ids = {result.experiment_id for result in ranked_stage1[:top_k]}
        ranked_index_by_id = {item.experiment_id: idx for idx, item in enumerate(ranked_stage1)}

        for result in stage1_results:
            if result.experiment_id in finalist_ids and result.status == "ok":
                if should_upgrade_candidate(
                    result,
                    ranked_index=int(ranked_index_by_id.get(result.experiment_id, 999)),
                    args=args,
                    screen_path_count=len(screen_path_specs),
                ):
                    result = upgrade_promising_experiment(
                        result,
                        args=args,
                        bonus_episodes=int(args.upgrade_extra_episodes),
                        bonus_time_budget_seconds=float(args.upgrade_time_budget_seconds),
                        bonus_process_timeout_seconds=float(args.upgrade_process_timeout_seconds),
                    )
                result = refine_experiment_result(
                    result,
                    args=args,
                    path_specs=path_specs,
                    eval_episodes=int(args.eval_episodes),
                    evaluation_label="stage2",
                    export_rollouts=True,
                )
            elif result.status == "ok":
                result.status = "screened_out"
                result.keep = False
                result.finished_at = now_text()

            current_best = load_current_best_state()
            result.keep = should_keep(result, current_best, score_epsilon=float(args.score_epsilon))
            if result.keep:
                promote_candidate(Path(result.config_path), result)
                archive_promoted_result(result)

            upsert_result_row(result)
            history = read_results_history()
            refresh_workspace_reports(history, load_current_best_state())
            print(
                "[ITER] id={id} candidate={candidate} status={status} keep={keep} score={score:.3f} pass={pass_count}".format(
                    id=result.experiment_id,
                    candidate=result.candidate,
                    status=result.status,
                    keep=result.keep,
                    score=float(result.score),
                    pass_count=int(result.pass_count),
                )
            )

        experiment_counter += len(candidate_batch)
        if remaining_runs is not None:
            remaining_runs -= len(candidate_batch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
