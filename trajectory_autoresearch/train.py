from __future__ import annotations

import argparse
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from prepare import (
    CURRENT_BEST_CONFIG,
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


AMP_MIN = 0.55
AMP_MAX = 1.60
AMP_SCORE_GAIN_SCALE = 30.0


def _scale_up(base_delta: float, amplitude: float) -> float:
    return 1.0 + float(base_delta) * float(amplitude)


def _scale_down(base_delta: float, amplitude: float) -> float:
    return max(0.05, 1.0 - float(base_delta) * float(amplitude))


def _int_delta(base_delta: int, amplitude: float) -> int:
    return max(1, int(round(float(base_delta) * max(0.5, float(amplitude)))))


def candidate_specs() -> list[CandidateSpec]:
    def baseline(cfg: dict, amplitude: float) -> None:
        return

    def lookahead_aggressive(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_up(0.12, amplitude), low=0.35, high=0.90)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", _scale_up(0.12, amplitude), low=1.4, high=4.5)
        mul_nested(cfg, "reward_weights.lookahead_reward.corner_target", _scale_up(0.08, amplitude), low=0.45, high=0.95)

    def lookahead_conservative(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", _scale_down(0.12, amplitude), low=0.25, high=0.90)
        mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", _scale_up(0.12, amplitude), low=0.8, high=3.0)
        mul_nested(cfg, "reward_weights.lookahead_reward.w_straight", _scale_up(0.20, amplitude), low=0.1, high=2.5)

    def smoother_corner(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.cornerness.w_track_min", _scale_up(0.30, amplitude), low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", _scale_up(0.25, amplitude), low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", _scale_up(0.15, amplitude), low=0.05, high=0.30)

    def stall_strict(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.p4.stall_steps", _scale_down(0.25, amplitude), low=200.0, high=5000.0)
        mul_nested(cfg, "reward_weights.p4.stall_progress_eps", _scale_up(1.00, amplitude), low=1e-8, high=1e-2)
        mul_nested(cfg, "reward_weights.p4.stall_v_eps", _scale_up(0.50, amplitude), low=1e-4, high=0.5)

    def stall_relaxed(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.p4.stall_steps", _scale_up(0.25, amplitude), low=200.0, high=5000.0)
        mul_nested(cfg, "reward_weights.p4.stall_progress_eps", _scale_down(0.40, amplitude), low=1e-8, high=1e-2)
        mul_nested(cfg, "reward_weights.p4.stall_v_eps", _scale_down(0.20, amplitude), low=1e-4, high=0.5)

    def exploration_low(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.ent_coef", _scale_down(0.50, amplitude), low=0.0, high=0.05)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", max(4, epochs - _int_delta(2, amplitude)))
        mul_nested(cfg, "reward_weights.control_authority.tangent_blend", _scale_up(0.20, amplitude), low=0.0, high=0.30)

    def exploration_high(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.ent_coef", _scale_up(1.00, amplitude), low=0.0, high=0.05)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", min(14, epochs + _int_delta(2, amplitude)))
        mul_nested(cfg, "reward_weights.control_authority.tangent_blend", _scale_down(0.15, amplitude), low=0.0, high=0.30)

    def critic_faster(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "ppo.critic_lr", _scale_up(0.20, amplitude), low=5e-5, high=5e-4)
        mul_nested(cfg, "ppo.actor_lr", _scale_down(0.05, amplitude), low=1e-5, high=5e-4)
        mul_nested(cfg, "ppo.lmbda", _scale_up(0.02, amplitude), low=0.90, high=0.99)

    def smoother_actions(cfg: dict, amplitude: float) -> None:
        mul_nested(cfg, "reward_weights.w_smooth", _scale_up(0.30, amplitude), low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p6_1.w_du", _scale_up(0.50, amplitude), low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.p6_1.v_target_tau", _scale_up(0.10, amplitude), low=0.02, high=0.50)

    return [
        CandidateSpec("baseline", "保留当前最优配置，建立或刷新基线", baseline),
        CandidateSpec("lookahead_aggressive", "增强转角前瞻", lookahead_aggressive),
        CandidateSpec("lookahead_conservative", "增强直线稳定性", lookahead_conservative),
        CandidateSpec("smoother_corner", "强化角区平滑", smoother_corner),
        CandidateSpec("stall_strict", "更早打断 stall", stall_strict),
        CandidateSpec("stall_relaxed", "放宽 stall 触发", stall_relaxed),
        CandidateSpec("exploration_low", "减小探索，偏向收敛", exploration_low),
        CandidateSpec("exploration_high", "增大探索，争取跳出局部最优", exploration_high),
        CandidateSpec("critic_faster", "提高 critic 拟合速度", critic_faster),
        CandidateSpec("smoother_actions", "加大动作平滑权重", smoother_actions),
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
        top_window = tried[: max(batch_size, min(4, len(tried)))]
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

    if result.pass_count > best_pass:
        return True
    if result.pass_count < best_pass:
        return False
    if result.score > best_score + score_epsilon:
        return True
    if abs(result.score - best_score) <= score_epsilon:
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
    parser.add_argument("--eval-seed", type=int, default=43, help="统一评测随机种子")
    parser.add_argument("--seed", type=int, default=42, help="训练配置中写入的基础随机种子")
    parser.add_argument("--paths", type=str, default="square,s_shape,butterfly,trapezoid,circle", help="逗号分隔路径列表")
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
        top_k = max(1, min(int(args.screen_top_k), len(ranked_stage1))) if ranked_stage1 else 0
        finalist_ids = {result.experiment_id for result in ranked_stage1[:top_k]}

        for result in stage1_results:
            if result.experiment_id in finalist_ids and result.status == "ok":
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

            append_result_row(result)
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
