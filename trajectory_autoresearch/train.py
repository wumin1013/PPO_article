from __future__ import annotations

import argparse
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
    read_results_history,
    refresh_workspace_reports,
    load_yaml,
    mul_nested,
    next_experiment_id,
    now_text,
    promote_candidate,
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
    apply: Callable[[dict], None]


def candidate_specs() -> list[CandidateSpec]:
    def baseline(cfg: dict) -> None:
        return

    def lookahead_aggressive(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", 1.12, low=0.35, high=0.90)
        mul_nested(cfg, "reward_weights.lookahead_control.corner_dist", 1.12, low=1.4, high=4.5)
        mul_nested(cfg, "reward_weights.lookahead_reward.corner_target", 1.08, low=0.45, high=0.95)

    def lookahead_conservative(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.lookahead_control.mix_gain", 0.88, low=0.25, high=0.90)
        mul_nested(cfg, "reward_weights.lookahead_control.straight_dist", 1.12, low=0.8, high=3.0)
        mul_nested(cfg, "reward_weights.lookahead_reward.w_straight", 1.20, low=0.1, high=2.5)

    def smoother_corner(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.cornerness.w_track_min", 1.30, low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.cornerness.w_smooth0", 1.25, low=0.2, high=8.0)
        mul_nested(cfg, "reward_weights.p8.ang_cap_min_ratio", 1.15, low=0.05, high=0.30)

    def stall_strict(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.p4.stall_steps", 0.75, low=200.0, high=5000.0)
        mul_nested(cfg, "reward_weights.p4.stall_progress_eps", 2.00, low=1e-8, high=1e-2)
        mul_nested(cfg, "reward_weights.p4.stall_v_eps", 1.50, low=1e-4, high=0.5)

    def stall_relaxed(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.p4.stall_steps", 1.25, low=200.0, high=5000.0)
        mul_nested(cfg, "reward_weights.p4.stall_progress_eps", 0.60, low=1e-8, high=1e-2)
        mul_nested(cfg, "reward_weights.p4.stall_v_eps", 0.80, low=1e-4, high=0.5)

    def exploration_low(cfg: dict) -> None:
        mul_nested(cfg, "ppo.ent_coef", 0.50, low=0.0, high=0.05)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", max(4, epochs - 2))
        mul_nested(cfg, "reward_weights.control_authority.tangent_blend", 1.20, low=0.0, high=0.30)

    def exploration_high(cfg: dict) -> None:
        mul_nested(cfg, "ppo.ent_coef", 2.00, low=0.0, high=0.05)
        epochs = int(cfg.get("ppo", {}).get("epochs", 6))
        set_nested(cfg, "ppo.epochs", min(14, epochs + 2))
        mul_nested(cfg, "reward_weights.control_authority.tangent_blend", 0.85, low=0.0, high=0.30)

    def critic_faster(cfg: dict) -> None:
        mul_nested(cfg, "ppo.critic_lr", 1.20, low=5e-5, high=5e-4)
        mul_nested(cfg, "ppo.actor_lr", 0.95, low=1e-5, high=5e-4)
        mul_nested(cfg, "ppo.lmbda", 1.02, low=0.90, high=0.99)

    def smoother_actions(cfg: dict) -> None:
        mul_nested(cfg, "reward_weights.w_smooth", 1.30, low=0.02, high=1.50)
        mul_nested(cfg, "reward_weights.p6_1.w_du", 1.50, low=1e-4, high=0.20)
        mul_nested(cfg, "reward_weights.p6_1.v_target_tau", 1.10, low=0.02, high=0.50)

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


def choose_candidate(iteration: int, has_best: bool, history: Sequence[dict]) -> CandidateSpec:
    specs = candidate_specs()
    if not has_best:
        return specs[0]

    history_stats = summarize_candidate_history(history)
    pool = specs[1:] if len(specs) > 1 else specs

    untried = [spec for spec in pool if int(history_stats.get(spec.name, {}).get("tries", 0)) == 0]
    if untried:
        return untried[0]

    order = {spec.name: idx for idx, spec in enumerate(specs)}

    def _score_key(spec: CandidateSpec) -> tuple:
        stat = history_stats.get(spec.name, {})
        best_score = float(stat.get("best_score", float("-inf")))
        if best_score == float("-inf"):
            best_score = -1e18
        return (
            int(stat.get("recent_non_keep_streak", 0)),
            -int(stat.get("keep_count", 0)),
            -float(stat.get("ok_rate", 0.0)),
            -best_score,
            int(stat.get("tries", 0)),
            order.get(spec.name, 999),
        )

    ranked = sorted(pool, key=_score_key)
    top_n = max(1, min(3, len(ranked)))
    chosen = ranked[int(iteration) % top_n]

    if history:
        last_candidate = str(history[-1].get("candidate", "")).strip()
        if chosen.name == last_candidate and top_n > 1:
            chosen = ranked[(int(iteration) + 1) % top_n]
    return chosen


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


def run_single_experiment(
    *,
    iteration: int,
    candidate: CandidateSpec,
    parent_config_path: Path,
    parent_state: dict,
    args: argparse.Namespace,
    path_specs: Sequence[dict],
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
        "description": candidate.description,
        "parent_experiment_id": str(parent_state.get("experiment_id", "")),
        "parent_run_dir": str(parent_state.get("run_dir", "")),
        "git_head": git_head,
        "created_at": started_at,
    }
    config["autoresearch"] = metadata
    write_yaml(config_path, config)

    eval_summary_path = run_dir / "evaluation" / "summary.json"
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
        rollouts_summary_path=str(rollouts_summary_path),
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
            out_dir=run_dir / "evaluation",
            path_specs=path_specs,
            episodes=args.eval_episodes,
            deterministic=args.deterministic_eval,
            seed=args.eval_seed,
            conda_env=args.conda_env,
        )
        export_summary = export_best_rollouts(
            config_path=run_dir / "config.yaml",
            run_dir=run_dir,
            out_dir=run_dir / "best_rollouts",
            conda_env=args.conda_env,
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
        result.eval_summary_path = str(run_dir / "evaluation" / "summary.json")
        result.rollouts_summary_path = str(export_summary)

        write_json(
            run_dir / "experiment_summary.json",
            {
                "result": result.__dict__,
                "evaluation": eval_payload,
                "metadata": metadata,
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "experiment_summary.json",
            {
                "result": result.__dict__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "metadata": metadata,
            },
        )

    result.finished_at = now_text()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous experiment loop for PPO trajectory smoothing.")
    parser.add_argument("--setup-only", action="store_true", help="仅初始化工作区，不运行实验")
    parser.add_argument("--max-experiments", type=int, default=1, help="运行多少轮实验；0 表示无限循环")
    parser.add_argument("--extra-episodes", type=int, default=40, help="每轮在父代基础上追加的 episode 数")
    parser.add_argument("--time-budget-seconds", type=float, default=900.0, help="每轮训练的墙钟时间预算")
    parser.add_argument("--process-timeout-seconds", type=float, default=7200.0, help="单个训练子进程超时")
    parser.add_argument("--eval-episodes", type=int, default=8, help="每条路径的评测 episode 数")
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

    remaining = None if int(args.max_experiments) == 0 else int(args.max_experiments)
    iteration = len(history)
    while remaining is None or remaining != 0:
        best_state = load_current_best_state()
        has_best = bool(str(best_state.get("experiment_id", "")).strip())
        parent_config_path = CURRENT_BEST_CONFIG
        candidate = choose_candidate(iteration, has_best=has_best, history=history)

        result = run_single_experiment(
            iteration=iteration,
            candidate=candidate,
            parent_config_path=parent_config_path,
            parent_state=best_state,
            args=args,
            path_specs=path_specs,
        )
        result.keep = should_keep(result, best_state, score_epsilon=float(args.score_epsilon))
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

        iteration += 1
        if remaining is not None and remaining > 0:
            remaining -= 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
