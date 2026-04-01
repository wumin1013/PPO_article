from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from prepare import (
    CURRENT_BEST_CONFIG,
    RESEARCH_ROOT,
    archive_promoted_result,
    build_selected_paths,
    load_current_best_state,
    now_text,
    promote_candidate,
    read_results_history,
    refresh_workspace_reports,
    upsert_result_row,
    write_json,
)
from train import refine_experiment_result, should_keep, should_upgrade_candidate, upgrade_promising_experiment


STATE_PATH = RESEARCH_ROOT / "workspace" / "refine_worker_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel long-train worker for promising stage1 candidates.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--conda-env", type=str, default="PPO")
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--paths", type=str, default="square,circle,butterfly")
    parser.add_argument("--deadline-time", type=str, default="")
    parser.add_argument("--upgrade-top-k", type=int, default=2)
    parser.add_argument("--upgrade-min-pass-count", type=int, default=2)
    parser.add_argument("--upgrade-progress-threshold", type=float, default=0.985)
    parser.add_argument("--upgrade-extra-episodes", type=int, default=120)
    parser.add_argument("--upgrade-time-budget-seconds", type=float, default=1800.0)
    parser.add_argument("--upgrade-process-timeout-seconds", type=float, default=10800.0)
    parser.add_argument("--score-epsilon", type=float, default=1e-6)
    return parser.parse_args()


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {"processed": {}, "updated_at": ""}
    import json

    return json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))


def _save_state(state: dict) -> None:
    state["updated_at"] = now_text()
    write_json(STATE_PATH, state)


def _args_like(base: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        conda_env=base.conda_env,
        deterministic_eval=bool(base.deterministic_eval),
        eval_seed=int(base.eval_seed),
        eval_episodes=int(base.eval_episodes),
        upgrade_top_k=int(base.upgrade_top_k),
        upgrade_min_pass_count=int(base.upgrade_min_pass_count),
        upgrade_progress_threshold=float(base.upgrade_progress_threshold),
        upgrade_extra_episodes=int(base.upgrade_extra_episodes),
        upgrade_time_budget_seconds=float(base.upgrade_time_budget_seconds),
        upgrade_process_timeout_seconds=float(base.upgrade_process_timeout_seconds),
        score_epsilon=float(base.score_epsilon),
    )


def _deadline_reached(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    return datetime.now() >= datetime.fromisoformat(raw)


def _timestamp_on_or_after(raw: str, lower_bound: str) -> bool:
    text = str(raw or "").strip()
    bound = str(lower_bound or "").strip()
    if not text or not bound:
        return False
    try:
        return datetime.fromisoformat(text) >= datetime.fromisoformat(bound)
    except ValueError:
        return text >= bound


def _promising_stage1_rows(history: list[dict], args: argparse.Namespace, launch_started_at: str) -> list[dict]:
    rows = []
    for row in history:
        if str(row.get("evaluation_stage", "")).lower() != "stage1":
            continue
        if str(row.get("status", "")).lower() not in {"ok", "screened_out"}:
            continue
        if not _timestamp_on_or_after(str(row.get("finished_at", "")), launch_started_at):
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(row.get("pass_count", 0)),
            -float(row.get("mean_progress_final", 0.0)),
            float(row.get("mean_error_ratio", 999.0)),
            str(row.get("finished_at", "")),
        ),
        reverse=False,
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            -int(row.get("pass_count", 0)),
            -float(row.get("mean_progress_final", 0.0)),
            float(row.get("mean_error_ratio", 999.0)),
        ),
    )
    selected = []
    for idx, row in enumerate(ranked):
        dummy = SimpleNamespace(
            upgrade_top_k=int(args.upgrade_top_k),
            upgrade_min_pass_count=int(args.upgrade_min_pass_count),
            upgrade_progress_threshold=float(args.upgrade_progress_threshold),
        )
        from prepare import ExperimentResult

        probe = ExperimentResult(
            experiment_id=str(row.get("experiment_id", "")),
            candidate=str(row.get("candidate", "")),
            parent_experiment_id=str(row.get("parent_experiment_id", "")),
            status="ok",
            keep=False,
            score=float(row.get("score", float("-inf"))),
            pass_count=int(row.get("pass_count", 0)),
            mean_success_rate=float(row.get("mean_success_rate", 0.0)),
            mean_progress_final=float(row.get("mean_progress_final", 0.0)),
            mean_stall_rate=float(row.get("mean_stall_rate", 1.0)),
            mean_error_ratio=float(row.get("mean_error_ratio", 999.0)),
            max_error_ratio=float(row.get("max_error_ratio", 999.0)),
            mean_completion_time_seconds=float(row.get("mean_completion_time_seconds", 999999.0)),
            max_completion_time_seconds=float(row.get("max_completion_time_seconds", 999999.0)),
            git_head=str(row.get("git_head", "")),
            description=str(row.get("description", "")),
            config_path=str(row.get("config_path", "")),
            run_dir=str(row.get("run_dir", "")),
            model_path=str(row.get("model_path", "")),
            latest_checkpoint=str(row.get("latest_checkpoint", "")),
            eval_summary_path=str(row.get("eval_summary_path", "")),
            rollouts_summary_path=str(row.get("rollouts_summary_path", "")),
            started_at=str(row.get("started_at", "")),
            finished_at=str(row.get("finished_at", "")),
        )
        if should_upgrade_candidate(probe, ranked_index=idx, args=dummy, screen_path_count=3):
            selected.append(row)
    return selected


def _row_to_result(row: dict):
    from prepare import ExperimentResult

    return ExperimentResult(
        experiment_id=str(row.get("experiment_id", "")),
        candidate=str(row.get("candidate", "")),
        parent_experiment_id=str(row.get("parent_experiment_id", "")),
        status="ok",
        keep=False,
        score=float(row.get("score", float("-inf"))),
        pass_count=int(row.get("pass_count", 0)),
        mean_success_rate=float(row.get("mean_success_rate", 0.0)),
        mean_progress_final=float(row.get("mean_progress_final", 0.0)),
        mean_stall_rate=float(row.get("mean_stall_rate", 1.0)),
        mean_error_ratio=float(row.get("mean_error_ratio", 999.0)),
        max_error_ratio=float(row.get("max_error_ratio", 999.0)),
        mean_completion_time_seconds=float(row.get("mean_completion_time_seconds", 999999.0)),
        max_completion_time_seconds=float(row.get("max_completion_time_seconds", 999999.0)),
        git_head=str(row.get("git_head", "")),
        description=str(row.get("description", "")),
        config_path=str(row.get("config_path", "")),
        run_dir=str(row.get("run_dir", "")),
        model_path=str(row.get("model_path", "")),
        latest_checkpoint=str(row.get("latest_checkpoint", "")),
        eval_summary_path=str(row.get("eval_summary_path", "")),
        rollouts_summary_path=str(row.get("rollouts_summary_path", "")),
        started_at=str(row.get("started_at", "")),
        finished_at=str(row.get("finished_at", "")),
    )


def main() -> int:
    args = parse_args()
    state = _load_state()
    current_deadline = str(args.deadline_time).strip()
    current_paths = str(args.paths).strip()
    reset_state = (
        not str(state.get("launch_started_at", "")).strip()
        or str(state.get("deadline_time", "")).strip() != current_deadline
        or str(state.get("paths", "")).strip() != current_paths
    )
    if reset_state:
        state = {
            "processed": {},
            "launch_started_at": now_text(),
            "deadline_time": current_deadline,
            "paths": current_paths,
            "updated_at": "",
        }
        _save_state(state)

    launch_started_at = str(state.get("launch_started_at", "")).strip()
    processed = set(str(key) for key in state.get("processed", {}).keys())
    base_args = _args_like(args)
    handled = 0
    path_specs = build_selected_paths([item.strip() for item in str(args.paths).split(",") if item.strip()])

    while int(args.max_candidates) <= 0 or handled < int(args.max_candidates):
        if _deadline_reached(args.deadline_time):
            break
        history = read_results_history()
        candidates = _promising_stage1_rows(history, args, launch_started_at)
        picked = None
        for row in candidates:
            experiment_id = str(row.get("experiment_id", ""))
            if experiment_id in processed:
                continue
            picked = row
            break

        if picked is None:
            import time

            time.sleep(max(15, int(args.poll_seconds)))
            continue

        source_experiment_id = str(picked.get("experiment_id", "")).strip()
        result = _row_to_result(picked)
        result = upgrade_promising_experiment(
            result,
            args=base_args,
            bonus_episodes=int(args.upgrade_extra_episodes),
            bonus_time_budget_seconds=float(args.upgrade_time_budget_seconds),
            bonus_process_timeout_seconds=float(args.upgrade_process_timeout_seconds),
        )
        result = refine_experiment_result(
            result,
            args=base_args,
            path_specs=path_specs,
            eval_episodes=int(args.eval_episodes),
            evaluation_label="full",
            export_rollouts=True,
        )

        current_best = load_current_best_state()
        result.keep = should_keep(result, current_best, score_epsilon=float(args.score_epsilon))
        if result.keep:
            promote_candidate(Path(result.config_path), result)
            archive_promoted_result(result)

        upsert_result_row(result)
        refresh_workspace_reports(read_results_history(), load_current_best_state())
        processed.add(source_experiment_id)
        state.setdefault("processed", {})[source_experiment_id] = {
            "finished_at": now_text(),
            "refined_experiment_id": result.experiment_id,
            "status": result.status,
            "keep": bool(result.keep),
            "score": float(result.score),
        }
        _save_state(state)
        handled += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
