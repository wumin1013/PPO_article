from __future__ import annotations

import argparse
import json
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from prepare import (
    BASE_CONFIG_COPY,
    CURRENT_BEST_CONFIG,
    RESEARCH_ROOT,
    build_selected_paths,
    build_train_config,
    ensure_workspace,
    evaluate_model_across_paths,
    export_best_rollouts,
    find_latest_checkpoint,
    find_model_checkpoint,
    get_git_head,
    get_nested,
    load_current_best_state,
    load_yaml,
    now_tag,
    now_text,
    resolve_python_command,
    run_command,
    set_nested,
    train_candidate,
    write_json,
    write_yaml,
)


PAPER_RUNS_DIR = RESEARCH_ROOT / "paper_runs"
PAPER_SYNC_SCRIPT = RESEARCH_ROOT / "paper_sync.py"


@dataclass(frozen=True)
class PaperVariantSpec:
    name: str
    label: str
    description: str
    source: str
    mode: str
    apply: Callable[[dict], None]


def variant_specs() -> list[PaperVariantSpec]:
    def noop(cfg: dict) -> None:
        return

    def fixed_lookahead(cfg: dict) -> None:
        min_dist = float(get_nested(cfg, "reward_weights.lookahead_control.min_dist", 0.8) or 0.8)
        max_dist = float(get_nested(cfg, "reward_weights.lookahead_control.max_dist", 4.0) or 4.0)
        straight_dist = float(get_nested(cfg, "reward_weights.lookahead_control.straight_dist", 1.2) or 1.2)
        corner_dist = float(get_nested(cfg, "reward_weights.lookahead_control.corner_dist", 2.4) or 2.4)
        default_dist = float(get_nested(cfg, "reward_weights.lookahead.distance", 2.5) or 2.5)
        fixed_dist = min(max(default_dist, min_dist), max_dist)
        fixed_dist = min(max((fixed_dist + 0.5 * (straight_dist + corner_dist)) / 2.0, min_dist), max_dist)
        set_nested(cfg, "reward_weights.lookahead.distance", fixed_dist)
        set_nested(cfg, "reward_weights.lookahead_control.enabled", False)
        set_nested(cfg, "reward_weights.lookahead_control.policy_action", False)

    def no_kcm(cfg: dict) -> None:
        set_nested(cfg, "experiment.enable_kcm", False)

    def no_cornerness(cfg: dict) -> None:
        set_nested(cfg, "reward_weights.cornerness.enabled", False)
        set_nested(cfg, "reward_weights.cornerness.keep_legacy_smooth", True)

    return [
        PaperVariantSpec(
            name="full_method_snapshot",
            label="本文最终方法",
            description="直接评估当前搜索最优模型快照",
            source="current_best",
            mode="evaluate_only",
            apply=noop,
        ),
        PaperVariantSpec(
            name="baseline_policy",
            label="基线策略",
            description="以基线配置重新训练并评测",
            source="base_config",
            mode="train_eval",
            apply=noop,
        ),
        PaperVariantSpec(
            name="abl_fixed_lookahead",
            label="固定前瞻",
            description="关闭学习型前瞻距离，仅保留固定前瞻长度",
            source="current_best",
            mode="train_eval",
            apply=fixed_lookahead,
        ),
        PaperVariantSpec(
            name="abl_no_kcm",
            label="无KCM",
            description="关闭执行侧 KCM 约束投影",
            source="current_best",
            mode="train_eval",
            apply=no_kcm,
        ),
        PaperVariantSpec(
            name="abl_no_cornerness",
            label="无拐角感知平滑",
            description="关闭 cornerness 动态平滑与动态跟踪权重",
            source="current_best",
            mode="train_eval",
            apply=no_cornerness,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-oriented baseline and ablation experiments.")
    parser.add_argument("--suite-name", type=str, default="", help="论文实验套件名称")
    parser.add_argument("--total-episodes", type=int, default=220, help="每个训练型配置的最大 episode 数")
    parser.add_argument("--time-budget-seconds", type=float, default=600.0, help="每个训练型配置的墙钟时间预算")
    parser.add_argument("--process-timeout-seconds", type=float, default=5400.0, help="单个训练型配置的进程超时")
    parser.add_argument("--eval-episodes", type=int, default=3, help="每条路径的评测 episode 数")
    parser.add_argument("--eval-seed", type=int, default=43, help="统一评测随机种子")
    parser.add_argument("--seed", type=int, default=42, help="训练配置中写入的种子")
    parser.add_argument("--paths", type=str, default="square,circle,butterfly", help="评测路径列表")
    parser.add_argument("--conda-env", type=str, default="PPO", help="使用的 conda 环境")
    parser.add_argument("--deterministic-eval", action="store_true", help="评测时使用确定性策略")
    parser.add_argument("--sync-after-each", action="store_true", help="每完成一个变体即刷新论文产物")
    parser.add_argument("--variants", type=str, default="", help="只运行指定变体，逗号分隔；为空表示运行全部")
    return parser.parse_args()


def _suite_id(run_name: str) -> str:
    tag = str(run_name).strip().replace(" ", "_")
    return f"{now_tag()}_{tag or 'paper_suite'}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_manifest_path(variant_dir: Path) -> Path:
    return variant_dir / "variant_manifest.json"


def _write_suite_manifest(path: Path, payload: dict) -> None:
    write_json(path, payload)


def _run_paper_sync(conda_env: str) -> None:
    cmd = resolve_python_command(conda_env) + [str(PAPER_SYNC_SCRIPT), "--once"]
    run_command(cmd, cwd=RESEARCH_ROOT.parent, check=False)


def _prepare_train_config(
    *,
    base_config_path: Path,
    variant: PaperVariantSpec,
    experiment_name: str,
    path_specs: Sequence[dict],
    args: argparse.Namespace,
) -> dict:
    base_cfg = load_yaml(base_config_path)
    cfg = build_train_config(
        base_cfg,
        experiment_name=experiment_name,
        path_specs=path_specs,
        total_episodes=int(args.total_episodes),
        time_budget_seconds=float(args.time_budget_seconds),
        seed=int(args.seed),
    )
    cfg.setdefault("experiment", {})
    cfg["experiment"]["category"] = "paper_suite"
    cfg["experiment"]["paper_variant"] = variant.name
    cfg["experiment"]["paper_label"] = variant.label
    cfg["experiment"]["paper_description"] = variant.description
    cfg.setdefault("training", {})
    cfg["training"]["enable_best_trajectory_snapshot"] = True
    cfg["training"]["enable_latest_trajectory"] = False
    cfg["training"]["traj_write_interval_steps"] = 0
    cfg["training"]["step_log_interval_steps"] = max(10, int(cfg["training"].get("step_log_interval_steps", 10) or 10))
    variant.apply(cfg)
    return cfg


def _variant_base_path(
    variant: PaperVariantSpec,
    *,
    current_best_snapshot: Path,
    base_snapshot: Path,
) -> Path:
    if variant.source == "current_best":
        return current_best_snapshot
    if variant.source == "base_config":
        return base_snapshot
    raise ValueError(f"unsupported source: {variant.source}")


def _evaluate_existing_best(
    *,
    variant: PaperVariantSpec,
    variant_dir: Path,
    path_specs: Sequence[dict],
    args: argparse.Namespace,
    current_best_state: dict,
    current_best_snapshot: Path,
) -> dict:
    source_run_dir = Path(str(current_best_state.get("run_dir", "")).strip())
    source_model_path = Path(str(current_best_state.get("model_path", "")).strip())
    source_latest = Path(str(current_best_state.get("latest_checkpoint", "")).strip())
    if not source_run_dir.exists() or not source_model_path.exists():
        raise FileNotFoundError("current best run/model path missing; cannot build paper full-method snapshot")

    config_path = variant_dir / "config.yaml"
    cfg = _prepare_train_config(
        base_config_path=current_best_snapshot,
        variant=variant,
        experiment_name=variant_dir.name,
        path_specs=path_specs,
        args=args,
    )
    write_yaml(config_path, cfg)

    eval_payload = evaluate_model_across_paths(
        trained_config_path=config_path,
        model_path=source_model_path,
        out_dir=variant_dir / "evaluation",
        path_specs=path_specs,
        episodes=int(args.eval_episodes),
        deterministic=bool(args.deterministic_eval),
        seed=int(args.eval_seed),
        conda_env=args.conda_env,
        score_profile="stage2",
    )
    rollouts_summary = export_best_rollouts(
        config_path=config_path,
        run_dir=source_run_dir,
        out_dir=variant_dir / "best_rollouts",
        conda_env=args.conda_env,
    )
    return {
        "config_path": str(config_path),
        "run_dir": str(source_run_dir),
        "model_path": str(source_model_path),
        "latest_checkpoint": str(source_latest) if source_latest.exists() else "",
        "source_experiment_id": str(current_best_state.get("experiment_id", "")),
        "eval_summary_path": str(variant_dir / "evaluation" / "summary.json"),
        "rollouts_summary_path": str(rollouts_summary),
        "aggregated": eval_payload.get("aggregated", {}),
    }


def _train_and_evaluate_variant(
    *,
    variant: PaperVariantSpec,
    variant_dir: Path,
    path_specs: Sequence[dict],
    args: argparse.Namespace,
    current_best_snapshot: Path,
    base_snapshot: Path,
) -> dict:
    config_path = variant_dir / "config.yaml"
    cfg = _prepare_train_config(
        base_config_path=_variant_base_path(variant, current_best_snapshot=current_best_snapshot, base_snapshot=base_snapshot),
        variant=variant,
        experiment_name=variant_dir.name,
        path_specs=path_specs,
        args=args,
    )
    write_yaml(config_path, cfg)
    train_candidate(
        config_path=config_path,
        run_dir=variant_dir,
        conda_env=args.conda_env,
        resume_path=None,
        timeout_seconds=float(args.process_timeout_seconds),
    )
    model_path = find_model_checkpoint(variant_dir)
    latest_checkpoint = find_latest_checkpoint(variant_dir)
    eval_payload = evaluate_model_across_paths(
        trained_config_path=config_path,
        model_path=model_path,
        out_dir=variant_dir / "evaluation",
        path_specs=path_specs,
        episodes=int(args.eval_episodes),
        deterministic=bool(args.deterministic_eval),
        seed=int(args.eval_seed),
        conda_env=args.conda_env,
        score_profile="stage2",
    )
    rollouts_summary = export_best_rollouts(
        config_path=config_path,
        run_dir=variant_dir,
        out_dir=variant_dir / "best_rollouts",
        conda_env=args.conda_env,
    )
    return {
        "config_path": str(config_path),
        "run_dir": str(variant_dir),
        "model_path": str(model_path),
        "latest_checkpoint": str(latest_checkpoint),
        "eval_summary_path": str(variant_dir / "evaluation" / "summary.json"),
        "rollouts_summary_path": str(rollouts_summary),
        "aggregated": eval_payload.get("aggregated", {}),
    }


def main() -> int:
    args = parse_args()
    ensure_workspace()
    current_best_state = load_current_best_state()
    if not current_best_state:
        raise RuntimeError("current best state is missing; run trajectory_autoresearch first")

    suite_id = _suite_id(args.suite_name)
    suite_dir = PAPER_RUNS_DIR / suite_id
    suite_dir.mkdir(parents=True, exist_ok=True)
    path_specs = build_selected_paths([item.strip() for item in str(args.paths).split(",") if item.strip()])

    current_best_snapshot = suite_dir / "snapshot_current_best.yaml"
    base_snapshot = suite_dir / "snapshot_base_config.yaml"
    shutil.copy2(CURRENT_BEST_CONFIG, current_best_snapshot)
    shutil.copy2(BASE_CONFIG_COPY, base_snapshot)

    suite_manifest_path = suite_dir / "suite_manifest.json"
    suite_manifest = {
        "suite_id": suite_id,
        "status": "running",
        "git_head": get_git_head(),
        "started_at": now_text(),
        "finished_at": "",
        "current_best_experiment_id": str(current_best_state.get("experiment_id", "")),
        "paths": [str(item.get("name") or item.get("type")) for item in path_specs],
        "variants": {},
    }
    _write_suite_manifest(suite_manifest_path, suite_manifest)

    selected_variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    all_variants = variant_specs()
    if selected_variants:
        selected_set = set(selected_variants)
        variants_to_run = [variant for variant in all_variants if variant.name in selected_set]
        missing = sorted(selected_set.difference({variant.name for variant in variants_to_run}))
        if missing:
            raise ValueError(f"unknown paper variants: {', '.join(missing)}")
    else:
        variants_to_run = all_variants

    for variant in variants_to_run:
        variant_dir = suite_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": variant.name,
            "label": variant.label,
            "description": variant.description,
            "mode": variant.mode,
            "status": "running",
            "started_at": now_text(),
            "finished_at": "",
            "git_head": get_git_head(),
        }
        write_json(_variant_manifest_path(variant_dir), manifest)
        suite_manifest["variants"][variant.name] = manifest
        _write_suite_manifest(suite_manifest_path, suite_manifest)

        try:
            if variant.mode == "evaluate_only":
                payload = _evaluate_existing_best(
                    variant=variant,
                    variant_dir=variant_dir,
                    path_specs=path_specs,
                    args=args,
                    current_best_state=current_best_state,
                    current_best_snapshot=current_best_snapshot,
                )
            else:
                payload = _train_and_evaluate_variant(
                    variant=variant,
                    variant_dir=variant_dir,
                    path_specs=path_specs,
                    args=args,
                    current_best_snapshot=current_best_snapshot,
                    base_snapshot=base_snapshot,
                )
            manifest.update(payload)
            manifest["status"] = "completed"
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest["traceback"] = traceback.format_exc()

        manifest["finished_at"] = now_text()
        write_json(_variant_manifest_path(variant_dir), manifest)
        suite_manifest["variants"][variant.name] = manifest
        _write_suite_manifest(suite_manifest_path, suite_manifest)

        if args.sync_after_each:
            _run_paper_sync(args.conda_env)

    suite_manifest = _read_json(suite_manifest_path)
    suite_manifest["status"] = "completed"
    suite_manifest["finished_at"] = now_text()
    _write_suite_manifest(suite_manifest_path, suite_manifest)
    _run_paper_sync(args.conda_env)
    print(f"suite_dir={suite_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
