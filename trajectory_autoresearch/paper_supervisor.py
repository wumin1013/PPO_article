from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path

from prepare import RESEARCH_ROOT, get_git_head, load_current_best_state, now_tag, now_text, resolve_python_command, run_command, write_json


PAPER_RUNS_DIR = RESEARCH_ROOT / "paper_runs"
ACTIVE_LOGS_DIR = PAPER_RUNS_DIR / "active_logs"
PAPER_SUITE_SCRIPT = RESEARCH_ROOT / "paper_suite.py"
PAPER_SYNC_SCRIPT = RESEARCH_ROOT / "paper_sync.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep paper suite aligned with the active search window.")
    parser.add_argument("--deadline-time", type=str, required=True, help="统一截止时间，格式 YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--status-path", type=str, default="", help="监督器状态文件路径")
    parser.add_argument("--suite-name-prefix", type=str, default="paper_live", help="自动生成的论文套件名前缀")
    parser.add_argument("--interval-seconds", type=int, default=300, help="轮询 current_best 的周期")
    parser.add_argument("--paths", type=str, default="square,circle,butterfly", help="论文路径列表")
    parser.add_argument("--total-episodes", type=int, default=220, help="每轮论文套件的最大 episode 数")
    parser.add_argument("--time-budget-seconds", type=float, default=600.0, help="每个训练型变体的时间预算")
    parser.add_argument("--process-timeout-seconds", type=float, default=5400.0, help="每个训练型变体的进程超时")
    parser.add_argument("--eval-episodes", type=int, default=3, help="论文评测 episode 数")
    parser.add_argument("--conda-env", type=str, default="PPO", help="conda 环境名")
    parser.add_argument("--variants", type=str, default="", help="限定论文变体")
    parser.add_argument("--deterministic-eval", action="store_true", help="论文评测时启用确定性策略")
    return parser.parse_args()


def _parse_deadline(text: str) -> datetime:
    return datetime.fromisoformat(str(text).strip())


def _remaining_seconds(deadline: datetime) -> float:
    return max(0.0, (deadline - datetime.now()).total_seconds())


def _status_path(args: argparse.Namespace) -> Path:
    raw = str(args.status_path or "").strip()
    if raw:
        return Path(raw)
    return PAPER_RUNS_DIR / f"{now_tag()}_paper_supervisor_status.json"


def _write_status(path: Path, payload: dict) -> None:
    payload["updated_at"] = now_text()
    write_json(path, payload)


def _run_once_sync(args: argparse.Namespace, status: dict, status_path: Path) -> int:
    log_path = ACTIVE_LOGS_DIR / f"{now_tag()}_paper_sync.log"
    cmd = resolve_python_command(args.conda_env) + [str(PAPER_SYNC_SCRIPT), "--once"]
    exit_code = run_command(cmd, cwd=RESEARCH_ROOT.parent, log_path=log_path, check=False)
    status["last_sync_exit_code"] = int(exit_code)
    status["last_sync_log"] = str(log_path)
    status["last_sync_at"] = now_text()
    _write_status(status_path, status)
    return int(exit_code)


def _run_paper_suite(args: argparse.Namespace, *, best_id: str, status: dict, status_path: Path) -> int:
    ACTIVE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    suite_tag = f"{args.suite_name_prefix}_{best_id}"
    log_path = ACTIVE_LOGS_DIR / f"{now_tag()}_{best_id}_paper_suite.log"
    cmd = [
        *resolve_python_command(args.conda_env),
        str(PAPER_SUITE_SCRIPT),
        "--suite-name",
        suite_tag,
        "--paths",
        args.paths,
        "--total-episodes",
        str(int(args.total_episodes)),
        "--time-budget-seconds",
        str(float(args.time_budget_seconds)),
        "--process-timeout-seconds",
        str(float(args.process_timeout_seconds)),
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--conda-env",
        args.conda_env,
        "--deadline-time",
        args.deadline_time,
        "--sync-after-each",
    ]
    if str(args.variants or "").strip():
        cmd += ["--variants", str(args.variants).strip()]
    if bool(args.deterministic_eval):
        cmd.append("--deterministic-eval")

    remaining = _remaining_seconds(_parse_deadline(args.deadline_time))
    timeout_seconds = max(600.0, min(float(args.process_timeout_seconds) * 6.0, remaining + 300.0))
    status["active_suite"] = {
        "best_experiment_id": best_id,
        "suite_name": suite_tag,
        "log_path": str(log_path),
        "started_at": now_text(),
        "status": "running",
    }
    _write_status(status_path, status)
    exit_code = run_command(cmd, cwd=RESEARCH_ROOT.parent, log_path=log_path, timeout_seconds=timeout_seconds, check=False)
    status["active_suite"]["status"] = "completed" if int(exit_code) == 0 else "failed"
    status["active_suite"]["exit_code"] = int(exit_code)
    status["active_suite"]["finished_at"] = now_text()
    status["last_completed_best_experiment_id"] = best_id if int(exit_code) == 0 else status.get("last_completed_best_experiment_id", "")
    _write_status(status_path, status)
    return int(exit_code)


def main() -> int:
    args = parse_args()
    deadline = _parse_deadline(args.deadline_time)
    status_path = _status_path(args)
    status = {
        "status": "running",
        "started_at": now_text(),
        "deadline_time": args.deadline_time,
        "git_head": get_git_head(),
        "interval_seconds": int(args.interval_seconds),
        "paths": [item.strip() for item in str(args.paths).split(",") if item.strip()],
        "last_seen_best_experiment_id": "",
        "last_completed_best_experiment_id": "",
        "last_sync_at": "",
        "last_sync_exit_code": None,
        "last_sync_log": "",
        "active_suite": {},
        "last_error": "",
    }
    _write_status(status_path, status)

    try:
        while _remaining_seconds(deadline) > 0.0:
            best_state = load_current_best_state()
            best_id = str(best_state.get("experiment_id", "")).strip() if isinstance(best_state, dict) else ""
            status["last_seen_best_experiment_id"] = best_id
            _write_status(status_path, status)

            if best_id:
                if best_id != str(status.get("last_completed_best_experiment_id", "")).strip():
                    _run_once_sync(args, status, status_path)
                    if _remaining_seconds(deadline) > 240.0:
                        _run_paper_suite(args, best_id=best_id, status=status, status_path=status_path)
                        _run_once_sync(args, status, status_path)
                    else:
                        break
                else:
                    _run_once_sync(args, status, status_path)

            sleep_seconds = min(max(30, int(args.interval_seconds)), max(1, int(_remaining_seconds(deadline))))
            time.sleep(sleep_seconds)
    except Exception:
        status["status"] = "failed"
        status["last_error"] = traceback.format_exc()
        _write_status(status_path, status)
        raise

    status["status"] = "completed"
    status["finished_at"] = now_text()
    _write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
