from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def _to_float(value: str) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _read_column(csv_path: Path, column: str) -> list[float]:
    if not csv_path.exists():
        return []
    values: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = _to_float(row.get(column, ""))
            if v is not None:
                values.append(v)
    return values


def _tail_mean(values: list[float], k: int) -> float | None:
    if not values:
        return None
    k = max(1, min(k, len(values)))
    return mean(values[-k:])


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = min(len(xs), len(ys))
    if n < 2:
        return None
    xs = xs[:n]
    ys = ys[:n]
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return cov / math.sqrt(vx * vy)


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.6f}"


def summarize_run(run_dir: Path) -> dict[str, float | None]:
    logs_dir = run_dir / "logs"
    rewards = _read_column(logs_dir / "training_log.csv", "reward")
    rmse = _read_column(logs_dir / "paper_metrics_train_square.csv", "rmse_error")
    max_error = _read_column(logs_dir / "paper_metrics_train_square.csv", "max_error")
    mean_jerk = _read_column(logs_dir / "paper_metrics_train_square.csv", "mean_jerk")
    la_u = _read_column(logs_dir / "step_metrics_train_square.csv", "lookahead_u_policy")
    la_dist = _read_column(logs_dir / "step_metrics_train_square.csv", "lookahead_dist_active")

    summary: dict[str, float | None] = {
        "episodes": float(len(rewards)) if rewards else None,
        "reward_best": max(rewards) if rewards else None,
        "reward_last": rewards[-1] if rewards else None,
        "reward_tail20_mean": _tail_mean(rewards, 20),
        "rmse_mean": mean(rmse) if rmse else None,
        "rmse_min": min(rmse) if rmse else None,
        "max_error_mean": mean(max_error) if max_error else None,
        "mean_jerk_mean": mean(mean_jerk) if mean_jerk else None,
        "lookahead_u_std": _std(la_u),
        "lookahead_dist_std": _std(la_dist),
        "lookahead_u_dist_corr": _corr(la_u, la_dist),
    }
    return summary


def render_report(old_dir: Path, new_dir: Path, old_s: dict[str, float | None], new_s: dict[str, float | None]) -> str:
    keys = [
        ("episodes", "训练回合数"),
        ("reward_best", "最佳回合奖励"),
        ("reward_last", "最终回合奖励"),
        ("reward_tail20_mean", "末20回合奖励均值"),
        ("rmse_mean", "RMSE均值"),
        ("rmse_min", "RMSE最小值"),
        ("max_error_mean", "最大误差均值"),
        ("mean_jerk_mean", "平均Jerk均值"),
        ("lookahead_u_std", "前瞻动作u标准差"),
        ("lookahead_dist_std", "生效前瞻距离标准差"),
        ("lookahead_u_dist_corr", "corr(u,dist_active)"),
    ]
    lines = [
        "# 训练对比报告",
        "",
        f"- 旧实验: `{old_dir}`",
        f"- 新实验: `{new_dir}`",
        "",
        "| 指标 | 旧 | 新 | 新-旧 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in keys:
        ov = old_s.get(key)
        nv = new_s.get(key)
        dv = None
        if ov is not None and nv is not None:
            dv = nv - ov
        lines.append(f"| {label} | {_fmt(ov)} | {_fmt(nv)} | {_fmt(dv)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old/new run metrics from logs.")
    parser.add_argument("--old", required=True, help="旧实验目录（包含 logs）")
    parser.add_argument("--new", required=True, help="新实验目录（包含 logs）")
    parser.add_argument("--out", required=True, help="输出 Markdown 报告路径")
    args = parser.parse_args()

    old_dir = Path(args.old).resolve()
    new_dir = Path(args.new).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    old_s = summarize_run(old_dir)
    new_s = summarize_run(new_dir)
    report = render_report(old_dir, new_dir, old_s, new_s)
    out_path.write_text(report, encoding="utf-8")
    print(f"report_written={out_path}")


if __name__ == "__main__":
    main()
