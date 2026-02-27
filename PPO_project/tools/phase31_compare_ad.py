from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.p0_gold_pipeline import _load_yaml, _rollout_trace, _set_seed, _write_trace_csv


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return (PROJECT_ROOT.parent / path).resolve()
    return (PROJECT_ROOT.parent / path).resolve()


def _safe_float(value: object) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if math.isfinite(v):
        return v
    return None


def _metric_from_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    corner_mask: List[bool] = []
    omega_vals: List[float] = []
    domega_vals: List[float] = []
    velocity_vals: List[float] = []
    abs_error_vals: List[float] = []
    progress_vals: List[float] = []

    for row in rows:
        corner_flag = False
        corner_raw = _safe_float(row.get("corner_mask"))
        phase_raw = _safe_float(row.get("corner_phase"))
        if corner_raw is not None and corner_raw >= 0.5:
            corner_flag = True
        if phase_raw is not None and phase_raw >= 0.5:
            corner_flag = True
        corner_mask.append(corner_flag)

        omega = _safe_float(row.get("omega"))
        domega = _safe_float(row.get("domega"))
        velocity = _safe_float(row.get("velocity"))
        contour_error = _safe_float(row.get("contour_error"))
        progress = _safe_float(row.get("progress"))

        omega_vals.append(float("nan") if omega is None else omega)
        domega_vals.append(float("nan") if domega is None else domega)
        velocity_vals.append(float("nan") if velocity is None else velocity)
        if contour_error is not None:
            abs_error_vals.append(abs(contour_error))
        if progress is not None:
            progress_vals.append(progress)

    corner_idx = [i for i, is_corner in enumerate(corner_mask) if is_corner]
    corner_peak_abs_omega = float("nan")
    corner_mean_abs_domega = float("nan")
    corner_min_velocity = float("nan")

    if corner_idx:
        omega_corner = [abs(omega_vals[i]) for i in corner_idx if math.isfinite(omega_vals[i])]
        if omega_corner:
            corner_peak_abs_omega = max(omega_corner)

        domega_corner = [abs(domega_vals[i]) for i in corner_idx if math.isfinite(domega_vals[i])]
        if domega_corner:
            corner_mean_abs_domega = sum(domega_corner) / float(len(domega_corner))

        velocity_corner = [velocity_vals[i] for i in corner_idx if math.isfinite(velocity_vals[i])]
        if velocity_corner:
            corner_min_velocity = min(velocity_corner)

    max_abs_error = max(abs_error_vals) if abs_error_vals else float("nan")
    progress_final = progress_vals[-1] if progress_vals else float("nan")

    return {
        "corner_peak_abs_omega": float(corner_peak_abs_omega),
        "corner_mean_abs_domega": float(corner_mean_abs_domega),
        "corner_min_velocity": float(corner_min_velocity),
        "max_abs_error": float(max_abs_error),
        "progress_final": float(progress_final),
        "trace_steps": float(len(rows)),
    }


def _finite_delta(a: float, b: float) -> float:
    if math.isfinite(a) and math.isfinite(b):
        return b - a
    return float("nan")


def _find_model_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_model.pth",
        run_dir / "checkpoints" / "tracking_model_final.pth",
        run_dir / "checkpoints" / "latest_checkpoint.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No model checkpoint found under: {run_dir}")


def _build_bundle(
    run_dir: Path,
    out_bundle: Path,
    *,
    deterministic: bool,
    seed_override: Optional[int],
) -> tuple[Path, Dict[str, float]]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    model_path = _find_model_path(run_dir)
    config = _load_yaml(config_path)

    seed = int(seed_override) if seed_override is not None else int(config.get("seed", 42))
    _set_seed(seed)
    rows = _rollout_trace(config, model_path, deterministic=deterministic)

    out_bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, out_bundle / "config.yaml")
    trace_path = out_bundle / "trace.csv"
    _write_trace_csv(rows, trace_path)

    metrics = _metric_from_rows(rows)
    metrics.update(
        {
            "seed_trace": float(seed),
            "deterministic": 1.0 if deterministic else 0.0,
        }
    )
    with (out_bundle / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return trace_path, metrics


def _write_compare_csv(
    out_path: Path,
    baseline_label: str,
    candidate_label: str,
    baseline_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    metric_names: Iterable[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                baseline_label,
                candidate_label,
                "delta_candidate_minus_baseline",
            ],
        )
        writer.writeheader()
        for name in metric_names:
            a = float(baseline_metrics.get(name, float("nan")))
            b = float(candidate_metrics.get(name, float("nan")))
            writer.writerow(
                {
                    "metric": name,
                    baseline_label: a,
                    candidate_label: b,
                    "delta_candidate_minus_baseline": _finite_delta(a, b),
                }
            )


def _run_plot_compare(
    baseline_bundle: Path,
    candidate_bundle: Path,
    out_dir: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "plot_bundle_compare.py"),
        "--baseline_bundle",
        str(baseline_bundle),
        "--candidate_bundle",
        str(candidate_bundle),
        "--baseline_label",
        baseline_label,
        "--candidate_label",
        candidate_label,
        "--out_dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _read_episode_series(run_dir: Path) -> Dict[str, List[float]]:
    episode_csv = run_dir / "logs" / "episode_metrics_train_square.csv"
    if not episode_csv.exists():
        return {"episode_idx": [], "total_reward": [], "actor_loss": [], "critic_loss": []}

    episodes: List[float] = []
    rewards: List[float] = []
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    with episode_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = _safe_float(row.get("episode_idx"))
            rew = _safe_float(row.get("total_reward"))
            a_loss = _safe_float(row.get("actor_loss"))
            c_loss = _safe_float(row.get("critic_loss"))
            if ep is not None:
                episodes.append(ep)
            else:
                episodes.append(float(len(episodes)))
            rewards.append(float("nan") if rew is None else rew)
            actor_losses.append(float("nan") if a_loss is None else a_loss)
            critic_losses.append(float("nan") if c_loss is None else c_loss)
    return {
        "episode_idx": episodes,
        "total_reward": rewards,
        "actor_loss": actor_losses,
        "critic_loss": critic_losses,
    }


def _plot_training_curves(
    baseline_run: Path,
    candidate_run: Path,
    out_dir: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = _read_episode_series(baseline_run)
    d = _read_episode_series(candidate_run)

    def _plot_two(
        key: str,
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        if a["episode_idx"]:
            ax.plot(
                a["episode_idx"],
                a[key],
                color="#6c757d",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=baseline_label,
            )
        if d["episode_idx"]:
            ax.plot(
                d["episode_idx"],
                d[key],
                color="#e03131",
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=candidate_label,
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    _plot_two("total_reward", "Total Reward", "Training Curve: Total Reward", "training_total_reward.png")
    _plot_two("actor_loss", "Actor Loss", "Training Curve: Actor Loss", "training_actor_loss.png")
    _plot_two("critic_loss", "Critic Loss", "Training Curve: Critic Loss", "training_critic_loss.png")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase31 A vs D trace export + metric compare + plots.")
    parser.add_argument("--baseline_run", required=True, help="Baseline run dir (A)")
    parser.add_argument("--candidate_run", required=True, help="Candidate run dir (D)")
    parser.add_argument("--out_dir", required=True, help="Output root directory")
    parser.add_argument("--baseline_label", default="A_legacy", help="Label for baseline")
    parser.add_argument("--candidate_label", default="D_blend", help="Label for candidate")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for rollout trace")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling")
    args = parser.parse_args(argv)

    baseline_run = _resolve_path(args.baseline_run)
    candidate_run = _resolve_path(args.candidate_run)
    out_dir = _resolve_path(args.out_dir)

    deterministic = not bool(args.stochastic)
    baseline_bundle = out_dir / "baseline_bundle"
    candidate_bundle = out_dir / "candidate_bundle"
    plots_dir = out_dir / "plots"

    _, baseline_metrics = _build_bundle(
        baseline_run,
        baseline_bundle,
        deterministic=deterministic,
        seed_override=args.seed,
    )
    _, candidate_metrics = _build_bundle(
        candidate_run,
        candidate_bundle,
        deterministic=deterministic,
        seed_override=args.seed,
    )

    compare_metrics = [
        "corner_peak_abs_omega",
        "corner_mean_abs_domega",
        "corner_min_velocity",
        "max_abs_error",
        "progress_final",
    ]
    _write_compare_csv(
        out_dir / "compare_metrics.csv",
        args.baseline_label,
        args.candidate_label,
        baseline_metrics,
        candidate_metrics,
        compare_metrics,
    )

    payload = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "deterministic": deterministic,
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": {
            name: _finite_delta(
                float(baseline_metrics.get(name, float("nan"))),
                float(candidate_metrics.get(name, float("nan"))),
            )
            for name in compare_metrics
        },
        "files": {
            "baseline_trace": str((baseline_bundle / "trace.csv").resolve()),
            "candidate_trace": str((candidate_bundle / "trace.csv").resolve()),
            "compare_csv": str((out_dir / "compare_metrics.csv").resolve()),
        },
    }
    with (out_dir / "compare_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    _run_plot_compare(
        baseline_bundle=baseline_bundle,
        candidate_bundle=candidate_bundle,
        out_dir=plots_dir,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    _plot_training_curves(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        out_dir=plots_dir,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[done] outputs saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
