from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELD_MAP = {
    "timestamp": ["timestamp", "time", "t", "step", "env_step"],
    "pos_x": ["position_x", "pos_x", "x"],
    "pos_y": ["position_y", "pos_y", "y"],
    "ref_x": ["reference_x", "ref_x"],
    "ref_y": ["reference_y", "ref_y"],
    "velocity": ["velocity", "speed"],
    "contour_error": ["contour_error", "error", "e_n"],
    "omega": ["omega", "angular_vel", "omega_exec"],
    "domega": ["domega", "angular_acc", "omega_dot"],
}


@dataclass
class TraceData:
    label: str
    time: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    velocity: np.ndarray
    contour_error: np.ndarray
    omega: np.ndarray
    domega: np.ndarray


def _safe_float(value: object) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    if math.isfinite(val):
        return val
    return None


def _pick_value(row: dict, keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key not in row:
            continue
        val = _safe_float(row.get(key))
        if val is not None:
            return val
    return None


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return (PROJECT_ROOT.parent / path).resolve()
    return (PROJECT_ROOT / path).resolve()


def _find_trace_path(bundle_path: Path) -> Path:
    if bundle_path.is_file():
        return bundle_path
    candidate = bundle_path / "rollout_det" / "trace.csv"
    if candidate.exists():
        return candidate
    fallback = bundle_path / "trace.csv"
    return fallback


def _read_trace(path: Path, label: str) -> TraceData:
    if not path.exists():
        raise FileNotFoundError(f"trace.csv not found: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(row)
    n = len(rows)
    if n == 0:
        empty = np.asarray([], dtype=float)
        return TraceData(label, empty, empty, empty, empty, empty, empty, empty, empty, empty)

    time: List[float] = []
    pos_x: List[float] = []
    pos_y: List[float] = []
    ref_x: List[float] = []
    ref_y: List[float] = []
    velocity: List[float] = []
    contour_error: List[float] = []
    omega: List[float] = []
    domega: List[float] = []

    for idx, row in enumerate(rows):
        t_val = _pick_value(row, FIELD_MAP["timestamp"])
        if t_val is None:
            t_val = float(idx)
        time.append(t_val)
        pos_x.append(_pick_value(row, FIELD_MAP["pos_x"]) or float("nan"))
        pos_y.append(_pick_value(row, FIELD_MAP["pos_y"]) or float("nan"))
        ref_x.append(_pick_value(row, FIELD_MAP["ref_x"]) or float("nan"))
        ref_y.append(_pick_value(row, FIELD_MAP["ref_y"]) or float("nan"))
        velocity.append(_pick_value(row, FIELD_MAP["velocity"]) or float("nan"))
        contour_error.append(_pick_value(row, FIELD_MAP["contour_error"]) or float("nan"))
        omega.append(_pick_value(row, FIELD_MAP["omega"]) or float("nan"))
        domega.append(_pick_value(row, FIELD_MAP["domega"]) or float("nan"))

    return TraceData(
        label=label,
        time=np.asarray(time, dtype=float),
        pos_x=np.asarray(pos_x, dtype=float),
        pos_y=np.asarray(pos_y, dtype=float),
        ref_x=np.asarray(ref_x, dtype=float),
        ref_y=np.asarray(ref_y, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
        contour_error=np.asarray(contour_error, dtype=float),
        omega=np.asarray(omega, dtype=float),
        domega=np.asarray(domega, dtype=float),
    )


def _filter_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _plot_overlay(baseline: TraceData, candidate: TraceData, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ref_x, ref_y = _filter_xy(candidate.ref_x, candidate.ref_y)
    if ref_x.size == 0:
        ref_x, ref_y = _filter_xy(baseline.ref_x, baseline.ref_y)
    if ref_x.size > 0:
        ax.plot(ref_x, ref_y, "--", color="#1f77b4", linewidth=1.4, label="Reference")

    bx, by = _filter_xy(baseline.pos_x, baseline.pos_y)
    if bx.size > 0:
        ax.plot(bx, by, color="#6c757d", linewidth=1.8, label=baseline.label)

    cx, cy = _filter_xy(candidate.pos_x, candidate.pos_y)
    if cx.size > 0:
        ax.plot(cx, cy, color="#e03131", linewidth=2.0, label=candidate.label)

    ax.set_title("Overlay: Baseline vs Candidate")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_series(
    baseline: TraceData,
    candidate: TraceData,
    *,
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for series, color in ((baseline, "#6c757d"), (candidate, "#e03131")):
        data = getattr(series, key)
        mask = np.isfinite(series.time) & np.isfinite(data)
        if not mask.any():
            continue
        ax.plot(series.time[mask], data[mask], linewidth=1.8, color=color, label=series.label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_omega_domega(baseline: TraceData, candidate: TraceData, out_path: Path) -> None:
    has_omega = np.isfinite(baseline.omega).any() or np.isfinite(candidate.omega).any()
    has_domega = np.isfinite(baseline.domega).any() or np.isfinite(candidate.domega).any()
    if not (has_omega or has_domega):
        return

    panels = []
    if has_omega:
        panels.append(("omega", "Omega (rad/s)"))
    if has_domega:
        panels.append(("domega", "dOmega (rad/s^2)"))

    fig, axes = plt.subplots(len(panels), 1, figsize=(6.5, 3.2 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (key, ylabel) in zip(axes, panels):
        for series, color in ((baseline, "#6c757d"), (candidate, "#e03131")):
            data = getattr(series, key)
            mask = np.isfinite(series.time) & np.isfinite(data)
            if not mask.any():
                continue
            ax.plot(series.time[mask], data[mask], linewidth=1.6, color=color, label=series.label)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Angular Motion")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs candidate trace plots.")
    parser.add_argument("--baseline_bundle", required=True, help="Baseline bundle dir or trace.csv")
    parser.add_argument("--candidate_bundle", required=True, help="Candidate bundle dir or trace.csv")
    parser.add_argument("--baseline_label", default="baseline", help="Legend label for baseline")
    parser.add_argument("--candidate_label", default="candidate", help="Legend label for candidate")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: candidate_bundle/rollout_det/plots/compare)",
    )
    args = parser.parse_args()

    baseline_bundle = _resolve_path(args.baseline_bundle)
    candidate_bundle = _resolve_path(args.candidate_bundle)
    baseline_trace = _find_trace_path(baseline_bundle)
    candidate_trace = _find_trace_path(candidate_bundle)

    baseline = _read_trace(baseline_trace, args.baseline_label)
    candidate = _read_trace(candidate_trace, args.candidate_label)

    if args.out_dir:
        out_dir = _resolve_path(args.out_dir)
    else:
        if candidate_bundle.is_dir():
            out_dir = candidate_bundle / "rollout_det" / "plots" / "compare"
        else:
            out_dir = candidate_trace.parent / "plots" / "compare"

    _plot_overlay(baseline, candidate, out_dir / "overlay.png")
    _plot_series(baseline, candidate, key="velocity", ylabel="Velocity", title="v(t)", out_path=out_dir / "v_t.png")
    _plot_series(
        baseline,
        candidate,
        key="contour_error",
        ylabel="Contour Error",
        title="e_n(t)",
        out_path=out_dir / "e_n_t.png",
    )
    _plot_omega_domega(baseline, candidate, out_dir / "omega_domega_t.png")

    print(f"[done] plots saved to {out_dir}")


if __name__ == "__main__":
    main()
