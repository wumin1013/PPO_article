from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.geometry import generate_offset_paths
    from src.utils.path_generator import get_path_by_name
except Exception:
    generate_offset_paths = None
    get_path_by_name = None

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


def _as_xy(points: Optional[Sequence[Sequence[float]]]) -> tuple[np.ndarray, np.ndarray]:
    if points is None:
        empty = np.asarray([], dtype=float)
        return empty, empty
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        empty = np.asarray([], dtype=float)
        return empty, empty
    return _filter_xy(arr[:, 0], arr[:, 1])


def _subsample_xy(x: np.ndarray, y: np.ndarray, *, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    step = max(1, int(math.ceil(x.size / max_points)))
    return x[::step], y[::step]


def _save_figure(fig: plt.Figure, out_path: Path, *, dpi: int = 200, svg: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if svg:
        fig.savefig(out_path.with_suffix(".svg"))


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _bundle_root(bundle_path: Path) -> Path:
    if bundle_path.is_file():
        root = bundle_path.parent
        if root.name == "rollout_det":
            root = root.parent
        return root
    return bundle_path


def _load_bundle_config(bundle_path: Path) -> dict:
    root = _bundle_root(bundle_path)
    config_path = root / "config.yaml"
    if config_path.exists():
        return _load_yaml(config_path)
    config_eval_path = root / "config_eval.yaml"
    if config_eval_path.exists():
        return _load_yaml(config_eval_path)
    return {}


def _load_summary_half_epsilon(bundle_path: Path) -> Optional[float]:
    root = _bundle_root(bundle_path)
    candidates = [
        root / "eval" / "run2" / "summary_raw.json",
        root / "eval" / "run1" / "summary_raw.json",
        root / "smoke" / "summary_raw.json",
        root / "summary_raw.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        val = _safe_float(summary.get("half_epsilon"))
        if val is not None:
            return val
    return None


def _resolve_half_epsilon(config: dict, summary_half_epsilon: Optional[float]) -> Optional[float]:
    if summary_half_epsilon is not None:
        return summary_half_epsilon
    if isinstance(config, dict):
        val = _safe_float(config.get("half_epsilon"))
        if val is not None:
            return val
        env = config.get("environment")
        if isinstance(env, dict):
            epsilon = _safe_float(env.get("epsilon"))
            if epsilon is not None:
                return epsilon * 0.5
    return None


def _build_reference_from_config(config: dict) -> Optional[np.ndarray]:
    if not isinstance(config, dict) or get_path_by_name is None:
        return None
    path_cfg = config.get("path")
    if not isinstance(path_cfg, dict):
        return None
    path_type = path_cfg.get("type") or path_cfg.get("name")
    if not isinstance(path_type, str) or not path_type:
        return None
    scale = _safe_float(path_cfg.get("scale"))
    if scale is None:
        scale = 10.0
    num_points = path_cfg.get("num_points")
    try:
        num_points_int = int(num_points) if num_points is not None else 200
    except Exception:
        num_points_int = 200
    kwargs = path_cfg.get(path_type, {})
    if not isinstance(kwargs, dict):
        kwargs = {}
    if "closed" not in kwargs and "closed" in path_cfg:
        kwargs["closed"] = bool(path_cfg.get("closed"))
    try:
        points = get_path_by_name(path_type, scale=scale, num_points=num_points_int, **kwargs)
    except Exception:
        return None
    if not points:
        return None
    return np.asarray(points, dtype=float)


def _to_path_array(points: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    data = [np.asarray(p, dtype=float) for p in points if p is not None]
    if not data:
        return None
    return np.vstack(data)


def _resolve_reference_and_band(
    baseline: TraceData,
    candidate: TraceData,
    bundle_path: Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    config = _load_bundle_config(bundle_path)
    summary_half_epsilon = _load_summary_half_epsilon(bundle_path)
    half_epsilon = _resolve_half_epsilon(config, summary_half_epsilon)

    reference = _build_reference_from_config(config)
    if reference is None:
        ref_x, ref_y = _filter_xy(candidate.ref_x, candidate.ref_y)
        if ref_x.size == 0:
            ref_x, ref_y = _filter_xy(baseline.ref_x, baseline.ref_y)
        if ref_x.size > 0:
            reference = np.column_stack([ref_x, ref_y])

    left_band = None
    right_band = None
    if (
        reference is not None
        and half_epsilon is not None
        and half_epsilon > 0.0
        and generate_offset_paths is not None
    ):
        try:
            left_raw, right_raw = generate_offset_paths(reference, half_epsilon)
            left_band = _to_path_array(left_raw)
            right_band = _to_path_array(right_raw)
        except Exception:
            left_band = None
            right_band = None
    return reference, left_band, right_band


def _plot_tolerance_band(
    ax: plt.Axes,
    left_band: Optional[np.ndarray],
    right_band: Optional[np.ndarray],
) -> None:
    lx, ly = _as_xy(left_band)
    rx, ry = _as_xy(right_band)
    if lx.size == 0 or rx.size == 0:
        return
    has_fill = False
    if lx.size == rx.size:
        band_x = np.concatenate([lx, rx[::-1]])
        band_y = np.concatenate([ly, ry[::-1]])
        ax.fill(band_x, band_y, color="#74c0fc", alpha=0.12, linewidth=0, label="Tolerance band")
        has_fill = True
    line_label = None if has_fill else "Tolerance band"
    ax.plot(lx, ly, color="#74c0fc", linewidth=0.9, alpha=0.6, label=line_label)
    ax.plot(rx, ry, color="#74c0fc", linewidth=0.9, alpha=0.6)


def _plot_overlay(
    baseline: TraceData,
    candidate: TraceData,
    reference: Optional[np.ndarray],
    left_band: Optional[np.ndarray],
    right_band: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    _plot_tolerance_band(ax, left_band, right_band)
    ref_x, ref_y = _as_xy(reference)
    if ref_x.size > 0:
        ax.plot(ref_x, ref_y, "--", color="#1f77b4", linewidth=1.6, alpha=0.75, label="Reference")

    bx, by = _filter_xy(baseline.pos_x, baseline.pos_y)
    if bx.size > 0:
        ax.plot(bx, by, color="#6c757d", linewidth=1.6, linestyle="--", alpha=0.85, label=baseline.label)

    cx, cy = _filter_xy(candidate.pos_x, candidate.pos_y)
    if cx.size > 0:
        ax.plot(cx, cy, color="#e03131", linewidth=2.2, alpha=0.95, label=candidate.label)
        mx, my = _subsample_xy(cx, cy, max_points=400)
        if mx.size > 0:
            ax.scatter(mx, my, s=4, color="#e03131", alpha=0.5, edgecolors="none", zorder=3)

    ax.set_title("Overlay: Baseline vs Candidate")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=240, svg=True)
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
    _save_figure(fig, out_path, dpi=200)
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
    _save_figure(fig, out_path, dpi=200)
    plt.close(fig)


def _plot_points(
    baseline: TraceData,
    candidate: TraceData,
    reference: Optional[np.ndarray],
    left_band: Optional[np.ndarray],
    right_band: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    _plot_tolerance_band(ax, left_band, right_band)
    ref_x, ref_y = _as_xy(reference)
    if ref_x.size > 0:
        ax.plot(ref_x, ref_y, "--", color="#1f77b4", linewidth=1.4, alpha=0.6, label="Reference")

    bx, by = _filter_xy(baseline.pos_x, baseline.pos_y)
    if bx.size > 0:
        bx, by = _subsample_xy(bx, by, max_points=3000)
        ax.scatter(
            bx,
            by,
            s=3,
            marker="o",
            color="#6c757d",
            alpha=0.45,
            edgecolors="none",
            label=f"{baseline.label} points",
        )

    cx, cy = _filter_xy(candidate.pos_x, candidate.pos_y)
    if cx.size > 0:
        cx, cy = _subsample_xy(cx, cy, max_points=3000)
        ax.scatter(
            cx,
            cy,
            s=4,
            marker="^",
            color="#e03131",
            alpha=0.55,
            edgecolors="none",
            label=f"{candidate.label} points",
        )

    ax.set_title("Trajectory Points (Baseline vs Candidate)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=300, svg=True)
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

    ref_path, left_band, right_band = _resolve_reference_and_band(baseline, candidate, candidate_bundle)

    _plot_overlay(baseline, candidate, ref_path, left_band, right_band, out_dir / "overlay.png")
    _plot_points(baseline, candidate, ref_path, left_band, right_band, out_dir / "trajectory_points.png")
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
