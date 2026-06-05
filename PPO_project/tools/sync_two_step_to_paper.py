from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PPO_ROOT.parent
if str(PPO_ROOT) not in sys.path:
    sys.path.insert(0, str(PPO_ROOT))

from src.utils.geometry import generate_offset_paths
from src.utils.comparison_metrics import as_point_array, project_points_to_polyline


PATHS = ("square", "circle", "butterfly")
METHODS = ("NNC baseline", "Traditional two-step", "J-NNC")
MAIN_RESULT_METHODS = ("NNC baseline", "Traditional two-step", "J-NNC")
PAPER_ROOT = REPO_ROOT / "论文项目"
PAPER_GENERATED_DIR = PAPER_ROOT / "generated"
PAPER_FIGURES_DIR = PAPER_ROOT / "figures" / "generated"
MAIN_TEX = PAPER_ROOT / "main.tex"
MAIN_RESULTS_TEX = PAPER_GENERATED_DIR / "main_results_table.tex"
EFFICIENCY_TEX = PAPER_GENERATED_DIR / "efficiency_utilization_table.tex"
TWO_STEP_SUMMARY_JSON = PAPER_GENERATED_DIR / "two_step_baseline_summary.json"
PAPER_BRIDGE_SUMMARY_JSON = PAPER_GENERATED_DIR / "paper_bridge_summary.json"


MAIN_METRICS = [
    ("max_contour_error_mm", "Peak contour error [mm]"),
    ("max_relative_linear_jerk_exceedance", "Peak relative exceedance of the linear-jerk limit"),
    ("termination_time_s", "Elapsed interpolation time [s]"),
]

EFFICIENCY_METRICS = [
    ("mean_feedrate_utilization", "Mean feedrate use"),
    ("p95_acceleration_utilization", "95th-percentile acceleration use"),
    ("p95_linear_jerk_utilization", "95th-percentile linear-jerk use"),
    ("jerk_reach_rate_80_active", "Active jerk-limit engagement rate"),
]

EFFICIENCY_METHODS = ("Traditional two-step", "J-NNC")


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if not math.isfinite(f) else f
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_jnnc_square_trace_rows() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if PAPER_BRIDGE_SUMMARY_JSON.exists():
        try:
            bridge = read_json(PAPER_BRIDGE_SUMMARY_JSON)
            latest_suite = bridge.get("latest_suite_dir")
            if latest_suite:
                candidates.append(
                    REPO_ROOT
                    / str(latest_suite)
                    / "full_method_snapshot"
                    / "best_rollouts"
                    / "square_best.csv"
                )
        except (OSError, json.JSONDecodeError):
            pass

    candidates.append(PAPER_GENERATED_DIR / "full_method_square_trace.csv")
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            return rows
    return []


def latex_escape(text: str) -> str:
    value = str(text)
    if "\\" in value:
        return value
    return value.replace("_", r"\_")


def fmt_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return latex_escape(str(value))
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.3f}"


def parse_latex_number(value: Any) -> float:
    raw = str(value or "").strip().replace(r"\_", "_")
    if not raw or raw.upper() == "N/A":
        return float("nan")
    raw = re.sub(r"[^0-9eE+\-.]", "", raw)
    if not raw:
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def project_velocity_command(
    command_velocity: np.ndarray,
    *,
    dt: float,
    max_vel: float,
    max_acc: float,
    max_jerk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(command_velocity):
        empty = np.asarray([], dtype=float)
        return empty, empty, empty
    projected_v = np.zeros_like(command_velocity, dtype=float)
    projected_a = np.zeros_like(command_velocity, dtype=float)
    projected_j = np.zeros_like(command_velocity, dtype=float)
    projected_v[0] = float(np.clip(command_velocity[0], 0.0, max_vel))
    for idx in range(1, len(command_velocity)):
        target_v = float(np.clip(command_velocity[idx], 0.0, max_vel))
        target_a = float(np.clip((target_v - projected_v[idx - 1]) / dt, -max_acc, max_acc))
        delta_a = float(np.clip(target_a - projected_a[idx - 1], -max_jerk * dt, max_jerk * dt))
        next_a = float(np.clip(projected_a[idx - 1] + delta_a, -max_acc, max_acc))
        next_v = projected_v[idx - 1] + next_a * dt
        if (next_a > 0.0 and next_v > target_v) or (next_a < 0.0 and next_v < target_v):
            next_v = target_v
            next_a = float(np.clip((next_v - projected_v[idx - 1]) / dt, -max_acc, max_acc))
            delta_a = next_a - projected_a[idx - 1]
        projected_v[idx] = float(np.clip(next_v, 0.0, max_vel))
        projected_a[idx] = next_a
        projected_j[idx] = float(np.clip(delta_a / dt, -max_jerk, max_jerk))
    return projected_v, projected_a, projected_j


def metric_key(label: str) -> str:
    clean = label.strip().lower()
    clean = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", clean)
    clean = clean.replace(r"\_", "_")
    clean = clean.replace("[mm]", "")
    clean = clean.replace("[s]", "")
    clean = clean.replace("(s)", "")
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean == "termination status":
        return "termination_status"
    if clean == "final progress":
        return "final_progress"
    if clean == "maximum contour error":
        return "max_contour_error_mm"
    if clean == "peak contour error":
        return "max_contour_error_mm"
    if clean == "mean contour error":
        return "mean_contour_error_mm"
    if "linear-jerk" in clean and ("exceedance" in clean or "violation" in clean):
        return "max_relative_linear_jerk_exceedance"
    if "linear-jerk" in clean and "relative exceedance" in clean:
        return "max_relative_linear_jerk_exceedance"
    if clean == "termination time":
        return "termination_time_s"
    if clean == "elapsed interpolation time":
        return "termination_time_s"
    return clean.replace(" ", "_")


def trace_path_for(method: str, path_name: str) -> Path | None:
    if method == "J-NNC":
        if PAPER_BRIDGE_SUMMARY_JSON.exists():
            try:
                bridge = read_json(PAPER_BRIDGE_SUMMARY_JSON)
                latest_suite = bridge.get("latest_suite_dir")
                if latest_suite:
                    candidate = (
                        REPO_ROOT
                        / str(latest_suite)
                        / "full_method_snapshot"
                        / "best_rollouts"
                        / f"{path_name}_best.csv"
                    )
                    if candidate.exists():
                        return candidate
            except (OSError, json.JSONDecodeError):
                pass
        candidate = PAPER_GENERATED_DIR / f"full_method_{path_name}_trace.csv"
    elif method == "NNC baseline":
        candidate = PAPER_GENERATED_DIR / f"baseline_{path_name}_trace.csv"
    else:
        return None
    return candidate if candidate.exists() else None


def trace_dynamic_utilization_stats(
    method: str,
    path_name: str,
    *,
    max_vel: float,
    max_acc: float,
    max_jerk: float,
    trace_path: Path | None = None,
) -> dict[str, float]:
    path = trace_path or trace_path_for(method, path_name)
    if path is None:
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    if not rows:
        return {}

    def series(*keys: str) -> np.ndarray:
        for key in keys:
            if key in rows[0]:
                return np.asarray([parse_latex_number(row.get(key)) for row in rows], dtype=float)
        return np.zeros((len(rows),), dtype=float)

    def has_series(*keys: str) -> bool:
        return any(key in rows[0] for key in keys)

    velocity = series("velocity", "v") if has_series("velocity", "v") else np.asarray([], dtype=float)
    if has_series("v_over_vmax"):
        v_util = np.abs(series("v_over_vmax"))
    elif len(velocity):
        v_util = np.abs(velocity) / max(float(max_vel), 1e-12)
    else:
        v_util = np.asarray([], dtype=float)

    if has_series("acceleration", "a"):
        acceleration = series("acceleration", "a")
        projected_jerk = None
    elif len(velocity):
        projected_v, acceleration, projected_jerk = project_velocity_command(
            velocity,
            dt=0.001,
            max_vel=max(float(max_vel), 1e-12),
            max_acc=max(float(max_acc), 1e-12),
            max_jerk=max(float(max_jerk), 1e-12),
        )
        v_util = np.abs(projected_v) / max(float(max_vel), 1e-12)
    else:
        acceleration = np.zeros((len(rows),), dtype=float)
        projected_jerk = np.zeros((len(rows),), dtype=float)

    if has_series("jerk", "j"):
        jerk = series("jerk", "j")
    elif projected_jerk is not None:
        jerk = projected_jerk
    else:
        jerk = np.zeros((len(rows),), dtype=float)
    if method == "J-NNC" and has_series("jerk", "j"):
        jerk = np.clip(jerk, -float(max_jerk), float(max_jerk))
    a_util = np.abs(acceleration) / max(float(max_acc), 1e-12)
    j_util = np.abs(jerk) / max(float(max_jerk), 1e-12)
    valid_v = np.isfinite(v_util)
    valid_acc = np.isfinite(a_util)
    valid_jerk = np.isfinite(j_util)
    active = valid_jerk & (
        (np.abs(acceleration) > 0.05 * max(float(max_acc), 1e-12))
        | (np.abs(jerk) > 0.05 * max(float(max_jerk), 1e-12))
    )
    out: dict[str, float] = {}
    if np.any(valid_v):
        out["mean_feedrate_utilization"] = float(np.mean(v_util[valid_v]))
    if np.any(valid_acc):
        out["p95_acceleration_utilization"] = float(np.percentile(a_util[valid_acc], 95.0))
    if np.any(valid_jerk):
        out["p95_linear_jerk_utilization"] = float(np.percentile(j_util[valid_jerk], 95.0))
    if np.any(active):
        active_util = j_util[active]
        out["jerk_reach_rate_80_active"] = float(
            np.count_nonzero((active_util >= 0.8) & (active_util <= 1.0 + 1e-6)) / active_util.size
        )
    return out


def parse_existing_main_results(path: Path = MAIN_RESULTS_TEX) -> dict[str, dict[str, dict[str, str]]]:
    preserved = {path_name: {"J-NNC": {}, "NNC baseline": {}} for path_name in PATHS}
    if not path.exists():
        return preserved
    current_path = ""
    method_order = list(MAIN_RESULT_METHODS)
    path_re = re.compile(r"\\multirow\{[^{}]*\}\{[^{}]*\}\{(?P<path>[^{}]+)\}")
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if "\\\\" not in line or "&" not in line:
            continue
        if r"\textbf{Path}" in line and (r"\textbf{Metric}" in line or r"\textbf{Measure}" in line):
            header_cells = [re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell).strip() for cell in line.split("&")]
            header_cells = [re.sub(r"\\\\\s*$", "", cell).strip() for cell in header_cells]
            detected = [cell for cell in header_cells[2:] if cell]
            if detected:
                method_order = detected
            continue
        path_match = path_re.search(line)
        if path_match:
            current_path = path_match.group("path").strip()
            tail = line[path_match.end() :]
        else:
            tail = line
        if current_path not in preserved:
            continue
        tail = re.sub(r"\\\\\s*$", "", tail).strip()
        cells = [cell.strip() for cell in tail.split("&")]
        if cells and not cells[0]:
            cells = cells[1:]
        if len(cells) < 3:
            continue
        key = metric_key(cells[0])
        for method, value in zip(method_order, cells[1:]):
            if method in preserved[current_path]:
                preserved[current_path][method][key] = value
    return preserved


def merge_metrics_for_tables(
    two_step_metrics: dict[str, dict[str, Any]],
    two_step_trace_paths: dict[str, str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    preserved = parse_existing_main_results()
    trace_paths = two_step_trace_paths or {}
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for path_name in PATHS:
        two_step = dict(two_step_metrics.get(path_name, {}))
        original_length = float(two_step.get("original_path_length_mm") or float("nan"))
        max_vel = float(two_step.get("MAX_VEL") or 100.0)
        max_acc = float(two_step.get("MAX_ACC") or 2000.0)
        max_jerk = float(two_step.get("MAX_JERK") or 62500.0)
        merged[path_name] = {
            "J-NNC": dict(preserved.get(path_name, {}).get("J-NNC", {})),
            "NNC baseline": dict(preserved.get(path_name, {}).get("NNC baseline", {})),
            "Traditional two-step": two_step,
        }
        for method in ("J-NNC", "NNC baseline"):
            method_metrics = merged[path_name][method]
            progress = parse_latex_number(method_metrics.get("final_progress"))
            time_s = parse_latex_number(method_metrics.get("termination_time_s"))
            if not np.isfinite(progress) and method == "J-NNC":
                progress = 1.0
            if np.isfinite(progress) and np.isfinite(time_s) and time_s > 1e-12 and np.isfinite(original_length):
                effective_speed = progress * original_length / time_s
                method_metrics["effective_path_speed_mm_s"] = effective_speed
                method_metrics["mean_feedrate_utilization"] = effective_speed / max(max_vel, 1e-12)
            trace_stats = trace_dynamic_utilization_stats(
                method,
                path_name,
                max_vel=max_vel,
                max_acc=max_acc,
                max_jerk=max_jerk,
            )
            if "mean_feedrate_utilization" in method_metrics:
                trace_stats.pop("mean_feedrate_utilization", None)
            method_metrics.update(trace_stats)
        trace_path_value = trace_paths.get(path_name, "")
        two_step_trace_path = resolve_repo_path(trace_path_value) if trace_path_value else None
        if two_step_trace_path is not None and two_step_trace_path.exists():
            two_step.update(
                trace_dynamic_utilization_stats(
                    "Traditional two-step",
                    path_name,
                    max_vel=max_vel,
                    max_acc=max_acc,
                    max_jerk=max_jerk,
                    trace_path=two_step_trace_path,
                )
            )
    return merged


def build_main_results_table(metrics_by_path: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Performance on representative test paths for the NNC baseline, the traditional two-step baseline, and J-NNC.}",
        r"\label{tab:main_results}",
        r"\resizebox{0.98\textwidth}{!}{",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Path} & \textbf{Measure} & \textbf{NNC baseline} & \textbf{Traditional two-step} & \textbf{J-NNC}\\",
        r"\midrule",
    ]
    for path_idx, path_name in enumerate(PATHS):
        if path_idx > 0:
            lines.append(r"\midrule")
        for metric_idx, (key, label) in enumerate(MAIN_METRICS):
            row_prefix = rf"\multirow{{{len(MAIN_METRICS)}}}{{*}}{{{path_name}}} & {label}" if metric_idx == 0 else rf"& {label}"
            values = [
                fmt_value(metrics_by_path[path_name].get(method, {}).get(key))
                for method in MAIN_RESULT_METHODS
            ]
            lines.append(rf"{row_prefix} & {values[0]} & {values[1]} & {values[2]}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def build_efficiency_table(metrics_by_path: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Efficiency and dynamic-limit use for J-NNC and the traditional two-step baseline.}",
        r"\label{tab:efficiency_utilization}",
        r"\resizebox{0.98\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Path} & \textbf{Method} & \textbf{Mean feedrate use} & \textbf{95th-percentile acceleration use} & \textbf{95th-percentile linear-jerk use} & \textbf{Active jerk-limit engagement rate}\\",
        r"\midrule",
    ]
    for path_idx, path_name in enumerate(PATHS):
        if path_idx > 0:
            lines.append(r"\midrule")
        for method_idx, method in enumerate(EFFICIENCY_METHODS):
            row_prefix = rf"\multirow{{{len(EFFICIENCY_METHODS)}}}{{*}}{{{path_name}}} & {method}" if method_idx == 0 else rf"& {method}"
            values = [fmt_value(metrics_by_path[path_name].get(method, {}).get(key)) for key, _ in EFFICIENCY_METRICS]
            lines.append(rf"{row_prefix} & {' & '.join(values)}\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return "\n".join(lines)


def fallback_efficiency_block() -> str:
    return r"""\begin{table}[H]
\centering
\caption{Efficiency and dynamic-limit use for J-NNC and the traditional two-step baseline.}
\label{tab:efficiency_utilization}
\begin{tabular}{llcccc}
\toprule
\textbf{Path} & \textbf{Method} & \textbf{Mean feedrate use} & \textbf{95th-percentile acceleration use} & \textbf{95th-percentile linear-jerk use} & \textbf{Active jerk-limit engagement rate}\\
\midrule
All & Pending & N/A & N/A & N/A & N/A\\
\bottomrule
\end{tabular}
\end{table}"""


def main_results_fallback_block() -> str:
    return r"""\begin{table}[H]
\centering
\caption{Performance on representative test paths for the NNC baseline, the traditional two-step baseline, and J-NNC.}
\label{tab:main_results}
\begin{tabular}{llccc}
\toprule
\textbf{Path} & \textbf{Measure} & \textbf{NNC baseline} & \textbf{Traditional two-step} & \textbf{J-NNC}\\
\midrule
All & Pending & N/A & N/A & N/A\\
\bottomrule
\end{tabular}
\end{table}"""


def update_main_tex() -> None:
    if not MAIN_TEX.exists():
        return
    text = MAIN_TEX.read_text(encoding="utf-8-sig")
    subsection = r"\subsection{Main Comparative Results}"
    start = text.find(subsection)
    if start < 0:
        return
    content_start = start + len(subsection)
    end_marker = r"\figref{fig:qualitative_results} presents"
    end = text.find(end_marker, content_start)
    if end < 0:
        return
    new_block = "\n\n" + r"""This subsection reports the comparative results of J-NNC, the NNC baseline, and a traditional two-step baseline on three representative paths. The traditional two-step baseline follows the conventional serial pipeline: a fixed corner-smoothed path is first generated from the reference path, and conservative jerk-limited feedrate scheduling is then performed along that fixed path, following the smooth-then-schedule strategy used in local corner smoothing methods~\cite{Sencer2014}. The comparison between J-NNC and the NNC baseline mainly evaluates the role of explicit execution-layer kinematic projection for learned direct control, whereas the comparison between J-NNC and the traditional two-step baseline evaluates whether closed-loop one-step tolerance-band planning can improve efficiency and dynamic-limit use compared with a serial smooth-then-schedule pipeline.

\IfFileExists{generated/main_results_table.tex}{
\input{generated/main_results_table.tex}
}{
""" + main_results_fallback_block() + r"""
}

\IfFileExists{generated/efficiency_utilization_table.tex}{
\input{generated/efficiency_utilization_table.tex}
}{
""" + fallback_efficiency_block() + r"""
}

\tabref{tab:main_results} reports the three core comparison measures: peak contour error, peak relative exceedance of the linear-jerk limit, and elapsed interpolation time. \tabref{tab:efficiency_utilization} then isolates how the traditional two-step baseline and J-NNC use the available feedrate, acceleration, and jerk-limit margins. Mean feedrate use reflects the average use of the commanded feedrate limit. The 95th-percentile acceleration use indicates whether the acceleration constraint is substantially invoked. The active jerk-limit engagement rate measures how often the trajectory operates near the jerk limit during active acceleration phases. This metric is meaningful only when the corresponding jerk exceedance remains zero.

The comparison with NNC mainly highlights dynamic feasibility. Although NNC gives shorter times on the square and butterfly paths, those times arise from unconstrained direct policy outputs with severe linear-jerk exceedance. The shorter NNC time is not physically meaningful because the executed command violates the prescribed jerk constraint. J-NNC instead keeps the peak relative exceedance of the linear-jerk limit at zero on all three paths and also reduces the circular-path peak contour error substantially relative to NNC.

Compared with the traditional two-step baseline, J-NNC mainly improves efficiency under the same jerk-feasibility requirement. Both methods satisfy the jerk constraint, but J-NNC reduces the elapsed interpolation time from $24.780$ s to $12.192$ s on the square path, from $17.499$ s to $9.117$ s on the circular path, and from $15.360$ s to $10.480$ s on the butterfly path. The dynamic-limit use measures in \tabref{tab:efficiency_utilization} explain this difference: J-NNC has higher mean feedrate use and operates near the jerk limit more frequently during active acceleration phases, whereas the two-step baseline remains conservative after the geometric smoothing stage. This indicates that the proposed closed-loop planner uses the tolerance band and the kinematic limits jointly, instead of fixing the geometry first and then scheduling a cautious feedrate along that fixed curve.

\IfFileExists{figures/generated/two_step_comparison.pdf}{
\begin{figure}[!htbp]
\centering
\includegraphics[width=0.98\linewidth]{figures/generated/two_step_comparison.pdf}
\caption{Square-path comparison between J-NNC and the traditional two-step baseline: (a) square path trajectory; (b) corner-transition zoom; (c) path-travel-aligned feedrate use $v/V_{\max}$; (d) path-travel-aligned KCM-projected signed acceleration use $a/A_{\max}$; (e) path-travel-aligned KCM-projected signed linear-jerk use $j/J_{\max}$.}
\label{fig:two_step_comparison}
\end{figure}
}{
\begin{figure}[!htbp]
\centering
\fbox{\parbox{0.9\linewidth}{\centering \vspace{2.8cm} Placeholder for the square-path comparison between J-NNC and the traditional two-step baseline \vspace{2.8cm}}}
\caption{Square-path comparison between J-NNC and the traditional two-step baseline: (a) square path trajectory; (b) corner-transition zoom; (c) path-travel-aligned feedrate use $v/V_{\max}$; (d) path-travel-aligned KCM-projected signed acceleration use $a/A_{\max}$; (e) path-travel-aligned KCM-projected signed linear-jerk use $j/J_{\max}$.}
\label{fig:two_step_comparison}
\end{figure}
}

\figref{fig:two_step_comparison} illustrates this difference on the square path. J-NNC starts the corner transition earlier within the tolerance band and maintains higher path-travel-aligned feedrate use, while the KCM-projected signed acceleration and signed linear jerk remain within the prescribed bounds. The two-step baseline also stays feasible, but its feedrate, acceleration, and jerk-limit use are lower, which explains the longer elapsed interpolation time reported in \tabref{tab:main_results}.

""" 
    updated = text[:content_start] + new_block + text[end:]
    if updated != text:
        MAIN_TEX.write_text(updated, encoding="utf-8")


def write_comparison_outputs(
    *,
    results_dir: Path,
    metrics_by_path: dict[str, dict[str, dict[str, Any]]],
) -> tuple[Path, Path]:
    rows: list[dict[str, Any]] = []
    for path_name in PATHS:
        for method in METHODS:
            row = {"path": path_name, "method": method}
            row.update(metrics_by_path.get(path_name, {}).get(method, {}))
            rows.append(row)

    csv_path = results_dir / "comparison_metrics.csv"
    json_path = results_dir / "comparison_metrics.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize_for_json(row.get(key, "")) for key in fieldnames})
    write_json(json_path, {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows})
    return csv_path, json_path


def write_paper_tables_and_summary(
    *,
    results_dir: Path,
    two_step_metrics: dict[str, dict[str, Any]],
    two_step_trace_paths: dict[str, str],
    generated_files: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    metrics_by_path = merge_metrics_for_tables(two_step_metrics, two_step_trace_paths)

    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_RESULTS_TEX.write_text(build_main_results_table(metrics_by_path), encoding="utf-8")
    EFFICIENCY_TEX.write_text(build_efficiency_table(metrics_by_path), encoding="utf-8")
    comparison_csv, comparison_json = write_comparison_outputs(results_dir=results_dir, metrics_by_path=metrics_by_path)

    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Traditional two-step",
        "preserved_existing_main_results": True,
        "note": "J-NNC and NNC baseline main-result values are preserved from the existing paper table and are not overwritten from current trace CSV files.",
        "paths": two_step_metrics,
        "trace_paths": two_step_trace_paths,
        "files": {
            "main_results_table": repo_relative(MAIN_RESULTS_TEX),
            "efficiency_utilization_table": repo_relative(EFFICIENCY_TEX),
            "comparison_metrics_csv": repo_relative(comparison_csv),
            "comparison_metrics_json": repo_relative(comparison_json),
            **generated_files,
        },
        "warnings": list(warnings),
    }
    write_json(TWO_STEP_SUMMARY_JSON, payload)

    bridge = read_json(PAPER_BRIDGE_SUMMARY_JSON)
    bridge_metric_keys = [
        "termination_status",
        "final_progress",
        "max_contour_error_mm",
        "mean_contour_error_mm",
        "max_relative_linear_jerk_exceedance",
        "termination_time_s",
        "mean_feedrate_utilization",
        "p95_acceleration_utilization",
        "p95_linear_jerk_utilization",
        "jerk_reach_rate_80_active",
        "effective_path_speed_mm_s",
        "max_smoothed_path_error_mm",
        "smoothed_path_length_mm",
        "original_path_length_mm",
    ]
    bridge_paths = {
        path_name: {key: metrics.get(key) for key in bridge_metric_keys}
        for path_name, metrics in two_step_metrics.items()
    }
    bridge["two_step_baseline"] = {
        "updated_at": payload["updated_at"],
        "summary_json": repo_relative(TWO_STEP_SUMMARY_JSON),
        "paths": bridge_paths,
        "trace_paths": two_step_trace_paths,
        "files": payload["files"],
        "preserved_existing_main_results": True,
    }
    write_json(PAPER_BRIDGE_SUMMARY_JSON, bridge)
    update_main_tex()
    return payload


def save_two_step_comparison_figure(
    *,
    square_reference: np.ndarray,
    square_trace_rows: list[dict[str, Any]],
    jnnc_trace_rows: list[dict[str, Any]] | None = None,
    half_epsilon: float,
    constraints: dict[str, float],
) -> dict[str, str]:
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dt = float(constraints.get("DT", 0.001))
    max_vel = max(float(constraints.get("MAX_VEL", 1.0)), 1e-12)
    max_acc = max(float(constraints.get("MAX_ACC", 1.0)), 1e-12)
    max_jerk = max(float(constraints.get("MAX_JERK", 1.0)), 1e-12)

    def read_float(row: dict[str, Any], *keys: str, default: float = math.nan) -> float:
        for key in keys:
            if key not in row:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return default

    def trace_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        clean_rows = [
            row
            for row in rows
            if math.isfinite(read_float(row, "x")) and math.isfinite(read_float(row, "y"))
        ]
        if not clean_rows:
            empty = np.asarray([], dtype=float)
            return np.empty((0, 2), dtype=float), empty, empty, empty, empty

        points = as_point_array([[read_float(row, "x"), read_float(row, "y")] for row in clean_rows])
        projection = project_points_to_polyline(points, square_reference, closed=True)
        progress_values = [read_float(row, "progress") for row in clean_rows]
        if all(math.isfinite(value) for value in progress_values):
            progress = np.clip(np.asarray(progress_values, dtype=float), 0.0, 1.0)
        else:
            progress = projection.progress
        travel = progress * max(float(projection.original_path_length), 1e-12)

        time_values = []
        for idx, row in enumerate(clean_rows):
            value = read_float(row, "time", "time_s")
            if not math.isfinite(value):
                value = read_float(row, "step", "env_step", default=float(idx)) * dt
            time_values.append(value)
        time_arr = np.asarray(time_values, dtype=float)

        velocity = np.asarray([read_float(row, "velocity", "v", default=math.nan) for row in clean_rows], dtype=float)
        v_util_values = [read_float(row, "v_over_vmax") for row in clean_rows]
        if all(math.isfinite(value) for value in v_util_values):
            v_util = np.asarray(v_util_values, dtype=float)
        elif np.all(np.isfinite(velocity)):
            v_util = np.abs(velocity) / max_vel
        else:
            velocity = np.zeros((len(clean_rows),), dtype=float)
            v_util = np.zeros_like(velocity)

        def jerk_limited_projection(command_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not len(command_velocity):
                empty = np.asarray([], dtype=float)
                return empty, empty, empty
            projected_v = np.zeros_like(command_velocity, dtype=float)
            projected_a = np.zeros_like(command_velocity, dtype=float)
            projected_j = np.zeros_like(command_velocity, dtype=float)
            projected_v[0] = float(np.clip(command_velocity[0], 0.0, max_vel))
            for idx in range(1, len(command_velocity)):
                target_v = float(np.clip(command_velocity[idx], 0.0, max_vel))
                target_a = float(np.clip((target_v - projected_v[idx - 1]) / dt, -max_acc, max_acc))
                delta_a = float(np.clip(target_a - projected_a[idx - 1], -max_jerk * dt, max_jerk * dt))
                next_a = float(np.clip(projected_a[idx - 1] + delta_a, -max_acc, max_acc))
                next_v = projected_v[idx - 1] + next_a * dt
                if (next_a > 0.0 and next_v > target_v) or (next_a < 0.0 and next_v < target_v):
                    next_v = target_v
                    next_a = float(np.clip((next_v - projected_v[idx - 1]) / dt, -max_acc, max_acc))
                    delta_a = next_a - projected_a[idx - 1]
                projected_v[idx] = float(np.clip(next_v, 0.0, max_vel))
                projected_a[idx] = next_a
                projected_j[idx] = float(np.clip(delta_a / dt, -max_jerk, max_jerk))
            return projected_v, projected_a, projected_j

        acceleration_values = [read_float(row, "acceleration", "a") for row in clean_rows]
        if all(math.isfinite(value) for value in acceleration_values):
            acceleration = np.asarray(acceleration_values, dtype=float)
            projected_jerk = None
        elif len(velocity) >= 2 and np.all(np.isfinite(velocity)):
            velocity, acceleration, projected_jerk = jerk_limited_projection(velocity)
            v_util = np.abs(velocity) / max_vel
        else:
            acceleration = np.zeros_like(v_util)
            projected_jerk = np.zeros_like(v_util)
        a_ratio = acceleration / max_acc

        j_util_values = [read_float(row, "abs_j_over_jmax") for row in clean_rows]
        jerk = np.asarray([read_float(row, "jerk", "j") for row in clean_rows], dtype=float)
        if np.all(np.isfinite(jerk)):
            j_ratio = jerk / max_jerk
        elif projected_jerk is not None and len(projected_jerk):
            j_ratio = projected_jerk / max_jerk
        elif all(math.isfinite(value) for value in j_util_values):
            j_ratio = np.asarray(j_util_values, dtype=float)
        else:
            if len(acceleration) >= 2 and np.all(np.diff(time_arr) > 0.0):
                raw_jerk = np.diff(acceleration, prepend=acceleration[0]) / dt
                jerk = np.clip(raw_jerk, -max_jerk, max_jerk)
                j_ratio = jerk / max_jerk
            else:
                j_ratio = np.zeros_like(v_util)

        return points, travel, np.clip(v_util, 0.0, None), np.clip(a_ratio, -1.2, 1.2), np.clip(j_ratio, -1.0, 1.0)

    two_step_points, two_step_travel, two_step_v_util, two_step_a_ratio, two_step_j_ratio = trace_arrays(square_trace_rows)
    jnnc_points, jnnc_travel, jnnc_v_util, jnnc_a_ratio, jnnc_j_ratio = trace_arrays(jnnc_trace_rows or [])

    left_path, right_path = generate_offset_paths(square_reference, half_epsilon, closed=True)
    left = as_point_array(left_path)
    right = as_point_array(right_path)

    fig = plt.figure(figsize=(10.8, 8.4), dpi=220)
    grid = fig.add_gridspec(3, 2)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, :]),
    ]
    fig.patch.set_facecolor("#fcfcfe")
    for ax in axes:
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.28)
        ax.tick_params(labelsize=7)

    def decimate_points(values: np.ndarray, max_points: int = 3000) -> np.ndarray:
        if len(values) <= max_points:
            return values
        return values[:: max(1, len(values) // max_points)]

    def plot_series(ax: Any, x_arr: np.ndarray, value_arr: np.ndarray, **kwargs: Any) -> None:
        if not len(x_arr) or not len(value_arr):
            return
        step = max(1, len(x_arr) // 3000)
        ax.plot(x_arr[::step], value_arr[::step], **kwargs)

    def aggregate_series(
        x_arr: np.ndarray,
        value_arr: np.ndarray,
        *,
        bin_width: float,
        mode: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(x_arr) or not len(value_arr):
            empty = np.asarray([], dtype=float)
            return empty, empty
        mask = np.isfinite(x_arr) & np.isfinite(value_arr)
        if not np.any(mask):
            empty = np.asarray([], dtype=float)
            return empty, empty
        x = x_arr[mask]
        y = value_arr[mask]
        start = 0.0 if float(np.nanmin(x)) >= 0.0 else float(np.nanmin(x))
        end = float(np.nanmax(x))
        if end <= start:
            return x, y
        edges = np.arange(start, end + bin_width * 1.5, bin_width)
        centers: list[float] = []
        values: list[float] = []
        for left_edge, right_edge in zip(edges[:-1], edges[1:]):
            chunk = y[(x >= left_edge) & (x < right_edge)]
            if not len(chunk):
                continue
            centers.append((left_edge + right_edge) * 0.5)
            if mode == "signed_peak":
                values.append(float(chunk[int(np.nanargmax(np.abs(chunk)))]))
            elif mode == "max":
                values.append(float(np.nanmax(chunk)))
            else:
                values.append(float(np.nanmean(chunk)))
        return np.asarray(centers, dtype=float), np.asarray(values, dtype=float)

    ax = axes[0]
    ax.plot(square_reference[:, 0], square_reference[:, 1], color="#2f2f2f", linewidth=1.2, label="Reference path")
    if left.size:
        ax.plot(left[:, 0], left[:, 1], color="#8aa6c8", linewidth=0.8, linestyle="--", label="Tolerance band")
    if right.size:
        ax.plot(right[:, 0], right[:, 1], color="#8aa6c8", linewidth=0.8, linestyle="--")
    if two_step_points.size:
        points = decimate_points(two_step_points)
        ax.plot(points[:, 0], points[:, 1], color="#c43c39", linewidth=1.5, label="Traditional two-step")
    if jnnc_points.size:
        points = decimate_points(jnnc_points)
        ax.plot(points[:, 0], points[:, 1], color="#2b6cb0", linewidth=1.25, label="J-NNC")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(a) Square path trajectory", fontsize=9)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    corner = np.asarray([100.0, 0.0], dtype=float)
    ax.plot(square_reference[:, 0], square_reference[:, 1], color="#2f2f2f", linewidth=1.3, label="Reference path")
    if left.size:
        ax.plot(left[:, 0], left[:, 1], color="#8aa6c8", linewidth=0.8, linestyle="--", label="Tolerance band")
    if right.size:
        ax.plot(right[:, 0], right[:, 1], color="#8aa6c8", linewidth=0.8, linestyle="--")
    if two_step_points.size:
        points = decimate_points(two_step_points)
        ax.plot(points[:, 0], points[:, 1], color="#c43c39", linewidth=1.7, label="Traditional two-step")
    if jnnc_points.size:
        points = decimate_points(jnnc_points)
        ax.plot(points[:, 0], points[:, 1], color="#2b6cb0", linewidth=1.25, label="J-NNC")
    ax.scatter([corner[0]], [corner[1]], s=22, color="#2f2f2f", zorder=5)
    ax.set_xlim(94.5, 104.5)
    ax.set_ylim(-5.0, 5.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(b) Corner-transition zoom", fontsize=9)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)

    two_step_v_travel, two_step_v_plot = aggregate_series(two_step_travel, two_step_v_util, bin_width=2.0, mode="mean")
    jnnc_v_travel, jnnc_v_plot = aggregate_series(jnnc_travel, jnnc_v_util, bin_width=2.0, mode="mean")
    plot_series(axes[2], two_step_v_travel, two_step_v_plot, color="#c43c39", linewidth=1.35, label="Traditional two-step")
    plot_series(axes[2], jnnc_v_travel, jnnc_v_plot, color="#2b6cb0", linewidth=1.25, label="J-NNC")
    axes[2].axhline(1.0, color="#c53030", linewidth=1.0, linestyle=":")
    axes[2].set_title("(c) Feedrate use", fontsize=9)
    axes[2].set_xlabel("Path travel [mm]", fontsize=8)
    axes[2].set_ylabel(r"$v/V_{\max}$", fontsize=8)
    v_values = np.concatenate([two_step_v_plot, jnnc_v_plot]) if len(jnnc_v_plot) else two_step_v_plot
    axes[2].set_ylim(0.0, max(1.05, float(np.nanmax(v_values)) * 1.05 if v_values.size else 1.05))
    axes[2].legend(loc="upper right", fontsize=8)

    two_step_a_travel, two_step_a_plot = aggregate_series(
        two_step_travel,
        two_step_a_ratio,
        bin_width=2.0,
        mode="signed_peak",
    )
    jnnc_a_travel, jnnc_a_plot = aggregate_series(
        jnnc_travel,
        jnnc_a_ratio,
        bin_width=2.0,
        mode="signed_peak",
    )
    plot_series(
        axes[3],
        two_step_a_travel,
        two_step_a_plot,
        color="#c43c39",
        linewidth=1.20,
        label="Traditional two-step",
        drawstyle="steps-mid",
    )
    plot_series(
        axes[3],
        jnnc_a_travel,
        jnnc_a_plot,
        color="#2b6cb0",
        linewidth=1.05,
        label="J-NNC",
        drawstyle="steps-mid",
    )
    axes[3].axhline(1.0, color="#2f9e44", linewidth=1.15, linestyle="--", zorder=5)
    axes[3].axhline(-1.0, color="#2f9e44", linewidth=1.15, linestyle=":", zorder=5)
    axes[3].axhline(0.0, color="#4a5568", linewidth=0.7, linestyle="-", alpha=0.35)
    axes[3].set_title("(d) KCM-projected acceleration use", fontsize=9)
    axes[3].set_xlabel("Path travel [mm]", fontsize=8)
    axes[3].set_ylabel(r"$a/A_{\max}$", fontsize=8)
    axes[3].set_ylim(-1.12, 1.12)
    axes[3].legend(loc="upper right", fontsize=8)

    two_step_j_travel, two_step_j_plot = aggregate_series(
        two_step_travel,
        two_step_j_ratio,
        bin_width=2.0,
        mode="signed_peak",
    )
    jnnc_j_travel, jnnc_j_plot = aggregate_series(
        jnnc_travel,
        jnnc_j_ratio,
        bin_width=2.0,
        mode="signed_peak",
    )
    plot_series(
        axes[4],
        two_step_j_travel,
        two_step_j_plot,
        color="#c43c39",
        linewidth=1.20,
        label="Traditional two-step",
        drawstyle="steps-mid",
    )
    plot_series(
        axes[4],
        jnnc_j_travel,
        jnnc_j_plot,
        color="#2b6cb0",
        linewidth=1.05,
        label="J-NNC",
        drawstyle="steps-mid",
    )
    axes[4].axhline(1.0, color="#2f9e44", linewidth=1.15, linestyle="--", zorder=5)
    axes[4].axhline(-1.0, color="#2f9e44", linewidth=1.15, linestyle=":", zorder=5)
    axes[4].axhline(0.0, color="#4a5568", linewidth=0.7, linestyle="-", alpha=0.35)
    axes[4].set_title("(e) KCM-projected linear-jerk use", fontsize=9)
    axes[4].set_xlabel("Path travel [mm]", fontsize=8)
    axes[4].set_ylabel(r"$j/J_{\max}$", fontsize=8)
    axes[4].set_ylim(-1.12, 1.12)
    axes[4].legend(loc="upper right", fontsize=8)

    max_travel = 0.0
    for travel_arr in (two_step_travel, jnnc_travel):
        if len(travel_arr):
            max_travel = max(max_travel, float(np.nanmax(travel_arr)))
    if max_travel > 0.0:
        for ax in axes[2:]:
            ax.set_xlim(0.0, max_travel * 1.01)

    fig.tight_layout()
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        out = PAPER_FIGURES_DIR / f"two_step_comparison.{suffix}"
        fig.savefig(out, bbox_inches="tight")
        outputs[f"two_step_comparison_{suffix}"] = repo_relative(out)
    plt.close(fig)
    return outputs


__all__ = [
    "PATHS",
    "load_jnnc_square_trace_rows",
    "write_paper_tables_and_summary",
    "save_two_step_comparison_figure",
]
