"""Draw the registered v8 Figure 9 from all recorded 1 ms command trajectories.

The full 28-cell input is required and validated before any output is written.
This script changes neither manuscript text nor experiment completion criteria.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from generate_frozen_main_v6 import COLOR, LABEL, STYLE


ROOT = Path(__file__).resolve().parents[1]
CASES = ("piecewise_curvature_coupon_scale10", "asymmetric_neck_contour_scale10")
TITLES = ("Curvature coupon", "Neck contour")
METHODS = ("JNN", "FIXED", "S2", "S2R", "SCMO", "COPP", "NNC")
RESOLUTIONS = (0.001, 0.0005)
PRIMARY_DT = 0.001
CONTRACT = Path(__file__).with_name("figure9_scaled_v8_contract.md")
STYLE_SOURCE = Path(__file__).with_name("generate_frozen_main_v6.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bound_path(name: str, expected: str) -> Path:
    path = ROOT / name
    require(path.is_file(), f"Missing input: {path}")
    require(sha(path) == expected, f"Input SHA mismatch: {path}")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_validated(comparison_path: Path, registration_path: Path) -> tuple:
    comparison, registration = read_json(comparison_path), read_json(registration_path)
    require(comparison["registration_sha256"] == sha(registration_path), "Registration SHA mismatch")
    require(registration["study_id"] == "scaled_representative_v8", "Wrong study")
    require(registration["geometry_scale"] == 10, "Geometry scale changed")
    require(registration["nominal_tolerance_mm"] == 1.25, "Nominal tolerance changed")
    require(registration["planar_vector_limits"] == [100., 2000., 62500.], "Vector limits changed")
    require(comparison["all_28_cells_included"] is True, "Incomplete comparison")
    expected = {(case, method, dt) for case in CASES for method in METHODS for dt in RESOLUTIONS}
    grid = {(r["case"], r["method"].upper(), r["dt_s"]) for r in registration["grid"]}
    rows = comparison["rows"]
    actual = {(r["case"], r["method"], r["dt_s"]) for r in rows}
    require(len(rows) == len(actual) == 28 and actual == grid == expected, "Wrong or duplicated comparison grid")
    definitions = {item["id"]: item for item in registration["cases"]}
    require(set(definitions) == set(CASES), "Wrong case definitions")
    references, sources, plots, audit = {}, {}, {}, []
    for case, item in definitions.items():
        path = bound_path(item["reference"], item["reference_sha256"])
        bound_path(item["geometry"], item["geometry_sha256"])
        reference = np.loadtxt(path, delimiter=",", skiprows=1)
        require(reference.ndim == 2 and reference.shape[1] == 2 and np.isfinite(reference).all(), "Invalid reference")
        references[case] = reference
        sources[item["reference"]] = item["reference_sha256"]
        sources[item["geometry"]] = item["geometry_sha256"]
    for row in rows:
        case, method, dt = row["case"], row["method"], row["dt_s"]
        key = f"{case}__{method}__dt{round(dt * 1e6)}us"
        metric = row["metrics"]
        success = metric["success"]
        require(isinstance(success, bool), f"Invalid completion status: {key}")
        require(success == (metric["complete_time_s"] is not None), f"Completion/prefix mix-up: {key}")
        if row.get("record"):
            bound_path(row["record"], row["record_sha256"])
            sources[row["record"]] = row["record_sha256"]
        entry = dict(case=case, method=method, dt_s=dt, success=success,
                     complete_time_s=metric["complete_time_s"], reason=metric.get("reason"),
                     trace=row.get("trace"), trace_sha256=row.get("trace_sha256"))
        series = dict(positions=None, time_s=None, speed_mm_s=None,
                      distance_mm=None, deficit_mm=None, success=success)
        if not row.get("trace"):
            require(not success, f"Complete execution has no trace: {key}")
            entry.update(trace_available=False, command_intervals=0)
        else:
            trace_path = bound_path(row["trace"], row["trace_sha256"])
            sources[row["trace"]] = row["trace_sha256"]
            with np.load(trace_path, allow_pickle=False) as archive:
                positions, reference = archive["positions"].copy(), archive["reference"].copy()
            require(np.array_equal(reference, references[case]), f"Reference changed: {key}")
            require(positions.ndim == 2 and positions.shape[1] == 2,
                    f"Invalid position dimensions: {key}")
            require(np.isfinite(positions).all(), f"Nonfinite position: {key}")
            if len(positions):
                require(np.array_equal(positions[0], reference[0]), f"Wrong initial position: {key}")
            n = max(0, len(positions) - 1)
            entry.update(trace_available=True, position_samples=len(positions), command_intervals=n)
            series["positions"] = positions
            if not n:
                require(not success, f"Empty complete trace: {key}")
            else:
                velocity = np.diff(positions, axis=0) / dt
                acceleration = np.diff(np.vstack((np.zeros(2), velocity)), axis=0) / dt
                jerk = np.diff(np.vstack((np.zeros(2), acceleration)), axis=0) / dt
                norms = np.column_stack([np.linalg.norm(a, axis=1) for a in (velocity, acceleration, jerk)])
                speed = norms[:, 0]
                distance = np.r_[0., np.cumsum(speed) * dt]
                deficit = np.r_[0., np.cumsum(100. - speed) * dt]
                elapsed, length, total_deficit = n * dt, float(distance[-1]), float(deficit[-1])
                require(np.allclose(norms.max(axis=0), metric["vector_peaks"], rtol=1e-8, atol=2e-4),
                        f"Vector metrics mismatch: {key}")
                require(abs(elapsed - metric["elapsed_prefix_s"]) < 1e-8, f"Elapsed time mismatch: {key}")
                require(abs(length - metric["executed_length_mm"]) < 1e-6, f"Length mismatch: {key}")
                require(abs(elapsed - (length + total_deficit) / 100.) < 1e-7, f"Time identity mismatch: {key}")
                if success:
                    require(abs(elapsed - metric["complete_time_s"]) < 1e-8, f"Completion time mismatch: {key}")
                    require(metric["kinematic_pass"] and metric["rest_pass"], f"Complete acceptance mismatch: {key}")
                    require(metric["geometry"]["ordered_complete"] and metric["geometry"]["corridor_inside"],
                            f"Complete geometry mismatch: {key}")
                entry.update(elapsed_prefix_s=elapsed, executed_length_mm=length,
                             speed_deficit_mm=total_deficit, vector_peaks=norms.max(axis=0).tolist())
                series.update(time_s=np.arange(1, n + 1) * dt, speed_mm_s=speed,
                              distance_mm=distance, deficit_mm=deficit)
        audit.append(entry)
        if dt == PRIMARY_DT:
            plots[(case, method)] = series
    return references, plots, audit, sources


def generate(comparison: Path, registration: Path, output: Path, validate_only: bool = False) -> dict:
    references, plots, audit, sources = load_validated(comparison, registration)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   comparison=str(comparison.resolve()), comparison_sha256=sha(comparison),
                   registration_sha256=sha(registration), validated_cells=28,
                   primary_dt_s=PRIMARY_DT, selected_primary_cells=14,
                   plotted_cells=sum(s["positions"] is not None and len(s["positions"]) > 0 for s in plots.values()),
                   methods=list(METHODS),
                   input_sha256=sources, generator_sha256=sha(Path(__file__)),
                   style_dependency_sha256=sha(STYLE_SOURCE), contract_sha256=sha(CONTRACT),
                   smoothing=False, decimation=False, temporal_cropping=False,
                   manual_visual_review="pending", scientific_release_validated=False)
    if validate_only:
        return {**receipt, "status": "validated_without_output"}
    # Input validation is complete before any artifact directory is created.
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "path.simplify": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 133 / 25.4))
    fig.subplots_adjust(left=.085, right=.985, bottom=.105, top=.83, wspace=.5, hspace=.68)
    handles = [Line2D([], [], color="#4c4c4c", ls="--", lw=.85, label="Reference")]
    handles += [Line2D([], [], color=COLOR[m], ls=STYLE[m], lw=1.05, label=LABEL[m]) for m in METHODS]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.53, .993),
               columnspacing=1.45, handlelength=2.7, handletextpad=.55)
    max_speed = max([100.] + [float(s["speed_mm_s"].max()) for s in plots.values() if s["speed_mm_s"] is not None])
    source_arrays = {}
    failed_primary = False
    for row_index, (case, title) in enumerate(zip(CASES, TITLES)):
        geo_ax, speed_ax, deficit_ax = axes[row_index]
        reference = references[case]
        source_arrays[case + "__reference"] = reference
        geo_ax.plot(*reference.T, color="#4c4c4c", lw=.8, ls="--", zorder=1)
        no_intervals = []
        for method in METHODS:
            series = plots[(case, method)]
            key = case + "__" + method
            positions = series["positions"]
            kw = dict(color=COLOR[method], ls=STYLE[method], lw=.85, rasterized=True)
            if positions is not None:
                source_arrays[key + "__positions"] = positions
                geo_ax.plot(*positions.T, **kw)
                if len(positions) and not series["success"]:
                    geo_ax.plot(*positions[-1], marker="x", color=COLOR[method], ms=4.2, mew=.85, zorder=8)
            if series["speed_mm_s"] is None:
                no_intervals.append(LABEL[method] + (": no trace" if positions is None or not len(positions) else ": no command interval"))
            else:
                speed_ax.plot(series["time_s"], series["speed_mm_s"], **kw)
                deficit_ax.plot(series["distance_mm"] / 1000., series["deficit_mm"] / 1000., **kw)
                for name in ("time_s", "speed_mm_s", "distance_mm", "deficit_mm"):
                    source_arrays[key + "__" + name] = series[name]
                if not series["success"]:
                    speed_ax.plot(series["time_s"][-1], series["speed_mm_s"][-1], marker="x",
                                  color=COLOR[method], ms=4.2, mew=.85, zorder=8)
                    deficit_ax.plot(series["distance_mm"][-1] / 1000., series["deficit_mm"][-1] / 1000.,
                                    marker="x", color=COLOR[method], ms=4.2, mew=.85, zorder=8)
            failed_primary |= not series["success"]
        geo_ax.plot(*reference[0], marker="^", markerfacecolor="white", markeredgecolor="#333333",
                    ms=3.5, mew=.7, zorder=9)
        geo_ax.set_aspect("equal", adjustable="datalim")
        geo_ax.set(xlabel="X (mm)", ylabel="Y (mm)")
        geo_ax.margins(.06)
        speed_ax.axhline(100., color="#777777", lw=.6, ls=":", zorder=0)
        speed_ax.set(xlabel="Elapsed time (s)", ylabel="Speed (mm/s)", ylim=(-.025 * max_speed, 1.06 * max_speed))
        speed_ax.set_xlim(left=0)
        deficit_ax.set(xlabel="Executed distance (m)", ylabel="Speed deficit (m)")
        deficit_ax.set_xlim(left=0)
        for axis in (speed_ax, deficit_ax):
            axis.grid(axis="y", color="#ececec", lw=.4)
        letters = "abc" if row_index == 0 else "def"
        for axis, letter, heading in zip(axes[row_index], letters, (title, "Command speed", "Speed deficit")):
            axis.set_title(letter + "  " + heading, loc="left", fontweight="bold", pad=8)
        if no_intervals:
            speed_ax.text(.015, .045, "\n".join(no_intervals), transform=speed_ax.transAxes,
                          fontsize=5.8, va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    note = "Triangle: program start"
    if failed_primary:
        note += "    x: incomplete execution endpoint (all recorded commands shown)"
    fig.text(.5, .016, note, ha="center", va="bottom", fontsize=6.4)
    outputs = []
    for extension in ("pdf", "svg", "png"):
        path = output / ("figure9_scaled_v8." + extension)
        fig.savefig(path, dpi=400, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    np.savez_compressed(output / "figure9_scaled_v8_source.npz", **source_arrays)
    outputs.append(output / "figure9_scaled_v8_source.npz")
    source_index = dict(primary_dt_s=PRIMARY_DT, figure_rows=[r for r in audit if r["dt_s"] == PRIMARY_DT],
                        all_comparison_rows=audit, source_arrays=list(source_arrays),
                        dimensions_mm=[183, 133], colors=COLOR, line_styles=STYLE)
    index_path = output / "figure9_scaled_v8_data.json"
    index_path.write_text(json.dumps(source_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs.append(index_path)
    caption = ("Complete command motion on the tenfold curvature coupon (top) and neck contour (bottom) at 1 ms. "
               "The columns show complete reference and recorded command geometry, speed over elapsed recorded time, "
               "and accumulated speed deficit over executed distance. The triangle marks the program start. "
               + ("Crosses mark the final recorded command of incomplete executions; no completion time is assigned to those prefixes. " if failed_primary else "")
               + "The physical scale is changed while the nominal tolerance and planar vector limits are retained.\n")
    caption_path = output / "figure9_scaled_v8_caption.tex"
    caption_path.write_text(caption, encoding="utf-8")
    outputs.append(caption_path)
    contract_copy = output / "figure9_scaled_v8_contract.md"
    if contract_copy.resolve() != CONTRACT.resolve():
        shutil.copy2(CONTRACT, contract_copy)
    outputs.append(contract_copy)
    receipt.update(status="generated_pending_visual_review", outputs_sha256={p.name: sha(p) for p in outputs},
                   rasterization="Dense line artists only, 400 dpi; axes, labels and legends remain vector text")
    (output / "figure9_scaled_v8_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=ROOT / "audit/scaled_representative_v8/comparison.json")
    parser.add_argument("--registration", type=Path, default=ROOT / "evidence/scaled_representative_v8_registration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "shared_manuscript/representative_scaled_v8")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    receipt = generate(args.comparison, args.registration, args.output, args.validate_only)
    print(json.dumps({key: receipt[key] for key in ("status", "validated_cells", "plotted_cells", "primary_dt_s")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
