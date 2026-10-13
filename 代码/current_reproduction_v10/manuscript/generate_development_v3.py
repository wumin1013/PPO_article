"""Generate development motion panels from complete version-3 execution records.

No experiment is executed here. Selection is fixed by geometry and seed, never by
the observed ranking. Full-resolution traces generate all reported quantities.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shared_manuscript"
AUDIT = ROOT / "audit"
DT = 0.001
SEED = 101
METHODS = ("adaptive", "fixed", "S2R", "SCMO")
COLORS = {"adaptive": "#28688a", "fixed": "#ba7944", "S2R": "#577862", "SCMO": "#806595"}
LABELS = {"adaptive": "J-NNC", "fixed": "Fixed preview", "S2R": "S2-R", "SCMO": "SCMO"}
STYLES = {"adaptive": "-", "fixed": (0, (4, 1.5)), "S2R": (0, (2, 1.3)), "SCMO": (0, (5, 1.3, 1, 1.3))}
CASES = {"original": {"label": "Long mixed path", "corner": "corner_02"},
         "dense": {"label": "Dense corner path", "corner": "corner_01"}}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def result_path(case, method):
    if method in ("adaptive", "fixed"):
        return ROOT / "results/development_v3/final_model" / f"{method}_{SEED}_{case}_{DT}.json"
    return ROOT / "results/baseline_v3/development_unified" / f"{method}_{case}_dt1000us_scale1000.json"


def local_frame(reference, geometry, corner_id):
    corner = next(c for c in geometry["sharp_corners"] if c["id"] == corner_id)
    vertex = int(corner["reference_vertex_index"])
    center = np.asarray(corner["point_mm"], dtype=float)
    require(np.allclose(reference[vertex], center, rtol=0, atol=1e-8), "Corner-reference mismatch")
    incoming = reference[vertex] - reference[vertex - 1]
    outgoing = reference[vertex + 1] - reference[vertex]
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    normal = np.array([-incoming[1], incoming[0]])
    require(np.dot(outgoing, normal) > .99, "Registered corner must be a left 90-degree turn")
    return dict(corner=corner, center=center, incoming=incoming, outgoing=outgoing, normal=normal,
                before=float(corner["window_before_mm"]), after=float(corner["window_after_mm"]))


def load_trace(case, method, reference, frame, hashes):
    record_file = result_path(case, method)
    trace_file = record_file.with_suffix(".npz")
    require(record_file.is_file() and trace_file.is_file(), "Required new-version result absent: " + str(record_file))
    record = read(record_file)
    require(record["method"] == method and record.get("path", record.get("path_id")) == case,
            "Unexpected result identity: " + rel(record_file))
    require(record["dt_s"] == DT, "Unexpected timestep")
    require(np.array_equal(record["limits"], [100., 2000., 62500.]), "Constraint mismatch")
    if method in ("adaptive", "fixed"):
        require(record["seed"] == SEED and record["runtime_version"] == "v3", "Incorrect learned-policy version")
        require(record["scope"] in ("development; not final holdout",
                "Previously observed development illustration; not independent test evidence"), "Development scope mismatch")
        require(sha(trace_file) == record["trace_sha256"], "Stored trace hash mismatch")
        model_file = ROOT / "results/unified_v3/paired/seed_101" / method / "final.pt"
        hashes[rel(model_file)] = sha(model_file)
        if "model_sha256" in record:
            require(record["model_sha256"] == hashes[rel(model_file)], "Model hash mismatch")
        else:
            provenance_file = AUDIT / "development_v3_replay_provenance.json"
            provenance = read(provenance_file)
            matching = [r for r in provenance["records"] if r["method"] == method
                        and r["seed"] == SEED and r["path"] == case and r["dt_s"] == DT]
            require(len(matching) == 1, "Exactly one development provenance record is required")
            identity = matching[0]
            require(identity["model_sha256"] == hashes[rel(model_file)]
                    and identity["result_sha256"] == sha(record_file)
                    and identity["trace_sha256"] == sha(trace_file), "Supplemental provenance hash mismatch")
            require(identity["reference_sha256"] == sha(ROOT / identity["reference"]),
                    "Supplemental reference provenance mismatch")
            hashes[rel(provenance_file)] = sha(provenance_file)
        if "reference_sha256" in record:
            reference_file = ROOT / "baseline_records/supplementary_reproduction/cases" / case / "reference.csv"
            require(record["reference_sha256"] == sha(reference_file), "Reference-file hash mismatch")
    for p in (record_file, trace_file):
        hashes[rel(p)] = sha(p)
    with np.load(trace_file, allow_pickle=False) as data:
        positions = data["positions"].copy()
        require(np.array_equal(data["reference"], reference), "Methods must use identical reference arrays")
        if "dt_s" in data:
            require(float(data["dt_s"]) == DT, "NPZ timestep mismatch")
    require(positions.ndim == 2 and positions.shape[1] == 2 and len(positions) == record["samples"],
            "Sample count mismatch")
    require(np.isfinite(positions).all(), "Non-finite trajectory")
    require(np.allclose(positions[0], reference[0], rtol=0, atol=1e-8), "Initial point mismatch")
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = np.r_[0., displacement / DT]
    elapsed = (len(positions) - 1) * DT
    length = float(displacement.sum())
    cumulative = np.r_[0., np.cumsum(displacement)]
    deficit = np.r_[0., np.cumsum(100. - speed[1:]) * DT]
    require(abs(elapsed - record["elapsed_prefix_s"]) < 1e-8, "Elapsed-time mismatch")
    require(abs(length - record["executed_length_mm"]) < 1e-7, "Executed-length mismatch")
    require(abs(float(speed.max()) - record["vector_peaks"][0]) < 1e-7, "Speed-peak mismatch")
    require(abs((length + deficit[-1]) / 100. - elapsed) < 1e-7, "Time-accounting identity failed")
    if record["success"]:
        require(abs(elapsed - record["complete_time_s"]) < 1e-8, "Completion-time mismatch")
        require(record["geometry"]["ordered_complete"] and record["geometry"]["corridor_inside"]
                and record["kinematic_pass"] and record["rest_pass"], "Inconsistent success audit")
    else:
        require(record["complete_time_s"] is None, "Incomplete trace cannot have completion time")
    delta = positions - frame["center"]
    xy = np.column_stack((delta @ frame["incoming"], delta @ frame["normal"]))
    # This continuous coordinate is exactly signed reference arc length on either
    # adjacent straight. It avoids a discontinuous nearest-segment switch inside a blend.
    progress = delta @ ((frame["incoming"] + frame["outgoing"]) /
                        (1. + np.dot(frame["incoming"], frame["outgoing"])))
    padding = 2. * record["tolerance_mm"]
    local = ((progress >= -frame["before"]) & (progress <= frame["after"]) &
             (xy[:, 0] >= -frame["before"] - padding) & (xy[:, 0] <= padding) &
             (xy[:, 1] >= -padding) & (xy[:, 1] <= frame["after"] + padding))
    indices = np.flatnonzero(local)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []
    groups = [g for g in groups if len(g)]
    chosen = groups[0] if groups else np.empty(0, dtype=int)
    metrics = dict(case=case, method=method, seed=SEED if method in ("adaptive", "fixed") else None,
                   statistical_unit="one complete development-program execution; seed 101 for learned methods",
                   success=record["success"], reason=record["reason"], complete_time_s=record["complete_time_s"],
                   elapsed_prefix_s=elapsed, sample_count=len(positions), dt_s=DT,
                   executed_length_mm=length, peak_speed_mm_s=float(speed.max()),
                   time_weighted_mean_speed_mm_s=length / elapsed, speed_deficit_distance_mm=float(deficit[-1]),
                   tolerance_mm=record["tolerance_mm"], terminal_rest_s=record["terminal_rest_s"],
                   max_polyline_deviation_mm=record["max_polyline_deviation_mm"],
                   vector_limit_ratios=record["vector_limit_ratios"], geometry=record["geometry"],
                   local_corner_id=frame["corner"]["id"], local_visit_count=len(groups),
                   local_visit_rule="first chronological visit to the fixed reference-corner window",
                   local_sample_count=len(chosen),
                   local_time_interval_s=[float(chosen[0] * DT), float(chosen[-1] * DT)] if len(chosen) else None,
                   result_file=rel(record_file), trace_file=rel(trace_file))
    if method in ("adaptive", "fixed"):
        metrics["model_sha256_recorded_in_result"] = record.get("model_sha256")
        metrics["current_model_sha256"] = hashes[rel(model_file)]
        metrics["model_identity_source"] = ("run record" if "model_sha256" in record
                                             else rel(AUDIT / "development_v3_replay_provenance.json"))
    return dict(record=record, positions=positions, speed=speed, cumulative=cumulative, deficit=deficit,
                local_indices=chosen, local_xy=xy[chosen], local_progress=progress[chosen], metrics=metrics)


def main():
    source_hashes = {rel(Path(__file__)): sha(Path(__file__))}
    trajectories, frames, references = {}, {}, {}
    for case, specification in CASES.items():
        base = ROOT / "baseline_records/supplementary_reproduction/cases" / case
        reference_file, geometry_file = base / "reference.csv", base / "geometry.json"
        reference = np.loadtxt(reference_file, delimiter=",", skiprows=1)
        geometry = read(geometry_file)
        for p in (reference_file, geometry_file):
            source_hashes[rel(p)] = sha(p)
        references[case] = reference
        frames[case] = local_frame(reference, geometry, specification["corner"])
        for method in METHODS:
            trajectories[case, method] = load_trace(case, method, reference, frames[case], source_hashes)
        tolerances = [trajectories[case, m]["record"]["tolerance_mm"] for m in METHODS]
        require(max(tolerances) - min(tolerances) < 1e-12, "Methods use different geometric budgets")

    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
                        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
                        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                        "axes.spines.right": False, "axes.linewidth": .65, "path.simplify": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2))
    fig.subplots_adjust(left=.075, right=.985, bottom=.09, top=.88, hspace=.50, wspace=.38)
    handles = [Line2D([0], [0], color="#4c4c4c", ls="--", lw=.85, label="Reference")]
    handles += [Line2D([0], [0], color=COLORS[m], ls=STYLES[m], lw=1.2, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(.53, .997), frameon=False,
               columnspacing=1.2, handlelength=2.3)
    for row, (case, specification) in enumerate(CASES.items()):
        spatial, speed_ax, accounting = axes[row]
        frame = frames[case]
        spatial.plot([-frame["before"], 0., 0.], [0., 0., frame["after"]],
                     color="#4c4c4c", lw=.9, ls="--", zorder=1)
        for method in reversed(METHODS):
            tr = trajectories[case, method]
            kw = dict(color=COLORS[method], ls=STYLES[method], lw=1.1)
            spatial.plot(*tr["local_xy"].T, **kw)
            speed_ax.plot(tr["local_progress"], tr["speed"][tr["local_indices"]], **kw)
            # The dense and long cases use the same metre units; plotted data are
            # complete and are not decimated before output or metric calculation.
            accounting.plot(tr["cumulative"] / 1000., tr["deficit"] / 1000., **kw)
            if not tr["record"]["success"]:
                accounting.plot(tr["cumulative"][-1] / 1000., tr["deficit"][-1] / 1000.,
                                marker="x", color=COLORS[method], ms=5, mew=1.2)
        spatial.set_aspect("equal", adjustable="box")
        padding = .75
        spatial.set_xlim(-frame["before"] - padding, padding)
        spatial.set_ylim(-padding, frame["after"] + padding)
        spatial.set_xlabel("Incoming-tangent coordinate (mm)")
        spatial.set_ylabel("Normal coordinate (mm)", labelpad=2)
        speed_ax.set_xlabel("Local projected progress (mm)")
        speed_ax.set_ylabel("Speed (mm/s)", labelpad=2)
        speed_ax.set_xlim(-frame["before"], frame["after"])
        speed_ax.set_ylim(0., 105.)
        speed_ax.set_yticks([0., 25., 50., 75., 100.])
        speed_ax.axhline(100., c="#777777", ls=":", lw=.6)
        speed_ax.grid(axis="y", alpha=.16, lw=.5)
        accounting.set_xlabel("Executed distance (m)")
        accounting.set_ylabel("Accumulated speed deficit (m)", labelpad=2)
        accounting.set_xlim(left=0.)
        accounting.set_ylim(bottom=0.)
        accounting.grid(axis="y", alpha=.16, lw=.5)
        titles = [specification["label"] + ": corner", "Corner speed", "Full-path speed accounting"]
        for col, (ax, title) in enumerate(zip(axes[row], titles)):
            ax.set_title(f"{'abcdef'[row * 3 + col]}  {title}", loc="left", fontweight="bold", pad=7)
            ax.tick_params(length=2.5, width=.6, pad=2)
        # A geometric locator keeps the corner choice visible without crowding
        # the local comparison or treating it as a full-path spatial display.
        inset = spatial.inset_axes([.06, .59, .33, .35])
        ref = references[case]
        inset.plot(ref[:, 0], ref[:, 1], c="#a0a0a0", lw=.45)
        inset.plot(*frame["center"], "o", color="#222222", ms=2.5)
        inset.set_aspect("equal")
        inset.set_axis_off()

    stem = OUT / "development_motion_v3"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)

    arrays, csv_rows = {}, []
    for (case, method), tr in trajectories.items():
        prefix = case + "__" + method + "__"
        arrays[prefix + "positions_mm"] = tr["positions"]
        arrays[prefix + "speed_mm_s"] = tr["speed"]
        arrays[prefix + "executed_distance_mm"] = tr["cumulative"]
        arrays[prefix + "speed_deficit_mm"] = tr["deficit"]
        arrays[prefix + "local_indices"] = tr["local_indices"]
        arrays[prefix + "local_xy_mm"] = tr["local_xy"]
        arrays[prefix + "local_projected_progress_mm"] = tr["local_progress"]
        for index, xy, progress in zip(tr["local_indices"], tr["local_xy"], tr["local_progress"]):
            csv_rows.append([case, method, int(index), index * DT, *xy, progress, tr["speed"][index]])
    for case, reference in references.items():
        arrays[case + "__reference_mm"] = reference
    np.savez_compressed(OUT / "development_motion_v3_source.npz", **arrays)
    with (OUT / "development_motion_v3_local.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "method", "sample_index", "time_s", "local_x_mm", "local_y_mm",
                         "local_projected_progress_mm", "speed_mm_s"])
        writer.writerows(csv_rows)
    metrics = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "development cases; not final test",
               "selection": {"seed": SEED, "case_order": list(CASES), "methods": list(METHODS),
                             "corners": {c: s["corner"] for c, s in CASES.items()},
                             "rule": "case and seed fixed in advance; corners selected from reference geometry"},
               "time_accounting": "D=sum((100-speed_i)*dt); T=(executed_length+D)/100; initial sample excluded",
               "local_progress": "(p-c) dot (t_in+t_out)/(1+t_in dot t_out); signed reference distance on either adjacent straight",
               "data_policy": "All input samples retained. Incomplete runs retain prefixes and end with an x; no completion-time imputation.",
               "metrics": [tr["metrics"] for tr in trajectories.values()], "source_sha256": source_hashes}
    (OUT / "development_motion_v3_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    contract = """# Development motion figure contract

The two rows are registered development programs, not the independent final test.
Learned methods use seed 101 of the common final configuration, at 1 ms. S2-R and
SCMO use the new development_unified results with identical reference arrays,
kinematic bounds, geometric budgets and completion audit. Old model 923202 and
old nominal-polygon S2 records are never read by this generator.

The first two panels in each row use a reference-defined left 90-degree corner:
original/corner_02 and dense/corner_01. The first chronological visit in the
reference-defined window is retained; all visit counts are disclosed in the
metrics JSON. The continuous local progress equals signed reference arc length
on either adjacent straight. It is not a global nearest-reference projection.

The third panel uses every command interval in the complete recorded trace.
With V=100 mm/s, D=integral(V-v)dt and T=(L+D)/V. No value is clipped to make D
positive. Final rest is included. Failed executions are prefixes, marked x;
they are not presented as completed times. No mean, uncertainty interval or
multi-seed inference is drawn from these mechanism examples.

No curve smoothing or sample decimation is performed. Full-resolution plot
arrays are retained in the NPZ; the local CSV is an additional readable export.
All inputs are hashed and independently checked against recorded durations,
lengths, peaks, sample counts and (for learned traces) stored trace hashes.

Matplotlib uses the existing manuscript palette, Arial with DejaVu Sans fallback,
embedded TrueType PDF fonts, editable SVG text, and a white background. The
project's existing scientific-figure convention is used; no external visual
precedent is asserted. The final PNG must receive visual inspection.
"""
    (OUT / "development_motion_v3_contract.md").write_text(contract, encoding="utf-8")
    outputs = [stem.with_suffix("." + e) for e in ("pdf", "svg", "png")]
    outputs += [OUT / ("development_motion_v3" + suffix) for suffix in
                ("_source.npz", "_local.csv", "_metrics.json", "_contract.md")]
    receipt = {"created_utc": metrics["created_utc"], "generator": rel(Path(__file__)),
               "source_sha256": source_hashes, "output_sha256": {rel(p): sha(p) for p in outputs},
               "numerical_checks": "passed", "visual_inspection": "pending",
               "command": r"D:\Anaconda\envs\AFC\python.exe manuscript\generate_development_v3.py"}
    (AUDIT / "development_motion_v3_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(stem.with_suffix(".pdf")), "records": len(trajectories),
                      "completed": sum(t["record"]["success"] for t in trajectories.values())}))


if __name__ == "__main__":
    main()
