"""Separate full-command speed curves without changing the frozen v9 evidence.

Only the two explicitly supplied evidence roots are remapped.  Every canonical
cell is verified, including unsuccessful prefixes and the 0.5 ms checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OLD_ROOT = PureWindowsPath(r"D:\PPO_article_work\JNN_efficiency_20261012_v9")
OLD_BASE = PureWindowsPath(r"D:\PPO_article_work\JNN_dominance_20261011_v5")
CANONICAL_SHA = "22b7c88fd0298f88801ff3a5d5af79488d8f5b75b349330e0e01ce336d2f1725"
METHODS = ("JNN", "FIXED", "S2", "S2R", "SCMO", "COPP", "NNC")
CASES = ("original", "dense", "piecewise_curvature_coupon_scale10", "asymmetric_neck_contour_scale10")
LABEL = dict(JNN="J-NNC", FIXED="Fixed preview", S2="S2-Q", S2R="S2-R", SCMO="SCMO", COPP="COPP", NNC="NNC adapter")
COLOR = dict(JNN="#28688a", FIXED="#5b9098", S2="#ba7944", S2R="#577862", SCMO="#806595", COPP="#53565d", NNC="#b05664")
LIMITS = np.array([100.0, 2000.0, 62500.0])
WINDOWS = dict(original="corner_02", dense="corner_01")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_evidence(root: Path, base: Path):
    def resolve(value: str):
        original = PureWindowsPath(value)
        for old, new, label in ((OLD_ROOT, root, "root"), (OLD_BASE, base, "base")):
            try:
                relative = original.relative_to(old)
            except ValueError:
                continue
            local = new.joinpath(*relative.parts).resolve()
            if not local.is_relative_to(new):
                raise ValueError("Evidence path escapes supplied root: " + value)
            return local, label + "/" + relative.as_posix()
        raise ValueError("Evidence path is outside the two declared roots: " + value)

    canonical = root / "canonical/final_comparison.json"
    assert sha(canonical) == CANONICAL_SHA, "Frozen canonical bytes changed"
    evidence = json.loads(canonical.read_text("utf-8"))
    rows = evidence["rows"]
    assert len(rows) == 56
    assert {(r["case"], r["method"], r["dt_s"]) for r in rows} == {
        (c, m, dt) for c in CASES for m in METHODS for dt in (.001, .0005)}
    sources = {"root/canonical/final_comparison.json": CANONICAL_SHA}
    refs, geometry = {}, {}
    for case in CASES:
        definition = evidence["cases"][case]
        for field in ("reference", "geometry"):
            local, resource = resolve(definition[field])
            digest = sha(local)
            assert digest == definition[field + "_sha256"], resource
            sources[resource] = digest
            if field == "reference":
                refs[case] = np.loadtxt(local, delimiter=",", skiprows=1)
            else:
                geometry[case] = json.loads(local.read_text("utf-8"))
    series, audits = {}, []
    for row in rows:
        metric = row["metrics"]
        complete = metric["complete_time_s"]
        assert metric["success"] == (complete is not None)
        local_files = {}
        for field in ("record", "trace"):
            if row.get(field):
                local, resource = resolve(row[field])
                digest = sha(local)
                assert digest == row[field + "_sha256"], row["id"]
                sources[resource] = digest
                local_files[field] = local
        audit = dict(id=row["id"], case=row["case"], method=row["method"], dt_s=row["dt_s"],
                     success=metric["success"], complete_time_s=complete, elapsed_prefix_s=metric.get("elapsed_prefix_s"),
                     trace_sha256=row.get("trace_sha256"), record_sha256=row.get("record_sha256"))
        if "trace" in local_files:
            with np.load(local_files["trace"], allow_pickle=False) as trace:
                positions = trace["positions"].copy()
                assert np.array_equal(trace["reference"], refs[row["case"]]), row["id"]
            assert positions.ndim == 2 and positions.shape[1] == 2 and np.isfinite(positions).all()
            dt, n = row["dt_s"], len(positions) - 1
            if n:
                velocity = np.diff(positions, axis=0) / dt
                acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / dt
                jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / dt
                values = np.column_stack([np.linalg.norm(v, axis=1) for v in (velocity, acceleration, jerk)])
                peaks = values.max(axis=0)
                assert np.allclose(peaks, metric["vector_peaks"], rtol=1e-8, atol=2e-4), row["id"]
                assert abs(values[:, 0].sum() * dt - metric["executed_length_mm"]) < 1e-6, row["id"]
                assert abs(n * dt - metric["elapsed_prefix_s"]) < 1e-8
                if metric["success"]:
                    assert abs(n * dt - complete) < 1e-8
                    assert metric["kinematic_pass"] and metric["rest_pass"]
                    assert metric["geometry"]["ordered_complete"] and metric["geometry"]["corridor_inside"]
                audit.update(sample_count=len(positions), recomputed_vector_peaks=peaks.tolist())
                if row["case"] in WINDOWS and dt == .001:
                    series[(row["case"], row["method"])] = dict(
                        positions_mm=positions, time_s=np.arange(1, n + 1) * dt,
                        distance_mm=np.cumsum(values[:, 0]) * dt, norms=values,
                        peaks=peaks, metrics=metric, audit=audit)
            else:
                assert not metric["success"]
        else:
            assert not metric["success"]
        audits.append(audit)
    assert len(series) == 14
    return canonical, refs, geometry, series, audits, sources


def local_window(reference, geometry, positions, corner_id, allowance):
    """Use the same first geometry-defined visit as frozen Figure 8."""
    corner = next(c for c in geometry["sharp_corners"] if c["id"] == corner_id)
    vertex = int(corner["reference_vertex_index"])
    center = np.asarray(corner["point_mm"], dtype=float)
    assert np.allclose(reference[vertex], center, rtol=0, atol=1e-8)
    incoming = reference[vertex] - reference[vertex - 1]
    outgoing = reference[vertex + 1] - reference[vertex]
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    normal = np.array([-incoming[1], incoming[0]])
    assert np.dot(outgoing, normal) > .99
    before, after = float(corner["window_before_mm"]), float(corner["window_after_mm"])
    delta = positions - center
    xy = np.column_stack([delta @ incoming, delta @ normal])
    progress = delta @ ((incoming + outgoing) / (1.0 + incoming @ outgoing))
    pad = 2.0 * allowance
    selected = np.flatnonzero((progress >= -before) & (progress <= after) &
                             (xy[:, 0] >= -before - pad) & (xy[:, 0] <= pad) &
                             (xy[:, 1] >= -pad) & (xy[:, 1] <= after + pad))
    visits = [v for v in np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1) if len(v)]
    assert visits, "Registered Figure 8 window was not traversed"
    indices = visits[0]
    indices = indices[indices >= 1]
    return dict(indices=indices, progress_mm=progress[indices], before_mm=before, after_mm=after,
                corner=corner_id, visit_count=len(visits), center_mm=center.tolist())


def draw_case(case, reference, geometry, series, out):
    primary = {method: series[(case, method)] for method in METHODS}
    complete = [method for method in METHODS if primary[method]["metrics"]["success"]]
    assert complete == list(METHODS[:6])
    jnn = primary["JNN"]
    window = local_window(reference, geometry, jnn["positions_mm"], WINDOWS[case], jnn["metrics"]["tolerance_mm"])
    xmax = max(primary[m]["distance_mm"][-1] for m in complete) / 1000.0
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.1,
                         "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .55, "path.simplify": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(125 / 25.4, 180 / 25.4))
    # A direct-labelled speed strip for each completed method.
    speed_grid = fig.add_gridspec(6, 1, left=.105, right=.985, bottom=.550, top=.944, hspace=.52)
    fig.text(.018, .980, "a", weight="bold", fontsize=8.2, va="top")
    fig.text(.105, .980, "Complete speed profiles", weight="bold", fontsize=7.7, va="top")
    fig.text(.020, .750, "Speed (mm/s)", rotation=90, ha="center", va="center", fontsize=7.2)
    speed_axes = []
    for number, method in enumerate(complete):
        ax = fig.add_subplot(speed_grid[number, 0])
        item = primary[method]
        ax.plot(item["distance_mm"] / 1000.0, item["norms"][:, 0], color=COLOR[method], lw=.50, rasterized=True)
        ax.axhline(100, color="#8b8b8b", lw=.4, ls=(0, (2, 2)), zorder=0)
        ax.set(xlim=(0, xmax * 1.001), ylim=(-2, 105), yticks=[0, 100])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(length=2.0, width=.5, pad=1.5)
        ax.text(.005, 1.06, LABEL[method], transform=ax.transAxes, color=COLOR[method], fontsize=7.2, va="bottom", weight="bold")
        ax.text(.995, 1.06, f"{item['metrics']['complete_time_s']:.3f} s", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.0, color="#303030")
        if number < len(complete) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Executed distance (m)", labelpad=2)
        speed_axes.append(ax)

    # Complete J-NNC A/J profiles and the pre-existing geometry-defined corner.
    motion_grid = fig.add_gridspec(2, 2, left=.105, right=.985, bottom=.250, top=.477, hspace=.98, wspace=.29)
    motion_axes = []
    for col, (component, label, symbol) in enumerate(((1, "Acceleration", r"$\|a\|/A_{\max}$"), (2, "Jerk", r"$\|j\|/J_{\max}$"))):
        ax = fig.add_subplot(motion_grid[0, col])
        ax.plot(jnn["distance_mm"] / 1000.0, jnn["norms"][:, component] / LIMITS[component],
                color=COLOR["JNN"], lw=.45, rasterized=True)
        ax.set(xlim=(0, xmax * 1.001), xlabel="Executed distance (m)", ylabel=symbol)
        ax.set_title(f"{'bc'[col]}  J-NNC: complete {label.lower()}", loc="left", weight="bold", pad=4)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        local = fig.add_subplot(motion_grid[1, col])
        local.plot(window["progress_mm"], jnn["norms"][window["indices"] - 1, component] / LIMITS[component],
                   color=COLOR["JNN"], lw=.65)
        local.set(xlim=(-window["before_mm"], window["after_mm"]), xlabel="Local progress (mm)", ylabel=symbol)
        local.set_title(f"{'de'[col]}  {WINDOWS[case].replace('corner_', 'C')}: {label.lower()}", loc="left", weight="bold", pad=4)
        local.xaxis.set_major_locator(MaxNLocator(4))
        local.axvline(0, color="#dddddd", lw=.45, zorder=0)
        for a in (ax, local):
            a.set(ylim=(-.025, 1.05), yticks=[0, .5, 1])
            a.axhline(1, color="#8b8b8b", lw=.45, ls=(0, (2, 2)), zorder=0)
            a.tick_params(length=2.0, width=.5, pad=1.5)
            a.grid(axis="y", color="#e9e9e9", lw=.35)
        motion_axes.extend([ax, local])

    # Numerical peak ratios retain precision without an NNC-driven common range.
    table_ax = fig.add_axes([.105, .035, .880, .143])
    table_ax.axis("off")
    table_ax.text(-.099, 1.15, "f", weight="bold", fontsize=8.2, transform=table_ax.transAxes)
    table_ax.text(0, 1.15, "Recorded peak / common limit", weight="bold", fontsize=7.7, transform=table_ax.transAxes)
    table_rows = []
    for method in METHODS:
        ratio = primary[method]["peaks"] / LIMITS
        name = LABEL[method] + ("*" if not primary[method]["metrics"]["success"] else "")
        table_rows.append([name] + [f"{value:.4f}" for value in ratio])
    table = table_ax.table(cellText=table_rows, colLabels=["Method", "Speed", "Acceleration", "Jerk"],
                          colWidths=[.37, .19, .24, .20], cellLoc="right", colLoc="right", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = .07
        if col == 0:
            cell.get_text().set_ha("left")
        if row == 0:
            cell.get_text().set_weight("bold")
            cell.set_facecolor("#f2f3f4")
        elif col == 0:
            cell.get_text().set_color(COLOR[METHODS[row - 1]])
        if row == len(METHODS):
            cell.set_facecolor("#faf1f2")
    fig.text(.105, .011, "* NNC adapter: first 1 ms only; incomplete execution.", fontsize=7.0, va="bottom")

    outputs = []
    for extension in ("pdf", "png"):
        target = out / f"kinematics_{case}.{extension}"
        kwargs = dict(dpi=500, facecolor="white")
        if extension == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None, "Creator": "Matplotlib", "Title": f"Complete command kinematics: {case}"}
        fig.savefig(target, **kwargs)
        outputs.append(target)
    plt.close(fig)

    arrays = {"reference_mm": reference, "JNN_local_position_indices": window["indices"], "JNN_local_progress_mm": window["progress_mm"]}
    for method in METHODS:
        for key in ("positions_mm", "time_s", "distance_mm", "norms", "peaks"):
            arrays[method + "__" + key] = primary[method][key]
    source = out / f"kinematics_{case}_source.npz"
    np.savez_compressed(source, **arrays)
    outputs.append(source)
    metadata = dict(case=case, dt_s=.001, width_mm=125, height_mm=180,
                    full_speed_methods=complete, prefix_methods=["NNC"],
                    all_primary_records=[primary[m]["audit"] for m in METHODS],
                    limits=dict(speed_mm_s=100, acceleration_mm_s2=2000, jerk_mm_s3=62500),
                    local_window={k: v for k, v in window.items() if k not in ("indices", "progress_mm")},
                    local_selection="Same corner and first geometric-window visit as frozen Figure 8; position indices select norms at index minus one.",
                    source_arrays="All primary positions, all per-step vector norms, times and distances; no smoothing or decimation.",
                    plot_contract="Independent full-speed strips on common physical axes; complete J-NNC A/J plus geometric detail; all seven recorded peak ratios.",
                    statistics="One deterministic complete program per numerical entry; no error bars or seed-based inference.",
                    outputs_sha256={p.name: sha(p) for p in outputs})
    target = out / f"kinematics_{case}_source.json"
    write_json(target, metadata)
    outputs.append(target)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="v9 scientific evidence root, including canonical/")
    parser.add_argument("--base", type=Path, required=True, help="v5 evidence root or portable independent base/")
    parser.add_argument("--output", type=Path, required=True, help="New presentation-only output directory")
    args = parser.parse_args()
    root, base, out = args.root.resolve(), args.base.resolve(), args.output.resolve()
    if out == root or out == base or out.is_relative_to(root / "canonical"):
        raise ValueError("Output must not overwrite canonical evidence")
    out.mkdir(parents=True, exist_ok=True)
    canonical, refs, geometry, series, audits, sources = load_evidence(root, base)
    outputs = []
    for case in WINDOWS:
        outputs.extend(draw_case(case, refs[case], geometry[case], series, out))
    assert sha(canonical) == CANONICAL_SHA
    receipt = dict(generator_sha256=sha(Path(__file__)), canonical_sha256=CANONICAL_SHA,
                   verified_canonical_cells=len(audits), verified_completed_cells=sum(r["success"] for r in audits),
                   all_cell_audit=audits, input_sha256=sources,
                   outputs_sha256={p.name: sha(p) for p in outputs},
                   integrity=dict(data_unchanged=True, smoothing=False, decimation=False, complete_overview_cropped=False,
                                  same_limits=True, all_prefixes_retained=True),
                   visual_precedent="Retained the project's verified per-method palette, full-command axes and fixed Figure 8 corner windows; new work is a separation of existing overlapping curves.")
    write_json(out / "kinematics_v10_receipt.json", receipt)
    print(json.dumps({"verified_cells": len(audits), "output": str(out), "figures": [f"kinematics_{c}.png" for c in WINDOWS]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
