"""Plot complete v9 enlarged-program commands; layout inherited from the v8 figure contract."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_main_v9 import COLOR,LABEL,STYLE
CASES=('piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
TITLES=('Curvature coupon','Neck contour')
METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
PRIMARY_DT=.001
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate(references,plots,audit,sources,output):
    output.mkdir(parents=True,exist_ok=True)
    receipt=dict(validated_cells=len(audit),input_sha256=sources,primary_dt_s=PRIMARY_DT,
        smoothing=False,decimation=False,temporal_cropping=False)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
        "path.simplify": False,"svg.hashsalt":"jnn-v9", "legend.frameon": False})
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
        path = output / ("figure9_v9." + extension)
        fig.savefig(path, dpi=400, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    np.savez_compressed(output / "figure9_v9_source.npz", **source_arrays)
    outputs.append(output / "figure9_v9_source.npz")
    source_index = dict(primary_dt_s=PRIMARY_DT, figure_rows=[r for r in audit if r["dt_s"] == PRIMARY_DT],
                        all_comparison_rows=audit, source_arrays=list(source_arrays),
                        dimensions_mm=[183, 133], colors=COLOR, line_styles=STYLE)
    index_path = output / "figure9_v9_data.json"
    index_path.write_text(json.dumps(source_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    outputs.append(index_path)
    caption = ("Complete command motion on the tenfold curvature coupon (top) and neck contour (bottom) at 1 ms. "
               "The columns show complete reference and recorded command geometry, speed over elapsed recorded time, "
               "and accumulated speed deficit over executed distance. The triangle marks the program start. "
               + ("Crosses mark the final recorded command of incomplete executions; no completion time is assigned to those prefixes. " if failed_primary else "")
               + "The physical scale is changed while the nominal tolerance and planar vector limits are retained.\n")
    caption_path = output / "figure9_v9_caption.tex"
    caption_path.write_text(caption, encoding="utf-8")
    outputs.append(caption_path)
    receipt['outputs_sha256']={p.name:sha(p) for p in outputs}
    (output/'figure9_v9_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
