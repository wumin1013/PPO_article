"""Report existing reference geometry; no policy or physics execution."""
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

from design_composite_long_case import sha, write, wrap

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_evidence_review/natural_closed_case"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    if (OUT / "geometry_reporting.json").exists():
        raise RuntimeError("Reporting artifacts already exist; inspect instead of rerunning.")
    receipt = read("receipt.json")
    for path, digest in receipt["outputs_sha256"].items():
        assert sha(BASE / path) == digest
    g, a = read("geometry.json"), read("geometry_audit.json")
    spans = copy.deepcopy(g["primitives"])
    for i, p in enumerate(spans):
        p["original_primitive_indices"] = [i]
    merged = []
    # Merge only contiguous collinear forward-directed lines, including the seam.
    while True:
        found = False
        for i, p in enumerate(spans):
            j = (i + 1) % len(spans)
            q = spans[j]
            if p["kind"] != "line" or q["kind"] != "line":
                continue
            if abs(wrap(q["start_heading_rad"] - p["end_heading_rad"])) > 1e-10:
                continue
            assert np.linalg.norm(np.array(p["end_mm"]) - q["start_mm"]) < 1e-8
            merged.append(dict(original_primitive_indices=p["original_primitive_indices"] + q["original_primitive_indices"],
                               point_mm=p["end_mm"], across_start_end=(j == 0)))
            combined = dict(p, end_mm=q["end_mm"], end_heading_rad=q["end_heading_rad"],
                            length_mm=p["length_mm"] + q["length_mm"],
                            original_primitive_indices=p["original_primitive_indices"] + q["original_primitive_indices"])
            if j == 0:
                spans = [combined] + spans[1:i]
            else:
                spans = spans[:i] + [combined] + spans[j + 1:]
            found = True
            break
        if not found:
            break
    lines = [p for p in spans if p["kind"] == "line"]
    switches = sum(p["kind"] == q["kind"] == "line" and
                   (p["length_mm"] <= 700) != (q["length_mm"] <= 700)
                   for p, q in zip(spans, spans[1:] + spans[:1]))
    assert abs(sum(p["length_mm"] for p in spans) - a["analytic_length_mm"]) < 1e-6
    events = g["sharp_corners"]
    distinct_angles = sorted({round(e["interior_angle_deg"], 1) for e in events})
    reporting = dict(reference_sha256=sha(OUT / "reference.csv"), geometry_sha256=sha(OUT / "geometry.json"),
        count_convention="Maximal collinear straight spans on the closed geometry; start/end splits are merged for geometry statistics only.",
        short_threshold_mm=700, straight_span_count=len(lines),
        long_span_count=sum(p["length_mm"] > 700 for p in lines),
        short_span_count=sum(p["length_mm"] <= 700 for p in lines),
        actual_adjacent_short_long_switches=switches,
        minimum_straight_span_mm=min(p["length_mm"] for p in lines),
        maximum_straight_span_mm=max(p["length_mm"] for p in lines),
        original_registered_line_primitives=a["straight_count"], merged_collinear_junctions=merged,
        arc_count=a["arc_count"], reference_geometry_changed=False, actor_calls=0, physics_ticks=0,
        training_updates=0, distinct_included_angles_deg_rounded_0_1=distinct_angles,
        distinct_included_angle_count=len(distinct_angles), corner_count=len(events), spans=spans)
    write(OUT / "geometry_reporting.json", reporting)
    ref = np.loadtxt(OUT / "reference.csv", delimiter=",", skiprows=1)
    chosen = [next(e for e in events if e["id"] == cid) for cid in g["mechanism_columns"]]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False})
    fig = plt.figure(figsize=(11.5, 9.0), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[2.1, 1])
    ax = fig.add_subplot(grid[0, :])
    ax.plot(ref[:, 0], ref[:, 1], color="#444B52", lw=1.6)
    for p in g["primitives"]:
        if p["kind"] == "arc":
            xy = ref[p["start_reference_index"]:p["end_reference_index"] + 1]
            ax.plot(xy[:, 0], xy[:, 1], color="#0072B2", lw=1.7)
    ax.scatter(*ref[0], marker="s", s=30, color="black", zorder=5)
    ax.annotate("Start / end: one lap", ref[0], xytext=(600, 8500), arrowprops=dict(arrowstyle="-", lw=.8))
    ax.annotate("", (3000, 8150), (2150, 8150), arrowprops=dict(arrowstyle="->", color="#444B52", lw=1))
    ax.text(3150, 8150, "Travel direction", fontsize=8)
    ax.text(12450, 5350, "R = 600 mm", color="#0072B2", fontsize=8)
    ax.text(900, 2480, "R = 800 mm", color="#0072B2", fontsize=8)
    ax.text(6300, 4450,
            f"{len(lines)} straight spans: {reporting['long_span_count']} long + {reporting['short_span_count']} short\n"
            f"{switches} adjacent long/short switches; two arcs\n"
            f"Heading changes {a['min_heading_change_deg']:.1f}°–{a['max_heading_change_deg']:.1f}°",
            ha="center", va="center", fontsize=10)
    colors = ["#A76920", "#0072B2", "#785A99"]
    for event, letter, color, position in zip(chosen, "BCD", colors, [(3000, 6550), (4950, 6550), (7950, 6550)]):
        point = np.array(event["point_mm"])
        ax.scatter(*point, s=22, color=color, zorder=5)
        ax.annotate(letter, point, xytext=position, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", lw=.8, color=color))
    for idx, position in [(4, (5400, 9550)), (8, (8650, 8850)), (16, (10950, 400)),
                          (18, (8500, 2950)), (21, (5150, -400)), (22, (5800, 950)),
                          (26, (1850, -250)), (28, (2750, 3400)), (31, (-600, 6350))]:
        event = events[idx]
        ax.scatter(*event["point_mm"], s=9, color="#697580", zorder=4)
        ax.annotate(f"{event['interior_angle_deg']:.1f}°", event["point_mm"], xytext=position,
                    color="#444B52", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=.6, color="#697580"))
    ax.set(aspect="equal", xlabel="X (mm)", ylabel="Y (mm)", xlim=(-850, 13900), ylim=(-650, 9800))
    ax.set_title("a  One asymmetric closed contour: deep recesses, prominent corners, long/short lines, and two arcs", loc="left", fontsize=10)
    ax.grid(alpha=.16)
    for col, (event, color) in enumerate(zip(chosen, colors)):
        local_ax = fig.add_subplot(grid[1, col])
        center = np.array(event["point_mm"])
        i = event["reference_vertex_index"]
        local = ref[max(0, i - 12):min(len(ref), i + 13)]
        region = LineString(local).buffer(1.25, resolution=40)
        bx, by = region.exterior.xy
        local_ax.fill(bx, by, color="#E7EFF5")
        local_ax.plot(bx, by, color="#8095A8", ls="--", lw=.7)
        local_ax.plot(local[:, 0], local[:, 1], color=color, lw=1.2)
        local_ax.scatter(*center, s=16, color=color)
        local_ax.set(aspect="equal", xlim=(center[0] - 20, center[0] + 20),
                     ylim=(center[1] - 20, center[1] + 20), xlabel="X (mm)", ylabel="Y (mm)")
        label = ["Obtuse", "Right angle", "Acute"][col]
        local_ax.set_title(f"{'bcd'[col]}  {label}: included angle {event['interior_angle_deg']:.0f}°", loc="left", fontsize=9.5)
        local_ax.text(.03, .04, f"Heading change {event['signed_heading_change_deg']:+.0f}°\n1.25 mm corridor",
                      transform=local_ax.transAxes, fontsize=8,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    fig.suptitle("Natural composite contour — reference design, not a policy result\n"
                 f"Length {a['analytic_length_mm']/1000:.3f} m; complete-lap time lower bound "
                 f"{a['ordered_gate_time_lower_bound_s']:.3f} s at 100 mm/s", fontsize=12)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"trajectory_case.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 3.5), layout="constrained")
    ax.axhspan(0, 90, color="#785A99", alpha=.07)
    ax.axhspan(90, 180, color="#A76920", alpha=.05)
    ax.axhline(90, color="#0072B2", ls="--", lw=.8)
    category_colors = dict(obtuse=colors[0], right_angle=colors[1], acute=colors[2])
    for i, e in enumerate(events):
        beta = e["interior_angle_deg"]
        ax.vlines(i, 0, beta, color="#CDD2D6", lw=.7)
        ax.scatter(i, beta, color=category_colors[e["category"]], s=24, zorder=3)
        ax.annotate(f"{beta:.1f}°", (i, beta), xytext=(0, 8 if i % 2 == 0 else -16),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.set(xlim=(-.8, len(events) - .2), ylim=(0, 200), xlabel="Corner index along the ordered lap (00–33)",
           ylabel="Included angle β (degrees)", xticks=range(len(events)),
           xticklabels=[f"{i:02d}" for i in range(len(events))], yticks=[0, 45, 90, 135, 180])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(f"All {len(events)} corners: {len(distinct_angles)} distinct included angles (rounded to 0.1°)",
                 loc="left", fontsize=11)
    ax.text(.98, .06, "2 acute + 3 right-angle + 29 obtuse corners\nβ = 180° − |heading change|; geometry only",
            transform=ax.transAxes, ha="right", fontsize=8)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"angle_coverage.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    fields = ["id", "category", "interior_angle_deg", "signed_heading_change_deg", "reference_s_mm", "x_mm", "y_mm"]
    with (OUT / "corner_catalogue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for e in events:
            writer.writerow({**{k: e[k] for k in fields[:-2]}, "x_mm": e["point_mm"][0], "y_mm": e["point_mm"][1]})
    print(json.dumps({k: v for k, v in reporting.items() if k != "spans"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
