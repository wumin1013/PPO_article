"""Draw the selected event-preview execution diagram from saved configuration.

This is an offline drawing tool. It does not import a runner, environment or
policy, and never reads trajectory data or test results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath
import yaml

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "论文项目"
STUDY = PAPER / "revision_20260909_event_preview/study"
BLUE, ORANGE, GRAY = "#2467a4", "#d58a43", "#818181"
INK = "#263746"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box(ax, xy, size, title, detail, color=BLUE, fill="#f1f6fa"):
    x, y = xy; width, height = size
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.025,rounding_size=0.08",
                              edgecolor=color, facecolor=fill, linewidth=.8))
    ax.text(x + width / 2, y + height - .25, title, ha="center", va="top",
            fontsize=7.4, color=INK, fontweight="bold")
    ax.text(x + width / 2, y + height - .67, detail, ha="center", va="top",
            fontsize=6.6, color=INK, linespacing=1.35)


def arrow(ax, points, color=BLUE, dashed=False):
    # Orthogonal routes keep control and memory paths separate and readable.
    if len(points) > 2:
        path = PlotPath(points[:-1], [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 2))
        ax.add_patch(PathPatch(path, fill=False, color=color, linewidth=.85,
                               linestyle="--" if dashed else "-"))
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=7, linewidth=.85, color=color,
                                 linestyle="--" if dashed else "-", shrinkA=0, shrinkB=2))


def draw(cfg, figure_dir):
    dt_ms = 1000 * cfg["environment"]["interpolation_period"]
    period = cfg["revision"]["preview_period_steps"]
    hold_ms = period * dt_ms
    reward_window = cfg["reward_weights"]["cornerness"]["reference_distance"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(183 / 25.4, 116 / 25.4), facecolor="white")
    ax = fig.add_axes([.008, .015, .984, .97])
    ax.set(xlim=(0, 18.3), ylim=(0, 11.6)); ax.axis("off")
    ax.text(.3, 11.25, "Policy-selected preview with periodic holding", fontsize=9,
            color=INK, fontweight="bold", va="top")
    ax.text(.3, 10.7, f"Selected interval {hold_ms:g} ms; physical interpolation {dt_ms:g} ms",
            fontsize=7, color=GRAY, va="top")

    box(ax, (.3, 8.6), (2.6, 1.5), "26-D state", "$s_t^{\\mathrm{event}}$\n20 + 6 entries")
    box(ax, (3.65, 8.6), (2.05, 1.5), "PPO policy", "3 latent outputs\nsteer / feed / $L$")
    box(ax, (7.15, 8.6), (3.05, 1.5), "Motion intention", "Steer/feed + LOS\ncombined before KCM")
    box(ax, (11.05, 8.6), (2.15, 1.5), "KCM", "Scalar feed /\nheading bounds")
    box(ax, (14.1, 8.6), (3.35, 1.5), "Output integration", "Position, heading, motion\nat $t+1$")
    arrow(ax, [(2.9, 9.35), (3.65, 9.35)])
    arrow(ax, [(5.7, 9.35), (7.15, 9.35)])
    ax.text(6.425, 9.65, f"motion\n{dt_ms:g} ms", fontsize=6.2, ha="center", color=BLUE)
    arrow(ax, [(10.2, 9.35), (11.05, 9.35)])
    arrow(ax, [(13.2, 9.35), (14.1, 9.35)])

    box(ax, (3.3, 5.9), (2.8, 1.5), "Event + hold", f"PPO sample: {hold_ms:g} ms\nhold between events",
        ORANGE, "#fcf5ed")
    box(ax, (6.85, 5.9), (2.65, 1.5), "Preview distance", "$D_t=L_t^{\\mathrm{act}}$ (mm)\n$0.8+3.8\\,\\widetilde u_{L,t}$",
        ORANGE, "#fcf5ed")
    box(ax, (10.25, 5.9), (3.0, 1.5), "Geometry + LOS", "Multiscale geometry\nLOS filter: once / tick")
    box(ax, (14.05, 4.85), (3.4, 2.25), "Next observation", "$s_{t+1}^{\\mathrm{event}}$\nheld $D$, phase, filter\nnew geometry / motion")
    arrow(ax, [(4.68, 8.6), (4.68, 7.4)], ORANGE)
    ax.text(4.95, 8.05, "preview event", fontsize=6.4, color=ORANGE, va="center")
    arrow(ax, [(6.1, 6.65), (6.85, 6.65)], ORANGE)
    arrow(ax, [(9.5, 6.65), (10.25, 6.65)], ORANGE)
    arrow(ax, [(11.7, 7.4), (11.7, 8.05), (8.67, 8.05), (8.67, 8.6)])
    arrow(ax, [(13.25, 6.45), (14.05, 6.45)])
    arrow(ax, [(15.8, 8.6), (15.8, 7.1)])
    arrow(ax, [(4.68, 5.9), (4.68, 4.4), (15.2, 4.4), (15.2, 4.85)], ORANGE, True)
    ax.text(6.95, 4.59, "held value + schedule phase", fontsize=6.2, color=ORANGE)
    arrow(ax, [(16.4, 4.85), (16.4, 3.72), (1.6, 3.72), (1.6, 8.6)], GRAY, True)
    ax.text(.53, 6.62, "next\nstep", fontsize=6.5, color=GRAY, va="center")

    box(ax, (3.3, 1.32), (10.15, 1.65), "Reward evaluation", 
        f"Fixed {reward_window:g} mm corner-intensity window, shared across methods\n"
        "Ordered progress, tracking, smoothness, time and terminal terms", GRAY, "#f6f6f6")
    arrow(ax, [(17.45, 9.0), (17.87, 9.0), (17.87, 2.15), (13.45, 2.15)], GRAY)
    ax.text(15.7, 2.42, "output state\n+ reference path", fontsize=6.4, color=GRAY, ha="center")
    ax.text(8.38, .75, "Record reward components and transitions for PPO updates", fontsize=6.5,
            color=GRAY, ha="center")
    arrow(ax, [(8.38, 1.32), (8.38, 1.04)], GRAY)

    outputs = []
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        path = figure_dir / ("preview_event_framework." + ext)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=PAPER / "figures/generated")
    args = parser.parse_args()
    selection_path = STUDY / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["selected_hold"] != "hold100":
        raise ValueError("This registered diagram depicts the selected hold100 configuration")
    config_path = STUDY / "pilot/hold100_seed101/config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (cfg["revision"]["observation_dim"] != 26
        or cfg["revision"]["preview_period_steps"] != 100
        or cfg["environment"]["interpolation_period"] != .001
        or cfg["revision"]["preview_bounds_mm"] != [.8, 4.6]
        or cfg["reward_weights"]["cornerness"]["reference_distance"] != 4.6):
        raise ValueError("Saved configuration differs from the diagram's registered scope")
    outputs = draw(cfg, args.figure_dir)
    metadata_dir = PAPER / "revision_20260909_event_preview/mechanism"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sources = [selection_path, config_path, STUDY / "protocol.json",
               PAPER / "revision_20260909_event_preview/figure_contract.md",
               ROOT / "PPO_project/tools/preview_event_study.py",
               ROOT / "PPO_project/tools/preview_direct_research.py",
               ROOT / "PPO_project/src/environment/cnc_env.py"]
    metadata = dict(command="python " + " ".join(sys.argv), script_sha256=sha(__file__),
                    figure_role="Execution schematic, not a performance result", 
                    input_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                    figures=[dict(path=str(p.resolve()), sha256=sha(p)) for p in outputs],
                    dimensions_mm=[183, 116], editable_text=True,
                    selected_distance_source="PPO third output, clipped and periodically held",
                    reward_window_mm=4.6, independent_geometry_observation_window_mm=4.6,
                    no_policy_or_environment_execution=True, no_test_artifacts_read=True)
    (metadata_dir / "framework_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(figures=[str(p) for p in outputs], script_sha256=metadata["script_sha256"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
