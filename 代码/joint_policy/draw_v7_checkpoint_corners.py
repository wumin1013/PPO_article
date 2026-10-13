"""Retain the approved four-corner geometry for the paired 60k evaluation."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from study_io import BASE, ROOT, register, receipt, write_json, sha
from summarize_conditioned_training import distance_to_commands


def main():
    groups = [('Joint independent', 'joint_independent_refinement_v7', '#245784', '-'),
              ('Fixed3 independent', 'fixed3_independent_refinement_v7', '#cd7825', '-'),
              ('Joint gSDE8', 'joint_gsde_refinement_v7', '#628cae', '--'),
              ('Fixed3 gSDE8', 'fixed3_gsde_refinement_v7', '#875324', '--')]
    tag = 'validation_60000'
    converter_path = ROOT/'论文项目/revision_20260912_finish_v1/svg_to_pdf_v2.py'
    sources = [Path(__file__), converter_path, BASE/'v7_checkpoint_figure_contract.md',
        BASE/'v7_checkpoint_60000_analysis_v1/results.json', BASE/'v6_selected_corner_comparison_figure_v1/visual_qa.json',
        BASE/'summarize_conditioned_training.py']
    for _, name, _, _ in groups:
        sources += [BASE/name/tag/'summary.json', *[BASE/name/tag/f'{i}.npz' for i in range(4)]]
    if not all(p.exists() for p in sources):
        raise RuntimeError('All paired checkpoint analyses must be complete before plotting')
    stage = register('v7_60000_corner_figure_v1', sources,
        dict(saved_trace_files=16, svg_conversions=1, model_calls=0, physics_ticks=0),
        'Show all four isolated corners at the same v7 60k checkpoint, preserving the approved true-aspect trajectory layout.')
    mpl.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'svg.fonttype': 'none',
        'axes.linewidth': .65, 'mathtext.fontset': 'stix'})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.5))
    data_rows = []
    for i, (ax, angle) in enumerate(zip(axes.ravel(), [50, 95, 125, 150])):
        with np.load(BASE/groups[0][1]/tag/f'{i}.npz') as data:
            reference = data['reference']-data['reference'][1]
        region = LineString(reference).buffer(1.25, resolution=192)
        x, y = region.exterior.xy
        ax.fill(x, y, color='#edf4fa', zorder=0)
        ax.plot(x, y, color='#8db2d2', lw=.65, ls='--', zorder=1)
        ax.plot(reference[:, 0], reference[:, 1], color='#62676b', ls=':', lw=.9, label='Reference')
        ax.plot(0, 0, 'o', ms=3, color='#333333')
        for j, (label, name, color, linestyle) in enumerate(groups):
            with np.load(BASE/name/tag/f'{i}.npz') as data:
                xy = data['rows'][:, :2]-data['reference'][1]
                speeds = np.linalg.norm(data['rows'][:, 2:4], axis=1)
                initial = data['reference'][0]-data['reference'][1]
            result = json.loads((BASE/name/tag/f'{i}.json').read_text(encoding='utf-8'))
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.15, ls=linestyle, label=label)
            local = np.linalg.norm(xy, axis=1) < 10.
            row = dict(method=label, turn_deg=angle, success=result['success'],
                minimum_vertex_distance_mm=distance_to_commands(np.zeros(2), np.vstack([initial, xy])),
                local_median_speed_mm_s=float(np.median(speeds[local])) if local.any() else None)
            data_rows.append(row)
            value = f"{row['minimum_vertex_distance_mm']:.2f} / {row['local_median_speed_mm_s']:.1f}"
            if not result['success']:
                value += ' (prefix)'
            ax.text(.02+(j % 2)*.53, -.38-(j//2)*.11, value,
                transform=ax.transAxes, color=color, fontsize=8)
        ax.set(xlim=(-10, 10), ylim=(-3, 12), xlabel='x from vertex (mm)', ylabel='y from vertex (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'({chr(97+i)}) {angle}° turn', loc='left', fontsize=9)
        ax.grid(color='#e7eaed', lw=.35)
        ax.text(.02, -.27, 'Distance (mm) / local median speed (mm/s)', transform=ax.transAxes, fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.09, right=.985, top=.90, bottom=.15, hspace=.74, wspace=.28)
    svg = stage/'v7_60000_corners.svg'
    fig.savefig(svg)
    fig.savefig(stage/'v7_60000_corners.png', dpi=180)
    plt.close(fig)
    with (stage/'source_data.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    spec = importlib.util.spec_from_file_location('v7_svg_converter', converter_path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    external = Path('D:/PPO_article_joint_runs/v7_60000_corner_figure_v1')
    converter.convert(svg, external)
    pdf = stage/'v7_60000_corners.pdf'
    shutil.copy2(external/pdf.name, pdf)
    write_json(stage/'caption.json', dict(text='Saved deterministic transitions at the common 60k development checkpoint. All four single-corner paths are shown. Distances are measured to the complete command segments; local speeds use the 10 mm vertex neighbourhood. Values are arranged by independent methods in the first row and gSDE8 methods in the second row, joint then fixed. The shaded region is the 1.25 mm Euclidean tolerance neighbourhood. These intermediate results do not establish final superiority or maximum tolerance utilization.',
        svg_sha256=sha(svg), pdf_sha256=sha(pdf), preview=str(external/'preview.png'),
        full_pdf_visual_QA_pending=True))
    receipt(stage, dict(status='converted_saved_v7_corner_figure_pending_visual_QA',
        saved_trace_files=16, figures=1, model_calls=0, physics_ticks=0))
    print(external/'preview.png')


if __name__ == '__main__':
    main()
