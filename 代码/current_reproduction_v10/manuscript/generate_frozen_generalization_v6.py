"""All-case, equal-scale geometry figure from the frozen-model transfer traces."""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha, utc, write_json

DESTINATION = ROOT / 'shared_manuscript/frozen_main_v6'
COLORS = {1000: '#0072B2', 500: '#D55E00'}
TITLES = {
    'hilbert_cell_sweep': 'Hilbert sweep',
    'harmonic_radial_contour': 'Harmonic contour',
    'piecewise_curvature_coupon': 'Curvature coupon',
    'asymmetric_neck_contour': 'Neck contour',
    'annular_sector_raster': 'Annular raster',
}


def main():
    font = font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.titlesize': 7.5,
        'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'path.simplify': False})
    registration_path = ROOT / 'evidence/frozen_main_generalization_v6.json'
    report_path = ROOT / 'audit/frozen_main_generalization_v6_report.json'
    data_path = DESTINATION / 'generalization_geometry_data.npz'
    metadata_path = DESTINATION / 'generalization_geometry_metadata.json'
    registration = json.loads(registration_path.read_text('utf-8'))
    report = json.loads(report_path.read_text('utf-8'))
    metadata = json.loads(metadata_path.read_text('utf-8'))
    assert report['registration_sha256'] == metadata['registration_sha256'] == sha(registration_path)
    assert metadata['data_sha256'] == sha(data_path)
    assert report['complete_grid'] and len(report['rows']) == 10
    figure, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 132 / 25.4))
    figure.subplots_adjust(left=.072, right=.99, top=.935, bottom=.10, wspace=.31, hspace=.46)
    panels = []
    with np.load(data_path) as archive:
        for index, case in enumerate(registration['cases']):
            axis = axes.flat[index]
            base = case['id']
            reference = archive[base + '_dt1000us__reference']
            axis.plot(reference[:, 0], reference[:, 1], color='#A4A4A4', linewidth=1.25,
                solid_capstyle='round', label='Reference', zorder=1)
            all_positions = [reference]
            end_markers = []
            for resolution in (1000, 500):
                name = f'{base}_dt{resolution}us'
                result = next(x for x in report['rows'] if x['run_id'] == name)
                positions = archive[name + '__positions']
                all_positions.append(positions)
                style = '-' if resolution == 1000 else (0, (3.4, 2.0))
                axis.plot(positions[:, 0], positions[:, 1], color=COLORS[resolution],
                    linewidth=.95, linestyle=style, zorder=2 + (resolution == 500))
                marker = 'o' if result['success'] else 'x'
                size = 3.4 if result['success'] else (6.0 if resolution == 1000 else 4.5)
                axis.plot(positions[-1, 0], positions[-1, 1], marker=marker, linestyle='none',
                    color=COLORS[resolution], markersize=size, markeredgewidth=.9, zorder=6)
                end_markers.append(dict(run_id=name, marker=marker, endpoint_mm=positions[-1].tolist(),
                    positions_plotted=len(positions), full_recorded_prefix=True, success=result['success']))
            axis.plot(reference[0, 0], reference[0, 1], marker='^', markersize=5.0,
                markerfacecolor='none', markeredgecolor='#333333', markeredgewidth=.8,
                linestyle='none', zorder=5)
            # Every panel spans100mm in each direction; no hidden trajectory rescaling.
            joined = np.vstack(all_positions)
            center = (joined.min(axis=0) + joined.max(axis=0)) / 2.
            axis.set_xlim(center[0] - 50., center[0] + 50.)
            axis.set_ylim(center[1] - 50., center[1] + 50.)
            axis.set_aspect('equal', adjustable='box')
            for coordinate, setter in ((0, axis.set_xticks), (1, axis.set_yticks)):
                lower, upper = center[coordinate] - 50., center[coordinate] + 50.
                setter(np.arange(np.ceil(lower / 25.) * 25., upper + 1e-9, 25.))
            axis.set_xlabel('x (mm)', labelpad=2)
            axis.set_ylabel('y (mm)', labelpad=2)
            axis.set_title(TITLES[base], loc='left', pad=7)
            axis.text(-.20, 1.05, chr(97 + index), transform=axis.transAxes,
                weight='bold', size=8, ha='left', va='bottom')
            panels.append(dict(case=base, limits_x_mm=list(axis.get_xlim()), limits_y_mm=list(axis.get_ylim()),
                equal_metric_scale=True, ends=end_markers))
    legend_axis = axes.flat[5]
    legend_axis.axis('off')
    handles = [Line2D([], [], color='#A4A4A4', linewidth=1.25, label='Complete reference'),
        Line2D([], [], color=COLORS[1000], linewidth=.95, label='J-NNC, 1 ms'),
        Line2D([], [], color=COLORS[500], linewidth=.95, linestyle=(0, (3.4, 2.0)), label='J-NNC, 0.5 ms'),
        Line2D([], [], color='#333333', marker='^', markerfacecolor='none', linestyle='none',
            markersize=5, label='Start'),
        Line2D([], [], color='#333333', marker='o', linestyle='none', markersize=3.4, label='Completed endpoint'),
        Line2D([], [], color='#333333', marker='x', linestyle='none', markersize=5, label='Stopped before completion')]
    legend_axis.legend(handles=handles, loc='center left', bbox_to_anchor=(0., .5),
        handlelength=2.6, labelspacing=1.0, borderaxespad=0., fontsize=7)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    text_outside = []
    for obj in figure.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            text_outside.append(obj.get_text())
    if text_outside:
        raise ValueError('Text outside figure: ' + repr(text_outside))
    stem = DESTINATION / 'generalization'
    for suffix in ('.pdf', '.svg'):
        figure.savefig(stem.with_suffix(suffix), facecolor='white')
    figure.savefig(stem.with_suffix('.png'), dpi=400, facecolor='white')
    plt.close(figure)
    receipt = dict(created_utc=utc(), figure_id='figure9_frozen_generalization',
        generator=str(Path(__file__).relative_to(ROOT)), generator_sha256=sha(__file__),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (registration_path, report_path, data_path, metadata_path)},
        outputs={str(stem.with_suffix(suffix).relative_to(ROOT)): sha(stem.with_suffix(suffix))
                 for suffix in ('.pdf', '.svg', '.png')},
        font='Arial', font_file=font, width_mm=183, height_mm=132,
        backend='Python/matplotlib', panels=panels, full_resolution_positions=True,
        smoothing=False, decimation=False, no_case_exclusion=True,
        label_bounds_pass=True, observation_unit='One frozen controller execution on each full synthetic program and resolution.',
        visual_precedent='The requested all-case trajectory overlay and the existing project geometry convention determine the layout; no external scholarly visual precedent is asserted.')
    write_json(DESTINATION / 'generalization_figure_receipt.json', receipt)
    contract = ('# Frozen-model transfer figure contract\n\n'
        'Claim: the five full synthetic programs show the geometry-dependent completion of one frozen controller at two integration resolutions.\n'
        'Evidence: all ten registered trajectories or complete failure prefixes are shown with their full reference programs; no successful-only panel selection.\n'
        'Archetype: quantitative geometry grid. Five equally scaled100mm-by100mm panels plus one shared legend; each panel is one distinct sealed geometry family.\n'
        'Encoding: gray reference, blue solid1ms, orange dashed0.5ms, colored crosses for failed endpoints and dots for completed endpoints; open triangles mark starts.\n'
        'Export: Python/matplotlib, Arial,183mm by132mm, vector PDF and editable SVG plus400dpi PNG. No smoothing, point deletion, hidden scaling or trace crop.\n'
        'Precedent: the author specified the all-case geometry overlay. The existing project figure contract supplies typography and full-trace conventions. No unverified top-journal example is claimed or cited.\n'
        'Statistics: one frozen checkpoint, five programs, two paired numerical resolutions; no seed variability or efficiency-comparison claim.\n')
    (DESTINATION / 'generalization_figure_contract.md').write_text(contract, 'utf-8')
    print(json.dumps(dict(outputs=list(receipt['outputs']), label_bounds_pass=True), indent=2))


if __name__ == '__main__':
    main()
