"""Plot the frozen enlarged-path comparisons with separated command-speed axes.

Run after generate_paper_v9.py and before format_extension_results_v9.py.
Only the two declared evidence roots can be remapped in a portable bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator

OLD_ROOT = Path(r'D:\PPO_article_work\JNN_efficiency_20261012_v9')
OLD_BASE = Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
CANONICAL_SHA = '22b7c88fd0298f88801ff3a5d5af79488d8f5b75b349330e0e01ce336d2f1725'
CASES = ('piecewise_curvature_coupon_scale10', 'asymmetric_neck_contour_scale10')
TITLES = ('Curvature-variation path', 'Neck contour')
METHODS = ('JNN', 'FIXED', 'S2', 'S2R', 'SCMO', 'COPP', 'NNC')
SUCCESS_METHODS = METHODS[:5]
DT = (.001, .0005)
PRIMARY_DT = .001
LABEL = dict(JNN='J-NNC', FIXED='Fixed preview', S2='S2-Q', S2R='S2-R',
             SCMO='SCMO', COPP='COPP', NNC='NNC adapter')
COLOR = dict(JNN='#28688a', FIXED='#5b9098', S2='#ba7944', S2R='#577862',
             SCMO='#806595', COPP='#53565d', NNC='#b05664')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_evidence(root, base):
    """Verify all 28 cells, including failure prefixes and the resolution check."""
    def resolve(value):
        path = Path(value)
        for old, new in ((OLD_ROOT, root), (OLD_BASE, base)):
            try:
                local = new / path.relative_to(old)
            except ValueError:
                continue
            local = local.resolve()
            if not local.is_relative_to(new):
                raise ValueError('Evidence path escapes its declared root: ' + str(path))
            return local
        raise ValueError('Evidence path outside the two explicit roots: ' + str(path))

    canonical = root / 'canonical/final_comparison.json'
    assert sha(canonical) == CANONICAL_SHA, 'Frozen canonical bytes changed'
    data = json.loads(canonical.read_text('utf-8'))
    rows = [r for r in data['rows'] if r['case'] in CASES]
    assert len(rows) == 28
    assert {(r['case'], r['method'], r['dt_s']) for r in rows} == {
        (c, m, d) for c in CASES for m in METHODS for d in DT}
    sources = {str(canonical): CANONICAL_SHA}
    references, series, audit, arrays = {}, {}, [], {}
    for case in CASES:
        definition = data['cases'][case]
        for field in ('reference', 'geometry'):
            path = resolve(definition[field])
            digest = sha(path)
            assert digest == definition[field + '_sha256'], str(path)
            sources[str(path)] = digest
            if field == 'reference':
                references[case] = np.loadtxt(path, delimiter=',', skiprows=1)
                arrays[case + '__reference'] = references[case]
    for row in rows:
        metrics = row['metrics']
        complete = metrics['complete_time_s']
        assert metrics['success'] == (complete is not None), row['id']
        local = {}
        for field in ('record', 'trace'):
            if row.get(field):
                path = resolve(row[field])
                digest = sha(path)
                assert digest == row[field + '_sha256'], row['id']
                sources[str(path)] = digest
                local[field] = path
        item = dict(id=row['id'], case=row['case'], method=row['method'], dt_s=row['dt_s'],
                    success=metrics['success'], complete_time_s=complete,
                    reason=metrics.get('reason'), trace_sha256=row.get('trace_sha256'),
                    record_sha256=row.get('record_sha256'),
                    recorded_samples=metrics.get('samples'), positions=None, time_s=None, speed_mm_s=None)
        if 'trace' in local:
            with np.load(local['trace'], allow_pickle=False) as trace:
                positions = trace['positions'].copy()
                assert np.array_equal(trace['reference'], references[row['case']]), row['id']
            assert positions.ndim == 2 and positions.shape[1] == 2 and np.isfinite(positions).all()
            n = len(positions) - 1
            dt = row['dt_s']
            velocity = np.diff(positions, axis=0) / dt
            speed = np.linalg.norm(velocity, axis=1)
            assert n > 0
            acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / dt
            jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / dt
            peaks = [float(np.linalg.norm(a, axis=1).max()) for a in (velocity, acceleration, jerk)]
            assert np.allclose(peaks, metrics['vector_peaks'], rtol=1e-8, atol=2e-4), row['id']
            assert abs(n * dt - metrics['elapsed_prefix_s']) < 1e-8, row['id']
            assert abs(float(speed.sum() * dt) - metrics['executed_length_mm']) < 1e-6, row['id']
            if metrics['success']:
                assert abs(n * dt - complete) < 1e-8
                assert metrics['kinematic_pass'] and metrics['rest_pass']
                assert metrics['geometry']['ordered_complete'] and metrics['geometry']['corridor_inside']
            time = np.arange(1, n + 1) * dt
            item.update(positions=positions, time_s=time, speed_mm_s=speed)
            # The source archive contains every primary trace, including failed prefixes.
            if dt == PRIMARY_DT:
                for name, values in (('positions', positions), ('time_s', time), ('speed_mm_s', speed)):
                    arrays[row['case'] + '__' + row['method'] + '__' + name] = values
        else:
            assert not metrics['success']
        audit.append({k: v for k, v in item.items() if k not in ('positions', 'time_s', 'speed_mm_s')})
        if row['dt_s'] == PRIMARY_DT:
            series[(row['case'], row['method'])] = item
    return references, series, audit, arrays, sources, data['cases']


def maximum_turn_frame(reference, allowance):
    """Choose the first maximum turn using reference geometry only."""
    segments = np.diff(reference, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    assert np.all(lengths > 0)
    directions = segments / lengths[:, None]
    turns = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1))
    vertex = int(np.argmax(turns)) + 1
    incoming, outgoing = directions[vertex - 1], directions[vertex]
    normal = np.array([-incoming[1], incoming[0]])
    if outgoing @ normal < 0:
        normal = -normal
    half_width = 8 * allowance
    normal_rise = half_width * abs(outgoing @ normal) / max(abs(outgoing @ incoming), .05)
    return dict(vertex_index=vertex, turn_angle_deg=float(np.degrees(turns[vertex - 1])),
                center=reference[vertex], incoming=incoming, normal=normal,
                tangent_xlim_mm=[-half_width, half_width],
                normal_ylim_mm=[-1.2 * allowance, normal_rise + 1.2 * allowance],
                reference_length_mm=float(lengths.sum()), effective_allowance_mm=float(allowance),
                selection='First argmax of the unsigned consecutive reference-segment angle; no command data used.',
                window_rule='Tangent half-width is eight effective allowances; normal bounds enclose the two reference rays plus 1.2 allowances.')


def delta_label(seconds):
    if abs(seconds) < 5e-10:
        return '0 ms'
    if abs(seconds) < 1:
        return f'{seconds * 1000:+.0f} ms'
    return f'{seconds:+.3f} s'


def generate(root, base, output):
    root, base, output = root.resolve(), base.resolve(), output.resolve()
    references, series, audit, arrays, sources, definitions = load_evidence(root, base)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(
        conclusion='Complete command geometry and speed profiles explain the recorded completion-time comparisons; small differences remain explicitly small.',
        archetype='quantitative grid', backend='Python/matplotlib', dimensions_mm=[183, 170],
        evidence_chain=['Complete reference and J-NNC command geometry with a reference-selected maximum-turn local view',
                        'Five separate speed axes with a common physical scale and all command intervals',
                        'Exact completion times and differences, with failed methods explicitly retained'],
        visual_precedent=dict(status='User-prescribed layout; no external visual precedent applied',
                              limitation='This local revision does not retrieve or claim a journal precedent.'),
        integrity=dict(smoothing=False, decimation=False, temporal_cropping=False,
                       speed_amplitude_offsets=False, failed_runs_have_completion_times=False,
                       primary_resolution_s=PRIMARY_DT, numerical_check_resolution_s=.0005,
                       statistical_unit='One deterministic complete path execution; no statistical error bars.'))
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8, 'legend.fontsize': 6.5,
        'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'jnn-v9-extension-clarity',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'path.simplify': False, 'legend.frameon': False,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    })
    fig = plt.figure(figsize=(183 / 25.4, 170 / 25.4))
    outer = fig.add_gridspec(2, 3, width_ratios=[1.03, 1.10, 1.15],
                            left=.080, right=.982, bottom=.105, top=.942,
                            wspace=.49, hspace=.34)
    axes_manifest, geometry_windows, geometry_layout_artists = [], [], []
    for index, (case, title) in enumerate(zip(CASES, TITLES)):
        letters = 'abc' if index == 0 else 'def'
        reference = references[case]
        allowance = definitions[case]['evaluation_tolerance_mm']
        frame = maximum_turn_frame(reference, allowance)
        extent = np.ptp(reference, axis=0)
        geometry_ax = fig.add_subplot(outer[index, 0])
        geometry_ax.plot(*reference.T, color='#4c4c4c', lw=.7, ls=(0, (3, 2)), label='Reference')
        jnn = series[(case, 'JNN')]
        geometry_ax.plot(*jnn['positions'].T, color=COLOR['JNN'], lw=.9,
                         label=LABEL['JNN'], rasterized=True)
        geometry_ax.plot(*reference[0], marker='^', ms=3.7, markerfacecolor='white',
                         markeredgecolor='#333333', mew=.7, zorder=5)
        geometry_ax.set_aspect('equal', adjustable='box')
        if extent[0] / extent[1] > 1.5:
            geometry_ax.set_anchor((.5, .40))
        geometry_ax.set(xlabel='X (mm)', ylabel='Y (mm)')
        geometry_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        geometry_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        geometry_ax.margins(.08)
        geometry_box = outer[index, 0].get_position(fig)
        fig.text(geometry_box.x0, geometry_box.y1 + .017, letters[0] + '  ' + title,
                 fontsize=8, fontweight='bold')
        information_text = fig.text(geometry_box.x0, geometry_box.y1 - .012,
                                   rf'$L_{{\mathrm{{ref}}}}={frame["reference_length_mm"]:.2f}$ mm; $\varepsilon_{{\mathrm{{poly}}}}={allowance:.4f}$ mm',
                                   fontsize=6.3, color='#4c4c4c')
        if index == 0:
            geometry_ax.legend(loc='upper right', handlelength=1.8, borderaxespad=.3)
        # The local view is positioned in free canvas space; its selection uses only reference geometry.
        inset_box = [.10, 1.22, .80, .55] if extent[0] / extent[1] > 1.5 else [.24, .29, .45, .30]
        local_ax = geometry_ax.inset_axes(inset_box)
        projected = {}
        for name, positions, color, style, width in (
                ('reference', reference, '#4c4c4c', (0, (3, 2)), .65),
                ('JNN', jnn['positions'], COLOR['JNN'], '-', .8)):
            delta = positions - frame['center']
            xy = np.column_stack((delta @ frame['incoming'], delta @ frame['normal']))
            projected[name] = xy
            arrays[case + '__local_' + name + '_tangent_normal_mm'] = xy
            local_ax.plot(*xy.T, color=color, ls=style, lw=width, rasterized=name == 'JNN')
        local_ax.set(xlim=frame['tangent_xlim_mm'], ylim=frame['normal_ylim_mm'],
                     xlabel='Tangent (mm)', ylabel='Normal (mm)')
        local_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        local_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        local_ax.tick_params(labelsize=5.1, length=1.8, pad=1)
        local_ax.xaxis.label.set_size(5.5)
        local_ax.yaxis.label.set_size(5.5)
        local_ax.xaxis.labelpad = 1
        local_ax.yaxis.labelpad = 1
        local_title = local_ax.set_title(f'Local turn: {frame["turn_angle_deg"]:.2f}' + '\N{DEGREE SIGN}',
                                         loc='left', fontsize=5.7, pad=3)
        geometry_layout_artists.append((information_text, local_title))
        local_ax.set_facecolor('#fbfcfd')
        local_ax.grid(color='#e5e9ec', lw=.35)
        local_ax.spines[['top', 'right']].set_visible(True)
        for spine in local_ax.spines.values():
            spine.set_color('#acb4bb')
            spine.set_linewidth(.5)
        geometry_ax.plot(*frame['center'], marker='o', ms=4, markerfacecolor='none',
                         markeredgecolor='#353c42', mew=.65, zorder=6)
        connector = ConnectionPatch(xyA=frame['center'], coordsA=geometry_ax.transData,
                                    xyB=(1, .03), coordsB=local_ax.transAxes,
                                    color='#9ba5ad', linewidth=.5, linestyle=(0, (2.5, 2.5)),
                                    zorder=1, clip_on=False)
        fig.add_artist(connector)
        geometry_windows.append(dict(case=case, **{k: v.tolist() if isinstance(v, np.ndarray) else v
                                                   for k, v in frame.items()}, inset_box_axes_fraction=inset_box,
                                     coordinate_note='Tangent and normal projections use separately labelled scales; no tolerance buffer is drawn.'))
        speed_grid = outer[index, 1].subgridspec(5, 1, hspace=.45)
        max_time = max(series[(case, m)]['complete_time_s'] for m in SUCCESS_METHODS)
        for k, method in enumerate(SUCCESS_METHODS):
            item = series[(case, method)]
            assert item['success']
            speed_ax = fig.add_subplot(speed_grid[k, 0])
            # One value for every command interval. The same physical y-scale is used in all ten axes.
            speed_ax.plot(item['time_s'], item['speed_mm_s'], color=COLOR[method],
                          lw=.65, rasterized=True)
            speed_ax.set(xlim=(0, max_time * 1.015), ylim=(-.5, 100.5), yticks=[0, 100])
            speed_ax.set_facecolor('#f5f8fa' if k % 2 == 0 else '#fbfcfd')
            for level in (0, 50):
                speed_ax.axhline(level, lw=.35, color='#dfe5e9', zorder=0)
            speed_ax.axhline(100, lw=.6, ls=(0, (2, 2)), color='#9da8b1', zorder=0)
            speed_ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
            speed_ax.text(.035, .18, LABEL[method], transform=speed_ax.transAxes,
                          color=COLOR[method], fontsize=6.5)
            speed_ax.tick_params(axis='y', pad=2)
            if k == 0:
                speed_ax.set_title(letters[1] + '  Command speed (mm/s)',
                                   loc='left', fontweight='bold', pad=8)
            if k < 4:
                speed_ax.tick_params(axis='x', labelbottom=False)
            else:
                speed_ax.set_xlabel('Elapsed time (s)', labelpad=3)
            axes_manifest.append(dict(case=case, method=method, displayed_commands=len(item['time_s']),
                                      total_recorded_commands=len(item['positions']) - 1,
                                      xlim_s=[0, max_time * 1.015], ylim_mm_s=[-.5, 100.5],
                                      amplitude_offset_mm_s=0))
        right_grid = outer[index, 2].subgridspec(2, 1, height_ratios=[5.2, 1.2], hspace=.50)
        time_ax = fig.add_subplot(right_grid[0, 0])
        time_ax.set_title(letters[2] + '  Completion times', loc='left', fontweight='bold', pad=8)
        time_ax.set_xscale('symlog', linthresh=.02, linscale=1)
        time_ax.set(xlim=(-.001, 9), ylim=(-.52, 4.58), yticks=[],
                    xticks=[0, .01, .1, 1], xticklabels=['0', '0.01', '0.1', '1'])
        time_ax.spines['left'].set_visible(False)
        time_ax.grid(axis='x', color='#e1e6e9', linewidth=.4, zorder=0)
        time_ax.axvline(0, lw=.75, color='#606970')
        time_ax.set_xlabel(r'$\Delta T=T-T_{\mathrm{J\!\!-\!\!NNC}}$ (s)', labelpad=3)
        for k, method in enumerate(SUCCESS_METHODS):
            item = series[(case, method)]
            completion = item['complete_time_s']
            delta = completion - jnn['complete_time_s']
            y = 4 - k
            time_ax.text(.015, (y + .34 + .52) / 5.10,
                         f'{LABEL[method]}   {completion:.3f} s',
                         transform=time_ax.transAxes, va='center', fontsize=6.6)
            time_ax.plot([0, delta], [y - .12, y - .12], color=COLOR[method], lw=1.05)
            time_ax.plot(delta, y - .12, marker='o', ms=3.4, color=COLOR[method], clip_on=False)
            # Exact labels are independent of the nonlinear axis spacing.
            time_ax.annotate(delta_label(delta), (delta, y - .12),
                             xytext=(-4 if delta > .5 else 4, 0), textcoords='offset points',
                             ha='right' if delta > .5 else 'left', va='center',
                             fontsize=6.5, color=COLOR[method],
                             bbox=dict(facecolor='white', edgecolor='none', pad=.25, alpha=.9))
        failure_ax = fig.add_subplot(right_grid[1, 0])
        failure_ax.axis('off')
        for y, method in ((.77, 'NNC'), (.17, 'COPP')):
            assert not series[(case, method)]['success']
            assert series[(case, method)]['complete_time_s'] is None
            failure_ax.text(0, y, method + ': \u2014', transform=failure_ax.transAxes,
                            color=COLOR[method], fontsize=6.35, va='center')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for window, (information_text, local_title) in zip(geometry_windows, geometry_layout_artists):
        gap_mm = (information_text.get_window_extent(renderer).y0 - local_title.get_window_extent(renderer).y1) * 25.4 / fig.dpi
        window['information_to_inset_title_gap_mm'] = float(gap_mm)
        assert gap_mm >= 6, 'Geometry information and inset title must be separated by at least 6 mm'
    outputs = []
    for extension in ('pdf', 'png', 'svg'):
        path = output / ('figure9_v9.' + extension)
        fig.savefig(path, dpi=400, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    array_path = output / 'figure9_v9_source.npz'
    np.savez_compressed(array_path, **arrays)
    outputs.append(array_path)
    differences = []
    for case in CASES:
        jnn_time = series[(case, 'JNN')]['complete_time_s']
        for method in METHODS:
            item = series[(case, method)]
            complete = item['complete_time_s']
            differences.append(dict(case=case, method=method, complete_time_s=complete,
                                    delta_time_s=None if complete is None else complete - jnn_time,
                                    delta_label=None if complete is None else delta_label(complete - jnn_time)))
    data_path = output / 'figure9_v9_data.json'
    write_json(data_path, dict(primary_dt_s=PRIMARY_DT, figure_rows=[r for r in audit if r['dt_s'] == PRIMARY_DT],
                               all_comparison_rows=audit, source_arrays=list(arrays),
                               completion_comparison=differences, speed_axes=axes_manifest,
                               reference_selected_geometry_windows=geometry_windows,
                               dimensions_mm=[183, 170], colors=COLOR, figure_contract=contract))
    outputs.append(data_path)
    caption = ('Complete commands on the enlarged curvature-variation path (top) and neck contour (bottom) at 1 ms. '
               'Left: full geometry and local tangent/normal views at the first maximum reference turning angle (circles); triangles mark starts. '
               r'$\varepsilon_{\mathrm{poly}}$ is the effective one-sided polygonal tolerance. '
               'Middle: complete command speeds on separate axes with a common scale; dashed lines mark 100 mm/s. '
               'Right: completion times and differences from J-NNC. Difference axes are linear to 0.02 s and logarithmic thereafter. '
               'Dashes indicate no complete time; NNC denotes the project adapter.' + chr(10))
    caption_path = output / 'figure9_v9_caption.tex'
    caption_path.write_text(caption, encoding='utf-8')
    outputs.append(caption_path)
    # Never hash a previous receipt into its replacement.
    receipt = dict(validated_cells=len(audit), canonical_sha256=CANONICAL_SHA,
                   generator_sha256=sha(__file__), input_sha256=sources,
                   primary_dt_s=PRIMARY_DT, numerical_check_dt_s=.0005,
                   smoothing=False, decimation=False, temporal_cropping=False,
                   speed_amplitude_offsets=False, full_command_counts=axes_manifest,
                   reference_selected_geometry_windows=geometry_windows,
                   outputs_sha256={p.name: sha(p) for p in outputs})
    write_json(output / 'figure9_v9_receipt.json', receipt)
    print(json.dumps(dict(output=str(output), validated_cells=len(audit), full_speed_axes=len(axes_manifest),
                          canonical_sha256=CANONICAL_SHA)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    generate(args.root, args.base, args.output)
