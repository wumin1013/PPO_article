"""Replot retained development diagnostics; never execute or alter experiments."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import platform

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript'
BASE = ROOT / 'baseline_records/supplementary_reproduction'
STEM = 'development_constraint_comparison'
DT = 0.001
SAMPLES = 1000
WINDOW_MS = 80
LIMITS = np.array([2000.0, 62500.0])
COLORS = {'J-NNC': '#17689A', 'NNC adaptation': '#B85248'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def acceleration_and_jerk(positions):
    """Backward differences with the declared zero initial velocity/acceleration."""
    velocity = np.diff(positions, axis=0) / DT
    acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / DT
    jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / DT
    return np.column_stack([np.linalg.norm(acceleration, axis=1),
                            np.linalg.norm(jerk, axis=1)])


def main():
    sources, traces, metrics = {}, {}, []
    times = np.arange(1, SAMPLES + 1) * DT
    shown = times <= WINDOW_MS / 1000 + 1e-12
    for case in ['original', 'dense']:
        reference = None
        for method, relative in [
                ('NNC adaptation', f'nnc_diagnostic/{case}'),
                ('J-NNC', f'records/J-NNC/{case}_1ms')]:
            trace_file = BASE / (relative + '.npz')
            result_file = BASE / (relative + '.json')
            result = json.loads(result_file.read_text(encoding='utf-8'))
            require(sha(trace_file) == result['trace_sha256'], 'Trace hash mismatch: ' + relative)
            for source in [trace_file, result_file]:
                sources[source.relative_to(ROOT).as_posix()] = sha(source)
            with np.load(trace_file, allow_pickle=False) as z:
                current_reference = z['reference'].copy()
                if method == 'NNC adaptation':
                    require(result['physical_interval_s'] == DT and
                            result['observed_prefix_time_s'] == 1.0 and
                            result['recorded_ticks'] == SAMPLES,
                            'Unexpected NNC diagnostic interval')
                    positions = z['positions'].copy()
                    require(len(positions) == SAMPLES + 1 and
                            np.allclose(z['timestamps'], np.arange(SAMPLES + 1) * DT,
                                        rtol=0, atol=1e-14), 'Unexpected NNC timestamps')
                    stored = np.column_stack([z['acceleration'], z['jerk']])
                else:
                    require(result['success'] is True and float(z['dt_s']) == DT,
                            'Expected successful 1 ms historical J-NNC record')
                    positions = np.vstack([current_reference[0], z['rows'][:SAMPLES, :2]])
                    stored = np.column_stack([np.linalg.norm(z['rows'][:SAMPLES, 4:6], axis=1),
                                              np.linalg.norm(z['rows'][:SAMPLES, 6:8], axis=1)])
            require(positions.shape == (SAMPLES + 1, 2) and np.isfinite(positions).all(),
                    'Invalid position prefix: ' + relative)
            require(np.array_equal(positions[0], current_reference[0]), 'Initial position mismatch')
            if reference is not None:
                require(np.array_equal(reference, current_reference), 'Different reference programs')
            reference = current_reference
            motion = acceleration_and_jerk(positions)
            normalized_record_difference = np.max(np.abs(motion - stored) / LIMITS, axis=0)
            require(np.all(normalized_record_difference <= 1e-7),
                    'Position-derived dynamics disagree with retained records: ' + relative)
            normalized = motion / LIMITS
            traces[case, method] = (motion, normalized)
            metrics.append(dict(case=case, method=method, recorded_interval_s=1.0,
                displayed_interval_ms=[1, WINDOW_MS],
                displayed_acceleration_limit_ratio=float(normalized[shown, 0].max()),
                displayed_jerk_limit_ratio=float(normalized[shown, 1].max()),
                first_second_acceleration_limit_ratio=float(normalized[:, 0].max()),
                first_second_jerk_limit_ratio=float(normalized[:, 1].max()),
                after_20ms_jerk_limit_ratio=float(normalized[times > .020, 1].max()),
                position_reconstruction_verified=True,
                max_normalized_difference_from_retained_dynamics=normalized_record_difference.tolist(),
                retained_dynamics_comparison_absolute_ratio_tolerance=1e-7,
                jerk_violations_first_second=int(np.count_nonzero(normalized[:, 1] > 1 + 1e-8))))

    # All data checks precede output; this is a deterministic plotting-only entry.
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'path.simplify': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.9),
                             gridspec_kw={'hspace': .4, 'wspace': .25})
    labels = [r'$\|a\|/A_{\max}$', r'$\|j\|/J_{\max}$']
    for col, case in enumerate(['original', 'dense']):
        for row in range(2):
            ax = axes[row, col]
            for method in ['NNC adaptation', 'J-NNC']:
                normalized = traces[case, method][1]
                ax.plot(times[shown] * 1000, normalized[shown, row],
                        color=COLORS[method], lw=.9 if method == 'NNC adaptation' else 1.0)
            ax.axvspan(0, 20, color='#EDF1F4', zorder=-4)
            ax.axhline(1, color='#3F4549', ls='--', lw=.8)
            ax.set_yscale('symlog', linthresh=1, linscale=.7)
            ticks = [0, 1, 10, 30] if row == 0 else [0, 1, 10, 100, 1000]
            ax.set(xlim=(0, WINDOW_MS), ylim=(-.08, 45 if row == 0 else 2000),
                   xticks=[0, 20, 40, 60, 80], yticks=ticks)
            ax.set_yticklabels([str(tick) for tick in ticks])
            if col == 0:
                ax.set_ylabel(labels[row])
            if row == 1:
                ax.set_xlabel('Elapsed time (ms)')
            title = ('Long route' if case == 'original' else 'Dense case')
            title += ': ' + ('acceleration' if row == 0 else 'jerk')
            ax.set_title(chr(97 + row * 2 + col) + '  ' + title, loc='left', fontweight='bold', pad=7)
            peak = traces[case, 'NNC adaptation'][1][shown, row].max()
            ax.text(.98, .94, f'NNC adaptation peak: {peak:.1f} × limit', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1.5})
            ax.grid(axis='y', color='#E3E7EA', lw=.4)
    handles = [Line2D([], [], color=COLORS[method], lw=1.2, label=method)
               for method in ['J-NNC', 'NNC adaptation']]
    handles.append(Line2D([], [], color='#3F4549', ls='--', lw=.8, label='Common limit'))
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.53, .995))
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.11)
    outputs = []
    for extension in ['pdf', 'svg', 'png']:
        path = OUT / f'{STEM}.{extension}'
        fig.savefig(path, dpi=300, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    source_csv = OUT / 'development_constraint_source.csv'
    with source_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'method', 'time_s', 'acceleration_mm_s2', 'jerk_mm_s3',
                         'acceleration_limit_ratio', 'jerk_limit_ratio', 'in_displayed_window'])
        for case in ['original', 'dense']:
            for method in ['NNC adaptation', 'J-NNC']:
                motion, normalized = traces[case, method]
                for index, time in enumerate(times):
                    writer.writerow([case, method, time, *motion[index], *normalized[index], int(shown[index])])
    outputs.append(source_csv)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), figure=STEM,
        source_sha256=sources, script_sha256=sha(__file__),
        contract_sha256=sha(OUT / 'development_constraint_figure_contract.md'),
        outputs_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in outputs},
        metrics=metrics, source_csv_rows=2 * 2 * SAMPLES,
        limits={'acceleration_mm_s2': LIMITS[0], 'jerk_mm_s3': LIMITS[1]},
        metric_definition='Backward differences of XY positions, with zero initial velocity and acceleration; planar Euclidean norms.',
        rendering={'figure_inches': [7.25, 4.9], 'png_dpi': 300, 'dt_s': DT,
                   'displayed_window_ms': [1, WINDOW_MS], 'initial_shading_ms': [0, 20],
                   'symlog_linthresh': 1, 'symlog_linscale': .7, 'smoothing': False,
                   'resampling': False, 'path_simplification': False, 'editable_svg_text': True},
        software={'python': platform.python_version(), 'numpy': np.__version__,
                  'matplotlib': matplotlib.__version__},
        scope='Historical development-case commands; one retained run per method and path. NNC adaptation is not an exact published-NNC reproduction or an isolated KCM ablation. No servo or cutting measurements.',
        caption='Constraint diagnostics on the two historical development paths at 1 ms. Normalized planar acceleration (a,b) and jerk (c,d) are reconstructed from recorded positions with initial rest. The shaded region marks the first 20 ms and dashed lines show the common limits. Axes are linear from zero to one and logarithmic above one. Peak annotations apply only to the displayed 1--80 ms interval. The direct-action NNC adaptation differs from J-NNC in its checkpoint and execution scheme; this is not an isolated KCM ablation.')
    (OUT / 'development_constraint_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'figure': STEM, 'metrics': metrics, 'source_csv_rows': 4000}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
