"""Recompute every registered transfer outcome from the retained full command trace."""
import csv
import io
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate_frozen_main_generalization_v6 import REGISTRATION, OUT, verify
from frozen_policy import FrozenPolicy
from metrics import metrics


def main():
    torch.set_num_threads(1)
    registration = verify()
    registration_sha = sha(REGISTRATION)
    policy = FrozenPolicy(ROOT / registration['model'])
    rows = []
    for cell in registration['grid']:
        case = next(x for x in registration['cases'] if x['id'] == cell['case'])
        path = ROOT / cell['record']
        if not path.exists():
            raise FileNotFoundError('All ten registered outcomes are required: ' + str(path))
        result = json.loads(path.read_text('utf-8'))
        if result['registration_sha256'] != registration_sha:
            raise ValueError('Run belongs to another registration')
        if result['model_sha256'] != registration['model_sha256']:
            raise ValueError('Run belongs to another checkpoint')
        trace = ROOT / cell['trace']
        if sha(trace) != result['trace_sha256']:
            raise ValueError('Retained trace changed')
        with np.load(trace) as archive:
            positions = archive['positions'].copy()
            reference = archive['reference'].copy()
            observations = archive['observations'].copy()
            actions = archive['actions'].copy()
            commands = archive['rows'].copy()
        expected_reference = np.loadtxt(ROOT / case['reference'], delimiter=',', skiprows=1)
        if not np.array_equal(reference, expected_reference) or not np.array_equal(positions[0], reference[0]):
            raise ValueError('Reference or initial position mismatch')
        if len(observations) != len(actions) or len(actions) != result['submitted_actions']:
            raise ValueError('Policy call accounting mismatch')
        action_error = 0.
        if len(observations):
            predicted = np.vstack([policy.mean(batch).numpy() for batch in np.array_split(
                observations, max(1, int(np.ceil(len(observations) / 256))))])
            action_error = float(np.abs(predicted - actions).max())
            if action_error >= 2e-6:
                raise ValueError('Saved requests are not c16 policy outputs')
        independently = metrics(reference, positions, dt=cell['dt_s'],
            tolerance=case['evaluation_tolerance_mm'], declared_success=result['executor_reason'] == 'success',
            reason=result['executor_reason'])
        if independently['success'] != result['success']:
            raise ValueError('Independent acceptance changed')
        for key in ('complete_time_s', 'elapsed_prefix_s', 'max_polyline_deviation_mm', 'terminal_rest_s'):
            if key not in independently:
                continue
            a, b = independently[key], result[key]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                raise ValueError('Independent metric changed: ' + key)
        if 'vector_peaks' in independently and not np.allclose(
                independently['vector_peaks'], result['vector_peaks'], rtol=1e-10, atol=1e-7):
            raise ValueError('Independent planar metrics changed')
        gates = independently.get('geometry') or {}
        rows.append(dict(run_id=cell['run_id'], case=case['id'], origin='synthetic',
            dt_s=cell['dt_s'], success=result['success'], complete_time_s=result['complete_time_s'],
            retained_prefix_s=result['elapsed_prefix_s'], reason=result['reason'], error=result.get('error'),
            gates_passed=gates.get('gates_passed'), gate_count=gates.get('gate_count'),
            max_polyline_deviation_mm=independently.get('max_polyline_deviation_mm'),
            evaluation_tolerance_mm=case['evaluation_tolerance_mm'],
            nominal_tolerance_mm=case['nominal_tolerance_mm'],
            vector_peaks=independently.get('vector_peaks'),
            terminal_rest_s=independently.get('terminal_rest_s'),
            positions=len(positions), command_rows=len(commands), policy_calls=len(actions),
            max_c16_action_replay_error=action_error,
            wall_s=result['wall_s'], cpu_s=result['cpu_s'],
            cost=result.get('cost'), source_trace=cell['trace'], source_trace_sha256=sha(trace),
            source_record=cell['record'], source_record_sha256=sha(path),
            independent_audit=independently))
    comparisons = []
    for case in registration['cases']:
        nominal = next(x for x in rows if x['case'] == case['id'] and x['dt_s'] == .001)
        finer = next(x for x in rows if x['case'] == case['id'] and x['dt_s'] == .0005)
        paired = nominal['success'] and finer['success']
        comparisons.append(dict(case=case['id'], nominal_success=nominal['success'], finer_success=finer['success'],
            nominal_complete_s=nominal['complete_time_s'], finer_complete_s=finer['complete_time_s'],
            completion_time_change_percent=(100. * (finer['complete_time_s'] / nominal['complete_time_s'] - 1.) if paired else None),
            comparison_available=paired))
    summary = dict(created_utc=utc(), registration_sha256=registration_sha,
        model_sha256=registration['model_sha256'], execution_source_sha256=registration['source_sha256'],
        generator_sha256=sha(__file__), complete_grid=True, registered_cells=10,
        successful_cells=sum(x['success'] for x in rows),
        successful_by_resolution={str(dt): sum(x['success'] for x in rows if x['dt_s'] == dt) for dt in (.001, .0005)},
        paired_complete_cases=sum(x['comparison_available'] for x in comparisons),
        statistics='Five synthetic complete programs, one frozen trained checkpoint; two numerical resolutions are paired settings, not independent seeds. No successful-only aggregate of time.',
        rows=rows, resolution_comparisons=comparisons)
    write_json(ROOT / 'audit/frozen_main_generalization_v6_report.json', summary)
    stream = io.StringIO(newline='')
    keys = ['run_id', 'case', 'dt_s', 'success', 'complete_time_s', 'retained_prefix_s', 'reason',
            'gates_passed', 'gate_count', 'max_polyline_deviation_mm', 'evaluation_tolerance_mm',
            'terminal_rest_s', 'policy_calls', 'wall_s', 'cpu_s']
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows({k: row[k] for k in keys} for row in rows)
    (OUT / 'complete_results.csv').write_text(stream.getvalue(), encoding='utf-8')
    lines = ['# Frozen main-model synthetic transfer study', '',
        f"Registered grid: 5 complete synthetic programs x 2 resolutions. Completed {summary['successful_cells']}/10 executions.", '',
        '| Program | h (ms) | Complete | Complete time (s) | Retained prefix (s) | Ordered gates | Reason |',
        '|---|---:|---|---:|---:|---|---|']
    for row in rows:
        complete = f"{row['complete_time_s']:.4f}" if row['success'] else '--'
        lines.append(f"| {row['case']} | {row['dt_s']*1000:g} | {row['success']} | {complete} | {row['retained_prefix_s']:.4f} | {row['gates_passed']}/{row['gate_count']} | {row['reason']} |")
    lines.extend(['', 'Each result was recomputed from the complete retained command positions. Saved requests were replayed against the frozen c16 actor. Incomplete runs retain null completion time; no mean time over successful subsets is reported.',
        '', 'The numerical-resolution comparison is paired within each program. This study contains no new training, parameter selection, physical axis motion or hardware-in-the-loop experiment.', ''])
    (ROOT / 'audit/frozen_main_generalization_v6_report.md').write_text('\n'.join(lines), encoding='utf-8')
    destination = ROOT / 'shared_manuscript/frozen_main_v6'
    destination.mkdir(parents=True, exist_ok=True)
    labels = dict(hilbert_cell_sweep='Hilbert sweep', harmonic_radial_contour='Harmonic contour',
        piecewise_curvature_coupon='Curvature coupon', asymmetric_neck_contour='Neck contour',
        annular_sector_raster='Annular raster')
    tex = [r'\begin{table}[tb]', r'\centering', r'\small',
        r'\caption{Complete-program transfer of the frozen J-NNC on five synthetic programs. Times include braking and final rest. A dash denotes an incomplete execution; gate counts retain every registered case.}',
        r'\label{tab:frozen_generalization}', r'\begin{tabular}{lrrrr}', r'\hline',
        r'Program & $T_{1\,\mathrm{ms}}$ (s) & Gates & $T_{0.5\,\mathrm{ms}}$ (s) & Gates\\', r'\hline']
    figure_arrays = {}
    figure_metadata = dict(registration_sha256=registration_sha,
        selection='All five registered programs and both resolutions; no outcome selection.',
        units='mm', curves=[])
    for case in registration['cases']:
        cells = [next(x for x in rows if x['case'] == case['id'] and x['dt_s'] == dt) for dt in (.001, .0005)]
        text_times = [f"{x['complete_time_s']:.4f}" if x['success'] else r'--' for x in cells]
        gate_counts = [f"{x['gates_passed']}/{x['gate_count']}" for x in cells]
        tex.append(f"{labels[case['id']]} & {text_times[0]} & {gate_counts[0]} & {text_times[1]} & {gate_counts[1]}" + r'\\')
        for row in cells:
            with np.load(ROOT / row['source_trace']) as z:
                prefix = row['run_id']
                figure_arrays[prefix + '__reference'] = z['reference'].copy()
                figure_arrays[prefix + '__positions'] = z['positions'].copy()
            figure_metadata['curves'].append(dict(run_id=row['run_id'], label=labels[case['id']],
                complete=row['success'], curve_scope='complete trajectory' if row['success'] else 'entire retained failure prefix',
                source_trace=row['source_trace'], source_trace_sha256=row['source_trace_sha256']))
    tex.extend([r'\hline', r'\end{tabular}', r'\end{table}', ''])
    (destination / 'generalization_table.tex').write_text('\n'.join(tex), encoding='utf-8')
    np.savez_compressed(destination / 'generalization_geometry_data.npz', **figure_arrays)
    figure_metadata['data_sha256'] = sha(destination / 'generalization_geometry_data.npz')
    write_json(destination / 'generalization_geometry_metadata.json', figure_metadata)
    print(json.dumps({k: summary[k] for k in ('complete_grid', 'successful_cells', 'successful_by_resolution', 'paired_complete_cases')}, indent=2))


if __name__ == '__main__':
    main()
