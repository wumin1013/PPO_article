"""Re-audit four saved fixed-preview command traces; no rollout or training."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha, utc, write_json
from metrics_exact_index_v5 import metrics
from runtime_configuration import configure_runtime


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    package = ROOT / 'baseline_records/supplementary_reproduction'
    model = package / 'models/two.pt'
    expected_model = '40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469'
    assert sha(model) == expected_model
    sources = {str(model): sha(model)}
    started, cpu_started = time.perf_counter(), time.process_time()
    results = []
    for case in ('original', 'dense'):
        reference_file = package / 'cases' / case / 'reference.csv'
        geometry_file = reference_file.with_name('geometry.json')
        reference = np.loadtxt(reference_file, delimiter=',', skiprows=1)
        geometry = read(geometry_file)
        chord = max([p.get('chord_error_bound_mm', 0.) for p in geometry['primitives']] + [0.])
        tolerance = 1.25 - chord
        configure_runtime('v3', tolerance=tolerance)
        for dt, suffix in ((.001, '1ms'), (.0005, 'half_ms')):
            trace_file = package / 'records/Two-action' / f'{case}_{suffix}.npz'
            record_file = trace_file.with_suffix('.json')
            record = read(record_file)
            for file in (reference_file, geometry_file, trace_file, record_file):
                sources[str(file)] = sha(file)
            assert sha(trace_file) == record['trace_sha256']
            assert record['checkpoint_sha256'] == expected_model
            with np.load(trace_file) as data:
                assert np.array_equal(data['reference'], reference)
                assert float(data['dt_s']) == dt
                rows = data['rows']
                positions = np.vstack([reference[0], rows[:, :2]])
                observations, actions, latents = data['observations'], data['actions'], data['latents']
                assert observations.shape[1] == 14 and latents.shape[1] == 2
                assert actions.shape == (len(latents), 3) and np.all(actions[:, 2] == 3.)
            check_started, check_cpu = time.perf_counter(), time.process_time()
            audit = metrics(reference, positions, dt=dt, tolerance=tolerance,
                            declared_success=record['success'], reason=record['reason'])
            results.append(dict(case=case, dt_s=dt,
                reference=str(reference_file), reference_sha256=sha(reference_file),
                geometry_source=str(geometry_file), geometry_sha256=sha(geometry_file),
                trace=str(trace_file), trace_sha256=sha(trace_file),
                original_result=str(record_file), original_result_sha256=sha(record_file),
                original_result_success=record['success'],
                nominal_tolerance_mm=1.25, arc_chord_error_reserve_mm=chord,
                effective_polyline_tolerance_mm=tolerance,
                observation_dimension=14, policy_output_dimension=2, held_interface_dimension=3,
                all_held_preview_values_mm=3., decision_count=len(actions),
                stored_command_intervals=len(positions)-1,
                start_position_exactly_reference_start=bool(np.array_equal(positions[0], reference[0])),
                initial_velocity_mm_s=[0., 0.], initial_acceleration_mm_s2=[0., 0.],
                original_analytic_interval_audit=record['analytic'], metrics=audit,
                re_audit_wall_s=time.perf_counter()-check_started,
                re_audit_process_cpu_s=time.process_time()-check_cpu))
            print('READONLY_FIXED', case, dt, audit['success'], audit['complete_time_s'], flush=True)
    from runtime_bootstrap import BASE
    for relative in ('src/metrics_exact_index_v5.py', 'src/exact_distance_index_v5.py',
                     'src/corridor_v3.py', 'src/runtime_configuration.py',
                     'src/runtime_bootstrap.py', 'runtime/source_manifest.json'):
        sources[str(ROOT / relative)] = sha(ROOT / relative)
    sources[str(BASE / 'geometry_spec.py')] = sha(BASE / 'geometry_spec.py')
    assert all(sha(p) == h for p, h in sources.items())
    output = dict(schema=1, created_utc=utc(),
        scope='Read-only acceptance audit of every stored command interval in four frozen fixed-preview development traces; no policy invocation, no new trajectory, no training.',
        model=dict(path=str(model), sha256=expected_model,
                   architecture='14-64-64-2 tanh base actor plus 192 compact-support Wendland features',
                   fixed_preview_mm=3., role='independently developed neural comparator, not a budget-paired ablation'),
        executor_identity='historical DirectTwoRuntime/LocalizedTwoPolicy; original stored executions, not the v4b main-controller executor',
        acceptance='Current source-witness ordered gates and exact command-capsule union; nominal 1.25 mm less source arc chord-error bound; vector backward differences from rest; at least 20 ms final rest.',
        new_physical_rollouts=0, source_files_unchanged=True, sources_sha256=sources,
        all_four_success=all(r['metrics']['success'] for r in results), runs=results,
        total_command_intervals=sum(r['stored_command_intervals'] for r in results),
        wall_s=time.perf_counter()-started, process_cpu_s=time.process_time()-cpu_started)
    destination = ROOT / 'audit/fixed_comparator_complete_trace_audit_v5.json'
    write_json(destination, output)
    print('WROTE', destination, 'PASS', output['all_four_success'], flush=True)


if __name__ == '__main__':
    main()
