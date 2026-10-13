"""Versioned 0.1% reserve candidate: replay unchanged, fully saved SOCP profiles."""
import argparse
import math
from pathlib import Path
import time
import json
import numpy as np
import copp_path_adapter_v4 as core
import copp_geometry_only_v4 as frozen
import copp_adapter_v5b as parent
from copp_cases_v5 import load_case
from copp_time_projection_v4 import speed_peak
from metrics_exact_index_v5 import metrics
from runtime_configuration import configure_runtime

ROOT = Path(__file__).resolve().parents[1]
RESERVE = .001


def sample(curve, stations, profile, dt, settings=None):
    if settings is not None and settings != parent.SETTINGS:
        raise ValueError('This reserve version requires the fixed v5b planner settings')
    peak = speed_peak(stations, profile.a, profile.b, profile.num_stationary)*parent.SETTINGS['spatial_scale_mm']
    dilation = (1+RESERVE)*max(1., peak/(100.*parent.SETTINGS['velocity_reserve']))
    scaled = core.copp.Profile3rd(profile.a/dilation**2, profile.b/dilation**2, profile.num_stationary)
    positions, times, progress, execution = frozen._sample(curve, stations, scaled, dt, parent.SETTINGS)
    execution.update(time_dilation=dilation, raw_interpolated_peak_speed_mm_s=peak,
        extra_time_reserve_fraction=RESERVE,
        adaptation='Exact interval-speed projection and globally fixed 0.1% extra time reserve',
        wrapper_sha256=core.sha(__file__))
    return positions, times, progress, execution


def install_for_new_planning():
    """Bind the identical final executor for fresh registered programs."""
    core.prepare_geometry = parent.prepare_geometry
    core.grid_geometry = parent.grid_geometry
    core.sample = sample
    core.metrics = metrics
    return core.run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'results/copp_v5b_complete_development')
    parser.add_argument('--cases', nargs='+', default=['validation_trapezoid', 'validation_zigzag'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.input/'summary.json').exists():
        raise RuntimeError('Wait for the complete parent matrix before reserve validation')
    if (args.output/'registration.json').exists():
        raise FileExistsError('Preserve existing reserve evidence')
    parent_records = json.loads((args.input/'summary.json').read_text('utf-8'))
    by_name = {r['path_id']: r for r in parent_records}
    names = list(by_name) if args.cases == ['all'] else args.cases
    args.output.mkdir(parents=True, exist_ok=True)
    registration = dict(version='v5c-reserve0.001', extra_time_reserve_fraction=RESERVE,
        parent_version='v5b', parent_directory=str(args.input),
        parent_registration_sha256=core.sha(args.input/'registration.json'),
        policy='One reserve and unchanged frozen geometry/grid/three SOCP profiles for every case. Fixed third iteration. Every intermediate output/failure retained.',
        scope='Development; validation-only unless all cases explicitly registered. No prospective final programs.',
        source_sha256={name: core.sha(ROOT/'src'/name) for name in ['copp_reserve_replay_v5c.py',
            'copp_adapter_v5b.py', 'copp_signed_geometry_v5b.py', 'copp_adapter_v5.py', 'copp_cases_v5.py',
            'metrics.py', 'metrics_exact_index_v5.py', 'exact_distance_index_v5.py',
            'runtime_configuration.py', 'runtime_bootstrap.py']},
        evaluator_equivalence_review_sha256=core.sha(ROOT/'audit/distance_index_independent_review_v5.json'),
        cases=[dict(id=name, parent_record_sha256=core.sha(args.input/(name+'.json'))) for name in names])
    core.write(args.output/'registration.json', registration)
    records = []
    for name in names:
        case = load_case(name)
        source = by_name[name]
        record = dict(path_id=name, registered_adapter_version='v5c-reserve0.001', iterations=[],
            tolerance_mm=case['reference_tolerance_mm'], nominal_tolerance_mm=1.25,
            limits=[100., 2000., 62500.], chord_error_mm=case['chord_error_mm'],
            parent_record=str(args.input/(name+'.json')), parent_optimizer_status=source['status'],
            parent_optimizer_clocks=source.get('planner', {}).get('clocks'))
        started = time.perf_counter()
        try:
            curve, *_ = parent.prepare_geometry(case['reference'], tolerance=case['reference_tolerance_mm'],
                source_geometry=case['source_geometry'])
            sigma, *_ = parent.grid_geometry(curve, case['source_geometry'], parent.SETTINGS)
            stations = sigma/parent.SETTINGS['spatial_scale_mm']
            record['geometry_reconstruction_wall_s'] = time.perf_counter()-started
            record['planner'] = dict(source.get('planner', {}), execution_reserve_version='v5c-reserve0.001')
            for previous in source['iterations']:
                iteration = previous['iteration']
                profile_path = args.input/f'{name}_iteration{iteration}_profile.npz'
                raw = np.load(profile_path)
                if not np.array_equal(case['reference'], raw['reference']):
                    raise ValueError('Complete source reference differs from saved immutable optimizer input')
                if not np.array_equal(stations, raw['s']):
                    raise ValueError('Reconstructed stations differ from saved immutable optimizer input')
                profile = core.copp.Profile3rd(raw['a'], raw['b'], tuple(raw['num_stationary']))
                row = dict(iteration=iteration, solver_wall_s=previous['solver_wall_s'],
                    solver_rerun=False, parent_profile=str(profile_path), parent_profile_sha256=core.sha(profile_path), runs=[])
                record['iterations'].append(row)
                for dt in [.001, .0005]:
                    clock = time.perf_counter()
                    positions, times, progress, execution = sample(curve, stations, profile, dt)
                    configure_runtime('v3', tolerance=case['reference_tolerance_mm'])
                    result = metrics(case['reference'], positions, dt=dt, tolerance=case['reference_tolerance_mm'])
                    filename = f'{name}_iteration{iteration}_dt{round(dt*1e6)}us.npz'
                    np.savez_compressed(args.output/filename, positions=positions, time=times,
                        progress=progress, reference=case['reference'], dt_s=dt)
                    row['runs'].append(dict(dt_s=dt, metrics=result, execution=execution,
                        sample_audit_save_wall_s=time.perf_counter()-clock,
                        trace=filename, trace_sha256=core.sha(args.output/filename)))
                    core.write(args.output/(name+'.json'), record)
                    print(json.dumps(dict(path=name, iteration=iteration, dt=dt, success=result['success'],
                        T=result['complete_time_s'], peaks=result['vector_peaks'], reason=result['reason'])), flush=True)
            record['status'] = 'completed' if source['status'] == 'completed' else 'parent_optimizer_failed'
            if source['status'] != 'completed':
                record['error'] = source.get('error')
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            print(json.dumps(dict(path=name, status='failed', error=str(exc))), flush=True)
        record['total_wall_s'] = time.perf_counter()-started
        core.write(args.output/(name+'.json'), record)
        records.append(record)
    core.write(args.output/'summary.json', records)


if __name__ == '__main__':
    main()
