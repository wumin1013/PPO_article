"""Frozen traditional comparators on two explicitly selected v6 programs.

All inputs and settings are registered before execution. Existing evidence is
never overwritten; COPP keeps every SOCP iterate and projection candidate.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from common import ROOT, sha, utc, write_json

OUT = ROOT / 'results/new_representative_v7/traditional'
REG = OUT / 'registration.json'
SOURCE = ROOT / 'evidence/frozen_main_generalization_v6.json'
GLOBAL = ROOT / 'evidence/new_representative_v7_registration.json'
CASES = ('piecewise_curvature_coupon', 'asymmetric_neck_contour')
METHODS = ('S2', 'S2R', 'SCMO', 'COPP')
PERIODS = (.001, .0005)
LIMITS = (100., 2000., 62500.)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_id(case, method, dt):
    return f'{case}_{method}_dt{round(dt * 1e6)}us'


def register():
    if REG.exists():
        return verify()
    source = read(SOURCE)
    cases = [next(x for x in source['cases'] if x['id'] == case) for case in CASES]
    dependencies = ['compare_traditional_representative_v7.py', 'baselines/legacy_s2.py',
        'baselines/ruckig_s2.py', 'baselines/motion_overlap.py',
        'copp_fresh_planning_v5d_entry2.py', 'copp_fresh_planning_v5d_entry1.py',
        'copp_profile_projection_v5d_entry1.py', 'copp_command_projection_v5d.py',
        'copp_adapter_v5b.py', 'copp_signed_geometry_v5b.py', 'copp_cases_v5.py',
        'copp_reserve_replay_v5c.py', 'metrics_exact_index_v5.py',
        'exact_distance_index_v5.py', 'runtime_bootstrap.py', 'runtime_configuration.py']
    registration = dict(created_utc=utc(), study_id='traditional_representative_v7',
        scope='Detailed comparison on two synthetic programs selected after the existing five-program transfer results; not a newly sealed independent test.',
        source_registration=str(SOURCE.relative_to(ROOT)), source_registration_sha256=sha(SOURCE),
        global_registration=str(GLOBAL.relative_to(ROOT)), global_registration_sha256=sha(GLOBAL),
        cases=cases, methods=list(METHODS), periods_s=list(PERIODS), limits=list(LIMITS),
        fixed_s2_corner_scale=1., tuning='No new tuning; carry forward existing development-validation selection.',
        copp='Official0.2.3 TOPP3-SOCP; frozen v5d fresh entry v2; three iterations, fixed third; unchanged shared two-resolution projection rule.',
        physical_time_budget='5 + complete reference length / 5 seconds, copied from the existing frozen-policy registration.',
        acceptance='Zero initial velocity/acceleration, original ordered gates, complete command-interval capsule union, vector finite differences, same effective tolerance and at least20ms terminal rest.',
        retention='Every attempt, failure and full available prefix retained; failure completion time is null; no version or iterate chosen from observed time.',
        source_sha256={name:sha(ROOT / 'src' / name) for name in dependencies},
        runtime_manifest_sha256=sha(ROOT / 'runtime/source_manifest.json'),
        grid=[dict(run_id=run_id(case, method, dt), case=case, method=method, dt_s=dt)
              for case in CASES for method in METHODS for dt in PERIODS])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(REG, registration)
    return verify()


def verify():
    r = read(REG)
    if sha(SOURCE) != r['source_registration_sha256']:
        raise ValueError('Source study registration changed')
    if sha(GLOBAL) != r['global_registration_sha256']:
        raise ValueError('Global comparison registration changed')
    for item in r['cases']:
        for key in ('reference', 'geometry'):
            if sha(ROOT / item[key]) != item[key + '_sha256']:
                raise ValueError('Registered case changed: ' + item['id'])
    for name, expected in r['source_sha256'].items():
        if sha(ROOT / 'src' / name) != expected:
            raise ValueError('Registered source changed: ' + name)
    if sha(ROOT / 'runtime/source_manifest.json') != r['runtime_manifest_sha256']:
        raise ValueError('Runtime manifest changed')
    return r


def assess(reference, positions, case, dt, output):
    from metrics_exact_index_v5 import metrics
    budget = case['physical_time_budget_s']
    ticks = int(np.floor(budget / dt + 1e-9))
    exceeded = len(positions) - 1 > ticks
    if exceeded:
        np.savez_compressed(output.with_name(output.stem + '_full_planned.npz'),
                            positions=positions, reference=reference, dt_s=dt)
        positions = positions[:ticks + 1]
    np.savez_compressed(output, positions=positions, reference=reference,
                        time=np.arange(len(positions)) * dt, dt_s=dt)
    result = metrics(reference, positions, dt=dt, tolerance=case['evaluation_tolerance_mm'],
                     limits=LIMITS, declared_success=not exceeded,
                     reason='execution_budget_exhausted' if exceeded else None)
    return result


def save_cell(case, method, dt, result, trace, started, cpu, planner=None):
    name = run_id(case['id'], method, dt)
    record = dict(result, run_id=name, case=case['id'], method=method,
        dt_s=dt, trace=str(trace.relative_to(ROOT)) if trace.exists() else None,
        trace_sha256=sha(trace) if trace.exists() else None, planner=planner,
        wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu,
        reference_sha256=case['reference_sha256'], geometry_sha256=case['geometry_sha256'],
        nominal_tolerance_mm=case['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=case['arc_chord_error_bound_mm'],
        evaluation_tolerance_mm=case['evaluation_tolerance_mm'],
        physical_time_budget_s=case['physical_time_budget_s'], registration_sha256=sha(REG),
        origin='synthetic', actual_CAM=False, no_new_parameter_selection=True)
    write_json(OUT / (name + '.json'), record)
    print(json.dumps({key:record.get(key) for key in
        ('run_id', 'success', 'complete_time_s', 'reason', 'wall_s', 'error')}), flush=True)
    return record


def run(case_id, method, dt=None):
    registration = verify()
    case = next(x for x in registration['cases'] if x['id'] == case_id)
    periods = PERIODS if method == 'COPP' else (dt,)
    if any((OUT / (run_id(case_id, method, period) + '.json')).exists() for period in periods):
        raise FileExistsError('Never overwrite an attempted formal cell')
    started, cpu = time.perf_counter(), time.process_time()
    reference = np.loadtxt(ROOT / case['reference'], delimiter=',', skiprows=1)
    geometry = read(ROOT / case['geometry'])
    from runtime_bootstrap import setup
    setup()
    from runtime_configuration import configure_runtime
    configure_runtime('v3', tolerance=case['evaluation_tolerance_mm'])
    if method == 'COPP':
        from copp_fresh_planning_v5d_entry2 import run_case
        folder = OUT / 'copp_pipeline' / case_id
        result = run_case(case_id, reference, geometry, case['evaluation_tolerance_mm'], folder,
            input_provenance={key:case[key] for key in ('reference', 'reference_sha256', 'geometry', 'geometry_sha256')},
            nominal_tolerance=case['nominal_tolerance_mm'], chord_error=case['arc_chord_error_bound_mm'],
            scope=registration['scope'])
        third = next((x for x in result.get('iterations', []) if x['iteration'] == 3), None)
        for period in PERIODS:
            target = OUT / (run_id(case_id, method, period) + '.npz')
            source_run = next((x for x in (third or {}).get('runs', []) if x['dt_s'] == period), None)
            if source_run is None:
                metric = dict(success=False, complete_time_s=None, reason='planner_exception',
                              error=result.get('error', result.get('status')))
            else:
                with np.load(source_run['trace']) as data:
                    metric = assess(reference, data['positions'], case, period, target)
            save_cell(case, method, period, metric, target, started, cpu,
                dict(version='v5d-fresh-entry-v2', fixed_iteration=3,
                     full_record=str((folder / (case_id + '.json')).relative_to(ROOT)),
                     status=result.get('status')))
        return
    target = OUT / (run_id(case_id, method, dt) + '.npz')
    planner = None
    try:
        if method in ('S2', 'S2R'):
            sys.path.insert(0, str(ROOT / 'literature/ruckig_vendor'))
            if method == 'S2':
                from baselines.legacy_s2 import plan
            else:
                from baselines.ruckig_s2 import plan
            positions, planner = plan(reference, dt=dt, tolerance=case['evaluation_tolerance_mm'],
                limits=LIMITS, corner_scale=1., source_geometry=geometry)
        else:
            from baselines.motion_overlap import plan
            trajectory = plan(reference, dt=dt, tolerance=case['evaluation_tolerance_mm'],
                              limits=dict(zip(('v', 'a', 'j'), LIMITS)))
            positions, planner = trajectory['positions'], trajectory['metrics']
        metric = assess(reference, positions, case, dt, target)
    except Exception as exc:
        metric = dict(success=False, complete_time_s=None, reason='planner_or_audit_exception',
                      error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    save_cell(case, method, dt, metric, target, started, cpu, planner)


def summarize():
    registration = verify()
    records = []
    for cell in registration['grid']:
        path = OUT / (cell['run_id'] + '.json')
        if path.exists():
            row = read(path)
            if row['registration_sha256'] != sha(REG):
                raise ValueError('Attempt registration differs')
            records.append({key:row.get(key) for key in ('run_id', 'case', 'method', 'dt_s',
                'success', 'complete_time_s', 'reason', 'error', 'wall_s', 'cpu_s', 'trace',
                'trace_sha256', 'vector_limit_ratios', 'max_polyline_deviation_mm',
                'executed_length_mm', 'geometry', 'terminal_rest_s')})
    result = dict(created_utc=utc(), registration_sha256=sha(REG), expected_cells=16,
                  attempted_cells=len(records), successful_cells=sum(x['success'] for x in records),
                  all_registered_cells_present=len(records)==16, records=records)
    write_json(OUT / 'summary.json', result)
    return result


def run_all():
    registration = verify()
    for case in CASES:
        for method in METHODS:
            for dt in ((None,) if method == 'COPP' else PERIODS):
                names = [run_id(case, method, x) for x in (PERIODS if dt is None else (dt,))]
                if all((OUT / (name + '.json')).exists() for name in names):
                    continue
                log = OUT / (case + '_' + method + ('_both' if dt is None else f'_dt{round(dt*1e6)}us') + '.log')
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--case', case, '--method', method]
                if dt is not None:
                    command.extend(['--dt', str(dt)])
                with log.open('x', encoding='utf-8') as stream:
                    child = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                if child.returncode:
                    raise RuntimeError('Preserved worker failure: ' + str(log))
                summary = summarize()
                print(json.dumps(dict(attempted=summary['attempted_cells'], successful=summary['successful_cells'])), flush=True)
    summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--dt', type=float, choices=PERIODS)
    args = parser.parse_args()
    if args.register:
        registered = register()
        print(json.dumps(dict(registration=str(REG), sha256=sha(REG), cells=len(registered['grid']))))
    elif args.all:
        run_all()
    elif args.summary:
        print(json.dumps(summarize(), indent=2))
    elif args.case and args.method and (args.dt is not None or args.method == 'COPP'):
        run(args.case, args.method, args.dt)
        summarize()
    else:
        parser.error('Choose register, all, summary, or case/method/dt')
