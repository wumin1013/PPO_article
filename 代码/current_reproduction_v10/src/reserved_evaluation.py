"""Frozen, complete evaluation of the five prospectively registered programs.

Use only after a paired validation gate; freezing requires ten complete models.
The v1 experiments and their scientific inputs are never modified.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json

RESERVE = ROOT / 'prospective' / 'reserved_programs'
OUTPUT = ROOT / 'results' / 'reserved_v2'
FREEZE = ROOT / 'audit' / 'reserved_evaluation_freeze.json'
SEEDS = [101, 211, 307, 401, 503]


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def freeze(index):
    from model_io import load_model
    if FREEZE.exists() or OUTPUT.exists():
        raise FileExistsError('Prospective evaluation already frozen or started.')
    models = [dict(m, file=m['file'].replace('\\', '/')) for m in read(index)['models']]
    pairs = [(m['method'], m['seed']) for m in models]
    expected = [(arm, seed) for arm in ('adaptive', 'fixed') for seed in SEEDS]
    if len(pairs) != len(expected) or set(pairs) != set(expected):
        raise ValueError('Exactly five paired independent seeds are required.')
    if len({(ROOT / m['file']).resolve() for m in models}) != len(models):
        raise ValueError('Every model must have a distinct checkpoint path.')
    registration = read(RESERVE / 'registration.json')
    prior = read(ROOT / 'audit' / 'test_freeze.json')['sha256']
    for name, digest in prior.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Retained v1 scientific input changed: ' + name)
    files = set(prior)
    files.update(['src/reserved_evaluation.py', 'src/reserve_programs.py', 'src/model_io.py',
                  'prospective/reserved_programs/registration.json'])
    for file in (ROOT / 'src').glob('learning*.py'):
        files.add(file.relative_to(ROOT).as_posix())
    teacher_folder = ROOT / 'results' / 'development_v2' / 'learning_001'
    teacher = read(teacher_folder / 'teacher_selection.json')
    selected = [r for r in teacher['candidates'] if r['candidate'] == teacher['selected']]
    if len(selected) != 1 or not selected[0]['all_development_paths_pass']:
        raise ValueError('Shared teacher did not pass the registered selection gate.')
    dataset_hash = sha(teacher_folder / 'shared_teacher_records.npz')
    for arm in ('adaptive', 'fixed'):
        gate = read(teacher_folder / 'paired' / 'seed_101' / arm / 'validation_08192.json')
        if len(gate['records']) != 2 or not all(r['success'] for r in gate['records']):
            raise ValueError('Paired seed-101 validation gate has not passed.')
    # Freeze complete training provenance, including the common records, all
    # selected and rejected teacher candidates, budgets and validation traces.
    for folder in (teacher_folder, ROOT / 'results' / 'development_v2' / 'learning_002'):
        if not (folder / 'registration.json').exists():
            raise ValueError('Missing independent training registration: ' + str(folder))
        for file in folder.rglob('*'):
            if file.is_file() and file.suffix not in ('.log', '.md'):
                files.add(file.relative_to(ROOT).as_posix())
    for model in models:
        path = (ROOT / model['file']).resolve()
        if not path.is_relative_to(ROOT) or path.name != 'final.pt':
            raise ValueError('Model must be a registered final.pt inside the evidence tree.')
        receipt = read(path.with_name('receipt.json'))
        if not receipt['complete'] or receipt['model_sha256'] != sha(path):
            raise ValueError('Incomplete or changed model: ' + str(path))
        reg = read(path.with_name('registration.json'))
        _, saved = load_model(path)
        if any(obj['seed'] != model['seed'] or obj['arm'] != model['method'] for obj in (reg, saved)):
            raise ValueError('Checkpoint/registration identity differs from index: ' + str(path))
        if reg['dataset_sha256'] != dataset_hash or receipt['cost']['training_interactions'] != 8192:
            raise ValueError('Unequal records or interaction budget: ' + str(path))
        if not receipt['source_unchanged'] or not receipt['protected_sources_unchanged']:
            raise ValueError('Training source changed during execution: ' + str(path))
        for file in [path, path.with_name('receipt.json'), path.with_name('registration.json')]:
            files.add(file.relative_to(ROOT).as_posix())
    for row in registration['paths']:
        file = RESERVE / row['file']
        if sha(file) != row['sha256']:
            raise ValueError('Reserved program changed: ' + row['id'])
        files.add(file.relative_to(ROOT).as_posix())
    grid = []
    for row in registration['paths']:
        for dt in (.001, .0005):
            for model in models:
                if dt == .0005 and model['seed'] != 101:
                    continue
                grid.append(dict(path=row['id'], dt_s=dt, **model))
            for method in ('SCMO', 'S2'):
                grid.append(dict(path=row['id'], dt_s=dt, method=method, seed=None))
    write_json(FREEZE, dict(created_utc=utc(), models=models, grid=grid,
        sha256={name: sha(ROOT / name) for name in sorted(files)},
        constraints=registration['proposed_conditions'],
        scientific_scope='Prospective synthetic programs; all paths and all registered attempts retained.',
        result_selection='No replacement, post-test checkpoint selection or outcome-conditioned exclusion.',
        baseline_settings='SCMO overlap_fraction=0.5; original registered S2 implementation; both use common limits/tolerance.',
        validation_gate='Seed 101, both arms, final 8192 checkpoint, both validation programs. Later seed outcomes are not used to exclude completed training runs.',
        uncertainty='Per-path mean/SD conditional on completion; paired Student intervals with sample count and success fractions.'))
    print('Frozen cells:', len(grid))


def verify():
    frozen = read(FREEZE)
    for name, digest in frozen['sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return frozen


def stem(cell):
    return f"{cell['method']}_{cell['seed']}_{cell['path']}_{cell['dt_s']}"


def run(worker=0, workers=1):
    import torch
    from model_io import load_model
    from evaluate import rollout
    from metrics import metrics
    frozen = verify()
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Worker must be in [0, workers).')
    for index, cell in enumerate(frozen['grid']):
        if index % workers != worker:
            continue
        target = OUTPUT / (stem(cell) + '.npz')
        if target.with_suffix('.json').exists():
            continue
        ref = np.loadtxt(RESERVE / (cell['path'] + '.csv'), delimiter=',', skiprows=1)
        dt = cell['dt_s']
        if cell['method'] in ('adaptive', 'fixed'):
            began = time.perf_counter()
            try:
                if cell['file'] not in loaded:
                    loaded[cell['file']] = load_model(ROOT / cell['file'])
                actor, saved = loaded[cell['file']]
                result = rollout(actor, ref, target, dt=dt,
                                 fixed_d=saved['d'] if cell['method'] == 'fixed' else None)
            except Exception as exc:
                result = dict(success=False, complete_time_s=None,
                    reason='evaluation_setup_exception: ' + type(exc).__name__ + ': ' + str(exc),
                    trace_sha256=sha(target) if target.exists() else None,
                    created_utc=utc(), wall_s=time.perf_counter()-began)
        else:
            began = time.perf_counter()
            try:
                if cell['method'] == 'SCMO':
                    from baselines.motion_overlap import plan
                    planned = plan(ref, dt=dt, tolerance=1.25, limits=dict(v=100., a=2000., j=62500.))
                    positions = planned['positions']
                else:
                    from baselines.legacy_s2 import plan
                    positions, _ = plan(ref, dt=dt, tolerance=1.25, limits=(100., 2000., 62500.))
                np.savez_compressed(target, reference=ref, positions=positions, dt_s=dt)
                result = metrics(ref, positions, dt, 1.25, (100., 2000., 62500.))
                result['trace_sha256'] = sha(target)
            except Exception as exc:
                result = dict(success=False, complete_time_s=None, reason=type(exc).__name__ + ': ' + str(exc),
                              trace_sha256=sha(target) if target.exists() else None)
            result.update(wall_s=time.perf_counter()-began, created_utc=utc())
        result.update(path=cell['path'], method=cell['method'], seed=cell['seed'], dt_s=dt,
                      worker=worker, workers=workers,
                      wall_time_scope='Shared-host execution, not an isolated online-latency benchmark.')
        write_json(target.with_suffix('.json'), result)
        print(stem(cell), result['success'], result['complete_time_s'], flush=True)


def audit():
    from metrics import metrics
    from scipy import stats
    frozen = verify()
    rows, mismatches, missing = [], [], []
    for cell in frozen['grid']:
        file = OUTPUT / (stem(cell) + '.json')
        if not file.exists():
            missing.append(stem(cell)); continue
        try:
            result = read(file)
            required = ('success', 'complete_time_s', 'created_utc', 'wall_s', 'reason')
            if any(key not in result for key in required) or not isinstance(result['success'], bool):
                raise ValueError('Incomplete result schema')
            started = datetime.fromisoformat(result['created_utc']).timestamp() - result['wall_s']
        except Exception as exc:
            mismatches.append(stem(cell) + ':result_exception:' + type(exc).__name__ + ':' + str(exc))
            continue
        trace = file.with_suffix('.npz')
        for key in ('method', 'seed', 'path', 'dt_s'):
            if result.get(key) != cell[key]:
                mismatches.append(stem(cell) + ':identity:' + key)
        if not result.get('reason') or (not result['success'] and result['complete_time_s'] is not None):
            mismatches.append(stem(cell) + ':outcome_contract')
        freeze_time = datetime.fromisoformat(frozen['created_utc']).timestamp()
        if started < freeze_time:
            mismatches.append(stem(cell) + ':execution_before_freeze')
        if trace.exists():
            try:
                with np.load(trace, allow_pickle=False) as data:
                    declared = result.get('executor_reason', 'success') == 'success' and not result.get('error')
                    recovered = metrics(data['reference'], data['positions'], cell['dt_s'], 1.25,
                                        (100., 2000., 62500.), declared_success=declared,
                                        reason=result.get('executor_reason'))
                    expected_ref = np.loadtxt(RESERVE / (cell['path'] + '.csv'), delimiter=',', skiprows=1)
                    if sha(trace) != result['trace_sha256'] or not np.array_equal(data['reference'], expected_ref):
                        mismatches.append(stem(cell) + ':trace')
                    if not len(data['positions']) or not np.array_equal(data['positions'][0], expected_ref[0]):
                        mismatches.append(stem(cell) + ':initial_position')
                    if 'dt_s' in data and float(data['dt_s']) != cell['dt_s']:
                        mismatches.append(stem(cell) + ':trace_dt')
                    if 'rows' in data:
                        extra = len(data['positions']) - 1 - len(data['rows'])
                        if extra < 0 or extra != result.get('partial_decision_position_samples_retained', 0):
                            mismatches.append(stem(cell) + ':partial_rows')
                        if not np.array_equal(data['rows'][:, :2], data['positions'][1:len(data['rows'])+1]):
                            mismatches.append(stem(cell) + ':rows_positions')
                    for key, value in recovered.items():
                        if result.get(key) != value:
                            mismatches.append(stem(cell) + ':metrics:' + key)
            except Exception as exc:
                mismatches.append(stem(cell) + ':audit_exception:' + type(exc).__name__ + ':' + str(exc))
        elif result['success'] or result.get('trace_sha256') or result.get('complete_time_s') is not None or not result.get('reason'):
            mismatches.append(stem(cell) + ':missing_trace')
        rows.append(dict(path=cell['path'], method=cell['method'], seed=cell['seed'], dt_s=cell['dt_s'],
                         success=result['success'], time_s=result['complete_time_s'], reason=result.get('reason'),
                         result=file.relative_to(ROOT).as_posix()))
    summary = []
    for path in sorted({r['path'] for r in rows}):
        for method in ('adaptive', 'fixed', 'SCMO', 'S2'):
            subset = [r for r in rows if r['path'] == path and r['method'] == method and r['dt_s'] == .001]
            times = [r['time_s'] for r in subset if r['success']]
            summary.append(dict(path=path, method=method, attempts=len(subset), completed=len(times),
                                mean_s=float(np.mean(times)) if times else None,
                                sd_s=float(np.std(times, ddof=1)) if len(times)>1 else None))
    pairs = []
    for path in sorted({r['path'] for r in rows}):
        subset = {(r['method'], r['seed']): r for r in rows if r['path'] == path and r['dt_s'] == .001}
        deltas = [subset['fixed', s]['time_s']-subset['adaptive', s]['time_s'] for s in SEEDS
                  if ('fixed', s) in subset and ('adaptive', s) in subset and
                  subset['fixed', s]['success'] and subset['adaptive', s]['success']]
        mean = float(np.mean(deltas)) if deltas else None
        half = float(stats.t.ppf(.975, len(deltas)-1)*np.std(deltas, ddof=1)/np.sqrt(len(deltas))) if len(deltas)>1 else None
        pairs.append(dict(path=path, n_complete_pairs=len(deltas), fixed_minus_adaptive_s=mean,
                          t95_low_s=mean-half if half is not None else None,
                          t95_high_s=mean+half if half is not None else None))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'summary.json', dict(created_utc=utc(), expected=len(frozen['grid']), recorded=len(rows),
        valid_for_manuscript=not missing and not mismatches and len(rows)==len(frozen['grid']),
        missing=missing, mismatches=mismatches, results=rows, per_path=summary, paired=pairs,
        interval_scope='Descriptive paired Student intervals conditional on both arms completing; at most five seeds, approximate normality assumption.'))
    if rows:
        with (OUTPUT / 'all_runs.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Recorded:', len(rows), 'missing:', len(missing), 'mismatches:', len(mismatches))
    if missing or mismatches:
        raise RuntimeError('Evaluation grid is incomplete or inconsistent.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'run', 'audit'])
    parser.add_argument('--models', type=Path)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path, help='New result directory inside this evidence tree for an independent rerun.')
    args = parser.parse_args()
    if args.output is not None:
        OUTPUT = args.output.resolve()
        if not OUTPUT.is_relative_to(ROOT):
            parser.error('--output must stay inside the evidence tree for portable relative result references')
    if args.command == 'freeze':
        if args.models is None:
            parser.error('--models is required for freeze')
        freeze(args.models)
    elif args.command == 'run':
        run(args.worker, args.workers)
    else:
        audit()
