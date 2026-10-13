"""Single source of identity and complete-trace metrics for the restored main model."""
from pathlib import Path
import argparse
import copy
import json
import shutil
import numpy as np
from common import ROOT, sha, utc, write_json

MANIFEST = ROOT / 'evidence/main_model_v5.json'
MODEL_SHA = 'c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
MODEL_ID = 'jnnc_wendland96_192_c16d45e3'
EXECUTOR_ID = 'v4b_no_target_override'
METHODS = ('JNN', 'S2', 'S2R', 'SCMO')
FIGURE_DEPENDENCIES = {'local_frame': 'manuscript/generate_development_v3.py'}
EXPECTED_BINDINGS = ('figure8', 'main_table', 'main_text', 'abstract_numbers', 'highlights_numbers')
EXPECTED_RUNS = {f'{case}_{method}_dt1000us': (case, method)
                 for case in ('original', 'dense') for method in METHODS}
V4 = Path('D:/PPO_article_work/JNN_efficiency_20261011_v4')
V3 = Path('D:/PPO_article_work/JNN_revision_20261011_v3')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def import_bound(source, destination, sources):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        check(sha(source) == sha(destination), 'Existing import differs: '+str(destination))
    else:
        shutil.copy2(source, destination)
    relative = destination.relative_to(ROOT).as_posix()
    sources[relative] = dict(sha256=sha(destination), imported_from=str(source))
    return relative


def register():
    if MANIFEST.exists():
        return verify()
    sources, runs = {}, {}
    destination = ROOT / 'evidence/main_model_inputs'
    model = ROOT / 'baseline_records/supplementary_reproduction/models/main.pt'
    check(sha(model) == MODEL_SHA, 'Wrong main checkpoint')
    model_path = model.relative_to(ROOT).as_posix()
    sources[model_path] = dict(sha256=MODEL_SHA)
    for relative in FIGURE_DEPENDENCIES.values():
        sources[relative] = dict(sha256=sha(ROOT / relative))
    registration = read(V4 / 'results/v4b_no_target_override/registration.json')
    registration_path = import_bound(V4 / 'results/v4b_no_target_override/registration.json',
                                    destination / 'execution_registration.json', sources)
    for name, expected in registration['source_sha256'].items():
        local = ROOT / 'src' / name
        check(sha(local) == expected, 'Execution source differs: '+name)
        sources[local.relative_to(ROOT).as_posix()] = dict(sha256=expected)
    runtime_manifest = ROOT / 'runtime/source_manifest.json'
    check(sha(runtime_manifest) == registration['runtime_manifest_sha256'], 'Runtime snapshot differs')
    sources['runtime/source_manifest.json'] = dict(sha256=sha(runtime_manifest))
    import_bound(V4 / 'audit/original_strong_recipe_inventory.json',
                 destination / 'training_recipe_inventory.json', sources)
    for case in ('original', 'dense'):
        case_root = ROOT / 'baseline_records/supplementary_reproduction/cases' / case
        for filename in ('reference.csv', 'geometry.json'):
            path = case_root / filename
            sources[path.relative_to(ROOT).as_posix()] = dict(sha256=sha(path))
        for method in METHODS:
            source = (V4 / f'results/v4b_no_target_override/strong/{case}.json'
                      if method == 'JNN' else
                      V3 / f'results/baseline_v3/development_unified/{method}_{case}_dt1000us_scale1000.json')
            record_path = import_bound(source, destination / f'{case}_{method}.json', sources)
            trace_path = import_bound(source.with_suffix('.npz'), destination / f'{case}_{method}.npz', sources)
            record = read(source)
            run_id = f'{case}_{method}_dt1000us'
            runs[run_id] = dict(case=case, method=method, dt_s=.001,
                reference=(case_root / 'reference.csv').relative_to(ROOT).as_posix(),
                geometry=(case_root / 'geometry.json').relative_to(ROOT).as_posix(),
                record=record_path, trace=trace_path,
                model_id=MODEL_ID if method == 'JNN' else None,
                executor_id=EXECUTOR_ID if method == 'JNN' else 'baseline_development_unified_v3',
                tolerance_mm=record['tolerance_mm'], status=record['success'],
                statistical_unit='one complete development program with one fixed checkpoint or deterministic planner')
    manifest = dict(schema=1, created_utc=utc(), scope='historical main-model development programs',
        main_model_id=MODEL_ID, models={MODEL_ID:dict(checkpoint=model_path, sha256=MODEL_SHA,
            architecture='14-64-64-3 tanh; 96 preview and 192 joint Wendland features',
            feature_family='compact-support Wendland', training_seed=923202,
            training_recipe_inventory='evidence/main_model_inputs/training_recipe_inventory.json',
            training_identity='historical optimized development controller; no newly retrained seed is substituted')},
        executors={EXECUTOR_ID:dict(registration=registration_path,
            target_override=False, source_witness_initialization=True, local_disjoint_blending=True)},
        sources=sources, runs=runs, figure_dependencies=FIGURE_DEPENDENCIES,
        bindings={name:dict(main_model_id=MODEL_ID, executor_id=EXECUTOR_ID, runs=list(runs))
                  for name in ('figure8', 'main_table', 'main_text', 'abstract_numbers', 'highlights_numbers')},
        publication_ready=False,
        pending='Unified independent training and newly isolated final study remain separate unfinished work.')
    write_json(MANIFEST, manifest)
    return verify()


def verify(manifest=None, verify_files=True):
    manifest = read(MANIFEST) if manifest is None else manifest
    main_id = manifest['main_model_id']
    check(main_id == MODEL_ID, 'Main model identity changed')
    check(set(manifest['bindings']) == set(EXPECTED_BINDINGS), 'Required figure/table/text binding set differs')
    check(set(manifest['runs']) == set(EXPECTED_RUNS), 'Registered complete run set differs')
    check(manifest.get('figure_dependencies') == FIGURE_DEPENDENCIES,
          'Required figure dependency set differs')
    for relative in FIGURE_DEPENDENCIES.values():
        check(relative in manifest['sources'], 'Figure helper is not a bound source: '+relative)
    if verify_files:
        for relative, source in manifest['sources'].items():
            check(sha(ROOT / relative) == source['sha256'], 'Evidence source changed: '+relative)
        for item in read(ROOT / 'runtime/source_manifest.json'):
            check(sha(ROOT / 'runtime' / item['relative']) == item['sha256'],
                  'Runtime snapshot file changed: '+item['relative'])
    model = manifest['models'][main_id]
    check(model['sha256'] == MODEL_SHA, 'Wrong model hash')
    check(sha(ROOT / model['checkpoint']) == MODEL_SHA, 'Checkpoint path points to a different model')
    check(model['checkpoint'] in manifest['sources'], 'Checkpoint is not a bound source')
    check(model['feature_family'] == 'compact-support Wendland', 'Wrong feature-family description')
    check(model['architecture'] == '14-64-64-3 tanh; 96 preview and 192 joint Wendland features',
          'Wrong main-model architecture description')
    for name, binding in manifest['bindings'].items():
        check(binding['main_model_id'] == main_id, 'Model mismatch in '+name)
        check(binding['executor_id'] == EXECUTOR_ID, 'Executor mismatch in '+name)
        check(binding['runs'] == list(EXPECTED_RUNS), 'Run selection mismatch in '+name)
    output = {}
    for run_id, run in manifest['runs'].items():
        expected_case, expected_method = EXPECTED_RUNS[run_id]
        check(run['case'] == expected_case and run['method'] == expected_method,
              'Run ID, program and method do not agree: '+run_id)
        for key in ('reference', 'geometry', 'record', 'trace'):
            check(run[key] in manifest['sources'], 'Run input is not a bound source: '+run[key])
        record = read(ROOT / run['record'])
        check(record['method'] == ('strong' if expected_method == 'JNN' else expected_method),
              'Record belongs to another method: '+run_id)
        check(record.get('path', record.get('path_id')) == expected_case,
              'Record belongs to another program: '+run_id)
        if run['method'] == 'JNN':
            check(run['model_id'] == main_id and run['executor_id'] == EXECUTOR_ID, 'Main run identity mismatch')
            check(record['model_sha256'] == MODEL_SHA, 'Trace was produced by a different model')
            check(record['runtime_version'] == EXECUTOR_ID, 'Trace used a different executor')
            check(record['registration_sha256'] == sha(ROOT / manifest['executors'][EXECUTOR_ID]['registration']),
                  'Execution registration mismatch')
            check(record['trace_sha256'] == sha(ROOT / run['trace']), 'Recorded trace hash mismatch')
            check(record['reference_sha256'] == sha(ROOT / run['reference']), 'Recorded path mismatch')
        check(record['dt_s'] == run['dt_s'] == .001, 'Resolution mismatch')
        check(record['tolerance_mm'] == run['tolerance_mm'], 'Tolerance mismatch')
        check(record['limits'] == [100., 2000., 62500.], 'Vector limit mismatch')
        check(record['success'] == run['status'], 'Completion status mismatch')
        with np.load(ROOT / run['trace']) as z:
            p, reference = z['positions'], z['reference']
        check(np.array_equal(reference, np.loadtxt(ROOT / run['reference'], delimiter=',', skiprows=1)),
              'Reference array mismatch')
        check(len(p) == record['samples'] and np.isfinite(p).all(), 'Invalid command samples')
        velocity = np.diff(p, axis=0) / run['dt_s']
        acceleration = np.diff(np.vstack([np.zeros(2), velocity]), axis=0) / run['dt_s']
        jerk = np.diff(np.vstack([np.zeros(2), acceleration]), axis=0) / run['dt_s']
        speed = np.linalg.norm(velocity, axis=1)
        peaks = np.array([np.linalg.norm(x, axis=1).max() for x in (velocity, acceleration, jerk)])
        check(np.allclose(peaks, record['vector_peaks'], rtol=1e-10, atol=1e-7), 'Peak metric mismatch')
        elapsed = (len(p)-1) * run['dt_s']
        length = float(speed.sum() * run['dt_s'])
        deficit = float((100.-speed).sum() * run['dt_s'])
        check(abs(elapsed-record['elapsed_prefix_s']) < 1e-9, 'Elapsed metric mismatch')
        check(abs(length-record['executed_length_mm']) < 1e-7, 'Length metric mismatch')
        check(abs((length+deficit)/100.-elapsed) < 1e-8, 'Time accounting failed')
        if record['success']:
            check(record['kinematic_pass'] and record['rest_pass'] and record['geometry']['geometry_complete'],
                  'Incomplete acceptance')
            check(record['geometry']['gates_passed'] == record['geometry']['gate_count'], 'Skipped gate')
            check(abs(elapsed-record['complete_time_s']) < 1e-9, 'Completed duration mismatch')
        else:
            check(record['complete_time_s'] is None, 'Failed prefix has a completed duration')
        output[run_id] = dict(**run, complete_time_s=record['complete_time_s'],
            executed_length_mm=length, speed_deficit_mm=deficit,
            mean_speed_mm_s=length/elapsed, vector_peaks=peaks.tolist(),
            max_deviation_mm=record['max_polyline_deviation_mm'],
            geometry_metrics=record['geometry'], terminal_rest_s=record['terminal_rest_s'])
    for case in ('original', 'dense'):
        group = [r for r in output.values() if r['case'] == case]
        check(len({r['tolerance_mm'] for r in group}) == 1, 'Methods use different tolerance')
    return manifest, output


def negative_checks():
    manifest = read(MANIFEST)
    cases = []
    mutations = [('figure_model', lambda m: m['bindings']['figure8'].update(main_model_id='v3_seed101')),
                 ('table_executor', lambda m: m['bindings']['main_table'].update(executor_id='v3')),
                 ('mixed_selection', lambda m: m['bindings']['highlights_numbers']['runs'].pop()),
                 ('run_model', lambda m: m['runs']['dense_JNN_dt1000us'].update(model_id='v3_seed101'))]
    for name, mutation in mutations:
        changed = copy.deepcopy(manifest)
        mutation(changed)
        try:
            verify(changed, verify_files=False)
        except ValueError as exc:
            cases.append(dict(case=name, rejected=True, reason=str(exc)))
        else:
            raise AssertionError('Invalid binding was accepted: '+name)
    write_json(ROOT / 'audit/main_evidence_negative_checks_v5.json', dict(checks=cases, all_passed=True))
    return cases


def verify_policy_actions():
    import torch
    from frozen_policy import FrozenPolicy
    torch.set_num_threads(1)
    manifest, _ = verify()
    actor = FrozenPolicy(ROOT / manifest['models'][MODEL_ID]['checkpoint'])
    results = []
    for run_id, run in manifest['runs'].items():
        if run['method'] != 'JNN':
            continue
        with np.load(ROOT / run['trace']) as z:
            observations, actions = z['observations'], z['actions']
        predicted = np.vstack([actor.mean(x).numpy() for x in np.array_split(observations,
                                    max(1, int(np.ceil(len(observations)/256))))])
        error = float(np.abs(predicted-actions).max())
        check(error < 2e-6, 'Stored actions do not match the bound main model: '+run_id)
        results.append(dict(run_id=run_id, decisions=len(actions), max_raw_action_difference=error,
                            trace_sha256=sha(ROOT / run['trace'])))
    result = dict(model_sha256=MODEL_SHA,loader_sha256=sha(ROOT / 'src/frozen_policy.py'),
                  note='Batched float32 evaluation compared against scalar execution; tolerance 2e-6.',
                  all_passed=True,runs=results)
    write_json(ROOT / 'audit/main_policy_action_identity_v5.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register', 'verify', 'test', 'actions'))
    args = parser.parse_args()
    if args.mode == 'actions':
        print(json.dumps(verify_policy_actions(), indent=2))
    elif args.mode == 'test':
        print(json.dumps(negative_checks(), indent=2))
    else:
        manifest, records = register() if args.mode == 'register' else verify()
        print(json.dumps(dict(main_model=manifest['main_model_id'], executor=EXECUTOR_ID,
                             runs=len(records), completed=sum(r['status'] for r in records.values())), indent=2))
