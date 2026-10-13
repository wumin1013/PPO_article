"""Frozen learned comparators on two retrospectively selected complete programs.

No original runner, checkpoint, registered path or v6 result is modified.
NNC build and command loop are reused verbatim except for the open-path flag,
output location and result provenance. Fixed preview uses the frozen two-output
network with the same v5 execution layer as the frozen c16 transfer executions.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import csv
import inspect
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json

OUT = ROOT / 'results/new_representative_v7/learned_comparators'
REG = OUT / 'registration.json'
CASES = ('piecewise_curvature_coupon', 'asymmetric_neck_contour')
FIXED_SHA = '40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469'
NNC_SHA = '24d263a0e07fe471e7995ebae2d0a0ee7a2ee03e2feec80ea50aa5b94b0cfe6b'
SOURCE_REG = ROOT / 'evidence/frozen_main_generalization_v6.json'
SOURCE_REG_SHA = '9795cebd1dcc5186c558e66ed63f4f168c46a1f55a291551b424547143f23b2e'
GLOBAL_REG = ROOT / 'evidence/new_representative_v7_registration.json'
GLOBAL_REG_SHA = '96ca2d54ae1fbc618651923b1977c879aa627e8d4ce5692eec192958c09773c4'
NNC_ROOT = OUT / 'nnc_adapter'


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def copy_bound(source, target, copied):
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    copied.append(dict(source=source.relative_to(ROOT).as_posix(),
                       local=target.relative_to(OUT).as_posix(), sha256=sha(target)))


def register():
    assert not OUT.exists(), 'Never overwrite a prior study or an attempt.'
    from evaluate_frozen_main_generalization_v6 import verify as original_verify
    original = original_verify()
    assert sha(SOURCE_REG) == SOURCE_REG_SHA
    assert sha(GLOBAL_REG) == GLOBAL_REG_SHA
    import evaluate_restored_nnc_v6 as nnc
    previous_nnc = nnc.verify()
    files = []
    copy_bound(ROOT / 'evidence/frozen_main_v6/fixed_two.pt', OUT / 'fixed_two.pt', files)
    assert sha(OUT / 'fixed_two.pt') == FIXED_SHA
    for row in previous_nnc['source_files']:
        if row['local'].startswith('programs/'):
            continue
        copy_bound(nnc.OUT / row['local'], NNC_ROOT / row['local'], files)
    cases = [next(c for c in original['cases'] if c['id'] == name) for name in CASES]
    runs = []
    bound_main = []
    for case in cases:
        for key, filename in (('reference', 'reference.csv'), ('geometry', 'geometry.json')):
            copy_bound(ROOT / case[key], NNC_ROOT / 'programs' / case['id'] / filename, files)
        for dt in (.001, .0005):
            for method in ('FIXED', 'NNC'):
                run_id = f"{case['id']}_{method}_dt{round(dt*1e6)}us"
                folder = NNC_ROOT if method == 'NNC' else OUT / 'fixed'
                runs.append(dict(id=run_id, case=case['id'], method=method, dt_s=dt,
                    tolerance_mm=case['evaluation_tolerance_mm'],
                    nominal_tolerance_mm=case['nominal_tolerance_mm'],
                    chord_reserve_mm=case['arc_chord_error_bound_mm'],
                    physical_time_budget_s=case['physical_time_budget_s'], wall_budget_s=300.,
                    trace=(folder / (run_id+'.npz')).relative_to(ROOT).as_posix(),
                    record=(folder / (run_id+'.json')).relative_to(ROOT).as_posix()))
            cell = next(c for c in original['grid'] if c['case'] == case['id'] and c['dt_s'] == dt)
            record = read(ROOT / cell['record'])
            assert record['success'] and record['registration_sha256'] == SOURCE_REG_SHA
            assert sha(ROOT / cell['trace']) == record['trace_sha256']
            bound_main.append(dict(case=case['id'], method='JNN', dt_s=dt,
                trace=cell['trace'], record=cell['record'], trace_sha256=sha(ROOT / cell['trace']),
                record_sha256=sha(ROOT / cell['record']), model_sha256=original['model_sha256']))
    source_files = {name:sha(ROOT/'src'/name) for name in (
        'compare_learned_representative_v7.py', 'evaluate_restored_nnc_v6.py',
        'frozen_policy.py', 'metrics_exact_index_v5.py', 'exact_distance_index_v5.py')}
    source_files.update(original['source_sha256'])
    reg = dict(created_utc=utc(), study='new_representative_v7_learned_comparators',
        scope='Two complete synthetic programs retrospectively selected for detailed comparison after the complete five-program frozen-controller study; no new generalization claim.',
        source_registration=SOURCE_REG.relative_to(ROOT).as_posix(), source_registration_sha256=SOURCE_REG_SHA,
        global_registration=GLOBAL_REG.relative_to(ROOT).as_posix(), global_registration_sha256=GLOBAL_REG_SHA,
        selected_cases=cases, runs=runs, bound_main_runs=bound_main,
        main_model_sha256=original['model_sha256'], fixed_model_sha256=FIXED_SHA,
        nnc_model_sha256=NNC_SHA, new_training_updates=0, tuning_trials=0,
        fixed_preview_mm=3., fixed_policy_period_s=.02,
        fixed_executor='Unchanged runtime_physical_time_v5.install plus DirectTwoRuntime, no target override, shared local blending, common verified full terminal brake.',
        fixed_action='Original two-output deterministic mean, unchanged clip and physical decoder, configured constant 3 mm preview.',
        nnc_base_registration='results/restored_nnc_protected_v6/registration.json',
        nnc_base_registration_sha256=sha(nnc.OUT/'registration.json'),
        nnc_build_source_sha256=sha(ROOT/'src/evaluate_restored_nnc_v6.py'),
        nnc_worker_adaptations=['Corridor closed flag inferred from exact reference endpoints instead of True; required for the open curvature coupon.',
            'Runtime provenance label v7 and status uses independently audited success; actions, protections, initial state, sampling and first-infeasible-command stop are unchanged.'],
        copied_files=files, source_sha256=source_files,
        shared_contract=dict(vector_limits=[100.,2000.,62500.], initial_rest=True,
            minimum_final_rest_s=.02, geometric_tolerance='1.25 mm minus registered analytic chord reserve',
            geometry='Complete command interval, full directed ordered traversal and endpoint; no closure segment is invented for an open program.',
            vector_acceptance_ratio=1.00002, completion_time='Only complete accepted execution; failures retain every available actual command.'),
        order='Zero-command preflight, all four 1 ms comparator cells, inspect accounting, then all four 0.5 ms cells.',
        preservation='No result overwrite; subprocess logs and all failed attempts remain in this folder.')
    write_json(REG, reg)
    print('REGISTERED', sha(REG), flush=True)


def verify():
    reg = read(REG)
    assert sha(SOURCE_REG) == reg['source_registration_sha256']
    assert sha(ROOT/reg['global_registration']) == reg['global_registration_sha256']
    for name, expected in reg['source_sha256'].items():
        assert sha(ROOT/'src'/name) == expected, name
    for item in reg['copied_files']:
        assert sha(OUT/item['local']) == item['sha256'], item['local']
    assert sha(ROOT/reg['nnc_base_registration']) == reg['nnc_base_registration_sha256']
    for item in reg['bound_main_runs']:
        assert sha(ROOT/item['trace']) == item['trace_sha256']
        assert sha(ROOT/item['record']) == item['record_sha256']
    return reg


def fixed_worker(spec, preflight):
    from runtime_physical_time_v5 import install
    from frozen_policy import FrozenPolicy
    from evaluate import rollout
    reg = verify()
    case = next(c for c in reg['selected_cases'] if c['id'] == spec['case'])
    reference = np.loadtxt(ROOT/case['reference'], delimiter=',', skiprows=1)
    settings = install(dt=spec['dt_s'], tolerance=spec['tolerance_mm'])
    torch.set_num_threads(1)
    policy = FrozenPolicy(OUT/'fixed_two.pt')
    assert policy.outputs == 2 and policy.preview_mm == 3.
    prefile = OUT/'fixed'/(spec['id']+'_preflight.json')
    if preflight:
        from direct_two_action_control import DirectTwoRuntime
        env = DirectTwoRuntime(reference, preview_rule_mm=3., potential_speed=100.)
        observation = env.observation()
        action = policy.mean(observation).numpy()
        assert observation.shape == (14,) and action.shape == (2,)
        assert env.env.held[2] == 3. and env.env.fixed_d == 3.
        assert np.array_equal(env.position, reference[0])
        assert env.ticks == 0
        write_json(prefile, dict(passed=True, commands_executed=0,
            case=spec['case'], dt_s=spec['dt_s'], model_sha256=FIXED_SHA,
            observation_dimension=14, policy_outputs=2, fixed_preview_mm=3., settings=settings))
        return
    assert read(prefile)['passed']
    output = ROOT/spec['trace']
    assert not output.exists() and not output.with_suffix('.json').exists()
    class Actor:
        def act(self, observation, deterministic=True):
            return policy.mean(observation).numpy(), None, None
    started_cpu = time.process_time()
    result = rollout(Actor(), reference, output, fixed_d=3., dt=spec['dt_s'],
        wall_s=spec['wall_budget_s'], tolerance=spec['tolerance_mm'], runtime_version='v3')
    result.update(method='FIXED', case=spec['case'], path=spec['case'], model_sha256=FIXED_SHA,
        source_registration_sha256=SOURCE_REG_SHA, registration_sha256=sha(REG),
        runtime_version='v5_physical_time_fixed_two', executor_setup=settings,
        process_cpu_s=time.process_time()-started_cpu, new_training_updates=0,
        fixed_preview_mm=3., action_source='Unchanged historical two-output frozen policy deterministic mean.',
        physical_time_budget_s=spec['physical_time_budget_s'],
        comparison_scope='Retrospectively selected complete synthetic program; separately developed controller, not a paired preview ablation.')
    write_json(output.with_suffix('.json'), result)


def nnc_worker(spec, preflight):
    import evaluate_restored_nnc_v6 as nnc
    nnc.verify()
    nnc.OUT = NNC_ROOT
    def verify_adapter():
        reg = verify()
        return dict(runs=[r for r in reg['runs'] if r['method']=='NNC'])
    nnc.verify = verify_adapter
    original = inspect.getsource(nnc.worker)
    changes = {
        'closed=True)': 'closed=bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9))',
        "runtime_version='restored_project_NNC_protected_v6'": "runtime_version='restored_project_NNC_protected_v7'",
        "status='complete' if success else": "status='complete' if audit['success'] else",
    }
    for old, new in changes.items():
        assert original.count(old) == 1, old
        original = original.replace(old, new)
    namespace = {}
    exec(compile(original, str(ROOT/'src/evaluate_restored_nnc_v6.py'), 'exec'), nnc.__dict__, namespace)
    # The v6 worker binds its local registration; point it to this exact immutable
    # subordinate registration without changing its command execution function.
    local_reg = NNC_ROOT/'registration.json'
    assert local_reg.exists() and sha(local_reg) == sha(REG)
    namespace['worker'](spec['id'], preflight)


def worker(run_id, preflight):
    reg = verify()
    spec = next(r for r in reg['runs'] if r['id']==run_id)
    (fixed_worker if spec['method']=='FIXED' else nnc_worker)(spec, preflight)


def matrix(dt, preflight):
    reg = verify()
    if not (NNC_ROOT/'registration.json').exists():
        shutil.copy2(REG, NNC_ROOT/'registration.json')
    for spec in reg['runs']:
        if spec['dt_s'] != dt:
            continue
        suffix = '_preflight.log' if preflight else '.log'
        log = OUT/'logs'/(spec['id']+suffix)
        log.parent.mkdir(parents=True, exist_ok=True)
        assert not log.exists(), 'Never overwrite an attempt log'
        cmd = [sys.executable, '-B', str(Path(__file__)), 'worker', '--run-id', spec['id']]
        if preflight:
            cmd.append('--preflight')
        with log.open('w', encoding='utf-8') as stream:
            completed = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=spec['wall_budget_s']+120.)
        print('SUBPROCESS', spec['id'], 'preflight' if preflight else 'run', completed.returncode, flush=True)
        if completed.returncode:
            print(log.read_text('utf-8')[-6000:], flush=True)
            raise RuntimeError('Execution failed; preserved attempt log: '+str(log))


def report():
    from metrics_exact_index_v5 import metrics
    from frozen_policy import FrozenPolicy
    reg = verify()
    policy = FrozenPolicy(OUT/'fixed_two.pt')
    main_policy = FrozenPolicy(ROOT/'baseline_records/supplementary_reproduction/models/main.pt')
    torch.set_num_threads(1)
    rows = []
    for spec in list(reg['bound_main_runs'])+reg['runs']:
        case = next(c for c in reg['selected_cases'] if c['id']==spec['case'])
        record = read(ROOT/spec['record'])
        trace = ROOT/spec['trace']
        assert sha(trace) == record['trace_sha256']
        with np.load(trace) as z:
            positions, reference = z['positions'], z['reference']
            actions, observations = z['actions'], z['observations']
        expected = np.loadtxt(ROOT/case['reference'], delimiter=',', skiprows=1)
        assert np.array_equal(reference, expected)
        assert np.array_equal(positions[0], reference[0])
        closed = bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9)
        assert closed == case['static_audit']['closed']
        independent = metrics(reference, positions, dt=spec['dt_s'],
            tolerance=case['evaluation_tolerance_mm'], declared_success=record['success'], reason=record['reason'])
        assert independent['success'] == record['success']
        assert np.allclose(independent['vector_peaks'], record['vector_peaks'], rtol=1e-10, atol=1e-7)
        assert independent['geometry']['gates_passed'] == record['geometry']['gates_passed']
        action_error = None
        if spec['method'] in ('FIXED','JNN'):
            actor = policy if spec['method']=='FIXED' else main_policy
            predicted = np.vstack([actor.mean(b).numpy() for b in np.array_split(observations, max(1,len(observations)//256))])
            action_error = float(np.max(np.abs(predicted-actions)))
            assert action_error < 2e-6
        assert not independent['success'] or independent['complete_time_s'] <= case['physical_time_budget_s']+spec['dt_s']
        rows.append(dict(case=spec['case'], method=spec['method'], dt_s=spec['dt_s'],
            metrics=independent, trace=spec['trace'], trace_sha256=sha(trace),
            record=spec['record'], record_sha256=sha(ROOT/spec['record']),
            model_sha256=record['model_sha256'], reference_closed=closed,
            reference_sha256=case['reference_sha256'], tolerance_mm=case['evaluation_tolerance_mm'],
            max_action_replay_error=action_error, physical_time_budget_s=case['physical_time_budget_s'],
            wall_s=record.get('wall_s'), process_cpu_s=record.get('process_cpu_s',record.get('cpu_s')),
            new_execution=spec['method']!='JNN', first_failure=record.get('first_failure')))
    assert len(rows)==12
    summary = dict(created_utc=utc(), registration_sha256=sha(REG), complete_grid=True,
        selected_case_count=2, new_comparator_runs=8, reused_main_runs=4, all_attempts_retained=True,
        selection_scope=reg['scope'], all_program_closed_flags_match=True,
        whole_interval_geometry_and_order_recomputed=True, rows=rows)
    write_json(OUT/'trace_summary.json', summary)
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['case','method','dt_s','success','complete_time_s','prefix_time_s','gates_passed','gate_count','reason'])
    for row in rows:
        m=row['metrics'];g=m['geometry']
        writer.writerow([row['case'],row['method'],row['dt_s'],m['success'],m['complete_time_s'],
            m['elapsed_prefix_s'],g['gates_passed'],g['gate_count'],m['reason']])
    (OUT/'complete_results.csv').write_text(stream.getvalue(), encoding='utf-8')
    print(stream.getvalue(), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('register','preflight','run','worker','report'))
    parser.add_argument('--dt', type=float, choices=(.001,.0005))
    parser.add_argument('--run-id')
    parser.add_argument('--preflight', action='store_true')
    args=parser.parse_args()
    if args.mode=='register': register()
    elif args.mode=='worker': worker(args.run_id,args.preflight)
    elif args.mode=='report': report()
    else:
        assert args.dt is not None
        matrix(args.dt,args.mode=='preflight')
