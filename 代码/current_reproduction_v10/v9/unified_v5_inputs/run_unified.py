"""Registered common-v5 frozen-controller replays; preserve every attempted cell."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
PARENT = ROOT.parent
REG = ROOT/'registration.json'
MODEL_SOURCES = {'JNN': BASE/'baseline_records/supplementary_reproduction/models/main.pt',
                 'FIXED': BASE/'evidence/frozen_main_v6/fixed_two.pt'}
MODEL_EXPECTED = {'JNN':'c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764',
                 'FIXED':'40af7ea30bec92d9cd42e060b2d71ee72919c4cc50c3b28281ff036a0790f469'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False)
        stream.write('\n')


def register():
    assert not REG.exists(), 'Preserve the registered experiment.'
    parent_reg = PARENT/'registration.json'
    cases = [r for r in read(parent_reg)['cases'] if r['id'] in ('original','dense')]
    assert len(cases) == 2
    inputs = {}
    for method, source in MODEL_SOURCES.items():
        assert sha(source) == MODEL_EXPECTED[method]
        local = ROOT/'inputs/models'/f'{method}.pt'
        local.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,local)
        inputs[local.relative_to(ROOT).as_posix()] = dict(source=str(source), sha256=sha(local))
    for case in cases:
        for key in ('reference','geometry'):
            source = BASE/case[key]
            assert sha(source) == case[key+'_sha256']
            local = ROOT/'inputs/cases'/case['id']/source.name
            local.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,local)
            inputs[local.relative_to(ROOT).as_posix()] = dict(source=str(source),sha256=sha(local))
            case[key] = local.relative_to(ROOT).as_posix()
    jobs = [dict(case=case,method=method,dt_s=dt,
                 run_id=f'{case}_{method}_dt{round(dt*1e6)}us')
            for case in ('original','dense') for dt in (.001,.0005) for method in ('JNN','FIXED')]
    source_hashes = {p.relative_to(BASE).as_posix():sha(p) for p in (BASE/'src').glob('*.py')}
    runtime_manifest = BASE/'runtime/source_manifest.json'
    runtime_rows = read(runtime_manifest)
    for row in runtime_rows:
        assert sha(BASE/'runtime'/row['relative']) == row['sha256'], row['relative']
    write(REG,dict(created_utc=utc(),version='unified_original_dense_v5_replay',
        purpose='Same immutable model and runtime across all original/dense input cells before offline boundary refinement. No launch or combined runtime.',
        base=str(BASE),parent_registration_sha256=sha(parent_reg),runner_sha256=sha(__file__),
        source_sha256=source_hashes,runtime_manifest_sha256=sha(runtime_manifest),
        runtime_source_count=len(runtime_rows),inputs=inputs,cases=cases,jobs=jobs,
        models={m:dict(file=f'inputs/models/{m}.pt',sha256=MODEL_EXPECTED[m]) for m in MODEL_SOURCES},
        runtime='runtime_physical_time_v5.install',underlying_tracking_omega_s_inv=40.,
        fixed_preview_mm=3.,control_period_s=.02,heading_filter_tau_s=.012,
        vector_limits=[100.,2000.,62500.],nominal_tolerance_mm=1.25,
        geometry='Source-specific single-sided polyline tolerance is nominal minus registered arc chord error; full interval corridor and complete ordered traversal.',
        completion='Original complete start-to-final-rest definition, finite original cubic terminal brake, >=20ms qualified rest.',
        actor='FrozenPolicy.mean at every saved observation; raw deterministic action, no guide override, no training updates.',
        wall_budget_s=1800.,parallel_processes=4,all_attempts_retained=True,
        statistical_unit='Deterministic full replay of frozen controller; not an independent training seed.'))
    print(json.dumps(dict(registration=str(REG),sha256=sha(REG),jobs=len(jobs)),indent=2),flush=True)


def verify():
    reg = read(REG)
    assert sha(__file__) == reg['runner_sha256']
    assert sha(BASE/'runtime/source_manifest.json') == reg['runtime_manifest_sha256']
    for relative,digest in reg['source_sha256'].items():
        assert sha(BASE/relative) == digest, relative
    for relative,source in reg['inputs'].items():
        assert sha(ROOT/relative) == source['sha256'],relative
    return reg


def worker(run_id):
    reg = verify()
    job = next(j for j in reg['jobs'] if j['run_id'] == run_id)
    case = next(c for c in reg['cases'] if c['id'] == job['case'])
    output = ROOT/'results'/(run_id+'.npz')
    assert not output.exists() and not output.with_suffix('.json').exists()
    sys.path.insert(0,str(BASE/'src'))
    import torch
    torch.set_num_threads(1)
    from frozen_policy import FrozenPolicy
    from runtime_physical_time_v5 import install
    import evaluate
    from metrics_exact_index_v5 import metrics
    evaluate.metrics = metrics
    settings = install(dt=job['dt_s'],tolerance=case['evaluation_tolerance_mm'])
    assert settings['version'] == 'v5_physical_time'
    policy = FrozenPolicy(ROOT/reg['models'][job['method']]['file'])
    class Actor:
        def act(self,observation,deterministic=True):
            return policy.mean(observation).numpy(),None,None
    reference = np.loadtxt(ROOT/case['reference'],delimiter=',',skiprows=1)
    started_cpu = time.process_time()
    result = evaluate.rollout(Actor(),reference,output,
        fixed_d=reg['fixed_preview_mm'] if job['method']=='FIXED' else None,
        dt=job['dt_s'],wall_s=reg['wall_budget_s'],tolerance=case['evaluation_tolerance_mm'],
        runtime_version='v3')
    # evaluate already wrote the complete physical result. Enrich exactly that
    # newly produced record, without changing any result or trajectory field.
    result.update(method=job['method'],case=job['case'],path=job['case'],run_id=run_id,
        runtime_version='v5_physical_time',executor_setup=settings,
        model_sha256=reg['models'][job['method']]['sha256'],model_file=reg['models'][job['method']]['file'],
        reference_sha256=case['reference_sha256'],registration_sha256=sha(REG),
        nominal_tolerance_mm=case['nominal_tolerance_mm'],arc_chord_error_bound_mm=case['arc_chord_error_bound_mm'],
        process_cpu_s=time.process_time()-started_cpu,new_training_updates=0,
        action_source='Unchanged deterministic FrozenPolicy.mean; common v5_physical_time executor; omega40 original tracking.',
        scientific_scope='Unified original/dense input replay for offline boundary refinement; no independent training replicate.')
    output.with_suffix('.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    verify()
    print(json.dumps(dict(run_id=run_id,success=result['success'],time_s=result['complete_time_s'],reason=result['reason']),ensure_ascii=False),flush=True)


def run_matrix():
    reg = verify()
    started = time.perf_counter()
    def execute(job):
        log = ROOT/'logs'/(job['run_id']+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        t0 = time.perf_counter()
        try:
            with log.open('x',encoding='utf-8') as stream:
                done = subprocess.run([sys.executable,'-B',str(Path(__file__)),'worker','--run-id',job['run_id']],
                    stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,timeout=reg['wall_budget_s']+240.)
            code, error = done.returncode, None
        except Exception as exc:
            code, error = None, type(exc).__name__+': '+str(exc)
        row = dict(**job,exit_code=code,infrastructure_error=error,process_wall_s=time.perf_counter()-t0,
                   log=log.relative_to(ROOT).as_posix())
        print(json.dumps(row,ensure_ascii=False),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=reg['parallel_processes']) as pool:
        outcomes = list(pool.map(execute,reg['jobs']))
    write(ROOT/'matrix_receipt.json',dict(created_utc=utc(),registration_sha256=sha(REG),
        all_eight_attempts_retained=True,wall_s=time.perf_counter()-started,outcomes=outcomes))
    summarize()


def compare_arrays(left_path,right_path):
    with np.load(left_path,allow_pickle=False) as a, np.load(right_path,allow_pickle=False) as b:
        common = sorted(set(a.files)&set(b.files))
        rows = {}
        for key in common:
            x,y = a[key],b[key]
            n = min(len(x),len(y)) if x.ndim and y.ndim else None
            identical = bool(np.array_equal(x,y))
            first_difference = None
            if n is not None and x.shape[1:]==y.shape[1:]:
                mask = (x[:n] != y[:n]).reshape(n,-1).any(axis=1)
                difference = np.flatnonzero(mask)
                first_difference = int(difference[0]) if len(difference) else (n if len(x)!=len(y) else None)
            rows[key] = dict(left_shape=list(x.shape),right_shape=list(y.shape),bitwise_equal=identical,
                first_differing_row=first_difference)
        return dict(left=str(left_path),left_sha256=sha(left_path),right=str(right_path),right_sha256=sha(right_path),
            same_array_keys=set(a.files)==set(b.files),all_common_arrays_bitwise_equal=all(r['bitwise_equal'] for r in rows.values()),
            left_only_keys=sorted(set(a.files)-set(b.files)),right_only_keys=sorted(set(b.files)-set(a.files)),arrays=rows)


def summarize():
    reg = verify()
    target = ROOT/'trace_summary.json'
    assert not target.exists()
    frozen_root = BASE/'evidence/frozen_main_v6'
    frozen_manifest = read(frozen_root/'manifest.json')
    previous = {r['id']:r for r in frozen_manifest['runs']}
    rows = []
    for job in reg['jobs']:
        record_path = ROOT/'results'/(job['run_id']+'.json')
        if not record_path.exists():
            rows.append(dict(**job,success=False,complete_time_s=None,reason='worker_missing_record',
                log=str(ROOT/'logs'/(job['run_id']+'.log'))))
            continue
        record = read(record_path)
        trace_path = record_path.with_suffix('.npz')
        assert sha(trace_path) == record['trace_sha256']
        prior = previous[job['run_id']]
        prior_path = frozen_root/prior['trace']
        assert sha(prior_path) == prior['trace_sha256']
        old_comparison = compare_arrays(trace_path,prior_path)
        historical = BASE/'results/historical_strong_v5'/f"{job['case']}_dt{round(job['dt_s']*1e6)}us.npz"
        historical_comparison = compare_arrays(trace_path,historical) if job['method']=='JNN' and historical.exists() else None
        rows.append(dict(**job,success=record['success'],complete_time_s=record['complete_time_s'],reason=record['reason'],
            trace=str(trace_path),trace_sha256=sha(trace_path),record=str(record_path),record_sha256=sha(record_path),
            model_sha256=record['model_sha256'],runtime_version=record['runtime_version'],executor_setup=record['executor_setup'],
            reference_sha256=record['reference_sha256'],tolerance_mm=record['tolerance_mm'],
            vector_limit_ratios=record['vector_limit_ratios'],terminal_rest_s=record['terminal_rest_s'],
            geometry=record['geometry'],actor_calls=record['actor_calls'],cost=record['cost'],
            wall_s=record['wall_s'],process_cpu_s=record['process_cpu_s'],
            previous_frozen_identity=prior['identity'],previous_frozen_complete_time_s=prior['metrics']['complete_time_s'],
            versus_frozen_main=old_comparison,versus_existing_v5_replay=historical_comparison))
    result = dict(created_utc=utc(),registration_sha256=sha(REG),runner_sha256=sha(__file__),
        all_eight_registered_cells_retained=len(rows)==8,all_same_runtime_configuration=True,
        success_count=sum(r['success'] for r in rows),failure_count=sum(not r['success'] for r in rows),
        independent_training_replicates=0,new_training_updates=0,
        prior_frozen_manifest_sha256=sha(frozen_root/'manifest.json'),rows=rows)
    write(target,result)
    print(json.dumps(dict(summary=str(target),sha256=sha(target),success_count=result['success_count'],failure_count=result['failure_count'],
        rows=[{k:r.get(k) for k in ('run_id','success','complete_time_s','reason')} for r in rows]),indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','matrix','worker','summary'))
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.command=='register':register()
    elif args.command=='matrix':run_matrix()
    elif args.command=='worker':worker(args.run_id)
    else:summarize()
