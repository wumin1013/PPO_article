"""Unchanged J-NNC, fixed-preview and protected NNC on one fixed scale change."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
import compare_learned_representative_v7 as previous

OUT = ROOT/'results/representative_scaled_v8/learned'
REG = OUT/'registration.json'
MAIN_SHA = 'c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
METHODS = ('JNN', 'FIXED', 'NNC')


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def register(input_registration):
    assert not OUT.exists(), 'Preserve every earlier attempt; choose a new version.'
    source = read(input_registration)
    assert sha(input_registration) == 'c2aa5e6b087e346473406b5762e1b554b62c448a27e62ef92d04b4352b4cbe79'
    assert source['geometry_scale'] == 10. and source['planar_vector_limits'] == [100.,2000.,62500.]
    cases = source['cases']
    assert len(cases) == 2
    import evaluate_restored_nnc_v6 as nnc
    recovered = nnc.verify()
    frozen = read(ROOT/'evidence/frozen_main_generalization_v6.json')
    from evaluate_frozen_main_generalization_v6 import verify as verify_frozen_runtime
    verify_frozen_runtime()
    assert source['frozen_execution_source_sha256'] == frozen['source_sha256']
    files = []
    def copy(source, relative):
        destination = OUT/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copy2(source, destination)
        assert sha(destination) == sha(source)
        files.append(dict(local=relative, source=source.relative_to(ROOT).as_posix(), sha256=sha(destination)))
    copy(ROOT/'evidence/frozen_main_v6/main.pt', 'main.pt')
    copy(ROOT/'evidence/frozen_main_v6/fixed_two.pt', 'fixed_two.pt')
    assert sha(OUT/'main.pt') == MAIN_SHA
    assert sha(OUT/'fixed_two.pt') == previous.FIXED_SHA
    for item in recovered['source_files']:
        if not item['local'].startswith('programs/'):
            copy(nnc.OUT/item['local'], 'nnc_adapter/'+item['local'])
    runs = []
    for case in cases:
        assert case['nominal_tolerance_mm'] == 1.25
        assert abs(case['evaluation_tolerance_mm']+case['arc_chord_error_bound_mm']-1.25) < 1e-12
        for key, name in (('reference','reference.csv'),('geometry','geometry.json')):
            assert sha(ROOT/case[key]) == case[key+'_sha256']
            copy(ROOT/case[key], 'nnc_adapter/programs/'+case['id']+'/'+name)
        reference = np.loadtxt(ROOT/case['reference'], delimiter=',', skiprows=1)
        budget = 5.+float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())/5.
        assert abs(budget-case['physical_time_budget_s']) < 1e-8
        for dt in (.001,.0005):
            for method in METHODS:
                name = f"{case['id']}_{method}_dt{round(dt*1e6)}us"
                folder = 'nnc_adapter' if method=='NNC' else 'fixed' if method=='FIXED' else 'main'
                runs.append(dict(id=name, case=case['id'], method=method, dt_s=dt,
                    nominal_tolerance_mm=1.25, tolerance_mm=case['evaluation_tolerance_mm'],
                    chord_reserve_mm=case['arc_chord_error_bound_mm'],
                    physical_time_budget_s=budget, wall_budget_s=600.,
                    trace=(OUT/folder/(name+'.npz')).relative_to(ROOT).as_posix(),
                    record=(OUT/folder/(name+'.json')).relative_to(ROOT).as_posix()))
    sources = dict(frozen['source_sha256'])
    for name in ('compare_learned_scaled_v8.py','compare_learned_representative_v7.py',
                 'evaluate_restored_nnc_v6.py','metrics_exact_index_v5.py','exact_distance_index_v5.py'):
        sources[name] = sha(ROOT/'src'/name)
    result = dict(created_utc=utc(), study_id='representative_scaled_v8_learned',
        global_registration=input_registration.relative_to(ROOT).as_posix(),
        global_registration_sha256=sha(input_registration), selected_cases=cases, runs=runs,
        source_sha256=sources, copied_files=files,
        runtime_manifest_sha256=frozen['runtime_manifest_sha256'],
        main_model_sha256=MAIN_SHA, fixed_model_sha256=previous.FIXED_SHA,
        nnc_model_sha256=previous.NNC_SHA, main_runtime='v5_physical_time',
        fixed_runtime='v5_physical_time with unchanged DirectTwoRuntime and 3 mm preview',
        nnc_runtime='Exact protected v6 build/command loop with v7 open-reference flag correction',
        fixed_and_nnc_function_source='src/compare_learned_representative_v7.py',
        scope='Fixed uniformly enlarged direct variants of two previously evaluated synthetic programs; case studies, not independent holdout programs.',
        new_training_updates=0, parameter_tuning=0, scale_selection='Fixed by user/root before any enlarged-path outcome.',
        all_attempts_retained=True, initial_rest=True, final_rest_s=.02,
        limits=[100.,2000.,62500.], failure_completion_time=None,
        execution_order='Both J-NNC 1 ms preflight/execution first; inspect complete-task outcomes, then all remaining fixed comparator/resolution cells without changing scale or model.',
        nnc_base_registration_sha256=sha(nnc.OUT/'registration.json'))
    write_json(REG,result)
    shutil.copy2(REG,OUT/'nnc_adapter/registration.json')
    print('REGISTERED',sha(REG),flush=True)


def verify():
    reg=read(REG)
    assert sha(ROOT/reg['global_registration'])==reg['global_registration_sha256']
    for name,value in reg['source_sha256'].items():
        assert sha(ROOT/'src'/name)==value,name
    for item in reg['copied_files']:
        assert sha(OUT/item['local'])==item['sha256'],item['local']
    for case in reg['selected_cases']:
        for key in ('reference','geometry'):
            assert sha(ROOT/case[key])==case[key+'_sha256']
    assert sha(ROOT/'runtime/source_manifest.json')==reg['runtime_manifest_sha256']
    assert sha(OUT/'nnc_adapter/registration.json')==sha(REG)
    return reg


def bind_previous():
    reg=verify()
    previous.OUT=OUT
    previous.REG=REG
    previous.NNC_ROOT=OUT/'nnc_adapter'
    previous.SOURCE_REG_SHA=reg['global_registration_sha256']
    previous.verify=verify


def main_worker(spec,preflight):
    from frozen_policy import FrozenPolicy
    from runtime_physical_time_v5 import install
    from evaluate import rollout
    reg=verify();case=next(c for c in reg['selected_cases'] if c['id']==spec['case'])
    reference=np.loadtxt(ROOT/case['reference'],delimiter=',',skiprows=1)
    settings=install(dt=spec['dt_s'],tolerance=spec['tolerance_mm'])
    torch.set_num_threads(1)
    policy=FrozenPolicy(OUT/'main.pt')
    assert policy.outputs==3
    prefile=OUT/'main'/(spec['id']+'_preflight.json')
    if preflight:
        from compact_farvector_runtime import FarVectorRuntime
        env=FarVectorRuntime(reference,potential_speed=100.)
        observation=env.observation();action=policy.mean(observation).numpy()
        assert observation.shape==(14,) and action.shape==(3,)
        assert env.ticks==0 and np.array_equal(env.position,reference[0])
        write_json(prefile,dict(passed=True,commands_executed=0,model_sha256=MAIN_SHA,
            case=spec['case'],dt_s=spec['dt_s'],settings=settings))
        return
    assert read(prefile)['passed']
    output=ROOT/spec['trace']
    assert not output.exists() and not output.with_suffix('.json').exists()
    class Actor:
        def act(self,observation,deterministic=True):
            return policy.mean(observation).numpy(),None,None
    started_cpu=time.process_time()
    result=rollout(Actor(),reference,output,dt=spec['dt_s'],wall_s=spec['wall_budget_s'],
        tolerance=spec['tolerance_mm'],runtime_version='v3')
    result.update(method='JNN',case=spec['case'],path=spec['case'],model_sha256=MAIN_SHA,
        registration_sha256=sha(REG),source_registration_sha256=reg['global_registration_sha256'],
        runtime_version='v5_physical_time',executor_setup=settings,
        physical_time_budget_s=spec['physical_time_budget_s'],process_cpu_s=time.process_time()-started_cpu,
        new_training_updates=0,action_source='Unchanged c16 deterministic policy; no geometric recovery or feed tuning.')
    write_json(output.with_suffix('.json'),result)


def worker(run_id,preflight):
    reg=verify();spec=next(r for r in reg['runs'] if r['id']==run_id)
    bind_previous()
    function=main_worker if spec['method']=='JNN' else previous.fixed_worker if spec['method']=='FIXED' else previous.nnc_worker
    function(spec,preflight)


def matrix(dt,methods,preflight):
    reg=verify()
    for spec in reg['runs']:
        if spec['dt_s']!=dt or spec['method'] not in methods:continue
        log=OUT/'logs'/(spec['id']+('_preflight.log' if preflight else '.log'))
        log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B',str(Path(__file__)),'worker','--run-id',spec['id']]
        if preflight:command.append('--preflight')
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=spec['wall_budget_s']+120.)
        print(spec['id'],'preflight' if preflight else 'run',result.returncode,flush=True)
        if result.returncode:
            print(log.read_text('utf-8')[-6000:],flush=True)
            raise RuntimeError('Preserved failed attempt: '+str(log))


def report():
    from metrics_exact_index_v5 import metrics
    from frozen_policy import FrozenPolicy
    reg=verify();torch.set_num_threads(1)
    actors={'JNN':FrozenPolicy(OUT/'main.pt'),'FIXED':FrozenPolicy(OUT/'fixed_two.pt')}
    rows=[]
    for spec in reg['runs']:
        result=read(ROOT/spec['record']);case=next(c for c in reg['selected_cases'] if c['id']==spec['case'])
        trace=ROOT/spec['trace'];assert sha(trace)==result['trace_sha256']
        with np.load(trace) as z:
            reference,positions,observations,actions=z['reference'],z['positions'],z['observations'],z['actions']
        assert np.array_equal(reference,np.loadtxt(ROOT/case['reference'],delimiter=',',skiprows=1))
        assert np.array_equal(positions[0],reference[0])
        independent=metrics(reference,positions,dt=spec['dt_s'],tolerance=spec['tolerance_mm'],
            declared_success=result['success'],reason=result.get('executor_reason',result['reason'].split(';')[0]))
        assert independent['success']==result['success']
        assert np.allclose(independent['vector_peaks'],result['vector_peaks'],rtol=1e-10,atol=1e-7)
        error=None
        if spec['method'] in actors:
            predicted=np.vstack([actors[spec['method']].mean(batch).numpy() for batch in np.array_split(observations,max(1,len(observations)//256))])
            error=float(np.abs(predicted-actions).max());assert error<2e-6
        rows.append(dict(case=spec['case'],method=spec['method'],dt_s=spec['dt_s'],metrics=independent,
            trace=spec['trace'],trace_sha256=sha(trace),record=spec['record'],record_sha256=sha(ROOT/spec['record']),
            model_sha256=result['model_sha256'],max_action_replay_error=error,
            reference_closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9),
            wall_s=result['wall_s'],process_cpu_s=result.get('process_cpu_s'),first_failure=result.get('first_failure')))
    assert len(rows)==12
    result=dict(created_utc=utc(),registration_sha256=sha(REG),complete_grid=True,
        case_count=2,new_physical_executions=12,reused_runs=0,all_attempts_retained=True,
        scope=reg['scope'],all_complete_acceptance_recomputed=True,rows=rows)
    write_json(OUT/'trace_summary.json',result)
    print(json.dumps([dict(case=r['case'],method=r['method'],dt_s=r['dt_s'],
        success=r['metrics']['success'],time_s=r['metrics']['complete_time_s'],
        prefix_s=r['metrics']['elapsed_prefix_s'],reason=r['metrics']['reason']) for r in rows],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','preflight','run','worker','report'))
    parser.add_argument('--input-registration',type=Path)
    parser.add_argument('--dt',type=float,choices=(.001,.0005))
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--run-id');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.mode=='register':register(args.input_registration.resolve())
    elif args.mode=='worker':worker(args.run_id,args.preflight)
    elif args.mode=='report':report()
    else:
        assert args.dt is not None
        matrix(args.dt,args.methods,args.mode=='preflight')
