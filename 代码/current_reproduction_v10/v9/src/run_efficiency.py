"""Finite development/validation study; preserve every attempted full program."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get('JNN_V9_BASE', r'D:\PPO_article_work\JNN_dominance_20261011_v5'))
sys.path.insert(0, str(BASE/'src'))
import numpy as np
from common import sha, write_json, utc

REG = ROOT/'registration.json'
MODELS = {'JNN': BASE/'baseline_records/supplementary_reproduction/models/main.pt',
          'FIXED': BASE/'evidence/frozen_main_v6/fixed_two.pt'}


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def register():
    assert not REG.exists()
    from replay_strong_checkpoint_v4 import paths
    cases = paths()
    formal = read(BASE/'evidence/scaled_representative_v8_registration.json')['cases']
    cases += [dict(c, historical_split='observed_representative') for c in formal]
    holdout = read(BASE/'evidence/frozen_main_generalization_v6.json')
    write_json(ROOT/'parent_generalization_registration.json', holdout)
    # Bind all source files used by the archived runtime loader and new study.
    source = {str(p.relative_to(BASE)): sha(p) for p in (BASE/'src').glob('*.py')}
    own = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'src').glob('*.py')}
    write_json(REG, dict(created_utc=utc(), version='efficiency_v9', base=str(BASE),
        model_sha256={m:sha(p) for m,p in MODELS.items()}, source_sha256=source,
        own_source_sha256=own, runtime_manifest_sha256=sha(BASE/'runtime/source_manifest.json'),
        candidates=[40.,60.,80.], methods=['JNN','FIXED'], tuning_dt_s=.001,
        final_dt_s=[.001,.0005], cases=cases,
        selection='First reject candidates losing any successful baseline train/validation cell or increasing total failures for either method. Minimize JNN total validation completion time over the baseline-complete validation cells; tie chooses lower omega. One common omega for every route and JNN/FIXED.',
        no_path_specific_parameters=True, new_training_updates=0,
        limits=[100.,2000.,62500.], nominal_tolerance_mm=1.25,
        completion='Same initial rest, complete ordered gates, full command interval corridor, finite brake and final >=20ms rest as v8.',
        terminal_trigger='Unchanged 20ms policy entry and verified cubic.',
        all_attempts_retained=True, wall_budget_s=1800.))
    print('REGISTERED', sha(REG), flush=True)


def verify():
    reg = read(REG)
    for relative, value in reg['source_sha256'].items():
        assert sha(BASE/relative)==value, relative
    for relative, value in reg['own_source_sha256'].items():
        assert sha(ROOT/relative)==value, relative
    for method,path in MODELS.items():
        assert sha(path)==reg['model_sha256'][method]
    assert sha(BASE/'runtime/source_manifest.json')==reg['runtime_manifest_sha256']
    return reg


def output_path(case, method, omega, dt, phase):
    return ROOT/'results'/phase/f'omega{omega:g}'/f'{case}_{method}_dt{round(dt*1e6)}us.npz'


def worker(args):
    reg = verify()
    case = next(c for c in reg['cases'] if c['id']==args.case)
    reference_path = BASE/case['reference']
    assert sha(reference_path)==case['reference_sha256']
    reference = np.loadtxt(reference_path, delimiter=',', skiprows=1)
    output = output_path(args.case,args.method,args.omega,args.dt,args.phase)
    assert not output.exists() and not output.with_suffix('.json').exists()
    import torch
    torch.set_num_threads(1)
    from frozen_policy import FrozenPolicy
    from efficiency_runtime import install
    from evaluate import rollout
    # Exact nearest-segment indexing changes audit cost only, not acceptance.
    import evaluate
    from metrics_exact_index_v5 import metrics
    evaluate.metrics = metrics
    settings = install(omega=args.omega,dt=args.dt,tolerance=case['evaluation_tolerance_mm'])
    policy = FrozenPolicy(MODELS[args.method])
    class Actor:
        def act(self, observation, deterministic=True):
            return policy.mean(observation).numpy(),None,None
    started_cpu = time.process_time()
    result = rollout(Actor(),reference,output,
        fixed_d=3. if args.method=='FIXED' else None, dt=args.dt,
        wall_s=reg['wall_budget_s'], tolerance=case['evaluation_tolerance_mm'],runtime_version='v3')
    result.update(method=args.method,case=args.case,path=args.case,phase=args.phase,
        model_sha256=reg['model_sha256'][args.method],runtime_version=settings['version'],
        executor_setup=settings,omega=args.omega,registration_sha256=sha(REG),
        process_cpu_s=time.process_time()-started_cpu,new_training_updates=0,
        reference_sha256=sha(reference_path),
        action_source='Unchanged historical deterministic actor; versioned common tracking demand.')
    write_json(output.with_suffix('.json'),result)


def matrix(args):
    reg = verify()
    if args.phase=='tuning':
        cases=[c['id'] for c in reg['cases'] if c['historical_split'] in ('train','validation')]
        omegas=reg['candidates']; dts=[.001]
    else:
        selection=read(ROOT/'selection.json')
        assert selection['adoptable']
        cases=['original','dense','piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10']
        omegas=[selection['selected_omega']];dts=reg['final_dt_s']
    jobs=[(c,m,o,d) for o in omegas for c in cases for m in reg['methods'] for d in dts]
    def run(job):
        c,m,o,d=job
        output=output_path(c,m,o,d,args.phase)
        log=output.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-B',str(Path(__file__)),'worker','--case',c,'--method',m,
            '--omega',str(o),'--dt',str(d),'--phase',args.phase]
        with log.open('x',encoding='utf-8') as stream:
            completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,
                                     timeout=reg['wall_budget_s']+180.)
        print(c,m,o,d,'EXIT',completed.returncode,flush=True)
        if completed.returncode: print(log.read_text('utf-8')[-5000:],flush=True)
        return completed.returncode
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes=list(pool.map(run,jobs))
    assert not any(codes), 'Failed attempts preserved; inspect logs.'


def select():
    reg=verify();assert not (ROOT/'selection.json').exists()
    cases=[c for c in reg['cases'] if c['historical_split'] in ('train','validation')]
    records={}
    for o in reg['candidates']:
        for c in cases:
            for m in reg['methods']:
                records[o,c['id'],m]=read(output_path(c['id'],m,o,.001,'tuning').with_suffix('.json'))
    baseline={key[1:]:r for key,r in records.items() if key[0]==40.}
    validation=[(c['id'],'JNN') for c in cases if c['historical_split']=='validation' and baseline[c['id'],'JNN']['success']]
    assert validation, 'No baseline-complete validation path; cannot select by time.'
    rows=[]
    for o in reg['candidates']:
        lost=[list(k) for k,r in baseline.items() if r['success'] and not records[(o,)+k]['success']]
        totals={m:sum(not records[o,c['id'],m]['success'] for c in cases) for m in reg['methods']}
        failures_ok=all(totals[m]<=sum(not r['success'] for k,r in baseline.items() if k[1]==m) for m in reg['methods'])
        eligible=not lost and failures_ok
        score=sum(records[(o,)+k]['complete_time_s'] for k in validation) if eligible else None
        rows.append(dict(omega=o,eligible=eligible,lost_baseline_complete_cells=lost,
            failure_counts=totals,validation_sum_s=score))
    selected=min([r for r in rows if r['eligible']],key=lambda r:(r['validation_sum_s'],r['omega']))
    result=dict(created_utc=utc(),registration_sha256=sha(REG),selected_omega=selected['omega'],
        adoptable=selected['omega']!=40., selection_scope='Predeclared train/validation only',
        validation_cells=validation,candidates=rows,records=[dict(case=c,method=m,omega=o,
            success=r['success'],time_s=r['complete_time_s'],reason=r['reason']) for (o,c,m),r in records.items()])
    write_json(ROOT/'selection.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','matrix','worker','select'))
    parser.add_argument('--phase',choices=('tuning','formal'),default='tuning')
    parser.add_argument('--case');parser.add_argument('--method',choices=('JNN','FIXED'))
    parser.add_argument('--omega',type=float);parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='worker':worker(args)
    elif args.command=='matrix':matrix(args)
    else:select()
