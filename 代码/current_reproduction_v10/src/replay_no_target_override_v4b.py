"""Development candidate: source-witness initialization plus local blending only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from evaluate import rollout
from replay_strong_checkpoint_v4 import HistoricalActor,MODEL,EXPECTED_MODEL,paths,PROTECTED
from unified_training_v3 import Teacher,CANDIDATES
import runtime_configuration as runtime

VERSION='v4b_no_target_override'
OUT=ROOT/'results'/VERSION
TEACHER=next(c for c in CANDIDATES if c['id']=='f100_k1_h12_d3')
SOURCES=sorted(set(PROTECTED+['replay_no_target_override_v4b.py','unified_training_v3.py','paired_learning.py']))


def no_target_override(env):
    env.v4b_guide_calls=getattr(env,'v4b_guide_calls',0)+1
    return runtime._ORIGINAL_TARGET(env)


def register():
    file=OUT/'registration.json'
    if not file.exists():
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',version=VERSION,
            model_sha256=EXPECTED_MODEL,model_updated=False,dt_s=.001,
            source_sha256={name:sha(ROOT/'src'/name) for name in SOURCES},
            runtime_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'),cases=paths(),
            strong_cases='All six previously registered training/validation programs, then dense; original only if dense succeeds at historical 14.839 s.',
            teacher_cases='All six previously registered training/validation programs.',teacher=TEACHER,
            change='In this process only, replace runtime._ordered_target by no_target_override before configure_runtime(v3). Source witness initialization and disjoint local blending remain. No heading/target predictive guard is added.',
            unchanged='Frozen model; all gates; gate traversal semantics; continuous command-interval corridor checks; vector limits; action decoder; terminal brake; final rest; independent acceptance; nominal tolerance with arc chord error deduction.',
            source_mutation='No protected runtime/source file changes. Runtime function substitution is explicit and versioned.',
            completion='Retain every result and prefix. Prefix elapsed time is never completion time.',
            long_condition='dense success and |complete_time_s - 14.839| <= 1e-9; otherwise do not launch original.'))
    registered=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Historical model changed')
    for name,expected in registered['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Registered source changed: '+name)
    for row in registered['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return registered


def run(method,case):
    registration=register()
    row=next(r for r in registration['cases'] if r['id']==case)
    if method=='teacher' and row['historical_split'] not in ('train','validation'):
        raise ValueError('Teacher scope is six short development programs')
    if case=='original':
        dense=json.loads((OUT/'strong/dense.json').read_text('utf-8'))
        if not dense['success'] or abs(dense['complete_time_s']-14.839)>1e-9:
            raise ValueError('Dense restoration gate failed; no long replay authorized')
    runtime._ordered_target=no_target_override
    policy=HistoricalActor() if method=='strong' else Teacher(TEACHER)
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    output=OUT/method/(case+'.npz')
    result=rollout(policy,reference,output,fixed_d=None if method=='strong' else TEACHER['preview'],
        dt=.001,tolerance=row['evaluation_tolerance_mm'],wall_s=1800. if case=='original' else 180.,runtime_version='v3')
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Historical model changed during replay')
    with np.load(output) as f:
        rows=f['rows'];active=rows[:,12]<.5
        requests=np.linalg.norm(rows[active,8:10],axis=1)
        result['feed_summary']=dict(request_mean_mm_s=float(requests.mean()) if len(requests) else None,
            actual_active_mean_mm_s=float(np.linalg.norm(rows[active,2:4],axis=1).mean()) if active.any() else None)
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,base_runtime_configuration='v3',
        target_override=False,local_disjoint_blending=True,source_witness_initialization=True,
        method=method,path=case,model_sha256=EXPECTED_MODEL if method=='strong' else None,
        model_updated=False,teacher=TEACHER if method=='teacher' else None,
        reference_sha256=row['reference_sha256'],arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        registration_sha256=sha(OUT/'registration.json'),
        action_source='unchanged_923202_historical_actor' if method=='strong' else 'deterministic_geometric_teacher_f100_k1_h12_d3')
    write_json(output.with_suffix('.json'),result)
    print(json.dumps(dict(version=VERSION,method=method,path=case,success=result['success'],
        complete_time_s=result['complete_time_s'],prefix_time_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--method',choices=['strong','teacher'])
    parser.add_argument('--case')
    parser.add_argument('--short-batch',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.register:print(json.dumps(register(),indent=2))
    elif args.short_batch:
        for row in register()['cases']:
            if row['historical_split'] in ('train','validation'):run(args.method,row['id'])
        if args.method=='strong':run('strong','dense')
    elif args.case:run(args.method,args.case)
    else:parser.error('Choose register, short-batch or case')
