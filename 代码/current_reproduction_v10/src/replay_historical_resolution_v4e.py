"""Separately registered historical-weight control at fixed-width 0.5 ms."""
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
from runtime_resolution_fixed_v4e import install,VERSION

OUT=ROOT/'results/strong_checkpoint_v4/v4e_resolution_control'
SOURCES=sorted(set(PROTECTED+['replay_historical_resolution_v4e.py','runtime_resolution_fixed_v4e.py',
    'terminal_completion_v4d.py','runtime_no_target_override_v4b.py','replay_no_target_override_v4b.py',
    'unified_training_v3.py','paired_learning.py']))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,
            model_sha256=EXPECTED_MODEL,model_updated=False,seed=923202,dt_s=.0005,
            cases=[x for x in paths() if x['id'] in ('dense','original')],
            source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
            question='Separate unchanged historical policy from fresh-fit error under fixed-physical-width half-ms execution.',
            change='Executor version v4e, width 0.1 mm, integration dt 0.5 ms, 20 ms decision period, v4d verified terminal brake.',
            order='Dense first; original only after dense succeeds under full independent audit.',
            unchanged='Every gate, each command interval, nominal corridor minus chord error, vector limits, initial/final rest and complete program.',
            retained='All full traces and failures; prefix durations never interpreted as complete times.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Historical checkpoint changed')
    for n,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/n)!=expected:raise ValueError('Registered source changed: '+n)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case):
    r=register();row=next(x for x in r['cases'] if x['id']==case)
    if case=='original':
        dense=json.loads((OUT/'dense_dt500us.json').read_text('utf-8'))
        if not dense['success']:raise ValueError('Historical dense did not complete; long control not launched')
    setup=install(dt=.0005,tolerance=row['evaluation_tolerance_mm'])
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+'_dt500us.npz')
    result=rollout(HistoricalActor(),reference,file,dt=.0005,tolerance=row['evaluation_tolerance_mm'],
        wall_s=1800. if case=='original' else 300.,runtime_version='v3')
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Historical model changed during evaluation')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='historical_strong_JNNC_unchanged',path=case,seed=923202,model_sha256=EXPECTED_MODEL,
        model_updated=False,reference_sha256=row['reference_sha256'],
        nominal_tolerance_mm=row['nominal_tolerance_mm'],arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        action_source='unchanged_923202_historical_actor',registration_sha256=sha(OUT/'registration.json'))
    write_json(file.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case',choices=['dense','original']);a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.case:run(a.case)
    else:p.error('Select register or case')
