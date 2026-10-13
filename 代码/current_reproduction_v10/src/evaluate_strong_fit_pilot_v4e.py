"""Isolate dt with a fixed physical guide width; preserve all v4d records."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from evaluate import rollout
from strong_policy_v4 import load_strong_policy
from runtime_resolution_fixed_v4e import install,VERSION
from evaluate_strong_fit_pilot_v4d import MODEL,EXPECTED_MODEL,SOURCES as V4D_SOURCES,register as register_v4d

OUT=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4e'
V4D=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d'
SOURCES=sorted(set(V4D_SOURCES+['evaluate_strong_fit_pilot_v4e.py','runtime_resolution_fixed_v4e.py']))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        old=register_v4d();rows=old['cases'][:]
        rows.append(dict(id='train_sine',reference='data/train_sine.csv',reference_sha256=sha(ROOT/'data/train_sine.csv'),
            nominal_tolerance_mm=1.25,evaluation_tolerance_mm=1.25,arc_chord_error_bound_mm=0.,scope='Training program; 1 ms equality control only'))
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,
            cases=rows,model_sha256=EXPECTED_MODEL,seed=101,arm='adaptive',
            source_sha256={name:sha(ROOT/'src'/name) for name in SOURCES},
            prior_registration_sha256=sha(V4D/'registration.json'),
            change='Pin guide physical transition width to 0.1 mm regardless of integration dt or import order; preserve v4d terminal/executor and policy.',
            planned='1 ms train_sine exact trajectory control, all five observed v3 programs and dense at 0.5 ms. Optional original only after dense 0.5 ms succeeds.',
            inclusion='All six fine-resolution cases, successes and failures, retained without selection.',
            identity='Fresh-weight 12000-fit seed101 pilot; new explicit executor version, not frozen-original direct transfer.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Model changed')
    for name,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Source changed: '+name)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case,dt):
    registered=register()
    if not (V4D/'original_dt1000us.json').exists():raise ValueError('Complete the prior v4d long audit first')
    row=next(x for x in registered['cases'] if x['id']==case)
    if case=='original':
        dense=json.loads((OUT/'dense_dt500us.json').read_text('utf-8'))
        if not dense['success']:raise ValueError('No original rerun until fine dense completes')
    if case=='train_sine' and dt!=.001:raise ValueError('The equality control is at nominal 1 ms')
    setup=install(dt=dt,tolerance=row['evaluation_tolerance_mm'])
    policy=load_strong_policy(MODEL)[0]
    ref=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+f'_dt{round(dt*1e6)}us.npz')
    result=rollout(policy,ref,file,dt=dt,tolerance=row['evaluation_tolerance_mm'],runtime_version='v3',
        wall_s=1800. if case=='original' else 240.)
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Model changed during replay')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='fresh_seed101_fitted_strong_structure_adaptive',model_sha256=EXPECTED_MODEL,seed=101,
        path=case,reference_sha256=row['reference_sha256'],nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],registration_sha256=sha(OUT/'registration.json'),
        action_source='fresh_seed101_fitted_strong_structure_adaptive')
    write_json(file.with_suffix('.json'),result)
    if case=='train_sine':
        before=ROOT/'results/v4d_verified_terminal_preview/pilot_loader_retry/pilot/train_sine.npz'
        comparison=dict(v4d_trace_sha256=sha(before),v4e_trace_sha256=sha(file),arrays={})
        with np.load(before) as a,np.load(file) as b:
            for key in ('rows','positions','observations','actions'):
                comparison['arrays'][key]=dict(same_shape=a[key].shape==b[key].shape,
                    equal=bool(a[key].shape==b[key].shape and np.array_equal(a[key],b[key])),
                    max_abs=float(abs(a[key]-b[key]).max()) if a[key].shape==b[key].shape else None)
        write_json(OUT/'nominal_equality_control.json',comparison)
    print(json.dumps(dict(case=case,dt_s=dt,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case');p.add_argument('--dt',type=float,choices=[.001,.0005],default=.0005)
    p.add_argument('--five-observed',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.five_observed:
        for row in register()['cases']:
            if row['id'].startswith('final_'):run(row['id'],a.dt)
    elif a.case:run(a.case,a.dt)
    else:p.error('Select register, five-observed or case')
