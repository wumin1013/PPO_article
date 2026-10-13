"""Registered fixed-final second fitting stage; all prescribed development cases."""
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
from replay_strong_checkpoint_v4 import paths,PROTECTED

MODEL=ROOT/'results/final_function_fit_v4/stage2_seed101/adaptive/final.pt'
EXPECTED_MODEL='3efc90eba54cccde53cbe0260a32cb1504c25f898b0f943059dbe6b4bce15595'
OUT=MODEL.parent/'evaluation_v4e'
SOURCES=sorted(set(PROTECTED+['evaluate_stage2_fit_v4e.py','strong_policy_v4.py',
    'runtime_resolution_fixed_v4e.py','terminal_completion_v4d.py','runtime_no_target_override_v4b.py',
    'replay_no_target_override_v4b.py','unified_training_v3.py','paired_learning.py']))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        registered=paths();rows=[x for x in registered if x['id']=='dense']
        rows.extend(x for x in registered if x['historical_split'] in ('train','validation'))
        old=json.loads((ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/registration.json').read_text('utf-8'))
        rows.extend(x for x in old['cases'] if x['id'].startswith('final_'))
        rows.extend(x for x in registered if x['id']=='original')
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,
            model_sha256=EXPECTED_MODEL,seed=101,arm='adaptive',cases=rows,
            source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
            fitting_registration_sha256=sha(MODEL.parent/'registration.json'),
            fitting_receipt_sha256=sha(MODEL.parent/'receipt.json'),
            identity='Second fitting stage continuing the same fresh seed101 fit, 24000 cumulative offline iterations; not PPO and not a new independent seed.',
            selection='Use this final checkpoint only; no intermediate selection or new cases.',
            order='Dense, six short train/validation, five observed development programs at 1 ms. Original at 1 ms only after dense 1 ms succeeds. After all twelve non-original runs finish, dense at 0.5 ms regardless of nominal success.',
            unchanged='v4e fixed physical guide width and verified terminal brake, every original gate, every command interval, vector limits, start/final rest, and arc chord reservation.',
            retention='Every success and failure retained; failed prefix durations never treated as completion times.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Stage-two final checkpoint changed')
    for n,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/n)!=expected:raise ValueError('Registered source changed: '+n)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case,dt):
    r=register();row=next(x for x in r['cases'] if x['id']==case)
    if dt==.0005:
        if case!='dense':raise ValueError('Only dense fine-resolution check is registered')
        missing=[x['id'] for x in r['cases'] if x['id']!='original' and not (OUT/(x['id']+'_dt1000us.json')).exists()]
        if missing:raise ValueError('Complete all twelve nominal cases first: '+str(missing))
    if case=='original':
        dense=json.loads((OUT/'dense_dt1000us.json').read_text('utf-8'))
        if not dense['success']:raise ValueError('Nominal dense failed; long evaluation is not launched')
    setup=install(dt=dt,tolerance=row['evaluation_tolerance_mm'])
    policy,saved=load_strong_policy(MODEL)
    if (saved['seed'],saved['arm'])!=(101,'adaptive'):raise ValueError('Stage-two model identity mismatch')
    ref=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+f'_dt{round(dt*1e6)}us.npz')
    result=rollout(policy,ref,file,dt=dt,tolerance=row['evaluation_tolerance_mm'],
        runtime_version='v3',wall_s=1800. if case=='original' else 240.)
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Model changed during replay')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='fresh_seed101_stage2_fit24000_final',path=case,seed=101,arm='adaptive',model_sha256=EXPECTED_MODEL,
        reference_sha256=row['reference_sha256'],nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],registration_sha256=sha(OUT/'registration.json'),
        action_source='fixed_final_second_fit_stage_same_seed101',cumulative_offline_iterations=24000,new_ppo_updates=0)
    write_json(file.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,dt_s=dt,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case');p.add_argument('--dt',type=float,choices=[.001,.0005],default=.001)
    p.add_argument('--twelve-nominal',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.twelve_nominal:
        for row in register()['cases']:
            if row['id']!='original':run(row['id'],.001)
    elif a.case:run(a.case,a.dt)
    else:p.error('Choose register, twelve-nominal or case')
