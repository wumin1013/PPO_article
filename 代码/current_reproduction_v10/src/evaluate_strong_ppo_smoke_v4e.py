"""Full development execution of the fixed final 2048-decision PPO smoke model."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from evaluate import rollout
from strong_ppo_v4 import load_strong_ppo
from runtime_resolution_fixed_v4e import install,VERSION
from replay_strong_checkpoint_v4 import paths,PROTECTED

MODEL=ROOT/'results/strong_ppo_v4/smoke_seed101/adaptive/final.pt'
EXPECTED_MODEL='1a393dc79dcb1f8be5e74cb22034dac78c53556b4568d954a0b17be8c212aadd'
OUT=MODEL.parent/'evaluation_v4e'
SOURCES=sorted(set(PROTECTED+['evaluate_strong_ppo_smoke_v4e.py','strong_ppo_v4.py','strong_policy_v4.py',
    'runtime_resolution_fixed_v4e.py','terminal_completion_v4d.py','runtime_no_target_override_v4b.py',
    'replay_no_target_override_v4b.py','unified_training_v3.py','paired_learning.py']))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        all_rows=paths();rows=[x for x in all_rows if x['historical_split'] in ('train','validation')]
        old=json.loads((ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/registration.json').read_text('utf-8'))
        rows.extend(x for x in old['cases'] if x['id'].startswith('final_'))
        rows.extend(x for x in all_rows if x['id'] in ('dense','original'))
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,dt_s=.001,
            model_sha256=EXPECTED_MODEL,seed=101,arm='adaptive',cases=rows,
            source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
            ppo_registration_sha256=sha(MODEL.parent/'registration.json'),
            ppo_receipt_sha256=sha(MODEL.parent/'receipt.json'),
            selection='Only the predeclared final 2048-decision model; no checkpoint selection by evaluation outcomes.',
            order='All six short training/validation programs, all five already observed v3 development programs, dense; original only after full dense succeeds.',
            identity='Fresh-weight seed101 12000-fit parent followed by 2048 fresh PPO decisions; not the old historical checkpoint.',
            complete='Every original gate and every command interval, vector limits, start/final rest and arc chord error deduction remain unchanged.',
            retention='Keep all full and failed-prefix records; failed prefix durations are not completion times.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('PPO final checkpoint changed')
    for n,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/n)!=expected:raise ValueError('Registered source changed: '+n)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case):
    r=register();row=next(x for x in r['cases'] if x['id']==case)
    if case=='original':
        dense=json.loads((OUT/'dense_dt1000us.json').read_text('utf-8'))
        if not dense['success']:raise ValueError('Dense failed; original is not launched')
    setup=install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
    policy,saved=load_strong_ppo(MODEL)
    if (saved['seed'],saved['arm'])!=(101,'adaptive'):raise ValueError('PPO identity mismatch')
    ref=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+'_dt1000us.npz')
    result=rollout(policy,ref,file,dt=.001,tolerance=row['evaluation_tolerance_mm'],
        runtime_version='v3',wall_s=1800. if case=='original' else 240.)
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('PPO final changed during replay')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='fresh_seed101_strong_ppo2048_final',path=case,seed=101,arm='adaptive',model_sha256=EXPECTED_MODEL,
        reference_sha256=row['reference_sha256'],nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],registration_sha256=sha(OUT/'registration.json'),
        action_source='fixed_final_2048_fresh_PPO_actor')
    write_json(file.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case');p.add_argument('--short-all',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.short_all:
        for row in register()['cases']:
            if row['id']!='original':run(row['id'])
    elif a.case:run(a.case)
    else:p.error('Choose register, short-all or case')
