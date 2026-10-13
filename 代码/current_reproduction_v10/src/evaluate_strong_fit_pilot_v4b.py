"""Registered full development evaluation of the fresh seed-101 fitted pilot."""
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
from replay_strong_checkpoint_v4 import paths,PROTECTED
from runtime_no_target_override_v4b import install,VERSION

MODEL=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
EXPECTED_MODEL='188b02080069d7381a8ce9573ee936c89886e5b600e35f8e394e5f38f8256f94'
OUT=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4b'
SOURCES=sorted(set(PROTECTED+['evaluate_strong_fit_pilot_v4b.py','runtime_no_target_override_v4b.py',
    'replay_no_target_override_v4b.py','strong_policy_v4.py','train_strong_fit_v4.py',
    'unified_training_v3.py','paired_learning.py']))


def register():
    file=OUT/'registration.json'
    if not file.exists():
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,
            seed=101,arm='adaptive',model_sha256=EXPECTED_MODEL,
            identity='Fresh trainable weights, historical static features/assets, 12000 supervised fitting iterations, no fresh PPO.',
            model=str(MODEL.relative_to(ROOT)),cases=paths(),dt_s=.001,
            source_sha256={name:sha(ROOT/'src'/name) for name in SOURCES},
            fit_registration_sha256=sha(MODEL.parent/'registration.json'),
            fit_receipt_sha256=sha(MODEL.parent/'receipt.json'),
            action_source='fresh_seed101_fitted_strong_structure_adaptive',
            output_rule='All eight cases, including all failures. No success-dependent launch or reporting. No final-test paths.',
            completion='Common complete ordered traversal, every command interval corridor, vector limits, initial rest, terminal brake and final rest, with arc chord bound deducted.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Pilot checkpoint changed')
    for name,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Protected evaluation source changed: '+name)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case):
    r=register();row=next(x for x in r['cases'] if x['id']==case)
    setup=install(dt=.001,tolerance=row['evaluation_tolerance_mm'])
    policy,saved=load_strong_policy(MODEL)
    reference=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+'.npz')
    result=rollout(policy,reference,file,dt=.001,tolerance=row['evaluation_tolerance_mm'],
        wall_s=1800. if case=='original' else 180.,runtime_version='v3')
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Pilot changed during evaluation')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='fresh_seed101_fitted_strong_structure_adaptive',model_sha256=EXPECTED_MODEL,
        seed=101,arm='adaptive',path=case,reference_sha256=row['reference_sha256'],
        nominal_tolerance_mm=row['nominal_tolerance_mm'],arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        registration_sha256=sha(OUT/'registration.json'),action_source=r['action_source'],
        interpretation='A fitted-model execution test, not the unchanged historical checkpoint and not an independent final test.')
    write_json(file.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case');p.add_argument('--short-dense',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.short_dense:
        for row in register()['cases']:
            if row['id']!='original':run(row['id'])
    elif a.case:run(a.case)
    else:p.error('Select register, short-dense or case')
