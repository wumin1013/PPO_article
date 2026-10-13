"""Register/retain all v4d development checks, including observed v3 programs."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,sha,utc,write_json
from evaluate import rollout
from strong_policy_v4 import load_strong_policy
from replay_strong_checkpoint_v4 import paths,PROTECTED
from terminal_completion_v4d import install,VERSION

MODEL=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
EXPECTED_MODEL='188b02080069d7381a8ce9573ee936c89886e5b600e35f8e394e5f38f8256f94'
OUT=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d'
IMPORT=ROOT/'development_imports/observed_v3_final'
OLD=Path(r'D:\PPO_article_work\JNN_revision_20261011_v3/prospective/final_v3')
SOURCES=sorted(set(PROTECTED+['evaluate_strong_fit_pilot_v4d.py','runtime_no_target_override_v4b.py',
    'terminal_completion_v4d.py','replay_no_target_override_v4b.py','strong_policy_v4.py',
    'train_strong_fit_v4.py','unified_training_v3.py','paired_learning.py']))


def copy_inputs():
    source=OLD/'registration.json';target=IMPORT/'archived_v3_registration.json'
    IMPORT.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=sha(source):raise ValueError('Archived source registration mismatch')
    else:target.write_bytes(source.read_bytes())
    old=json.loads(source.read_text('utf-8'));rows=[]
    for item in old['paths']:
        original=OLD/item['file'];copied=IMPORT/item['file']
        if sha(original)!=item['sha256']:raise ValueError('Archived observed reference mismatch')
        if copied.exists():
            if sha(copied)!=item['sha256']:raise ValueError('Imported reference mismatch')
        else:copied.write_bytes(original.read_bytes())
        rows.append(dict(id=item['id'],reference=copied.relative_to(ROOT).as_posix(),
            reference_sha256=item['sha256'],group=item['group'],origin=item['origin'],
            scope='Already observed v3 program, DEVELOPMENT ONLY in v4',
            nominal_tolerance_mm=item['original_tolerance_mm'],
            evaluation_tolerance_mm=item['evaluation_tolerance_mm'],
            arc_chord_error_bound_mm=item['arc_chord_error_bound_mm'],
            exact_arc_primitives=item['exact_arc_primitives']))
    for item in paths():
        if item['id'] in ('original','dense'):
            item['scope']='Historical full development program';rows.append(item)
    return rows


def register():
    file=OUT/'registration.json'
    if not file.exists():
        rows=copy_inputs()
        write_json(file,dict(created_utc=utc(),scope='DEVELOPMENT ONLY',runtime_version=VERSION,
            seed=101,arm='adaptive',model_sha256=EXPECTED_MODEL,model=str(MODEL.relative_to(ROOT)),
            identity='Fresh trainable weights on historical static features/assets, 12000 supervised iterations; no fresh PPO. New executor version, never described as frozen-model direct transfer.',
            cases=rows,source_sha256={n:sha(ROOT/'src'/n) for n in SOURCES},
            imported_source_registry_sha256=sha(IMPORT/'archived_v3_registration.json'),
            resolutions_s=[.001,.0005],
            execution_order='All five observed v3 programs at 1 ms first; original at 1 ms may run in parallel. Inspect all five outcomes before any 0.5 ms phase; fine phase includes every one of the five, never only successes. Also dense at 0.5 ms and optional original at 0.5 ms.',
            inclusion='No route exclusion, success-only summary, shortened program, relaxed tolerance, or altered completion definition.',
            action_source='fresh_seed101_fitted_strong_structure_adaptive',
            executor_change='v4d verifies a candidate original terminal brake against every command interval and remaining ordered gate before installation; original brake generator and endpoint trigger threshold remain.',
            completion='Independent full-command-interval geometry, all ordered gates, planar vector limits, start rest, terminal brake and final rest; arc chord bound reserved.'))
    r=json.loads(file.read_text('utf-8'))
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Model changed')
    for name,expected in r['source_sha256'].items():
        if sha(ROOT/'src'/name)!=expected:raise ValueError('Registered source changed: '+name)
    for row in r['cases']:
        if sha(ROOT/row['reference'])!=row['reference_sha256']:raise ValueError('Reference changed')
    return r


def run(case,dt):
    r=register();row=next(x for x in r['cases'] if x['id']==case)
    if dt==.0005:
        missing=[x['id'] for x in r['cases'] if x['id'].startswith('final_') and not (OUT/(x['id']+'_dt1000us.json')).exists()]
        if missing:raise ValueError('All five 1 ms development cases must finish first: '+str(missing))
    setup=install(dt=dt,tolerance=row['evaluation_tolerance_mm'])
    policy=load_strong_policy(MODEL)[0]
    ref=np.loadtxt(ROOT/row['reference'],delimiter=',',skiprows=1)
    file=OUT/(case+f'_dt{round(dt*1e6)}us.npz')
    result=rollout(policy,ref,file,dt=dt,tolerance=row['evaluation_tolerance_mm'],
        wall_s=1800. if case=='original' else 240.,runtime_version='v3')
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Model changed during replay')
    result.update(scope='DEVELOPMENT ONLY',runtime_version=VERSION,executor_setup=setup,
        method='fresh_seed101_fitted_strong_structure_adaptive',model_sha256=EXPECTED_MODEL,
        seed=101,arm='adaptive',path=case,reference_sha256=row['reference_sha256'],
        nominal_tolerance_mm=row['nominal_tolerance_mm'],arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        registration_sha256=sha(OUT/'registration.json'),action_source=r['action_source'],
        interpretation='New-version development evaluation; already observed programs, not independent final tests.')
    write_json(file.with_suffix('.json'),result)
    print(json.dumps(dict(case=case,dt_s=dt,success=result['success'],time_s=result['complete_time_s'],
        prefix_s=result['elapsed_prefix_s'],reason=result['reason'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--register',action='store_true')
    p.add_argument('--case');p.add_argument('--dt',type=float,choices=[.001,.0005],default=.001)
    p.add_argument('--five-observed',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    if a.register:print(json.dumps(register(),indent=2))
    elif a.five_observed:
        for row in register()['cases']:
            if row['id'].startswith('final_'):run(row['id'],a.dt)
    elif a.case:run(a.case,a.dt)
    else:p.error('Select register, five-observed or case')
