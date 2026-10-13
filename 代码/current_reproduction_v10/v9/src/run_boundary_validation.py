"""Validate the fixed boundary planner on all original development controls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
sys.path.insert(0,str(BASE/'src'))
from common import sha,write_json,utc
from refine_boundaries import refine
OUT=ROOT/'boundary_validation'


def register():
    assert not (OUT/'registration.json').exists()
    source=ROOT/'results/tuning/omega40'
    files=sorted(source.glob('*.json'));assert len(files)==12
    inputs=[]
    for path in files:
        record=json.loads(path.read_text('utf-8'))
        assert record['omega']==40. and sha(path.with_suffix('.npz'))==record['trace_sha256']
        inputs.append(dict(record=str(path),record_sha256=sha(path),trace=str(path.with_suffix('.npz')),
            trace_sha256=record['trace_sha256'],case=record['case'],method=record['method']))
    files=[Path(__file__),ROOT/'src/refine_boundaries.py',ROOT/'src/boundary_timewarp.py',
        ROOT/'terminal_research/terminal_candidate.py']
    write_json(OUT/'registration.json',dict(created_utc=utc(),inputs=inputs,
        source_sha256={str(p):sha(p) for p in files},
        adoption='No loss of original complete cells; no increase in complete times; preserve all originally failed planning results.',
        parameter_selection=False,shared_by=['JNN','FIXED'],new_training_updates=0,
        plan_generation='Unchanged v5 executor, omega40, same frozen original and Fixed models.',
        refinement='One fixed quintic time map with integer-tick horizon search, followed by original-state terminal candidate selection; full audit and fallback original complete plan.',
        all_attempts_retained=True,command_observations='Original observation/action arrays describe the planning pass only.'))


def run():
    reg=json.loads((OUT/'registration.json').read_text('utf-8'))
    for p,value in reg['source_sha256'].items():assert sha(p)==value,p
    summaries=[]
    for spec in reg['inputs']:
        record=json.loads(Path(spec['record']).read_text('utf-8'))
        assert sha(spec['record'])==spec['record_sha256'] and sha(spec['trace'])==spec['trace_sha256']
        with np.load(spec['trace']) as z:
            reference=z['reference'];positions=z['positions'];rows=z['rows']
            observations=z['observations'];actions=z['actions']
        target=OUT/'results'/Path(spec['trace']).name
        target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
        new,audit,evidence,prefixes=refine(reference,positions,rows,dt=record['dt_s'],
            tolerance=record['tolerance_mm'],completed=record['success'])
        np.savez_compressed(target,reference=reference,positions=new,
            planning_observations=observations,planning_actions=actions,
            **{f'candidate_prefix_{i}':p for i,p in enumerate(prefixes)})
        result=dict(spec,metrics=audit,evidence=evidence,source_complete_time_s=record['complete_time_s'],
            trace=str(target),trace_sha256=sha(target),model_sha256=record['model_sha256'],
            registration_sha256=sha(OUT/'registration.json'))
        write_json(target.with_suffix('.json'),result);summaries.append(result)
        print(spec['case'],spec['method'],audit['success'],record['complete_time_s'],audit['complete_time_s'],flush=True)
    passed=all((r['metrics']['success'] and r['metrics']['complete_time_s']<=r['source_complete_time_s']+1e-12)
        if r['source_complete_time_s'] is not None else not r['metrics']['success'] for r in summaries)
    write_json(OUT/'summary.json',dict(adoptable=passed,rows=summaries,source_selection=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'));args=p.parse_args()
    register() if args.mode=='register' else run()
