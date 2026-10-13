"""Finite development check of a folded-target guard with frozen actors."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from filtered_target_v4g import install, VERSION
from replay_strong_checkpoint_v4 import HistoricalActor, MODEL, paths
from strong_policy_v4 import load_strong_policy
from evaluate import rollout

OUT = ROOT / 'results' / VERSION
PILOT = ROOT / 'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'


def register():
    path = OUT / 'registration.json'
    if not path.exists():
        cases = paths()
        for p in sorted((ROOT / 'development_imports/observed_v3_final').glob('final_*.csv')):
            cases.append(dict(id=p.stem, historical_split='observed_development',
                reference=str(p.relative_to(ROOT)), reference_sha256=sha(p), evaluation_tolerance_mm=1.25))
        sources = ['filtered_target_v4g.py', 'replay_filtered_target_v4g.py',
                   'runtime_resolution_fixed_v4e.py', 'terminal_completion_v4d.py',
                   'runtime_configuration.py', 'evaluate.py', 'metrics.py']
        write_json(path, dict(created_utc=utc(), scope='DEVELOPMENT ONLY', version=VERSION,
            cases=cases, source_sha256={s: sha(ROOT / 'src' / s) for s in sources},
            models=dict(strong=sha(MODEL), pilot=sha(PILOT)), dt_s=.001,
            pilot_scope='All six short, five previously observed programs, and original dense program. Original long only if dense succeeds.',
            strong_scope='Original dense; original long only if dense succeeds at historical 14.839 seconds.',
            change='Use exact next requested heading including held steering and 12ms filter for the v3 gate-channel ray test. A currently backward filtered heading is not grounds to replace a forward target. Keep v4e physical-width pin and fully checked finite terminal brake.',
            unchanged='Every gate and command interval, vector constraints, reference, nominal tolerance, weights, action map, final rest and acceptance.',
            retention='Preserve every success/failure and no outcome-dependent replacement.'))
    reg = json.loads(path.read_text('utf-8'))
    for source, expected in reg['source_sha256'].items():
        assert sha(ROOT / 'src' / source) == expected
    for case in reg['cases']:
        assert sha(ROOT / case['reference']) == case['reference_sha256']
    return reg


def run(method, case):
    reg = register()
    row = next(r for r in reg['cases'] if r['id'] == case)
    if case == 'original':
        dense = json.loads((OUT / method / 'dense.json').read_text('utf-8'))
        if not dense['success'] or (method == 'strong' and abs(dense['complete_time_s']-14.839)>1e-9):
            raise ValueError('Registered long condition not met')
    actor = HistoricalActor() if method == 'strong' else load_strong_policy(PILOT)[0]
    install(tolerance=row['evaluation_tolerance_mm'])
    result = rollout(actor, np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1),
        OUT / method / (case+'.npz'), runtime_version='v3', tolerance=row['evaluation_tolerance_mm'],
        wall_s=1800. if case == 'original' else 180.)
    result.update(scope='DEVELOPMENT ONLY', runtime_version=VERSION, path=case, method=method,
                  model_sha256=reg['models'][method], registration_sha256=sha(OUT / 'registration.json'))
    write_json(OUT / method / (case+'.json'), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['pilot', 'strong'], required=True)
    parser.add_argument('--case')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.case:
        run(args.method, args.case)
    elif args.method == 'strong':
        run('strong', 'dense')
    else:
        for row in register()['cases']:
            if row['id'] != 'original':
                run('pilot', row['id'])
