"""Registered development isolation of terminal timing from learned actions."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
from replay_strong_checkpoint_v4 import HistoricalActor, MODEL, EXPECTED_MODEL, paths, PROTECTED
from strong_policy_v4 import load_strong_policy
from unified_training_v3 import Teacher, CANDIDATES
from terminal_completion_v4c import VERSION, install

OUT = ROOT / 'results' / VERSION
PILOT = ROOT / 'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
TEACHER = next(c for c in CANDIDATES if c['id'] == 'f100_k1_h12_d3')
SOURCES = sorted(set(PROTECTED + ['terminal_completion_v4c.py', 'replay_terminal_v4c.py',
    'runtime_no_target_override_v4b.py', 'replay_no_target_override_v4b.py',
    'strong_policy_v4.py', 'unified_training_v3.py', 'paired_learning.py']))


def register():
    file = OUT / 'registration.json'
    if not file.exists():
        write_json(file, dict(created_utc=utc(), scope='DEVELOPMENT ONLY', version=VERSION,
            source_sha256={p: sha(ROOT / 'src' / p) for p in SOURCES},
            model_sha256=dict(strong=EXPECTED_MODEL, pilot=sha(PILOT)),
            cases=paths(), methods=['strong', 'pilot', 'teacher'], dt_s=.001,
            short_scope='All six registered training/validation programs for every method.',
            dense_scope='Strong and pilot, irrespective of short-case outcomes.',
            original_scope='Strong and pilot only if their v4c dense execution succeeds.',
            teacher=TEACHER,
            change='Same v4b executor, but before all remaining gates are crossed, permit the original finite brake only if a cloned unchanged GateTracker validates its entire trace, every remaining gate, and the existing 1mm trigger endpoint distance. Trigger horizon remains 12mm.',
            unchanged='All learned actions/weights, reference, tolerance, all gates, interval geometry, kinematic vector limits, brake family, full completion and final rest.',
            retention='All registered successes and failures retained; no prefix duration counted as completion.'))
    reg = json.loads(file.read_text('utf-8'))
    for p, expected in reg['source_sha256'].items():
        if sha(ROOT / 'src' / p) != expected:
            raise ValueError('Registered source changed: ' + p)
    for key, path in [('strong', MODEL), ('pilot', PILOT)]:
        if sha(path) != reg['model_sha256'][key]:
            raise ValueError('Registered model changed: ' + key)
    for item in reg['cases']:
        if sha(ROOT / item['reference']) != item['reference_sha256']:
            raise ValueError('Registered geometry changed')
    return reg


def run(method, case):
    reg = register()
    row = next(item for item in reg['cases'] if item['id'] == case)
    if case == 'original':
        dense = json.loads((OUT / method / 'dense.json').read_text('utf-8'))
        if not dense['success']:
            raise ValueError('Registered long-run condition not satisfied')
    if method == 'teacher' and row['historical_split'] not in ('train', 'validation'):
        raise ValueError('Teacher registered for short development programs only')
    policy = (HistoricalActor() if method == 'strong' else
              load_strong_policy(PILOT) if method == 'pilot' else Teacher(TEACHER))
    install(tolerance=row['evaluation_tolerance_mm'])
    reference = np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1)
    output = OUT / method / (case + '.npz')
    result = rollout(policy, reference, output,
        fixed_d=TEACHER['preview'] if method == 'teacher' else None,
        dt=.001, tolerance=row['evaluation_tolerance_mm'],
        wall_s=1800. if case == 'original' else 180., runtime_version='v3')
    result.update(runtime_version=VERSION, scope='DEVELOPMENT ONLY', method=method,
        path=case, model_sha256=reg['model_sha256'].get(method), model_weights_updated=False,
        evaluation_configuration_updated=True, reference_sha256=row['reference_sha256'],
        registration_sha256=sha(OUT / 'registration.json'))
    write_json(output.with_suffix('.json'), result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--method', choices=['strong', 'pilot', 'teacher'])
    parser.add_argument('--case')
    parser.add_argument('--short-batch', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.register:
        print(json.dumps(register(), indent=2))
    elif args.short_batch:
        for item in register()['cases']:
            if item['historical_split'] in ('train', 'validation'):
                run(args.method, item['id'])
        if args.method != 'teacher':
            run(args.method, 'dense')
    else:
        run(args.method, args.case)
