"""Registered development replay of the unchanged historical strong checkpoint."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from evaluate import rollout
from frozen_policy import FrozenPolicy

OUT = ROOT / 'results' / 'strong_checkpoint_v4'
MODEL = ROOT / 'baseline_records/supplementary_reproduction/models/main.pt'
EXPECTED_MODEL = 'c16d45e38bab022ce8f289148aac4f3dcb3559f98b18bdfa580e23aea4a99764'
PROTECTED = ['replay_strong_checkpoint_v4.py', 'frozen_policy.py', 'evaluate.py',
             'metrics.py', 'common.py', 'runtime_bootstrap.py',
             'runtime_configuration.py', 'corridor_v3.py', 'execution_settings.py']


class HistoricalActor:
    """Interface only: preserve checkpoint features and normalized action means."""
    def __init__(self):
        self.frozen = FrozenPolicy(MODEL)

    def act(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('This compatibility replay uses deterministic means')
        return self.frozen.mean(observation).numpy(), None, None


def paths():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text('utf-8'))
    rows = []
    for split in ('train', 'validation'):
        for item in manifest['paths']:
            if item['split'] == split:
                rows.append(dict(id=item['id'], historical_split=split,
                    reference=f"data/{item['id']}.csv", nominal_tolerance_mm=1.25,
                    arc_chord_error_bound_mm=0., evaluation_tolerance_mm=1.25))
    for case in ('original', 'dense'):
        base = ROOT / 'baseline_records/supplementary_reproduction/cases' / case
        geometry = json.loads((base / 'geometry.json').read_text('utf-8'))
        bound = max([p.get('chord_error_bound_mm', 0.)
                     for p in geometry['primitives']] + [0.])
        reference = base / 'reference.csv'
        if sha(reference) != geometry['reference_sha256']:
            raise ValueError('Development geometry/reference hash mismatch')
        rows.append(dict(id=case, historical_split='development',
            reference=reference.relative_to(ROOT).as_posix(),
            geometry=(base / 'geometry.json').relative_to(ROOT).as_posix(),
            nominal_tolerance_mm=1.25, arc_chord_error_bound_mm=bound,
            evaluation_tolerance_mm=1.25-bound))
    for row in rows:
        row['reference_sha256'] = sha(ROOT / row['reference'])
        if 'geometry' in row:
            row['geometry_sha256'] = sha(ROOT / row['geometry'])
    return rows


def register():
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Historical checkpoint changed')
    file = OUT / 'registration.json'
    if file.exists():
        registered = json.loads(file.read_text('utf-8'))
    else:
        registered = dict(created_utc=utc(), scope='v4 development ONLY',
            model=MODEL.relative_to(ROOT).as_posix(), model_sha256=EXPECTED_MODEL,
            model_updated=False, seed=923202, trained_weights_unchanged=True,
            historical_checkpoint='matched_direction_initialization/seed_923202/three/fit_12000.pt',
            old_feature_expression='LocalizedJointPolicy: 96 base plus 192 joint compact-support features',
            primary_executor='v3', dt_s=.001, cases=paths(),
            source_sha256={p: sha(ROOT / 'src' / p) for p in PROTECTED},
            runtime_manifest_sha256=sha(ROOT / 'runtime/source_manifest.json'),
            runtime_v3_difference='v3 adds disjoint local transition supports, ordered-channel initialization witness and next-gate target/recapture safeguard; policy tensors unchanged',
            legacy_v1_scope='Only a conditional development control after changed primary original/dense outcomes; restores historical transition/guide settings and strict original initialization',
            legacy_v1_tolerance_note='v1 executor uses its unchanged 1.25 mm native corridor; independent acceptance still deducts the arc chord bound. Such controls are version diagnostics, not formal common-executor comparisons.',
            completion='Common independent vector, all-command-interval geometry, complete ordered traversal, start rest and final rest audit; all failures and prefixes retained',
            output_rule='Never overwrite a result; no selection by outcome; each specified case remains in the summary',
            old_development_times_s=dict(original=444.177, dense=14.839),
            legacy_trigger='Run both original and dense with v1 if either v3 development case fails or differs in completion time from the archived nominal time by more than 1e-9 s')
        write_json(file, registered)
    if registered['model_sha256'] != sha(MODEL):
        raise ValueError('Registered model mismatch')
    for p, expected in registered['source_sha256'].items():
        if sha(ROOT / 'src' / p) != expected:
            raise ValueError('Registered replay source changed: ' + p)
    for row in registered['cases']:
        if sha(ROOT / row['reference']) != row['reference_sha256']:
            raise ValueError('Registered reference changed')
    return registered


def run(case, runtime):
    torch.set_num_threads(1)
    registration = register()
    row = next(p for p in registration['cases'] if p['id'] == case)
    if runtime == 'v1':
        if case not in ('original', 'dense'):
            raise ValueError('Legacy controls are restricted to the two development routes')
        results = [json.loads((OUT / 'v3' / (p+'.json')).read_text('utf-8'))
                   for p in ('original', 'dense')]
        changed = any(not r['success'] or abs(r['complete_time_s']-
                      registration['old_development_times_s'][r['path']]) > 1e-9 for r in results)
        if not changed:
            raise ValueError('No registered reason for a legacy-executor control')
    reference = np.loadtxt(ROOT / row['reference'], delimiter=',', skiprows=1)
    output = OUT / runtime / (case+'.npz')
    result = rollout(HistoricalActor(), reference, output, dt=.001,
                     tolerance=row['evaluation_tolerance_mm'], wall_s=1800.,
                     runtime_version=runtime)
    if sha(MODEL) != EXPECTED_MODEL:
        raise ValueError('Model changed during evaluation')
    with np.load(output) as trace:
        records = trace['rows']
        active = records[:,12] < .5 if len(records) else np.zeros(0, dtype=bool)
        requests = np.linalg.norm(records[active,8:10], axis=1)
        positions = trace['positions']
        speeds = np.linalg.norm(np.diff(positions, axis=0)/.001, axis=1)
        feed = dict(policy_calls=len(trace['actions']), active_ticks=int(active.sum()),
            request_mean_mm_s=float(requests.mean()) if len(requests) else None,
            request_max_mm_s=float(requests.max()) if len(requests) else None,
            actual_all_mean_mm_s=float(speeds.mean()) if len(speeds) else None,
            actual_active_mean_mm_s=float(np.linalg.norm(records[active,2:4],axis=1).mean()) if active.any() else None,
            terminal_ticks=int((~active).sum()),
            statistical_unit='One complete or retained-prefix development trajectory; ticks and decisions are descriptive samples, not independent replicates')
    result.update(scope='v4 development ONLY', path=case, seed=923202,
        method='historical_strong_JNNC_unchanged', model_sha256=EXPECTED_MODEL,
        model_updated=False, policy_loader='src/frozen_policy.py::FrozenPolicy.mean',
        reference_sha256=row['reference_sha256'],
        nominal_tolerance_mm=row['nominal_tolerance_mm'],
        arc_chord_error_bound_mm=row['arc_chord_error_bound_mm'],
        registration_sha256=sha(OUT/'registration.json'), feed_summary=feed,
        action_source='unchanged_923202_historical_actor; terminal rows are the common executor brake')
    write_json(output.with_suffix('.json'), result)
    print(json.dumps(dict(case=case,runtime=runtime,success=result['success'],
        time_s=result.get('complete_time_s'),reason=result['reason'],feed=feed)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--runtime', choices=['v3','v1'], default='v3')
    args = parser.parse_args()
    if args.register:
        print(json.dumps(register(),indent=2))
    elif args.case:
        run(args.case,args.runtime)
    else:
        parser.error('Specify --register or --case')
