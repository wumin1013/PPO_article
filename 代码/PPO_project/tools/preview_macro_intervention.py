"""Validation-only intervention on the preview output of one frozen policy.

The same actor continues to select feed and steering from the actual state.
Replacing D is a closed-loop mechanism diagnostic, not an independently
trained fixed-preview baseline. Constants come from the pre-existing grid.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

import preview_macro_study as study


def run(output):
    torch.set_num_threads(1)
    study.prepare()
    source_run = study.OUT/'pilot/joint_tracking_adaptive_seed101'
    cfg = yaml.safe_load((source_run/'config.yaml').read_text(encoding='utf-8'))
    source = study.source_state()
    provenance = study.old.read(source_run/'provenance.json')
    if provenance['source_sha256'] != source['source_sha256'] or provenance['config_sha256'] != study.old.sha(source_run/'config.yaml'):
        raise ValueError('Frozen policy source/config changed')
    checkpoint = source_run/'checkpoint.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if study.old.read(source_run/'training_summary.json')['checkpoint_sha256'] != study.old.sha(checkpoint):
        raise ValueError('Frozen checkpoint changed')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep prior intervention; use a new output directory')
    files = dict(source['source_files'])
    files[Path(__file__).relative_to(study.old.ROOT).as_posix()] = study.old.sha(__file__)
    binding = dict(source_files=files, checkpoint_sha256=study.old.sha(checkpoint),
                   config_sha256=study.old.sha(source_run/'config.yaml'),
                   manifest_sha256=study.old.sha(study.old.DATA/'manifest.json'),
                   source_run=source_run.relative_to(study.old.ROOT).as_posix(),
                   command='python '+' '.join(sys.argv), seed=101,
                   intervention_distances_mm=[None, 2.3, 3.0], split='validation',
                   purpose='Frozen weights; replace only executed preview output. Feed and steering respond to intervened states. No retraining or superiority claim.',
                   test_access=False, machine_execution_authorized=False)
    study.old.write(output/'protocol.json', binding)
    with zipfile.ZipFile(output/'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(study.old.ROOT/name, name)
    rows = []
    for distance in binding['intervention_distances_mm']:
        label = 'learned' if distance is None else f'clamp_D{distance:g}'
        for path_id in [p['id'] for p in study.old.manifest()['paths'] if p['split'] == 'validation']:
            env = study.MacroEnvironment(copy.deepcopy(cfg), path_id)
            agent = study.make_agent(cfg, env)
            agent.actor.load_state_dict(payload['actor'])
            agent.actor.eval()
            actions = []
            started = time.perf_counter()
            while not env.done:
                action = agent.take_action(env.state, deterministic=True)
                requested = float(.8+3.8*np.clip(action[2], 0., 1.))
                if distance is not None:
                    action[2] = (distance-.8)/3.8
                actions.append(dict(decision=len(actions), learned_D_mm=requested,
                                    executed_D_mm=float(.8+3.8*np.clip(action[2], 0., 1.)),
                                    feed_u=float(action[1]), steering_u=float(action[0])))
                env.step(action)
            stem = f'{label}_{path_id}'
            study.old.save_trace(output/f'{stem}.npz', env.trace)
            study.old.write(output/f'{stem}_decisions.json', actions)
            row = dict(env.result, intervention=label, path_id=path_id,
                       evaluation_wall_s=time.perf_counter()-started,
                       trace_sha256=study.old.sha(output/f'{stem}.npz'))
            if distance is None:
                prior = study.old.read(source_run/'validation'/f'{path_id}.json')
                for key in ('success', 'elapsed_time_s', 'contour_error_max_mm', 'ordered_progress'):
                    if row[key] != prior[key]:
                        raise ValueError(f'Frozen-policy replay differs: {path_id}/{key}')
            study.old.write(output/f'{stem}.json', row)
            rows.append(row)
            print(f'{stem}: success={row["success"]}, T={row["elapsed_time_s"]:.3f}', flush=True)
    comparisons = []
    learned = {r['path_id']: r for r in rows if r['intervention'] == 'learned'}
    for label in ('clamp_D2.3', 'clamp_D3'):
        controls = {r['path_id']: r for r in rows if r['intervention'] == label}
        paired = [p for p in learned if learned[p]['success'] and controls[p]['success']]
        denominator = sum(controls[p]['elapsed_time_s'] for p in paired)
        comparisons.append(dict(intervention=label, learned_successes=sum(r['success'] for r in learned.values()),
                                intervention_successes=sum(r['success'] for r in controls.values()),
                                common_success_paths=paired,
                                learned_time_gain_percent=100*(1-sum(learned[p]['elapsed_time_s'] for p in paired)/denominator) if denominator else None))
    study.old.write(output/'summary.json', dict(rows=rows, comparisons=comparisons,
        conclusion_scope='Single frozen policy, three development paths; separates output modulation from fixed substitution only.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=study.OUT.parent/'macro_intervention')
    run(parser.parse_args().output)
