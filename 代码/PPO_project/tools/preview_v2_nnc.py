"""Matched-budget retraining of the repository's existing no-KCM NNC control."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import preview_research_v2 as revision

OUT = revision.OUT / 'nnc_auxiliary'
STEPS = 98304


def config_for(variant, seed):
    cfg = revision.config_for('pretrain_selected', seed)
    cfg['revision'].update(variant='nnc', comparison_role='existing repository NNC, independently retrained; simulation only')
    cfg['environment'].update(lookahead_obs_enabled=False, lookahead_points=0)
    cfg['experiment']['enable_kcm'] = False
    cfg['reward_weights']['cornerness'].update(enabled=False, reference_distance=None, keep_legacy_smooth=True)
    cfg['reward_weights']['lookahead'].update(enabled=False, ref_in_control=False)
    cfg['reward_weights']['lookahead_reward']['enabled'] = False
    return cfg


def source_state():
    state = revision.source_state()
    state['source_files'][str(Path(__file__).relative_to(revision.ROOT)).replace('\\', '/')] = revision.backend.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'freeze', 'evaluate'])
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--split', choices=['validation', 'test', 'regression'], default='validation')
    args = parser.parse_args()
    revision.activate_backend()
    backend = revision.backend
    backend.OUT = OUT
    backend.config_for = config_for
    backend.source_state = source_state
    if args.phase == 'train':
        backend.train('nnc', args.seed, 'runs', STEPS)
    elif args.phase == 'freeze':
        models, configs = {}, {}
        for seed in revision.SEEDS:
            run = OUT / 'runs' / f'nnc_seed{seed}'
            summary = json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
            if summary['steps'] != STEPS: raise ValueError('Incomplete matched NNC training budget')
            key = str(run.relative_to(OUT)).replace('\\','/')
            models[key] = backend.sha(run/'checkpoint.pt')
            configs[key] = backend.sha(run/'config.yaml')
        target = OUT/'freeze.json'
        if target.exists(): raise FileExistsError(target)
        backend.write_json(target, dict(models=models, configs=configs,
            source_sha256=source_state()['source_sha256'],
            path_manifest_sha256=backend.sha(revision.DATA/'manifest.json'),
            source_protocol=str(revision.OUT/'protocol.json'),
            training_steps_per_seed=STEPS, pretraining_steps=0,
            timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            test_used_for_selection=False, note='Original NNC definition; same shared V2 objective and total learning-step budget'))
    else:
        backend.evaluate(OUT/'runs'/f'nnc_seed{args.seed}', args.split)


if __name__ == '__main__': main()
