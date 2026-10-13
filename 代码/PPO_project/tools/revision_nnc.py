"""Supplementary NNC reruns using the shared PPO runner and explicit config.

This is the repository's no-KCM NNC control, not a new external baseline.
The separate freeze does not alter the primary three-method test freeze.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import revision_experiment as study

BASE_CONFIG = study.config_for
BASE_SOURCE = study.source_state
PRIMARY_OUT = study.OUT
study.OUT = PRIMARY_OUT / 'nnc_auxiliary'


def nnc_config(variant, seed):
    cfg = BASE_CONFIG('fixed_short', seed)
    cfg['revision']['variant'] = 'nnc'
    cfg['environment']['lookahead_obs_enabled'] = False
    cfg['environment']['lookahead_points'] = 0
    cfg['experiment']['enable_kcm'] = False
    cfg['reward_weights']['cornerness']['enabled'] = False
    cfg['reward_weights']['cornerness']['reference_distance'] = None
    cfg['reward_weights']['cornerness']['keep_legacy_smooth'] = True
    cfg['reward_weights']['lookahead']['enabled'] = False
    cfg['reward_weights']['lookahead']['ref_in_control'] = False
    cfg['reward_weights']['lookahead_reward']['enabled'] = False
    cfg['revision']['comparison_role'] = 'supplementary repository NNC retraining; simulation only, no KCM'
    return cfg


def nnc_source():
    state = BASE_SOURCE()
    state['source_files'][str(Path(__file__).relative_to(study.ROOT)).replace('\\', '/')] = study.sha(__file__)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


study.config_for = nnc_config
study.source_state = nnc_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['train', 'freeze', 'evaluate'])
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--split', choices=['validation', 'test', 'regression'], default='validation')
    args = parser.parse_args()
    seeds = (11, 22, 33, 44, 55)
    if args.phase == 'train':
        study.train('nnc', args.seed, 'runs', 8192)
    elif args.phase == 'freeze':
        models, configs = {}, {}
        for seed in seeds:
            run = study.OUT / 'runs' / f'nnc_seed{seed}'
            summary = json.loads((run / 'training_summary.json').read_text(encoding='utf-8'))
            if summary['steps'] != 8192: raise ValueError('Incomplete training budget')
            key = str(run.relative_to(study.OUT)).replace('\\', '/')
            models[key] = study.sha(run / 'checkpoint.pt'); configs[key] = study.sha(run / 'config.yaml')
        output = study.OUT / 'freeze.json'
        if output.exists(): raise FileExistsError(output)
        study.write_json(output, dict(models=models, configs=configs, source_sha256=nnc_source()['source_sha256'],
                         path_manifest_sha256=study.sha(study.DEFAULT_DATA_DIR / 'manifest.json'),
                         source_protocol=str(PRIMARY_OUT / 'protocol.json'), test_used_for_selection=False,
                         timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                         note='No hyperparameter search; same 8192-step budget and training seeds, existing NNC definition'))
    else:
        run = study.OUT / 'runs' / f'nnc_seed{args.seed}'
        study.evaluate(run, args.split)


if __name__ == '__main__': main()
