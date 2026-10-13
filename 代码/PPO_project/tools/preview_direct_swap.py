"""Validation-only replacement of a trained policy's distance output.

This is a counterfactual diagnostic. It is never an independently trained fixed
baseline. The original initial observation is preserved; from the first action
onward, the replaced distance drives both guidance and the next observation.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

import preview_direct_research as direct


class ReplacedPreview:
    def __init__(self, env, distance_mm):
        if env.action_space_dim != 3 or not .8 <= distance_mm <= 4.6:
            raise ValueError('Requires an adaptive three-action policy and D in [0.8,4.6] mm')
        self.env = env
        self.distance_mm = float(distance_mm)

    def __getattr__(self, key):
        return getattr(self.env, key)

    def step(self, action):
        replaced = np.asarray(action, dtype=float).copy()
        if replaced.shape != (3,) or not np.isfinite(replaced).all():
            raise ValueError('Expected three finite policy actions')
        replaced[2] = (self.distance_mm - .8) / 3.8
        return self.env.step(replaced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['direct', 'direct-gain', 'direct-fine', 'direct-explore'], default='direct-explore')
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.study != 'direct':
        study = importlib.import_module('preview_' + args.study.replace('-', '_')).install()
    else:
        study = direct
    study.check_protocol()
    study.activate_backend()
    run = args.run.resolve()
    run.relative_to(study.OUT.resolve())
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    if not config['revision']['variant'].startswith('direct'):
        raise ValueError('Only adaptive policies have a preview output to replace')
    selection = json.loads((study.OUT / 'selection.json').read_text(encoding='utf-8'))
    distance_key = ('fixed_distance_no_anchor_mm' if config['revision']['motion_anchor_beta'] == 0.
                    and 'fixed_distance_no_anchor_mm' in selection else 'fixed_distance_mm')
    distance = float(selection[distance_key])
    destination = run / f'evaluation_validation_nominal_fixed_{distance:g}_diagnostic'
    protocol = dict(command='python ' + ' '.join(sys.argv), role='Output replacement only; no retraining',
        initial_preview_mm=config['revision']['initial_preview_mm'], executed_preview_mm=distance,
        selection_distance_key=distance_key,
        initial_observation='Unchanged from the trained policy; replacement starts at the first action',
        selection_sha256=study.backend.sha(study.OUT / 'selection.json'),
        checkpoint_sha256=study.backend.sha(run / 'checkpoint.pt'),
        script_sha256=study.backend.sha(__file__), study_source_sha256=study.source_state()['source_sha256'])
    protocol_file = destination / 'replacement_protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Preserve the previous replacement diagnostic')
    study.backend.write_json(protocol_file, protocol)
    (destination / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    original_make_env = study.backend.make_env

    def make_env(cfg, path_id, episode=0, reference=None):
        env, observation = original_make_env(cfg, path_id, episode, reference=reference)
        return ReplacedPreview(env, distance), observation

    study.backend.make_env = make_env
    try:
        study.backend.evaluate(run, 'validation', 'nominal', diagnostic_fixed=distance)
    finally:
        study.backend.make_env = original_make_env


if __name__ == '__main__':
    main()
