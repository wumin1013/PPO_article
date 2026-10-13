"""Replace only D from a saved adaptive actor on every development path.

The actor still sees actual intervened states and chooses feed. These closed
loop interventions are mechanism diagnostics, never independently trained
fixed-distance baselines. The initial D and all weights remain identical.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

DISTANCES = (None, 1.2, 3.0, 4.0)
RUNNERS = {'local': 'preview_local_corner_study',
           'full_batch': 'preview_full_batch_study',
           'target_tracking': 'preview_target_tracking_study'}


def replace_preview(action, distance):
    """Retain original latent feed/steering; replace executed preview only."""
    result = np.asarray(action, dtype=float).copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Finite three-vector required')
    if distance not in DISTANCES:
        raise ValueError('Use the registered short/medium/long grid')
    if distance is not None:
        result[2] = (distance - .8) / 3.8
    return result


def compare(rows):
    learned = {r['path_id']: r for r in rows if r['intervention'] == 'learned'}
    comparisons = []
    for distance in DISTANCES[1:]:
        label = f'fixed_D{distance:g}'
        control = {r['path_id']: r for r in rows if r['intervention'] == label}
        if set(learned) != set(control):
            raise ValueError('Every intervention must include the same paths')
        paired = [p for p in learned if learned[p]['success'] and control[p]['success']]
        denominator = sum(control[p]['elapsed_time_s'] for p in paired)
        comparisons.append(dict(intervention=label,
            learned_successes=sum(r['success'] for r in learned.values()),
            intervention_successes=sum(r['success'] for r in control.values()),
            path_count=len(learned), common_success_paths=paired,
            learned_time_gain_percent=(100 * (1 - sum(learned[p]['elapsed_time_s']
                for p in paired) / denominator)) if denominator else None))
    return comparisons


def run(runner_name, source_run, output):
    torch.set_num_threads(1)
    runner = importlib.import_module(RUNNERS[runner_name])
    installed = runner.install()
    core = getattr(installed, 'core', installed)
    protocol = runner.prepare()
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    cfg = yaml.safe_load((source_run / 'config.yaml').read_text(encoding='utf-8'))
    provenance = core.old.read(source_run / 'provenance.json')
    source = protocol['source']
    checkpoint = source_run / 'checkpoint.pt'
    if not cfg['revision']['adaptive_preview']:
        raise ValueError('A saved adaptive actor is required')
    if (provenance['source_sha256'] != source['source_sha256'] or
            provenance['config_sha256'] != core.old.sha(source_run / 'config.yaml') or
            core.old.read(source_run / 'training_summary.json')['checkpoint_sha256'] != core.old.sha(checkpoint)):
        raise ValueError('Saved source/config/checkpoint binding changed')
    validation = core.old.read(source_run / 'validation/summary.json')
    ids = [p['id'] for p in core.old.manifest()['paths'] if p['split'] == 'validation']
    original = {r['path_id']: r for r in validation['rows']}
    if set(ids) != set(original) or len(ids) != len(validation['rows']):
        raise ValueError('Complete prior validation required')
    if output.exists():
        raise FileExistsError('Preserve earlier intervention; choose a new directory')
    files = dict(source['source_files'])
    files[Path(__file__).relative_to(core.old.ROOT).as_posix()] = core.old.sha(__file__)
    binding = dict(version='continuation-preview-intervention-v1',
        source_files=files, model_source_sha256=source['source_sha256'],
        checkpoint_sha256=core.old.sha(checkpoint),
        config_sha256=core.old.sha(source_run / 'config.yaml'),
        validation_sha256=core.old.sha(source_run / 'validation/summary.json'),
        manifest_sha256=core.old.sha(core.old.DATA / 'manifest.json'),
        source_run=source_run.relative_to(core.old.ROOT).as_posix(),
        distances_mm=list(DISTANCES), initial_preview='Identical saved configuration; first policy output is intervened',
        split='validation', seed=cfg['seed'], runner=runner_name,
        command='python ' + ' '.join(sys.argv), test_access=False,
        purpose='Same actor with replaced D; feed responds to actual states; no retraining',
        machine_execution_authorized=False)
    core.old.write(output / 'protocol.json', binding)
    with zipfile.ZipFile(output / 'source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(core.old.ROOT / name, name)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    rows = []
    for distance in DISTANCES:
        label = 'learned' if distance is None else f'fixed_D{distance:g}'
        for path_id in ids:
            env = core.MacroEnvironment(copy.deepcopy(cfg), path_id)
            agent = core.make_agent(cfg, env)
            agent.actor.load_state_dict(payload['actor'])
            agent.actor.eval()
            decisions = []
            started = time.perf_counter()
            while not env.done:
                raw_action = agent.take_action(env.state, deterministic=True)
                action = replace_preview(raw_action, distance)
                decisions.append(dict(decision=len(decisions),
                    timestamp_s=env.trace[-1]['timestamp_s'],
                    requested_D_mm=float(.8 + 3.8 * np.clip(raw_action[2], 0., 1.)),
                    executed_D_mm=float(.8 + 3.8 * np.clip(action[2], 0., 1.)),
                    feed_logit=float(action[1])))
                env.step(action)
            stem = f'{label}_{path_id}'
            core.old.save_trace(output / f'{stem}.npz', env.trace)
            core.old.write(output / f'{stem}_decisions.json', decisions)
            row = dict(env.result, intervention=label, path_id=path_id,
                evaluation_wall_s=time.perf_counter() - started,
                checkpoint_sha256=binding['checkpoint_sha256'],
                trace_sha256=core.old.sha(output / f'{stem}.npz'),
                decisions_sha256=core.old.sha(output / f'{stem}_decisions.json'))
            if distance is None:
                for key in ('success', 'elapsed_time_s', 'ordered_progress', 'contour_error_max_mm'):
                    if row[key] != original[path_id][key]:
                        raise ValueError(f'Unmodified actor replay differs: {path_id}/{key}')
            core.old.write(output / f'{stem}.json', row)
            rows.append(row)
            print(f'{stem}: success={row["success"]}, T={row["elapsed_time_s"]:.3f}', flush=True)
    summary = dict(rows=rows, comparisons=compare(rows),
        scope='One saved adaptive actor on development paths; intervention, not independently trained fixed baseline')
    core.old.write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', choices=RUNNERS, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.runner, args.run, args.output)
