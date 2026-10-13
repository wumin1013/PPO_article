"""Serial CPU inference timing of every frozen final policy on shared inputs.

Run after revision training/evaluation workers finish. These latency samples
describe computation; they are not independent policy-training replicates.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['direct-fine', 'direct-explore'], default='direct-explore')
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--output', type=Path, help='New directory for an additional timing run')
    args = parser.parse_args()
    if args.repeats < 100:
        raise ValueError('Use at least 100 timed calls per independently trained policy')
    module = importlib.import_module('preview_' + args.study.replace('-', '_'))
    study = module.install()
    study.check_protocol()
    frozen = json.loads((study.OUT / 'freeze.json').read_text(encoding='utf-8'))
    protocol = json.loads((study.OUT / 'protocol.json').read_text(encoding='utf-8'))
    expected = {f'formal/{method}_seed{seed}' for method in protocol['formal_variants'] for seed in protocol['formal_seeds']}
    if (set(frozen['models']) != expected or set(frozen['configs']) != expected
            or frozen['source_sha256'] != study.source_state()['source_sha256']
            or frozen['protocol_sha256'] != study.backend.sha(study.OUT / 'protocol.json')
            or frozen['selection_sha256'] != study.backend.sha(study.OUT / 'selection.json')
            or frozen['path_manifest_sha256'] != study.backend.sha(study.DATA / 'manifest.json')):
        raise ValueError('Complete current-source model/configuration freeze is required')
    configs = {}
    for key in sorted(expected):
        run = study.OUT / key
        if (frozen['models'][key] != study.backend.sha(run / 'checkpoint.pt')
                or frozen['configs'][key] != study.backend.sha(run / 'config.yaml')):
            raise ValueError(f'Frozen model or configuration changed: {key}')
        configs[key] = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    torch.set_num_threads(1)
    out = args.output if args.output is not None else study.OUT / 'inference_benchmark'
    if out.exists():
        raise FileExistsError('Preserve the saved timing run; choose a new --output directory')
    # Identical representative training-path inputs for every network. No test
    # path or test performance is accessed to construct this timing workload.
    common_cfg = next(iter(configs.values()))
    path_id = common_cfg['revision']['training_path_ids'][0]
    env, observation = study.make_env(common_cfg, path_id)
    states = [observation]
    for _ in range(255):
        observation, _, done, _ = env.step(np.array([0., .05, (3. - .8) / 3.8])[:env.action_space_dim])
        states.append(observation)
        if done:
            break
    states = np.asarray(states, dtype=np.float32)
    out.mkdir(parents=True)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    np.save(out / 'shared_training_states.npy', states)
    results, samples = [], {}
    with torch.no_grad():
        for key, cfg in configs.items():
            agent = study.make_agent(cfg, None)
            checkpoint = torch.load(study.OUT / key / 'checkpoint.pt', map_location='cpu', weights_only=False)
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.actor.eval()
            for tick in range(200):
                agent.actor(torch.as_tensor(states[tick % len(states)]).unsqueeze(0))[0][0].numpy()
            durations = []
            for tick in range(args.repeats):
                observation = states[tick % len(states)]
                began = time.perf_counter_ns()
                agent.actor(torch.as_tensor(observation).unsqueeze(0))[0][0].numpy()
                durations.append((time.perf_counter_ns() - began) / 1000.)
            samples[key.replace('/', '__')] = np.asarray(durations)
            results.append(dict(run=key, method=cfg['revision']['variant'], seed=cfg['seed'],
                checkpoint_sha256=frozen['models'][key], config_sha256=frozen['configs'][key],
                calls=len(durations), median_us=float(np.median(durations)),
                p95_us=float(np.percentile(durations,95)), mean_us=float(np.mean(durations))))
    np.savez_compressed(out / 'latency_samples.npz', **samples)
    summary = dict(command='python ' + ' '.join(sys.argv),
        rows=results, scope='Serial CPU actor inference including tensor conversion and numpy output; excludes geometry, KCM, logging, and controller/servo latency',
        scheduling='Run after revision training/evaluation workers finish; other host activity is not controlled',
        state_source='Shared constant-action prefix of designated training path', path_id=path_id,
        state_count=len(states), torch_threads=1, device='cpu', torch_version=torch.__version__,
        cpu_processor=platform.processor(), logical_processors=os.cpu_count(),
        completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        warmup_calls_per_policy=200, timing_clock='perf_counter_ns',
        statistical_unit='Method-level variation is across independently trained seeds; within-model latency samples describe timing only',
        freeze_sha256=study.backend.sha(study.OUT / 'freeze.json'), script_sha256=study.backend.sha(__file__),
        states_sha256=study.backend.sha(out / 'shared_training_states.npy'),
        raw_sha256=study.backend.sha(out / 'latency_samples.npz'))
    study.backend.write_json(out / 'summary.json', summary)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
