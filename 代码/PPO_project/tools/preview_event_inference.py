"""Serial actor timing on common training-path inputs after the event freeze."""
import argparse
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import yaml
import preview_event_study as study
from preview_event_nnc_transfer import require_freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=2000)
    parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.repeats<100:raise ValueError('At least 100 calls per model')
    require_freeze(study)
    destination=args.output or study.OUT/'serial_inference'
    if destination.exists():raise FileExistsError('Preserve previous timing samples')
    torch.set_num_threads(1)
    cfg=study.config_for('selected_fixed',11);path_id=cfg['revision']['training_path_ids'][0]
    env=study.EventPreviewEnvironment(cfg,path_id);states=[env.state]
    for _ in range(255):
        obs,_,done,_=env.step(np.array([0.,.05,(3.-.8)/3.8]));states.append(obs)
        if done:break
    states=np.asarray(states,np.float32);destination.mkdir(parents=True)
    np.save(destination/'common_training_states.npy',states)
    samples={};rows=[];frozen=study.read(study.OUT/'freeze.json')
    with torch.no_grad():
        for variant in study.FORMAL:
            for seed in study.SEEDS:
                key=f'formal/{variant}_seed{seed}';run=study.OUT/key
                local=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
                observation_dim=local['revision']['observation_dim']
                class Space:pass
                space=Space();space.observation_dim=observation_dim
                agent=study.make_agent(local,space)
                checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
                agent.actor.load_state_dict(checkpoint['actor']);agent.actor.eval()
                for tick in range(200):agent.actor(torch.as_tensor(states[tick%len(states),:observation_dim]).unsqueeze(0))[0][0].numpy()
                durations=[]
                for tick in range(args.repeats):
                    obs=states[tick%len(states),:observation_dim];began=time.perf_counter_ns()
                    agent.actor(torch.as_tensor(obs).unsqueeze(0))[0][0].numpy()
                    durations.append((time.perf_counter_ns()-began)/1000.)
                samples[f'{variant}_seed{seed}']=np.asarray(durations)
                rows.append(dict(variant=variant,seed=seed,observation_dim=observation_dim,calls=args.repeats,
                    median_us=float(np.median(durations)),p95_us=float(np.percentile(durations,95)),
                    checkpoint_sha256=frozen['models'][key],config_sha256=frozen['configs'][key]))
    np.savez_compressed(destination/'latencies.npz',**samples)
    (destination/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    study.write(destination/'summary.json',dict(rows=rows,command='python '+' '.join(sys.argv),
        scope='Serial CPU actor, tensor conversion and numpy output only. Excludes geometry/state extraction/KCM/environment, I/O, controller and servo latency. No real-time guarantee.',
        input_scope='Common designated training-path prefix; legacy actor takes the shared first 20 coordinates, corrected actors all 26. No test observations.',
        statistical_unit='Independent trained model for method summaries; within-model calls are timing samples only',
        path_id=path_id,torch_threads=1,torch_version=torch.__version__,logical_processors=os.cpu_count(),processor=platform.processor(),
        scheduling='Execution workflow requires starting after revision training/evaluation processes finish; this script does not independently inspect other processes and host activity is uncontrolled',
        warmup_calls=200,freeze_sha256=study.sha(study.OUT/'freeze.json'),source_sha256=study.source_state()['source_sha256'],
        states_sha256=study.sha(destination/'common_training_states.npy'),samples_sha256=study.sha(destination/'latencies.npz'),
        script_sha256=study.sha(__file__),gate_script_sha256=study.sha(Path(__file__).with_name('preview_event_nnc_transfer.py'))))
    print(destination/'summary.json')


if __name__=='__main__':main()
