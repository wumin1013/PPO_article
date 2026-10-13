"""Summarize all registered replays, with historical loader/trace checks."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import json
import numpy as np
import torch
from common import ROOT,write_json,sha
from frozen_policy import FrozenPolicy

torch.set_num_threads(1)
base=ROOT/'results/strong_checkpoint_v4'
registration=json.loads((base/'registration.json').read_text('utf-8'))
model=ROOT/registration['model']
policy=FrozenPolicy(model)
loader=[]
for case in ('dense','original'):
    old_file=ROOT/f'baseline_records/supplementary_reproduction/records/J-NNC/{case}_1ms.npz'
    new_file=base/f'v1/{case}.npz'
    with np.load(old_file) as old,np.load(new_file) as new:
        predicted=np.stack([policy.mean(x).numpy() for x in old['observations']])
        arrays={}
        for source,destination in [('rows','rows'),('observations','observations'),('latents','actions')]:
            a=old[source];b=new[destination]
            same_shape=a.shape==b.shape
            arrays[f'{source}_vs_{destination}']=dict(historical_shape=list(a.shape),replay_shape=list(b.shape),
                equal=bool(same_shape and np.array_equal(a,b)),
                max_abs=float(np.max(abs(a-b))) if same_shape else None)
        loader.append(dict(case=case,historical_trace_sha256=sha(old_file),v1_trace_sha256=sha(new_file),
            normalized_policy_output_vs_historical_latents_scalar_max_abs=float(np.max(abs(predicted-old['latents']))),
            exact_scalar_output_match=bool(np.array_equal(predicted,old['latents'])),array_comparison=arrays,
            historical_actions_semantics='Decoded steering radians, feed mm/s, preview mm; compare raw policy output to historical latents.'))
records=[]
for runtime in ('v3','v1'):
    cases=registration['cases'] if runtime=='v3' else [r for r in registration['cases'] if r['id'] in ('original','dense')]
    for row in cases:
        file=base/runtime/(row['id']+'.json')
        result=json.loads(file.read_text('utf-8'))
        records.append(dict(runtime=runtime,path=row['id'],success=result['success'],
            complete_time_s=result['complete_time_s'],retained_prefix_time_s=result['elapsed_prefix_s'],
            reason=result['reason'],guide_interventions=result['guide_interventions'],
            feed=result['feed_summary'],json_sha256=sha(file),trace_sha256=result['trace_sha256']))
out=dict(model_sha256=sha(model),registration_sha256=sha(base/'registration.json'),
    scope='Development compatibility diagnosis; all registered cases retained.',
    loader_and_historical_replay=loader,all_registered_runs=records,
    first_divergence_report='audit/strong_first_divergence_v4.json')
write_json(ROOT/'audit/strong_checkpoint_compatibility_v4.json',out)
lines=['# Historical strong checkpoint compatibility audit (v4 development)','',
    f"Unchanged checkpoint SHA-256: `{sha(model)}`.",
    'This audit is a development/version compatibility diagnosis, not a new final-test claim. No checkpoint, protected replay source or executor source was changed for these controls.',
    '', '## All registered results','',
    '| Executor | Program | Complete | Completion time (s) | Retained prefix (s) | Guide interventions |',
    '|---|---|---:|---:|---:|---:|']
for r in records:
    lines.append(f"| {r['runtime']} | {r['path']} | {r['success']} | {r['complete_time_s'] if r['complete_time_s'] is not None else '—'} | {r['retained_prefix_time_s']:.3f} | {r['guide_interventions']} |")
lines+=['','Every v3 replay failed with corridor violation, incomplete terminal rest, and geometry/ordered-completion failure. Prefix durations are not completion times.',
    '','The unchanged historical executor completed both development programs at their archived times (444.177 s and 14.839 s), including complete ordered traversal, the continuous command-interval corridor audit, vector limits and final rest. The long-path independent acceptance allowance is 1.2480086195146 mm after subtracting the arc chord bound. The historical v1 executor itself remains native 1.25 mm; this explicit diagnostic is not a common-executor comparison.',
    '', '## Loader and trace correspondence','']
for r in loader:
    lines.append(f"- {r['case']}: scalar loader versus historical normalized latents maximum absolute error = {r['normalized_policy_output_vs_historical_latents_scalar_max_abs']:.12g}; full arrays: `{json.dumps(r['array_comparison'])}`.")
lines+=['','Historical `actions` are already decoded physical actions; historical `latents` are the normalized actor outputs. Comparing actor outputs to physical actions is invalid.',
    '', '## First material position/desired-command divergence','',
    'Read-only reconstruction is saved in `strong_first_divergence_v4.json`. At dense 0.151 s and long 34.668 s, the historical transition function and disjoint-local-transition function return identical targets with no active gate blend. The added `_ordered_target` rule replaces these targets, changing the desired command before the next actor decision.',
    '', 'The bare target ray and the steering-rotated target ray both miss the gate channel at these states, but the existing 12 ms heading filter yields an actual desired-heading ray that crosses inside the channel. Therefore checking only bare LOS, or only LOS plus steering, is not equivalent to checking the executed command direction. Dense preceding rows are exactly identical. The long preceding positions and desired commands agree to the reported thresholds; minor numerical jerk differences occur earlier and are separately retained.',
    '', 'No new guard was implemented in this audit. Any guard repair requires a new registered version and full checks; the original checkpoint and all failed v3 prefixes remain preserved.',
    '', '## Reproduction','', '```powershell',
    "& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\audit_strong_first_divergence_v4.py",
    "& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_strong_checkpoint_v4.py",'```',
    '', 'The registered replay command refuses to overwrite an existing run; its registration and all ten completed result files are under `results/strong_checkpoint_v4/`.']
(ROOT/'audit/strong_checkpoint_compatibility_v4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(loader,indent=2))
