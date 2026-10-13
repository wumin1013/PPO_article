"""Generate registered development comparisons without suppressing failures."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,sha,write_json

base=ROOT/'results/v4b_no_target_override'
registration=json.loads((base/'registration.json').read_text('utf-8'))
legacy=Path(r'D:\PPO_article_work\JNN_revision_20261011_v3/results/unified_v3/teachers/f100_k1_h12_d3')
records=[]
for method in ('strong','teacher'):
    for row in registration['cases']:
        if method=='teacher' and row['historical_split'] not in ('train','validation'):continue
        file=base/method/(row['id']+'.json')
        before=(ROOT/'results/strong_checkpoint_v4/v3' if method=='strong' else legacy)/(row['id']+'.json')
        r=json.loads(file.read_text('utf-8')) if file.exists() else None
        old=json.loads(before.read_text('utf-8')) if before.exists() else None
        records.append(dict(method=method,path=row['id'],current=r,before_target_override_removed=old,
            result_sha256=sha(file) if file.exists() else None,before_result_sha256=sha(before) if before.exists() else None))
equality={}
for case in ('dense','original'):
    file=base/'strong'/(case+'.npz')
    if not file.exists():continue
    with np.load(file) as current,np.load(ROOT/f'results/strong_checkpoint_v4/v1/{case}.npz') as old:
        equality[case]={}
        for key in ('rows','observations','actions'):
            a=current[key];b=old[key]
            equality[case][key]=dict(shape=list(a.shape),same_shape=a.shape==b.shape,
                equal=bool(a.shape==b.shape and np.array_equal(a,b)),
                max_abs=float(abs(a-b).max()) if a.shape==b.shape else None)
            if key=='rows' and a.shape==b.shape:
                equality[case][key]['max_abs_by_quantity']={name:float(abs(a[:,lo:hi]-b[:,lo:hi]).max())
                    for name,lo,hi in [('position_mm',0,2),('velocity_mm_s',2,4),('acceleration_mm_s2',4,6),
                                      ('jerk_mm_s3',6,8),('desired_velocity_mm_s',8,10),('heading_rate_rad_s',10,11)]}
result=dict(scope='DEVELOPMENT ONLY',registration_sha256=sha(base/'registration.json'),all_planned_results=records,
    v1_complete_development_comparisons=equality)
write_json(ROOT/'audit/no_target_override_v4b.json',result)
lines=['# v4b: initialization witness and local blending, without target override','',
    'The original model remains frozen. The only difference from the registered v3 replay is the removal of `_ordered_target` target replacement through an explicitly registered process-local wrapper. Source-witness initialization, disjoint local blending, every ordered gate, the complete command-interval corridor check, vector constraints, rest-to-rest completion, and arc-error reservation remain active.',
    '', '| Method | Program | v3 complete / time (s) | v4b complete / time (s) | v4b retained prefix (s) |',
    '|---|---|---|---|---:|']
for item in records:
    a=item['before_target_override_removed'];b=item['current']
    describe=lambda r:'pending' if r is None else (f"yes / {r['complete_time_s']:.3f}" if r['success'] else 'no / —')
    prefix='—' if b is None else f"{b['elapsed_prefix_s']:.3f}"
    lines.append(f"| {item['method']} | {item['path']} | {describe(a)} | {describe(b)} | {prefix} |")
lines+=['', 'All failed prefixes are retained; they are not counted as completed programs. This table is not a final-test or generalization result.',
    '', 'The target override explains failures that disappear when it is removed, including the archived dense program. It does not explain every high-feed failure: the strong checkpoint still fails on three short programs, while the f100_k1_h12_d3 teacher still fails on the sine and Bezier programs. Those outcomes remain part of the development record.',
    '', 'The teacher uses feed = 100 / (1 + maximum tangent angle over 12 mm), zero steering, and 3 mm fixed preview. Its configuration and all source/reference hashes were fixed before these replays.',
    '', '## Complete trajectory comparison with historical v1','',
    '```json',json.dumps(equality,indent=2),'```',
    '', '## Reproduction','', '```powershell',
    "& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_no_target_override_v4b.py",'```',
    '', 'The replay entry is `src/replay_no_target_override_v4b.py`; its registration and all immutable raw records are under `results/v4b_no_target_override/`. Existing replay records cannot be overwritten.']
(ROOT/'audit/no_target_override_v4b.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for r in records:
    a=r['before_target_override_removed'];b=r['current']
    print(r['method'],r['path'],None if a is None else (a['success'],a['complete_time_s']),None if b is None else (b['success'],b['complete_time_s']))
