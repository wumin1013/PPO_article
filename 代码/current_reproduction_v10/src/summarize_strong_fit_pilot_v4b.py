"""Summarize all fresh-pilot development runs and verified baseline context."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,sha,write_json

base=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4b'
registered=json.loads((base/'registration.json').read_text('utf-8'))
baseline_root=Path(r'D:\PPO_article_work\JNN_revision_20261011_v3')
records=[]
for row in registered['cases']:
    file=base/(row['id']+'.json')
    r=json.loads(file.read_text('utf-8')) if file.exists() else None
    records.append(dict(path=row['id'],result=r,result_sha256=sha(file) if file.exists() else None))
comparisons=[]
for case in ('dense','original'):
    learned=next(r['result'] for r in records if r['path']==case)
    reference=np.loadtxt(ROOT/next(r['reference'] for r in registered['cases'] if r['id']==case),delimiter=',',skiprows=1)
    for method in ('S2R','SCMO'):
        file=baseline_root/f'results/baseline_v3/development_unified/{method}_{case}_dt1000us_scale1000.json'
        baseline=json.loads(file.read_text('utf-8'))
        trace=baseline_root/baseline['trace']
        with np.load(trace) as f:
            if not np.array_equal(f['reference'],reference):raise ValueError('Baseline reference mismatch')
        comparisons.append(dict(case=case,method=method,baseline_success=baseline['success'],
            baseline_time_s=baseline['complete_time_s'],baseline_tolerance_mm=baseline['tolerance_mm'],
            fresh_success=None if learned is None else learned['success'],
            fresh_time_s=None if learned is None else learned['complete_time_s'],
            reduction_percent=None if learned is None or not learned['success'] or not baseline['success'] else 100*(1-learned['complete_time_s']/baseline['complete_time_s']),
            baseline_json=str(file),baseline_json_sha256=sha(file),baseline_trace_sha256=sha(trace),
            reference_equal=True,
            context='Previously completed deterministic trajectory; same path and independent audit conditions. Development comparison only. S2R combines Sencer-style geometry with Ruckig straight-block timing; SCMO remains a project comparator.'))
out=dict(scope='DEVELOPMENT ONLY',model_sha256=registered['model_sha256'],all_cases=records,
    registered_cases=len(records),completed_records=sum(r['result'] is not None for r in records),
    successful_records=sum(r['result'] is not None and r['result']['success'] for r in records),
    baseline_context=comparisons,training_receipt='results/strong_fit_v4/pilot_seed101/adaptive/receipt.json',
    inference='Fresh weights fitted on existing development assets; no fresh PPO, independent seeds, or final test claimed.')
write_json(ROOT/'audit/strong_fit_pilot_v4b.json',out)
lines=['# Fresh seed-101 strong-structure fitted pilot: v4b development evaluation','',
    'This model has newly initialized trainable tensors and the historical feature form/assets, followed by 12,000 supervised fitting iterations. It is distinct from the unchanged historical checkpoint. Fitting error alone is not evidence of complete execution; the following records come from complete-path attempts with the common independent checks.',
    '', '| Program | Complete | Completion time (s) | Retained prefix (s) | Ordered gates | Max command deviation (mm) |',
    '|---|---:|---:|---:|---:|---:|']
for item in records:
    r=item['result']
    if r is None:
        lines.append(f"| {item['path']} | pending | — | — | — | — |")
        continue
    time='—' if r['complete_time_s'] is None else f"{r['complete_time_s']:.3f}"
    geo=r['geometry']
    lines.append(f"| {item['path']} | {r['success']} | {time} | {r['elapsed_prefix_s']:.3f} | {geo['gates_passed']}/{geo['gate_count']} | {r['max_polyline_deviation_mm']:.6f} |")
lines+=['', 'All registered attempts, including failures, are retained. Failed prefix time is not completion time. No final-test path was evaluated.',
    '', '| Development program | Existing deterministic method | Baseline time (s) | Fresh fitted pilot time (s) | Reduction (%) |',
    '|---|---|---:|---:|---:|']
for c in comparisons:
    time='—' if c['fresh_time_s'] is None else f"{c['fresh_time_s']:.3f}"
    reduction='—' if c['reduction_percent'] is None else f"{c['reduction_percent']:.4f}"
    lines.append(f"| {c['case']} | {c['method']} | {c['baseline_time_s']:.3f} | {time} | {reduction} |")
lines+=['', 'The baseline reference arrays were checked for exact equality with the registered programs. These are development comparisons against existing validated deterministic records, not independent test estimates. SCMO is a project comparator, while S2R uses documented Sencer-style geometry with Ruckig straight-block scheduling; no relabeling as a full reproduction is implied.',
    '', 'The executor is v4b: source-witness initialization and disjoint local blending without the v3 target replacement. Complete ordered traversal, every command interval, vector limits, initial/final rest and arc chord reservation are preserved.',
    '', '## Reproduction','', '```powershell',
    "& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_strong_fit_pilot_v4b.py",'```',
    '', 'Replay entry: `src/evaluate_strong_fit_pilot_v4b.py`. Raw registered results are under `results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4b/`.']
(ROOT/'audit/strong_fit_pilot_v4b.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(completed=out['completed_records'],successes=out['successful_records'],comparisons=comparisons),indent=2))
