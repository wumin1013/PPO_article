"""Report every v4d development result and the paired resolution outcomes."""
import json
from common import ROOT,sha,write_json

base=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d'
reg=json.loads((base/'registration.json').read_text('utf-8'))
records=[]
for item in reg['cases']:
    for dt in (.001,.0005):
        path=base/(item['id']+f'_dt{round(dt*1e6)}us.json')
        if item['id']=='dense' and dt==.001:
            path=ROOT/'results/v4d_verified_terminal_preview/pilot_loader_retry/pilot/dense.json'
        result=json.loads(path.read_text('utf-8')) if path.exists() else None
        records.append(dict(path=item['id'],dt_s=dt,result=result,
            result_file=str(path.relative_to(ROOT)),result_sha256=sha(path) if path.exists() else None))
out=dict(scope='DEVELOPMENT ONLY',model_sha256=reg['model_sha256'],runtime_version=reg['runtime_version'],
    records=records,observed_v3_paths=[r['id'] for r in reg['cases'] if r['id'].startswith('final_')],
    resolution_note='The source-defined guide-transition width is 100 times the DT value at first module import (0.1 mm at 1 ms, 0.05 mm at 0.5 ms in fresh processes). These existing operational-resolution replays therefore should not be presented as an isolated ODE-integration convergence experiment; no causal explanation of the failures is inferred.',
    optional_original_half_ms_status='Not launched pending 1 ms result and diagnostic review; no completion is inferred.')
write_json(ROOT/'audit/strong_fit_pilot_v4d.json',out)
lines=['# Fresh seed-101 pilot with verified terminal preview (v4d)','',
    'All five previously observed v3 programs are treated as development data. The model is the independently identified fresh-weight, 12,000-iteration fitted pilot; it is not the unchanged historical checkpoint. The terminal rule uses the original bounded brake and accepts it only after a copied tracker verifies every command interval and every remaining ordered gate.',
    '', '| Program | Resolution (ms) | Complete | Completion time (s) | Retained prefix (s) | Ordered gates |',
    '|---|---:|---:|---:|---:|---:|']
for item in records:
    r=item['result']
    if r is None:
        lines.append(f"| {item['path']} | {item['dt_s']*1000:g} | not run/pending | — | — | — |")
        continue
    time='—' if r['complete_time_s'] is None else f"{r['complete_time_s']:.4f}"
    geometry=r['geometry']
    lines.append(f"| {item['path']} | {item['dt_s']*1000:g} | {r['success']} | {time} | {r['elapsed_prefix_s']:.4f} | {geometry['gates_passed']}/{geometry['gate_count']} |")
lines+=['', 'No failed prefix is counted as a completed trajectory. The raw results include the continuous command-interval geometry audit, vector limits, initial/final rest, arc error deduction, and all policy invocations.',
    '', 'The existing resolution change also changes the source-defined guide-transition width at import from 0.1 to 0.05 mm. This report therefore treats the pair as an operational-resolution sensitivity check, not a pure integration-convergence proof. It does not attribute observed failures to that width without a separately registered control.',
    '', 'These results do not establish complete-path reliability for this fresh fitted policy. Successful development trajectories may inform subsequent design, but neither unsuccessful programs nor the altered resolution outcomes are removed.',
    '', '## Reproduction','', '```powershell',
    "& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_strong_fit_pilot_v4d.py",'```',
    '', 'Registered replay: `src/evaluate_strong_fit_pilot_v4d.py`. Immutable result files and full traces: `results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/`.']
(ROOT/'audit/strong_fit_pilot_v4d.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for item in records:
    r=item['result'];print(item['path'],item['dt_s'],None if r is None else (r['success'],r['complete_time_s'],r['elapsed_prefix_s']))
