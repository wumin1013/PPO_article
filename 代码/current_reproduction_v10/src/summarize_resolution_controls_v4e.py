"""Summarize fixed-width controls, preserving all unsuccessful programs."""
import json
from common import ROOT,sha,write_json

fresh=ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4e'
old=ROOT/'results/strong_checkpoint_v4/v4e_resolution_control'
reg=json.loads((fresh/'registration.json').read_text('utf-8'))
records=[]
for row in reg['cases']:
    if row['id']=='original':continue
    dt=.001 if row['id']=='train_sine' else .0005
    file=fresh/(row['id']+f'_dt{round(dt*1e6)}us.json')
    value=json.loads(file.read_text('utf-8')) if file.exists() else None
    records.append(dict(model='fresh_fit_seed101',path=row['id'],dt_s=dt,result=value,
        result_file=str(file.relative_to(ROOT)),result_sha256=sha(file) if file.exists() else None))
for case in ('dense','original'):
    file=old/(case+'_dt500us.json');value=json.loads(file.read_text('utf-8')) if file.exists() else None
    records.append(dict(model='historical_strong_seed923202',path=case,dt_s=.0005,result=value,
        result_file=str(file.relative_to(ROOT)),result_sha256=sha(file) if file.exists() else None))
control=json.loads((fresh/'nominal_equality_control.json').read_text('utf-8'))
write_json(ROOT/'audit/resolution_controls_v4e.json',dict(scope='DEVELOPMENT ONLY',records=records,
    nominal_equality_control=control,guide_width_mm=.1,decision_period_s=.02,heading_filter_tau_s=.012))
lines=['# v4e fixed physical guide-width controls','',
    'The guide transition width is fixed at 0.1 mm for both integration steps. The 20 ms policy decision period, 12 ms heading filter, all ordered gates, command-interval geometry, planar limits, and verified terminal brake are unchanged.',
    '', 'The fresh-model 1 ms sine control is byte-identical to its v4d trace, including observations and actions. Thus the fixed-width wrapper preserves this nominal-resolution control exactly.',
    '', '| Model | Program | dt (ms) | Complete | Completion (s) | Retained prefix (s) |',
    '|---|---|---:|---:|---:|---:|']
for item in records:
    r=item['result']
    if r is None:
        lines.append(f"| {item['model']} | {item['path']} | {item['dt_s']*1000:g} | pending/not run | — | — |")
    else:
        complete='—' if r['complete_time_s'] is None else f"{r['complete_time_s']:.4f}"
        lines.append(f"| {item['model']} | {item['path']} | {item['dt_s']*1000:g} | {r['success']} | {complete} | {r['elapsed_prefix_s']:.4f} |")
lines+=['', 'Fixing the width does not restore the fresh fitted model on every fine-resolution program. The unchanged historical model completing dense at 0.5 ms shows that this failure is not universal to that program/executor. This control alone does not identify a single causal source of the fresh model’s sensitivity.',
    '', 'Every unsuccessful attempt is retained and none of its prefix durations is counted as completion. The fresh original 0.5 ms replay was not launched because its required dense control failed; the historical original 0.5 ms replay is conditional on its own successful dense control.',
    '', '```powershell',"& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_resolution_controls_v4e.py",'```']
(ROOT/'audit/resolution_controls_v4e.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for item in records:
    r=item['result'];print(item['model'],item['path'],None if r is None else (r['success'],r['complete_time_s']))
