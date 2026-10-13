"""Report every second-stage final-checkpoint attempt, including fine dense."""
import json
from common import ROOT,sha,write_json

base=ROOT/'results/final_function_fit_v4/stage2_seed101/adaptive/evaluation_v4e'
reg=json.loads((base/'registration.json').read_text('utf-8'))
records=[]
for case,dt in [(x['id'],.001) for x in reg['cases']]+[('dense',.0005)]:
    file=base/(case+f'_dt{round(dt*1e6)}us.json')
    value=json.loads(file.read_text('utf-8')) if file.exists() else None
    records.append(dict(path=case,dt_s=dt,result=value,file=str(file.relative_to(ROOT)),sha256=sha(file) if file.exists() else None))
write_json(ROOT/'audit/stage2_fit_v4e.json',dict(scope='DEVELOPMENT ONLY',model_sha256=reg['model_sha256'],
    cumulative_offline_iterations=24000,new_ppo_updates=0,seed=101,records=records,
    interpretation='Same seed continued from its first fitting stage; not an independent seed, not the PPO branch, no checkpoint selection.'))
lines=['# Second fitting stage: fixed-final v4e development evaluation','',
    'This is the same fresh seed-101 fitting lineage continued for 12,000 further offline iterations (24,000 total). It is not the PPO branch or a new independent seed. The final checkpoint was fixed before every prescribed execution.',
    '', '| Program | dt (ms) | Complete | Completion time (s) | Retained prefix (s) | Ordered gates |',
    '|---|---:|---:|---:|---:|---:|']
for item in records:
    r=item['result']
    if r is None:
        lines.append(f"| {item['path']} | {item['dt_s']*1000:g} | pending/not run | — | — | — |")
        continue
    time='—' if r['complete_time_s'] is None else f"{r['complete_time_s']:.4f}"
    geo=r['geometry']
    lines.append(f"| {item['path']} | {item['dt_s']*1000:g} | {r['success']} | {time} | {r['elapsed_prefix_s']:.4f} | {geo['gates_passed']}/{geo['gate_count']} |")
lines+=['', 'The five programs labelled final_* were already observed in v3 and are development data here. All failed prefixes remain included; no failed prefix time is used as a completion time. The 0.5 ms dense check was run irrespective of the 1 ms outcome.',
    '', 'Every original ordered gate, continuous command interval, planar vector limit, start/final rest condition, and circular chord-error allowance is unchanged. The v4e physical guide width is 0.1 mm at both resolutions.',
    '', '```powershell',"& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_stage2_fit_v4e.py",'```',
    '', 'Replay entry: `src/evaluate_stage2_fit_v4e.py`. Full source registration, result JSONs and NPZ traces: `results/final_function_fit_v4/stage2_seed101/adaptive/evaluation_v4e/`.']
(ROOT/'audit/stage2_fit_v4e.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(expected=len(records),completed=sum(r['result'] is not None for r in records),successes=sum(r['result'] is not None and r['result']['success'] for r in records)),indent=2))
