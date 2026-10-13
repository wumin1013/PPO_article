"""Compare the fixed final PPO smoke model with its fitted parent on all cases."""
import json
from common import ROOT,sha,write_json

base=ROOT/'results/strong_ppo_v4/smoke_seed101/adaptive/evaluation_v4e'
reg=json.loads((base/'registration.json').read_text('utf-8'))
rows=[]
for item in reg['cases']:
    case=item['id'];current=base/(case+'_dt1000us.json')
    if case.startswith('final_') or case=='original':
        parent=ROOT/f'results/strong_fit_v4/pilot_seed101/adaptive/evaluation_v4d/{case}_dt1000us.json'
    else:parent=ROOT/f'results/v4d_verified_terminal_preview/pilot_loader_retry/pilot/{case}.json'
    value=json.loads(current.read_text('utf-8')) if current.exists() else None
    fitted=json.loads(parent.read_text('utf-8')) if parent.exists() else None
    if value is not None and fitted is not None:
        if value['reference_sha256']!=fitted['reference_sha256']:raise ValueError('Parent/program reference mismatch')
    rows.append(dict(path=case,current=value,parent=fitted,
        current_file=str(current.relative_to(ROOT)),parent_file=str(parent.relative_to(ROOT)),
        current_sha256=sha(current) if current.exists() else None,
        parent_sha256=sha(parent) if parent.exists() else None))
write_json(ROOT/'audit/strong_ppo_smoke_v4e.json',dict(scope='DEVELOPMENT ONLY',records=rows,
    selection='The final 2048-decision model was fixed before all evaluation; intermediate checkpoints were not evaluated here.',
    runtime_comparison='PPO v4e at 1 ms versus fitted parent v4d at 1 ms; v4e pins the already nominal 0.1 mm guide width. An independent nominal sine control proved byte-identical traces.',
    training_receipt='results/strong_ppo_v4/smoke_seed101/adaptive/receipt.json'))
lines=['# Fixed-final PPO smoke: complete development evaluation','',
    'The model inherited the same seed-101 fitted training stage, then completed 2,048 fresh PPO decisions (8 updates). The checkpoint was fixed before these evaluations. No intermediate checkpoint was selected by development outcome, and no independent final-test claim is made.',
    '', '| Program | Fitted parent complete / time (s) | PPO final complete / time (s) | PPO retained prefix (s) |',
    '|---|---|---|---:|']
for row in rows:
    def describe(r):
        if r is None:return 'pending/not run'
        return f"yes / {r['complete_time_s']:.3f}" if r['success'] else 'no / —'
    prefix='—' if row['current'] is None else f"{row['current']['elapsed_prefix_s']:.3f}"
    lines.append(f"| {row['path']} | {describe(row['parent'])} | {describe(row['current'])} | {prefix} |")
lines+=['', 'All cases, including all failed prefixes, remain in the table and raw archive. Completion requires every gate, every command interval, vector limits, initial and terminal rest, and the registered arc-error reservation. Prefix times are not completion times.',
    '', 'The fitted parent used v4d at 1 ms and this replay uses v4e at 1 ms. The physical guide width is 0.1 mm in both; v4e makes it explicit and independent of import order. The separate sine equality control is byte-identical. The training model and executor version identities remain distinct in the raw records.',
    '', '```powershell',"& 'D:\\Anaconda\\envs\\PPO\\python.exe' -B src\\summarize_strong_ppo_smoke_v4e.py",'```',
    '', 'Replay entry: `src/evaluate_strong_ppo_smoke_v4e.py`. All registration, JSON audit results and full NPZ trajectories are under `results/strong_ppo_v4/smoke_seed101/adaptive/evaluation_v4e/`.']
(ROOT/'audit/strong_ppo_smoke_v4e.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for row in rows:
    r=row['current'];print(row['path'],None if r is None else (r['success'],r['complete_time_s'],r['elapsed_prefix_s']))
