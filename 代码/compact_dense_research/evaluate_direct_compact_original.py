"""Complete original-case audit of the14-input compressed policy, unchanged weights."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,HIST,sha,write,verify_bound_inputs

CASE=BASE/'paper_evidence_review/natural_closed_case'
MODEL=ROOT/'direct14_saved_motion_fit_seed922302_bound/fit_0600.pt'
STAGE=ROOT/'direct14_saved_motion_fit_seed922302_original_full'


def worker():
    import numpy as np
    import torch
    from compact_control import CompactPolicy
    from direct_compact_runtime import DirectCompactRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    verify_bound_inputs();torch.set_num_threads(1)
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    policy=CompactPolicy(14,3,922302)
    policy.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['policy'])
    env=DirectCompactRuntime(reference)
    start=time.perf_counter();result=evaluate(policy,env,STAGE/'full_trace.npz',start+1320.)
    data=np.load(STAGE/'full_trace.npz');rows=data['rows']
    if len(rows):
        result['analytic']=analytic_command_audit(reference[0],rows,geometry)
        if not result['analytic']['analytic_corridor_pass']:result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
    result.update(checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),observation_dimension=14,
        policy_memory_scalars=0,inherited_controller_at_execution=False,goal_complete=False,method_frozen=False,
        ppo_updates=0,training_kind='Offline compression from saved developed teacher motion',online_prediction_search=False)
    write(STAGE/'full_result.json',result);write(STAGE/'gate_events.json',env.tracker.events)
    if result['success']:
        m=metrics(reference,rows,geometry);write(STAGE/'full_metrics.json',m)
        old=json.loads((HIST/'full_metrics.json').read_text(encoding='utf-8'))
        delta=[x['speed_deficit_mm']-y['speed_deficit_mm'] for x,y in zip(m['corners'],old['corners'])]
        write(STAGE/'comparison.json',dict(all34_complete=m['all34_windows_complete'],
            maximum_deficit_increase_over_historical_mm=max(delta),
            deficit_not_worse_than_historical_count=sum(x<=1e-7 for x in delta),
            historical_time_s=old['complete_recorded_time_s'],new_time_s=result['complete_time_s'],
            scope='Preliminary comparison to the retained historical main, not final baseline acceptance.',goal_complete=False))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',ROOT.parent/'turn_aware_projector.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Unchanged direct14checkpoint on complete original44.420514m path',
        wall_budget_s=1500,child_timeout_s=1400,execution_deadline_s=1320,checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),
        source_sha256=hashes,inherited_teacher_manifest=manifest,reference_sha256=sha(CASE/'reference.csv'),
        ppo_updates=0,evidence_gap='Compact policy succeeded on the dense case but original full motion and all34windows remain unverified.',
        adoption='Full corridor/vectorV/A/J/ordered coverage/braking/rest success, then original time and all34baseline criteria; no teacher at execution.',
        stop='At physical failure or deadline; all original geometry and terminal motion retained; no source or weight edits.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1400)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-2500:])


if __name__=='__main__':main()
