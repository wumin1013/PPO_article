"""Two registered feed-head calibrations of the direct14model on both full cases."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,CASE,HIST,sha,write,verify_bound_inputs

MODEL=ROOT/'direct14_saved_motion_fit_seed922302_bound/fit_0600.pt'
STAGE=ROOT/'direct14_feed_head_calibration_seed922302'


def worker():
    import numpy as np
    import torch
    from compact_control import CompactPolicy
    from direct_compact_runtime import DirectCompactRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    torch.set_num_threads(1);verify_bound_inputs();start=time.perf_counter();deadline=start+1320.
    saved=torch.load(MODEL,map_location='cpu',weights_only=False);records=[]
    for bias in [.02,.04]:
        if time.perf_counter()>deadline-60.:break
        folder=STAGE/f'feed_plus_{50*bias:g}mm_s';folder.mkdir()
        policy=CompactPolicy(14,3,922302);policy.load_state_dict(saved['policy'])
        with torch.no_grad():policy.actor[-1].bias[1].add_(bias)
        checkpoint=folder/'policy.pt'
        torch.save(dict(policy=policy.state_dict(),source_checkpoint=str(MODEL),source_checkpoint_sha256=sha(MODEL),
            output_bias_added=[0.,bias,0.],training_kind='Finite feed-head calibration after offline compression; zeroPPOupdates',
            observation_dimension=14,policy_memory_scalars=0),checkpoint)
        record=dict(feed_bias=bias,requested_feed_increase_mm_s=50*bias,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),cases={})
        for label,case in [('dense',CASE),('original',BASE/'paper_evidence_review/natural_closed_case')]:
            reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
            geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
            trace=folder/f'{label}_trace.npz'
            result=evaluate(policy,DirectCompactRuntime(reference),trace,deadline)
            rows=np.load(trace)['rows']
            if len(rows):
                result['analytic']=analytic_command_audit(reference[0],rows,geometry)
                if not result['analytic']['analytic_corridor_pass']:result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            result.update(ppo_updates=0,training_kind='Finite output-head calibration',checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),goal_complete=False,method_frozen=False)
            write(folder/f'{label}_result.json',result);record['cases'][label]=result
            if label=='original' and result['success']:
                m=metrics(reference,rows,geometry);write(folder/'original_metrics.json',m)
                old=json.loads((HIST/'full_metrics.json').read_text(encoding='utf-8'))
                delta=[a['speed_deficit_mm']-b['speed_deficit_mm'] for a,b in zip(m['corners'],old['corners'])]
                record['all34_vs_historical']=dict(all34_complete=m['all34_windows_complete'],
                    not_worse_count=sum(x<=1e-7 for x in delta),maximum_increase_mm=max(delta))
            print(json.dumps(dict(bias=bias,case=label,success=result['success'],time_s=result['complete_time_s'],gates=result['geometry']['gates_passed'])),flush=True)
            write(STAGE/'live.json',dict(current=record,elapsed_wall_s=time.perf_counter()-start,goal_complete=False))
            if label=='dense' and not result['success']:break
        records.append(record);write(STAGE/'candidates.json',records)
    qualifying=[r for r in records if all(k in r['cases'] and r['cases'][k]['success'] for k in ['dense','original'])]
    best=min(qualifying,key=lambda r:(r['cases']['dense']['complete_time_s'],r['cases']['original']['complete_time_s'])) if qualifying else None
    write(STAGE/'result.json',dict(status='registered_candidates_completed' if len(records)==2 else 'wall_budget_completed',
        candidates=records,best_complete_both=best,ppo_updates=0,wall_s=time.perf_counter()-start,
        goal_complete=False,method_frozen=False,scope='Both-case physical success is necessary but not sufficient; all34baseline/time criteria and0.5ms checks still determine adoption.'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',ROOT.parent/'turn_aware_projector.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Finite direct14feed-head calibration; not PPO',
        wall_budget_s=1500,child_timeout_s=1400,execution_deadline_s=1320,
        normalized_bias_candidates=[.02,.04],requested_feed_increase_mm_s=[1.,2.],
        checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),source_sha256=hashes,inherited_teacher_manifest=manifest,
        ppo_updates=0,evidence_gap='14input offline actor succeeds on both cases but loses original cruising and corner efficiency; quantify whether feed-head fitting below saturation explains it.',
        adoption='Run full dense then full original for each candidate. Preserve original34corner and baseline comparisons. Refined verification required; no failed prefix time counts.',
        stop='Exactly2registered calibrations; no steering/preview changes, no additional candidates, no geometry/window/baseline changes; physical failure or deadline.'))
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
