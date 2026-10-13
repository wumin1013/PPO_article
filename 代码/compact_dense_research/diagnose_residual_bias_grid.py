"""Preregistered finite output-bias calibration, explicitly not PPO training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT, CASE, sha, write, verify_bound_inputs

MODEL=ROOT/'paired_three_action_guarded_seed922302_round2/checkpoint_0028.pt'
NOMINAL=MODEL.parent/'eval_0028.json'
STAGE=ROOT/'three_action_bias_grid_seed922302'
GRID=[(feed,preview) for feed in [0.,.01,.02,.04] for preview in [0.,.025,.05]]


def worker():
    import numpy as np
    import torch
    from historical_residual_ppo import ResidualPolicy,HistoricalResidualRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    torch.set_num_threads(1);verify_bound_inputs()
    start=time.perf_counter();deadline=start+1080.
    saved=torch.load(MODEL,map_location='cpu',weights_only=False)
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    records=[];best=None
    for index,(feed,preview) in enumerate(GRID):
        if time.perf_counter()>deadline-45.:break
        folder=STAGE/f'candidate_{index:02d}';folder.mkdir()
        if feed==preview==0.:
            result=json.loads(NOMINAL.read_text(encoding='utf-8'))
            record=dict(index=index,feed_bias=feed,preview_bias=preview,
                checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),result=result,
                source='Existing unmodified baseline; no repeated simulation')
        else:
            policy=ResidualPolicy(seed=922302);policy.load_state_dict(saved['policy'])
            with torch.no_grad():policy.actor[-1].bias.add_(torch.tensor([0.,feed,preview]))
            model=folder/'policy.pt'
            torch.save(dict(policy=policy.state_dict(),source_checkpoint=str(MODEL),
                source_checkpoint_sha256=sha(MODEL),output_bias_added=[0.,feed,preview],
                training_kind='Finite offline parameter selection after PPO; zero new PPO updates'),model)
            result=evaluate(policy,HistoricalResidualRuntime(reference),folder/'trace.npz',deadline)
            trace=np.load(folder/'trace.npz')
            if result['success']:
                result['analytic']=analytic_command_audit(reference[0],trace['rows'],geometry)
                if not result['analytic']['analytic_corridor_pass']:
                    result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
            write(folder/'result.json',result)
            record=dict(index=index,feed_bias=feed,preview_bias=preview,
                checkpoint=str(model),checkpoint_sha256=sha(model),result=result,
                source='New complete or failed closed-loop parameter diagnostic')
        records.append(record)
        if result['success'] and (best is None or result['complete_time_s']<best['result']['complete_time_s']):best=record
        write(STAGE/'candidates.json',records)
        print(json.dumps(dict(index=index,feed_bias=feed,preview_bias=preview,
            success=result['success'],time_s=result['complete_time_s'],gates=result['geometry']['gates_passed'])),flush=True)
    write(STAGE/'result.json',dict(status='grid_completed' if len(records)==len(GRID) else 'wall_budget_completed',
        evaluated_candidates=len(records),new_simulations=max(0,len(records)-1),best=best,
        ppo_updates=0,parameter_selection=True,wall_s=time.perf_counter()-start,
        goal_complete=False,method_frozen=False,
        adoption_scope='1ms development evidence only. Unchanged shared executor and baseline; refined validation required before adoption.'))


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs();STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_residual_ppo.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Finite parameter optimization / output amplitude diagnostic, not PPO',
        grid=[dict(feed_bias=f,preview_bias=d) for f,d in GRID],
        physical_interpretation='Output residual bias: feed +0/0.5/1/2 mm/s and preview +0/0.14/0.28 mm before existing saturation; steering unchanged.',
        source_checkpoint=str(MODEL),source_checkpoint_sha256=sha(MODEL),nominal_result=str(NOMINAL),
        source_sha256=hashes,inherited_manifest=manifest,wall_budget_s=1200,
        child_timeout_s=1150,execution_deadline_s=1080,ppo_updates=0,
        evidence_gap='Four scheduled PPO evaluations plateau near15.3s, short of the15.06985s target. Determine whether small feed/preview amplitude changes yield usable gains.',
        adoption='Complete84gate/analytic corridor/vectorV/A/J/rest success first, then shortest full time; preserve baseline and failures; 0.5ms verification before adopting.',
        stop='Only12registered combinations including the retained baseline; no adaptive expansion; at wall/physical failure limits; never change geometry, windows or baselines.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1150)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-5000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-2500:])


if __name__=='__main__':main()
