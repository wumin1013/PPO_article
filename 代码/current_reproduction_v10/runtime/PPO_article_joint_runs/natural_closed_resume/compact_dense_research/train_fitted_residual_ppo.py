"""Actual PPO after explicitly identified offline residual initialization."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from evaluate_historical_dense import ROOT,BASE,HIST,CASE,sha,write,verify_bound_inputs

STAGE=ROOT/'historical_fitted_residual_ppo_seed922302_pilot'
INITIAL=ROOT/'historical_predictive_residual_fit_seed922302/fit_0200.pt'


def worker():
    import torch
    import train_compact_ppo_mc as training
    from historical_residual_ppo import HistoricalResidualRuntime,ResidualPolicy
    from natural_case_audit import analytic_command_audit
    import numpy as np
    verify_bound_inputs()
    torch.set_num_threads(1)
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    class WarmStartedResidual(ResidualPolicy):
        def __init__(self,dimension=17,action_dim=3,seed=922302):
            super().__init__(dimension,action_dim,seed)
            self.load_state_dict(initial['policy'])
    training.TrainingRuntime=HistoricalResidualRuntime
    training.CompactPolicy=WarmStartedResidual
    args=SimpleNamespace(seed=922302,dimension=17,action_dim=3,no_swing=False,preview=3.,
        potential_speed=30.,remaining_encoding='log',resume=None,steps=16384,batch=1024,
        evaluate_every=4,budget=1050.)
    training.worker(args,STAGE)
    result=json.loads((STAGE/'result.json').read_text(encoding='utf-8'))
    result.update(training_kind='PPO of17-input residual initialized by offline fit on a successful forecast-diagnostic trajectory',
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        residual_observation_dimension=17,inherited_observation_dimension=72,
        independent_full_method_training=False,goal_complete=False,method_frozen=False,
        success_scope='Conditional inherited-controller pilot; not three independently trained complete methods.')
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    evaluations=json.loads((STAGE/'evaluations.json').read_text(encoding='utf-8'))
    best=None
    for record in evaluations:
        if not record['result']['success']:continue
        file=STAGE/f"eval_{record['update']:04d}.npz"
        rows=np.load(file)['rows'];ref=np.load(file)['reference']
        record['result']['analytic']=analytic_command_audit(ref[0],rows,geometry)
        if not record['result']['analytic']['analytic_corridor_pass']:
            record['result'].update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        elif best is None or record['result']['complete_time_s']<best['complete_time_s']:
            best=dict(checkpoint=record['checkpoint'],complete_time_s=record['result']['complete_time_s'],evaluation=str(file.with_suffix('.json')))
        write(file.with_suffix('.json'),record['result'])
    result['best']=best
    write(STAGE/'evaluations.json',evaluations);write(STAGE/'result.json',result)


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs()
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_residual_ppo.py',ROOT/'train_compact_ppo_mc.py',
        ROOT/'learning_update_compact_mc.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Fitted residual PPO pilot with frozen historical base',
        wall_budget_s=1200,child_timeout_s=1150,ppo_training_budget_s=1050,
        new_decisions=16384,batch_size=1024,scheduled_evaluation_every_updates=4,
        seed=922302,source_sha256=hashes,inherited_manifest=manifest,
        initial_checkpoint=str(INITIAL),initial_checkpoint_sha256=sha(INITIAL),
        initialization_scope='Actor/critic parameters inherited from offline fit epoch200; new PPO optimizer and RNG. Shared full historical controller, not an independent complete-method seed.',
        evidence_gap='One recorded-trajectory fit reaches62/84gates but fails in closed loop; more fitting epochs regressed.',
        plan='Learn from own on-policy physical states with existing audited raw Gaussian residual PPO loop, then scheduled complete evaluations.',
        adoption='Only complete geometry, raw vector dynamics, braking and rest qualify; compare complete times only after successful analytic corridor audit.',
        stop='Stop at16384 decisions or1050s training budget; no additional same-round search or model expansion.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1150)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()
