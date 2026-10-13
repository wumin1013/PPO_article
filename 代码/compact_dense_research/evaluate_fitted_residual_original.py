"""Same learned dense-case checkpoint on the complete unchanged original case."""
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
MODEL=ROOT/'historical_fitted_residual_ppo_seed922302_pilot/checkpoint_0016.pt'
STAGE=ROOT/'historical_fitted_residual_ppo_seed922302_original_full'


def worker():
    import numpy as np
    import torch
    from historical_residual_ppo import HistoricalResidualRuntime,ResidualPolicy
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    verify_bound_inputs();torch.set_num_threads(1)
    assert sha(MODEL)=='abe89eb959b0fa4ffbced2914c3278ca7aaccacfadf45b092e43ed768e4a4e93'
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    policy=ResidualPolicy(seed=922302)
    policy.load_state_dict(torch.load(MODEL,map_location='cpu',weights_only=False)['policy'])
    class CountRuntime(HistoricalResidualRuntime):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.clustered_decisions=0
        def step(self,action):
            self._prepare_nominal()
            self.clustered_decisions+=int(self.correction_active)
            return super().step(action)
    env=CountRuntime(reference)
    start=time.perf_counter()
    result=evaluate(policy,env,STAGE/'full_trace.npz',start+1620.)
    data=np.load(STAGE/'full_trace.npz');rows=data['rows']
    result['analytic']=analytic_command_audit(reference[0],rows,geometry)
    if not result['analytic']['analytic_corridor_pass']:
        result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
    result.update(checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),
        clustered_correction_decisions=env.clustered_decisions,goal_complete=False,method_frozen=False,
        ppo_updates=0,residual_observation_dimension=17,inherited_observation_dimension=72,
        online_prediction_search=False)
    write(STAGE/'full_result.json',result);write(STAGE/'gate_events.json',env.tracker.events)
    previous=np.load(HIST/'full_trace.npz');comparison={}
    for key in ['reference','rows','actions','executed_actions_1ms']:
        a,b=data[key],previous[key];same_shape=a.shape==b.shape
        finite=np.isfinite(a)&np.isfinite(b) if same_shape else None
        comparison[key]=dict(shape=list(a.shape),historical_shape=list(b.shape),
            equal_nan_aware=bool(same_shape and np.array_equal(a,b,equal_nan=True)),
            maximum_finite_difference=float(np.max(abs(a[finite]-b[finite]))) if same_shape and np.any(finite) else None)
    preserved=False
    if result['success']:
        m=metrics(reference,rows,geometry);write(STAGE/'full_metrics.json',m)
        old=json.loads((HIST/'full_metrics.json').read_text(encoding='utf-8'))
        delta=[x['speed_deficit_mm']-y['speed_deficit_mm'] for x,y in zip(m['corners'],old['corners'])]
        comparison['all34']=dict(all_windows_complete=m['all34_windows_complete'],
            maximum_speed_deficit_change_mm=max(delta),maximum_absolute_speed_deficit_change_mm=max(abs(x) for x in delta),
            cruise99_time_change_s=m['cruise']['99']['time_s']-old['cruise']['99']['time_s'])
        preserved=bool(m['all34_windows_complete'] and result['complete_time_s']==old['complete_recorded_time_s']
            and comparison['rows']['equal_nan_aware'] and max(abs(x) for x in delta)<=1e-8)
    write(STAGE/'comparison.json',dict(historical_motion_preserved=preserved,comparison=comparison,
        goal_complete=False,method_frozen=False,
        scope='Same learned checkpoint on both cases. No geometry identifier or region ID used; clustered-turn activation depends only on local reference geometry.'))
    verify_bound_inputs()
    print(json.dumps(dict(result=result,historical_motion_preserved=preserved),ensure_ascii=False),flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs()
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_residual_ppo.py',ROOT/'local_arc_support.py',ROOT/'compact_control.py']
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Same learned residual checkpoint,complete original-case verification',
        wall_budget_s=1800,child_timeout_s=1710,execution_deadline_s=1620,physics_dt_s=.001,ppo_updates=0,
        checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),source_sha256=hashes,inherited_manifest=manifest,
        reference_sha256=sha(CASE/'reference.csv'),geometry_sha256=sha(CASE/'geometry.json'),
        evidence_gap='Dense-case learned checkpoint has not been run on the original full44.420514m path.',
        adoption='Complete original physical and geometric contract,all34 windows and original baseline comparisons. Check motion identity separately; do not substitute a sparse-memory-only prior run.',
        stop='Physical failure or registered deadline; no source,weight,case or baseline changes.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=1710)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4500:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()
