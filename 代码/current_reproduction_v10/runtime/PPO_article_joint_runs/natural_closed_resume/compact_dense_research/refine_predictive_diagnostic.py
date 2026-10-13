"""Refine the successful predictive diagnostic to0.5ms, retaining20ms actions."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,HIST,CASE,sha,write,verify_bound_inputs

NOMINAL=ROOT/'historical_dense_predictive_action_diagnostic'
STAGE=ROOT/'historical_dense_predictive_action_half_ms'


def worker():
    import numpy as np
    import torch
    import turn_aware_projector as turn
    from run_refined_turn_candidate import configure_resolution,backup_check
    from run_natural_resolution import physical_audit
    from candidate_policy_factory import candidate_policy
    from local_arc_support import install_local_arc_support,ClusteredPreviewFloorPolicy
    from predictive_action_diagnostic import PredictiveActionDiagnostic
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    verify_bound_inputs()
    torch.set_num_threads(1)
    start=time.perf_counter();dt=.0005
    configure_resolution(dt)
    write(STAGE/'backup_checks.json',backup_check(dt,start+120.))
    selection=json.loads((HIST/'selected_policy.json').read_text(encoding='utf-8'))
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    metadata=torch.load(selection['model'],map_location='cpu',weights_only=False)
    base,_=install_local_arc_support(candidate_policy(metadata,selection))
    base=ClusteredPreviewFloorPolicy(base,3.)
    policy=PredictiveActionDiagnostic(base,start+750.)
    env=turn.environment_factory(reference)
    policy.bind(env)
    chunks=[];observations=[];actions=[];latents=[];times=[];executed=[]
    error=None;last=time.perf_counter()
    try:
        while not env.done:
            if time.perf_counter()>=start+750.:raise TimeoutError('Refined diagnostic budget')
            x=env.observation();latent=policy.act(x,True)[0]
            times.append(env.ticks*dt);observations.append(x);latents.append(latent)
            _,_,_,info=env.step(latent)
            chunks.append(info['rows']);actions.append(info['held_action']);executed.append(info['executed_actions_1ms'])
            if time.perf_counter()-last>40.:
                last=time.perf_counter();print('REFINED',env.ticks*dt,'gates',env.tracker.frontier,flush=True)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    np.savez_compressed(STAGE/'trace.npz',reference=reference,rows=rows,observations=np.array(observations),
        actions=np.array(actions),latents=np.array(latents),decision_times_s=np.array(times),dt_s=dt,
        executed_actions=np.vstack(executed) if executed else np.empty((0,3)))
    for record in policy.records:
        record['physical_time_s']*=dt/.001
    write(STAGE/'action_search.json',policy.records)
    physical=physical_audit(reference,rows,dt) if len(rows) else None
    geometry_result=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,geometry) if len(rows) else None
    success=bool(error is None and env.success and geometry_result['geometry_complete'] and
        physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=len(rows)*dt if success else None,
        elapsed_prefix_or_complete_s=len(rows)*dt,geometry=geometry_result,audit=physical,analytic=analytic,
        dt_s=dt,decision_dt_s=.02,forecast_dt_s=dt,forecast_horizon_s=.12,
        trace_sha256=sha(STAGE/'trace.npz'),actor_calls=len(actions),physics_ticks=len(rows),
        forecast_physics_ticks=policy.forecast_ticks,
        changed_decisions=sum(r['selected']!='nominal' for r in policy.records),
        no_feasible_forecast_decisions=sum(r['feasible_candidates']==0 for r in policy.records),
        goal_complete=False,method_frozen=False,ppo_updates=0,wall_s=time.perf_counter()-start,
        scope='Both plant and forecast integration refined; same120ms horizon, candidate search,20ms decisions, thresholds and inherited policy.')
    write(STAGE/'result.json',result)
    write(STAGE/'gate_events.json',env.tracker.events)
    if success:write(STAGE/'all84_metrics.json',analyze(STAGE/'trace.npz',STAGE/'result.json',dt))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    manifest=verify_bound_inputs()
    nominal=json.loads((NOMINAL/'result.json').read_text(encoding='utf-8'))
    assert nominal['success']
    STAGE.mkdir(exist_ok=False)
    files=[Path(__file__),ROOT/'predictive_action_diagnostic.py',ROOT/'local_arc_support.py',
        ROOT.parent/'run_refined_turn_candidate.py',ROOT.parent/'turn_aware_projector.py',BASE/'run_natural_resolution.py']
    hashes={str(p):sha(p) for p in files}
    (STAGE/'source').mkdir()
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Half-step predictive-diagnostic validation',
        wall_budget_s=900,child_timeout_s=850,execution_deadline_s=750,ppo_updates=0,
        dt_s=.0005,decision_dt_s=.02,forecast_horizon_s=.12,source_sha256=hashes,
        nominal_result_sha256=sha(NOMINAL/'result.json'),inherited_manifest=manifest,
        reference_sha256=sha(CASE/'reference.csv'),
        evidence_gap='15.285s complete success has only been checked at1ms; tight local corridor margins need refinement.',
        adoption='Require full geometry, independent raw XY vector dynamics and >=20ms physical rest; report change in complete time and all84 windows. No method freeze.',
        stop='Any real failure or execution deadline; no changes to search thresholds or parameters.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=850)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__=='__main__':main()
