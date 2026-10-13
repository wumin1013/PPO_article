"""Independent half-ms closed-loop evaluation of the learned residual checkpoint."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,CASE,sha,write,verify_bound_inputs

PARENT=ROOT/'historical_fitted_residual_ppo_seed922302_pilot'
STAGE=ROOT/'historical_fitted_residual_ppo_seed922302_half_ms'
MODEL=PARENT/'checkpoint_0016.pt'
NOMINAL=PARENT/'eval_0016.json'
ACTIONS=3
PREVIEW=3.
DIRECT=False


def worker():
    import numpy as np
    import torch
    from run_refined_turn_candidate import configure_resolution,backup_check
    from run_natural_resolution import physical_audit
    from historical_residual_ppo import HistoricalResidualRuntime,ResidualPolicy
    if ACTIONS==2:
        from historical_two_action_residual import HistoricalTwoActionRuntime,TwoActionResidualPolicy
        HistoricalResidualRuntime,ResidualPolicy=HistoricalTwoActionRuntime,TwoActionResidualPolicy
    if DIRECT:
        from compact_control import CompactPolicy
        from direct_compact_runtime import DirectCompactRuntime
        HistoricalResidualRuntime,ResidualPolicy=DirectCompactRuntime,CompactPolicy
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    verify_bound_inputs();torch.set_num_threads(1)
    start=time.perf_counter();dt=.0005;configure_resolution(dt)
    write(STAGE/'backup_checks.json',backup_check(dt,start+90.))
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    policy=ResidualPolicy(seed=922302)
    saved=torch.load(MODEL,map_location='cpu',weights_only=False)
    policy.load_state_dict(saved['policy'])
    env=HistoricalResidualRuntime(reference,preview_rule_mm=PREVIEW)
    chunks=[];observations=[];actions=[];latents=[];times=[];error=None
    while not env.done:
        try:
            if time.perf_counter()>start+240.:raise TimeoutError('Half-ms learned residual budget')
            x=env.observation();z=policy.act(x,True)[0]
            times.append(env.ticks*dt);observations.append(x);latents.append(z)
            _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
            if ACTIONS==2:assert info['held_action'][2]==PREVIEW and len(z)==2
        except Exception as exc:error=f'{type(exc).__name__}: {exc}';break
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    np.savez_compressed(STAGE/'trace.npz',reference=reference,rows=rows,observations=np.array(observations),
        actions=np.array(actions),latents=np.array(latents),decision_times_s=np.array(times),dt_s=dt)
    physical=physical_audit(reference,rows,dt) if len(rows) else None
    ordered=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,geometry) if len(rows) else None
    success=bool(error is None and env.success and ordered['geometry_complete'] and physical['kinematic_pass']
        and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=len(rows)*dt if success else None,
        elapsed_prefix_or_complete_s=len(rows)*dt,geometry=ordered,audit=physical,analytic=analytic,
        dt_s=dt,decision_dt_s=.02,actor_calls=len(actions),physics_ticks=len(rows),
        checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),trace_sha256=sha(STAGE/'trace.npz'),
        wall_s=time.perf_counter()-start,ppo_updates_this_evaluation=0,goal_complete=False,method_frozen=False,
        online_prediction_search=False,inherited_observation_dimension=0 if DIRECT else 72,
        policy_observation_dimension=14 if DIRECT else 17,policy_memory_scalars=0 if DIRECT else 474,
        inherited_controller_at_execution=not DIRECT,
        actor_outputs=ACTIONS,fixed_preview_mm=PREVIEW if ACTIONS==2 else None)
    write(STAGE/'result.json',result);write(STAGE/'gate_events.json',env.tracker.events)
    if success:write(STAGE/'all84_metrics.json',analyze(STAGE/'trace.npz',STAGE/'result.json',dt))
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    global STAGE,MODEL,NOMINAL,ACTIONS,PREVIEW,DIRECT
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default=STAGE.name)
    parser.add_argument('--checkpoint',type=Path,default=MODEL)
    parser.add_argument('--nominal',type=Path,default=NOMINAL)
    parser.add_argument('--actions',type=int,choices=[2,3],default=3)
    parser.add_argument('--preview',type=float,default=3.)
    parser.add_argument('--direct-policy',action='store_true')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    assert Path(args.name).name==args.name
    STAGE,MODEL,NOMINAL,ACTIONS,PREVIEW=ROOT/args.name,args.checkpoint,args.nominal,args.actions,args.preview
    DIRECT=args.direct_policy
    assert not DIRECT or ACTIONS==3
    if args.worker:worker();return
    manifest=verify_bound_inputs()
    assert json.loads(NOMINAL.read_text(encoding='utf-8'))['success']
    STAGE.mkdir(exist_ok=False);(STAGE/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_residual_ppo.py',ROOT/'historical_two_action_residual.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py',ROOT.parent/'run_refined_turn_candidate.py',BASE/'run_natural_resolution.py']
    if DIRECT:files.append(ROOT/'direct_compact_runtime.py')
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Half-ms closed-loop learned-residual validation',
        wall_budget_s=300,child_timeout_s=280,execution_deadline_s=240,dt_s=.0005,decision_dt_s=.02,
        ppo_updates=0,checkpoint=str(MODEL),checkpoint_sha256=sha(MODEL),source_sha256=hashes,
        inherited_manifest=manifest,nominal_evaluation=str(NOMINAL),nominal_evaluation_sha256=sha(NOMINAL),
        actor_outputs=ACTIONS,preview_argument_mm=PREVIEW,direct_policy=DIRECT,
        evidence_gap='New deterministic PPO checkpoint succeeds at1ms; refined physical feasibility is unverified.',
        adoption='Require complete84gate geometry, independent raw XY vector V/A/J and>=20ms physical rest; weights unchanged.',
        stop='At failure or240s execution deadline; no action search, parameter adjustment or retraining.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],stdout=out,stderr=err,timeout=280)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()
