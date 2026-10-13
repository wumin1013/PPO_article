"""Independent0.5ms dense evaluation of the14input30mm-horizon actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from evaluate_historical_dense import ROOT,BASE,CASE,sha,write,verify_bound_inputs


def worker(args):
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_longview_runtime import LongViewRuntime as DirectCompactRuntime
    from run_refined_turn_candidate import configure_resolution,backup_check
    from run_natural_resolution import physical_audit
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    torch.set_num_threads(1)
    stage=ROOT/args.name;started=time.perf_counter();dt=.0005
    configure_resolution(dt);write(stage/'backup_checks.json',backup_check(dt,started+90))
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf8'))
    policy=FineCompactPolicy(feed_gain=args.feed_gain)
    saved=torch.load(args.checkpoint,weights_only=False)
    assert saved.get('feed_gain',args.feed_gain)==args.feed_gain
    policy.load_state_dict(saved['policy']);env=DirectCompactRuntime(reference)
    chunks=[];observations=[];actions=[];latents=[];times=[];error=None
    try:
        while not env.done:
            if time.perf_counter()>started+240:raise TimeoutError('Registered240s refined evaluation budget')
            x=env.observation();z=policy.act(x,True)[0]
            observations.append(x);latents.append(z);times.append(env.ticks*dt)
            _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,observations=np.array(observations),
        actions=np.array(actions),latents=np.array(latents),decision_times_s=np.array(times),dt_s=dt)
    physical=physical_audit(reference,rows,dt) if len(rows) else None
    ordered=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,geometry) if len(rows) else None
    success=bool(error is None and env.success and ordered['geometry_complete'] and physical['kinematic_pass']
        and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=len(rows)*dt if success else None,
        elapsed_prefix_or_complete_s=len(rows)*dt,geometry=ordered,audit=physical,analytic=analytic,
        dt_s=dt,decision_dt_s=.02,actor_calls=len(actions),physics_ticks=len(rows),feed_gain=args.feed_gain,
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),trace_sha256=sha(stage/'trace.npz'),
        wall_s=time.perf_counter()-started,ppo_updates_this_evaluation=0,goal_complete=False,method_frozen=False,
        policy_observation_dimension=14,policy_memory_scalars=0,inherited_controller_at_execution=False,actor_outputs=3,far_sensor_mm=30)
    write(stage/'result.json',result);write(stage/'gate_events.json',env.tracker.events)
    if success:write(stage/'all84_metrics.json',analyze(stage/'trace.npz',stage/'result.json',dt))
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--nominal',type=Path,required=True);p.add_argument('--feed-gain',type=float,required=True)
    p.add_argument('--worker',action='store_true');args=p.parse_args()
    assert Path(args.name).name==args.name
    if args.worker:worker(args);return
    verify_bound_inputs();assert json.loads(args.nominal.read_text(encoding='utf8'))['success']
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    files=[Path(__file__),ROOT/'compact_fine_policy.py',ROOT/'compact_control.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_longview_runtime.py',
        ROOT.parent/'run_refined_turn_candidate.py',BASE/'run_natural_resolution.py',BASE/'natural_case_audit.py',ROOT/'analyze_dense_trace.py']
    hashes={str(f):sha(f) for f in files}
    for f in files:shutil.copy2(f,stage/'source'/f.name)
    write(stage/'registration.json',dict(kind='Explicit14input30mm-horizon policy, independent0.5ms closed loop',
        wall_budget_s=300,child_timeout_s=280,worker_deadline_s=240,checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        feed_gain=args.feed_gain,nominal=str(args.nominal),nominal_sha256=sha(args.nominal),source_sha256=hashes,
        evidence_gap='Complete nominal candidate requires independent finer-resolution physical confirmation.',
        adoption='Complete ordered geometry, analytic corridor, raw vector V/A/J, braking and20ms rest; no parameter search.',
        stop='One complete execution, physical failure, or registered deadline.',goal_complete=False,method_frozen=False))
    start=time.perf_counter();code=None
    with (stage/'stdout.txt').open('w',encoding='utf8') as out,(stage/'stderr.txt').open('w',encoding='utf8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],stdout=out,stderr=err,timeout=280)
            status='finished';code=run.returncode
        except subprocess.TimeoutExpired:status='timeout'
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(f)==h for f,h in hashes.items()),goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf8')[-4000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf8')[-3000:])


if __name__=='__main__':main()
