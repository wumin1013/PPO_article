"""Independent half-ms complete execution of an explicitly bound14input actor."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,CASE,write,sha,verify_bound_inputs


def worker(args):
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from run_refined_turn_candidate import configure_resolution,backup_check,corner_stats
    from run_natural_resolution import physical_audit
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze
    stage=ROOT/args.name
    started=time.perf_counter()
    deadline=started+(780. if args.case=='original' else 240.)
    dt=.0005
    torch.set_num_threads(1)
    configure_resolution(dt)
    write(stage/'backup_checks.json',backup_check(dt,started+90.))
    folder=BASE/'paper_evidence_review/natural_closed_case' if args.case=='original' else CASE
    reference=np.loadtxt(folder/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((folder/'geometry.json').read_text(encoding='utf-8'))
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert saved['interface'][:2]==[14,args.action_dim]
    feed_gain=float(saved.get('feed_gain',1.))
    assert np.isfinite(feed_gain) and feed_gain>0.
    if args.action_dim==3:
        assert saved['runtime_class']=='FarVectorRuntime'
        policy=FineCompactPolicy(feed_gain=feed_gain)
        env=FarVectorRuntime(reference)
    else:
        from direct_two_action_control import DirectTwoPolicy,DirectTwoRuntime
        assert saved['runtime_class']=='DirectTwoRuntime'
        assert feed_gain==1.,'Two-output activation must be explicitly supported before applying a gain'
        policy=DirectTwoPolicy(seed=saved['interface'][2])
        env=DirectTwoRuntime(reference,preview_rule_mm=saved['preview_rule_mm'])
    policy.load_state_dict(saved['policy'])
    if 'observation_features' in saved:
        assert saved['observation_features']==env.feature_names()
    nominal=json.loads(args.nominal.read_text(encoding='utf-8'))
    assert nominal['success'] and nominal['checkpoint_sha256']==sha(args.checkpoint)
    if 'observation_features' in nominal:
        assert nominal['observation_features']==env.feature_names()
    else:
        assert args.action_dim==2 and saved['observation_features']==env.feature_names()
    assert nominal['observation_dimension']==14 and nominal['actor_outputs']==args.action_dim
    chunks,observations,actions,latents,times,executed=[],[],[],[],[],[]
    error=None
    last=started
    try:
        while not env.done:
            if time.perf_counter()>deadline:
                raise TimeoutError('Registered refinement wall deadline')
            x=env.observation()
            z=policy.act(x,True)[0]
            observations.append(x)
            latents.append(z)
            times.append(env.ticks*dt)
            _,_,_,info=env.step(z)
            chunks.append(info['rows'])
            actions.append(info['held_action'])
            executed.append(info.get('executed_actions_1ms',np.tile(info['held_action'],(len(info['rows']),1))))
            if time.perf_counter()-last>40.:
                last=time.perf_counter()
                print('REFINING',args.case,'ticks',env.ticks,'gates',env.tracker.frontier,flush=True)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    held=np.vstack(executed) if executed else np.empty((0,3))
    changes=np.flatnonzero(np.any(held[1:]!=held[:-1],axis=1))+1
    if len(changes):
        changes=changes[(rows[changes,12]==0)&(rows[changes-1,12]==0)]
        assert np.all(changes%40==0)
    trace=stage/'trace.npz'
    np.savez_compressed(trace,reference=reference,rows=rows,observations=np.asarray(observations),
        actions=np.asarray(actions),latents=np.asarray(latents),decision_times_s=np.asarray(times),
        executed_actions=held,dt_s=dt)
    physical=physical_audit(reference,rows,dt) if len(rows) else None
    ordered=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,geometry) if len(rows) else None
    success=bool(error is None and env.success and ordered['geometry_complete'] and
        physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=len(rows)*dt if success else None,
        elapsed_prefix_or_complete_s=len(rows)*dt,geometry=ordered,audit=physical,analytic=analytic,
        dt_s=dt,decision_dt_s=.02,actor_calls=len(actions),physics_ticks=len(rows),ordinary_20ms_hold=True,
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),trace_sha256=sha(trace),
        nominal=str(args.nominal),nominal_sha256=sha(args.nominal),
        nominal_complete_time_s=nominal['complete_time_s'],case=args.case,
        wall_s=time.perf_counter()-started,ppo_updates_this_evaluation=0,
        observation_dimension=14,policy_memory_scalars=0,actor_outputs=args.action_dim,
        feed_gain=feed_gain,
        observation_features=env.feature_names(),inherited_controller_at_execution=False,
        goal_complete=False,method_frozen=False)
    if args.action_dim==2:
        result['global_preview_mm']=saved['preview_rule_mm']
    result['nominal_feature_names_available']='observation_features' in nominal
    if success:
        result['resolution_time_difference_s']=result['complete_time_s']-nominal['complete_time_s']
    write(stage/'result.json',result)
    write(stage/'gate_events.json',env.tracker.events)
    if success:
        metric=corner_stats(reference,rows,geometry,dt) if args.case=='original' else analyze(trace,stage/'result.json',dt)
        write(stage/'corner_metrics.json',metric)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--case',choices=['original','dense'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--nominal',type=Path,required=True)
    parser.add_argument('--action-dim',type=int,choices=[2,3],default=3)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    assert Path(args.name).name==args.name
    if args.worker:
        worker(args)
        return
    verify_bound_inputs()
    nominal=json.loads(args.nominal.read_text(encoding='utf-8'))
    assert nominal['success'] and nominal['checkpoint_sha256']==sha(args.checkpoint)
    stage=ROOT/args.name
    stage.mkdir(exist_ok=False)
    (stage/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_fine_policy.py',ROOT/'compact_farvector_runtime.py',
        ROOT/'compact_previewcue_runtime.py',ROOT/'direct_compact_runtime.py',ROOT/'compact_control.py',
        ROOT.parent/'run_refined_turn_candidate.py',BASE/'run_natural_resolution.py',
        BASE/'natural_case_audit.py',ROOT/'analyze_dense_trace.py']
    if args.action_dim==2:
        sources.append(ROOT/'direct_two_action_control.py')
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        shutil.copy2(p,stage/'source'/p.name)
    budget=900 if args.case=='original' else 300
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Independent0.5ms complete closed loop; no tuning',case=args.case,
        wall_budget_s=budget,child_timeout_s=budget-20,dt_s=.0005,decision_dt_s=.02,
        observation_dimension=14,actor_outputs=args.action_dim,
        checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
        nominal=str(args.nominal),nominal_sha256=sha(args.nominal),source_sha256=hashes,
        evidence_gap='A nominal candidate needs independent numerical-resolution and full physical checks.',
        adoption='Complete ordered coverage,unchanged1.25corridor,raw vector V/A/J,full brake and20ms rest; then paired time and corner gates. This execution alone never establishes joint acceptance.',
        stop='One complete execution,physical failure,or registered deadline. No model changes.',
        goal_complete=False,method_frozen=False))
    started=time.perf_counter()
    code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],
                stdout=out,stderr=err,timeout=budget-20)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:
            status='timeout'
    verify_bound_inputs()
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:
        print((stage/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':
    main()
