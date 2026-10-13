"""Bounded diagnostic from witnessed failed dense arrivals; never a full success."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import itertools
import json
import math
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT,CASE,write,sha,verify_bound_inputs
from localized_two_action_policy import policy_from
from compact_farvector_runtime import FarVectorRuntime
from run_refined_turn_candidate import configure_resolution
from natural_case_audit import analytic_command_audit


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=[923201,923202,923203],required=True)
    parser.add_argument('--dt',type=float,choices=[.001,.0005],required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    verify_bound_inputs()
    if args.dt!=.001:
        configure_resolution(args.dt)
    key='1ms' if args.dt==.001 else 'half_ms'
    parent=ROOT/f'supplement_margin_main_{args.seed}_ppo24'
    checkpoint=parent/'checkpoint_0024.pt'
    trace=parent/f'evaluation/eval_0024_dense_{key}.npz'
    result=read(trace.with_suffix('.json'))
    assert not result['success'] and result['trace_sha256']==sha(trace)
    assert result['checkpoint_sha256']==sha(checkpoint)
    gates=trace.with_name(trace.stem+'_gates.json')
    geometry_path=CASE/'geometry.json'
    geometry=read(geometry_path)
    teacher_path=ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'
    policy=policy_from(torch.load(checkpoint,map_location='cpu',weights_only=False),args.seed).eval()
    teacher=policy_from(torch.load(teacher_path,map_location='cpu',weights_only=False),args.seed).eval()
    with np.load(trace) as data:
        reference=data['reference'].copy();rows=data['rows'].copy()
        observations=data['observations'].copy();actions=data['actions'].copy();times=data['decision_times_s'].copy()
    events=read(gates)
    index=len(observations)-20
    assert index>0
    tick=round(times[index]/args.dt)
    runtime=FarVectorRuntime(reference)
    state=runtime.state();physical=state['physical'];row=rows[tick-1]
    frontier=int(np.searchsorted(np.array([e['command_segment'] for e in events]),tick,side='left'))
    assert abs(float((row[:2]-runtime.corridor.gates[frontier].point)@runtime.corridor.gates[frontier].normal))>1e-8
    assert np.linalg.norm(row[8:10])>1e-6
    physical.update(position=row[:2].copy(),velocity=row[2:4].copy(),acceleration=row[4:6].copy(),
        heading=math.atan2(row[9],row[8]),held=actions[index-1].copy(),ticks=tick,decisions=index,rest_ticks=0)
    physical['tracker'].update(position=row[:2].copy(),frontier=frontier,pending_negative=False,inside=True,
        events=copy.deepcopy(events[:frontier]),segment_count=tick)
    stage=ROOT/'experimental_supplement_20260923/dense_prefix_recovery_probe'/f'seed_{args.seed}'/key
    stage.mkdir(parents=True,exist_ok=False)
    bindings={str(p):sha(p) for p in [Path(__file__),checkpoint,trace,trace.with_suffix('.json'),gates,geometry_path,teacher_path]}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,dt_s=args.dt,
        source_sha256=bindings,source_complete_success=False,source_prefix_s=result['elapsed_prefix_s'],
        scope='Diagnostic local counterfactual from a witnessed failed-prefix state, not complete-case acceptance.',
        evidence_gap='Broad fitting/PPO remains fragile; determine whether local teacher feedback or small local request offsets can recover witnessed dense arrivals.',
        restore_decisions_before_failure=20,required_unmodified_replay_decisions=8,probe_horizon_decisions=40,
        offset_grid={'steer':[-.002,0.,.002],'feed':[-.02,0.,.02],'preview':[-.05,0.,.05]},
        additional_trial='Frozen73 teacher feedback from the same physical start.',wall_budget_s=300,
        adoption='None. Valid local recovery must have unchanged corridor/vectorVAJ, at least0.01mm sampled margin, and full40decisions. Any fitted policy still needs complete bothcases/bothsteps/all34/time evaluation.',
        stop='28trials or300s; preserve failed prefixes and no-completion scope.',new_ppo_updates=0))
    started=time.perf_counter();runtime.restore(state)
    observation_error=float(abs(runtime.observation()-observations[index]).max())
    assert observation_error<2e-6
    chunks=[]
    for _ in range(8):
        _,_,done,info=runtime.step(policy.act(runtime.observation(),True)[0])
        assert not done
        chunks.append(info['rows'])
    replay=np.vstack(chunks)
    error=float(abs(replay[:,:4]-rows[tick:tick+len(replay),:4]).max())
    assert error<2e-5,('Arrival replay not verified',error)
    arrival=runtime.state();initial_position=runtime.position.copy()
    initial_velocity=runtime.velocity.copy();initial_acceleration=runtime.acceleration.copy()
    records=[]
    trials=[('offset',np.array(delta)) for delta in itertools.product([-.002,0.,.002],[-.02,0.,.02],[-.05,0.,.05])]
    trials.append(('frozen_teacher',np.zeros(3)))
    for trial,(kind,delta) in enumerate(trials):
        if time.perf_counter()-started>300:
            break
        env=FarVectorRuntime(reference);env.restore(copy.deepcopy(arrival))
        chunks=[];xs=[];zs=[];error_text=None;done=False
        for _ in range(40):
            x=env.observation()
            z=(teacher if kind=='frozen_teacher' else policy).act(x,True)[0]+delta
            try:
                _,_,done,info=env.step(z)
            except Exception as exc:
                error_text=f'{type(exc).__name__}: {exc}'
                break
            xs.append(x);zs.append(z);chunks.append(info['rows'])
            if done:
                break
        actual=np.vstack(chunks) if chunks else np.empty((0,13))
        complete=len(xs)==40 and not done and error_text is None
        analytic=analytic_command_audit(initial_position,actual,geometry) if len(actual) else {}
        if len(actual):
            acceleration=np.diff(np.vstack([initial_velocity,actual[:,2:4]]),axis=0)/args.dt
            jerk=np.diff(np.vstack([initial_acceleration,acceleration]),axis=0)/args.dt
            peaks=[float(np.linalg.norm(v,axis=1).max()) for v in [actual[:,2:4],acceleration,jerk]]
            distance=max(env.corridor.point_distance(p) for p in actual[:,:2])
        else:
            peaks=[];distance=float('inf')
        valid=bool(complete and analytic.get('analytic_corridor_pass',False) and
                   np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
        output=stage/f'trial_{trial:02d}.npz'
        np.savez_compressed(output,rows=actual,observations=np.asarray(xs),latents=np.asarray(zs),
            initial_position=initial_position,initial_velocity=initial_velocity,initial_acceleration=initial_acceleration,dt_s=args.dt)
        record=dict(trial=trial,kind=kind,delta=delta.tolist(),local_complete=complete,local_valid=valid,
            sampled_margin_mm=1.25-distance,robust_local_candidate=bool(valid and distance<=1.24),
            elapsed_local_s=len(actual)*args.dt,progress_end_mm=env.progress()[0],reason=env.reason,error=error_text,
            peaks=peaks,analytic=analytic,trace=str(output),trace_sha256=sha(output))
        records.append(record);write(stage/'trials.json',records)
    write(stage/'replay_receipt.json',dict(status='finished' if len(records)==28 else 'timeout',
        wall_s=time.perf_counter()-started,trial_count=len(records),unmodified_replay_xyv_error=error,
        reconstructed_observation_error=observation_error,robust_local_trials=[r['trial'] for r in records if r['robust_local_candidate']],
        source_unchanged=all(sha(p)==h for p,h in bindings.items()),whole_case_success_demonstrated=False))
    print('PREFIX_PROBE',args.seed,key,'local_robust',[(r['trial'],r['kind'],r['delta'],r['sampled_margin_mm']) for r in records if r['robust_local_candidate']],flush=True)


if __name__=='__main__':
    main()
