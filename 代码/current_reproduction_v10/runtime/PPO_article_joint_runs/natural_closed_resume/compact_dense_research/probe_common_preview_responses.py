"""Find admissible common preview responses across all34development corners."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs
from probe_paired_local_repair import INITIAL,SOURCE


def worker(dt,stage,deadline):
    import numpy as np
    import torch
    from run_refined_turn_candidate import configure_resolution
    from compact_farvector_runtime import FarVectorRuntime
    from saved_local_contexts import from_complete_trace
    from corner_deficit_objective import CornerDeficitCost
    from train_mixed_resolution_ppo import HALF_METRICS,policy_from
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    from observable_feedback_gate import modify
    if dt!=.001:configure_resolution(dt)
    torch.set_num_threads(1);key='1ms' if dt==.001 else 'half_ms';folder=stage/key;folder.mkdir()
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    trace=SOURCE/'eval_8000_original.npz' if dt==.001 else ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz'
    gates=SOURCE/'eval_8000_original_gates.json' if dt==.001 else trace.parent/'gate_events.json'
    ref,_,contexts,audit=from_complete_trace(trace,gates,geometry,dt=dt)
    prior=ROOT/'direct14_preview_correction_ppo_round42'/f'context_audit_{key}.json'
    verified=json.loads(prior.read_text(encoding='utf-8'))
    assert verified['trace_sha256']==audit['trace_sha256'] and verified['maximum_replay_xyv_error']==0.
    audit['prior_four_step_replay_audit_sha256']=sha(prior);write(folder/'context_audit.json',audit)
    cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=HALF_METRICS if dt==.0005 else None)
    policy=policy_from(torch.load(INITIAL,map_location='cpu',weights_only=False),922302)
    records=[];ticks=0
    for context,corner in zip(contexts,geometry['sharp_corners']):
        for gain in [1.1,1.3]:
            if time.perf_counter()>deadline-10:raise TimeoutError('Registered local response deadline')
            env=FarVectorRuntime(ref);env.restore(context['state']);start=env.env.state()
            chunks=[];observations=[];actions=[];error=None
            try:
                for _ in range(64):
                    x=env.observation();z=modify(policy.act(x,True)[0],x,0.,gain)
                    _,_,done,info=env.step(z);chunks.append(info['rows']);observations.append(x);actions.append(z)
                    if done:break
            except ValueError as exc:error=str(exc)
            rows=np.vstack(chunks) if chunks else np.empty((0,13));ticks+=len(rows)
            valid=False;deficit=None;peaks=[];analytic={};complete=False
            if len(rows):
                progress=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10;hi=corner['reference_s_mm']+10
                mask=(progress>=lo)&(progress<=hi);complete=bool(progress.min()<lo and progress.max()>hi)
                a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt
                j=np.diff(np.vstack([start['acceleration'],a]),axis=0)/dt
                peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(a,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
                analytic=analytic_command_audit(start['position'],rows,geometry)
                valid=bool(error is None and not env.done and complete and analytic['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
                deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0).sum()*dt)
            path=folder/f'{context["id"]}_gain{gain:.1f}.npz'
            np.savez_compressed(path,observations=np.array(observations),latents=np.array(actions),rows=rows,
                start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
            records.append(dict(id=context['id'],gain=gain,valid=valid,budget_pass=bool(valid and deficit<=cost.budgets[context['id']]),
                deficit_mm=deficit,budget_mm=cost.budgets[context['id']],window_complete=complete,
                reason=env.reason,error=error,peaks=peaks,analytic=analytic,trace=str(path),trace_sha256=sha(path)))
        write(folder/'matrix.json',records)
        if len(records)%16==0:print('COMMON_PREVIEW',key,'corners',len(records)//2,flush=True)
    return dict(records=records,physics_ticks=ticks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    assert Path(args.name).name==args.name;verify_bound_inputs()
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+1600.
    sources=[Path(__file__)]+[ROOT/n for n in ['saved_local_contexts.py','corner_deficit_objective.py','observable_feedback_gate.py']]
    bindings={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Local response feasibility diagnostic;noPPO',
        checkpoint=str(INITIAL),checkpoint_sha256=sha(INITIAL),source_sha256=bindings,wall_budget_s=1800,deadline_s=1600,
        preview_gains=[1.1,1.3],feed_increment=0.,phase_ms=[0],dt_s=[.001,.0005],corners='All34',
        evidence_gap='Similar observable states get critical positive and noncritical zero correction labels;not all zero labels are proven necessary.',
        adoption='Diagnostic admissible teachers only;both dt valid window and budget required. Does not prove full success or decision-phase robustness.',
        stop='Finite2x34x2schedule or registered deadline. Do not change baselines or physics.',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,dt,stage,deadline) for dt in [.001,.0005]]
        results=[f.result() for f in futures]
    records=[r for result in results for r in result['records']];allowed={}
    for ident in sorted({r['id'] for r in records}):
        allowed[ident]=[gain for gain in [1.1,1.3] if all(r['budget_pass'] for r in records if r['id']==ident and r['gain']==gain)]
    write(stage/'result.json',dict(admissible_gains_at_both_dt=allowed,physics_ticks=sum(r['physics_ticks'] for r in results),
        wall_s=time.perf_counter()-started,new_ppo_updates=0,goal_complete=False,method_frozen=False))
    write(stage/'receipt.json',dict(status='finished',sources_unchanged=all(sha(p)==h for p,h in bindings.items()),wall_s=time.perf_counter()-started))
    print(json.dumps(allowed),flush=True)


if __name__=='__main__':main()
