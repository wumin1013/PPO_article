"""Local feed/preview teacher search at both dt;never a whole-path result."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import itertools
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs

SOURCE=ROOT/'direct14_farvector_local_feedback_fit_round21'
INITIAL=ROOT/'direct14_head_bias_search_round29/candidate_08/model.pt'
RESPONSES=list(itertools.product([-.04,-.02,0.,.02],[.9,1.,1.1,1.2,1.3]))
OBSERVABLE_RESPONSES=list(itertools.product([-.04,-.02,0.,.02],[1.,1.1,1.2,1.3,1.4,1.5]))
PHASE_MS=[0,4,8,12,16]


def worker(dt,stage,deadline,observable_only=False):
    import numpy as np
    import torch
    import joint_runtime
    from run_refined_turn_candidate import configure_resolution
    from compact_fine_policy import FineCompactPolicy
    from compact_farvector_runtime import FarVectorRuntime
    from saved_local_contexts import from_complete_trace
    from corner_deficit_objective import CornerDeficitCost
    from train_mixed_resolution_ppo import HALF_METRICS
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    from observable_feedback_gate import modify
    responses=OBSERVABLE_RESPONSES if observable_only else RESPONSES
    if dt!=.001:configure_resolution(dt)
    torch.set_num_threads(1);key='1ms' if dt==.001 else 'half_ms';folder=stage/key;folder.mkdir()
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    full=SOURCE/'eval_8000_original.npz' if dt==.001 else ROOT/'direct14_farvector_local_feedback_fit_round21_original_half_ms/trace.npz'
    gates=SOURCE/'eval_8000_original_gates.json' if dt==.001 else full.parent/'gate_events.json'
    ref,expected,contexts,audit=from_complete_trace(full,gates,geometry,dt=dt)
    corners={c['id']:c for c in geometry['sharp_corners']}
    cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=HALF_METRICS if dt==.0005 else None)
    policy=FineCompactPolicy();policy.load_state_dict(torch.load(INITIAL,map_location='cpu',weights_only=False)['policy'])
    original=FineCompactPolicy();original.load_state_dict(torch.load(SOURCE/'fit_08000.pt',map_location='cpu',weights_only=False)['policy'])
    matrix=[];physics_ticks=0
    for context in contexts:
        env=FarVectorRuntime(ref);env.restore(context['state']);rows=[]
        for _ in range(4):
            _,_,done,info=env.step(original.act(env.observation(),True)[0]);rows.append(info['rows']);assert not done
        rows=np.vstack(rows);physics_ticks+=len(rows)
        assert abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4]).max()<2e-5
    audit['four_decision_replays_all34_pass']=True;write(folder/'context_audit.json',audit)
    for ident in ['corner_06','corner_30']:
        context=next(c for c in contexts if c['id']==ident);corner=corners[ident]
        for index,(feed_delta,preview_gain) in enumerate(responses):
            for phase_ms in PHASE_MS:
                if time.perf_counter()>deadline-5:raise TimeoutError('Registered local probe deadline')
                env=FarVectorRuntime(ref);env.restore(context['state'])
                if phase_ms:
                    held=env.held.copy();latent=held/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
                    before=joint_runtime.MACRO_TICKS
                    try:
                        joint_runtime.MACRO_TICKS=round(phase_ms*.001/dt)
                        _,_,done,info=env.step(latent);physics_ticks+=len(info['rows']);assert not done
                    finally:joint_runtime.MACRO_TICKS=before
                start=env.env.state();chunks=[];observations=[];latents=[]
                for _ in range(64):
                    x=env.observation();z=policy.act(x,True)[0].copy()
                    if observable_only:z=modify(z,x,feed_delta,preview_gain)
                    else:
                        z[1]+=feed_delta;z[2]=((6.4+5.6*float(z[2]))*preview_gain-6.4)/5.6
                    _,_,done,info=env.step(z);chunks.append(info['rows']);observations.append(x);latents.append(z)
                    if done:break
                rows=np.vstack(chunks);physics_ticks+=len(rows)
                p=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10.;hi=corner['reference_s_mm']+10.
                mask=(p>=lo)&(p<=hi);complete=bool(p.min()<lo and p.max()>hi)
                a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt
                j=np.diff(np.vstack([start['acceleration'],a]),axis=0)/dt
                peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(a,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
                corridor=analytic_command_audit(start['position'],rows,geometry)
                valid=bool(not env.done and complete and corridor['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
                deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0.).sum()*dt)
                trace=folder/f'{ident}_r{index:02d}_p{phase_ms:02d}.npz'
                np.savez_compressed(trace,observations=np.array(observations),latents=np.array(latents),rows=rows,
                                    start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
                item=dict(id=ident,index=index,feed_delta=feed_delta,preview_gain=preview_gain,dt_s=dt,phase_ms=phase_ms,
                          valid=valid,reason=env.reason,window_complete=complete,deficit_mm=deficit,budget_mm=cost.budgets[ident],
                          budget_pass=bool(valid and deficit<=cost.budgets[ident]),peaks=peaks,
                          maximum_distance_mm=corridor['maximum_evaluated_distance_mm'],trace=str(trace),trace_sha256=sha(trace))
                matrix.append(item)
                if not item['budget_pass']:break
            write(folder/'matrix.json',matrix)
        print('LOCAL_PROBE',key,ident,'records',len(matrix),flush=True)
    report=dict(matrix=matrix,new_physics_ticks=physics_ticks,new_ppo_updates=0,goal_complete=False)
    write(folder/'result.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--observable-only',action='store_true');args=parser.parse_args()
    assert Path(args.name).name==args.name
    verify_bound_inputs();stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['saved_local_contexts.py','corner_deficit_objective.py','compact_fine_policy.py','compact_farvector_runtime.py','observable_feedback_gate.py']]
    responses=OBSERVABLE_RESPONSES if args.observable_only else RESPONSES
    hashes={str(p):sha(p) for p in sources}
    for p in sources:shutil.copy2(p,stage/'source'/p.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),wall_budget_s=1800,worker_deadline_s=1600,
        kind='Local physical response search,zeroPPO',checkpoint=str(INITIAL),checkpoint_sha256=sha(INITIAL),source_sha256=hashes,
        dt_s=[.001,.0005],ordinary_decision_s=.02,responses=responses,phase_ms=PHASE_MS,corners=['corner_06','corner_30'],
        observable_feedback_gate=args.observable_only,
        observation_gate='smoothstep((max(hypot(1-x5,x13),abs(x6))-.005)/.145),clipped0to1' if args.observable_only else None,
        evidence_gap='Global feed bias has good dense paired runs but misses06and30locally;need feasible paired local targets before fitting.',
        adoption='Teacher data only:require complete local window,all physical bounds,and registered deficit budget at every checked phase and dt. Never call a collection of local policies one full policy.',
        stop='Registered finite grid or deadline;skip remaining phases as soon as a response fails;no final acceptance.',goal_complete=False,method_frozen=False))
    started=time.perf_counter();deadline=started+1600.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,dt,stage,deadline,args.observable_only) for dt in [.001,.0005]]
        results=[f.result() for f in futures]
    matrix=[r for result in results for r in result['matrix']];selected=[]
    for ident in ['corner_06','corner_30']:
        eligible=[]
        for index,response in enumerate(responses):
            rows=[r for r in matrix if r['id']==ident and r['index']==index]
            if len(rows)==2*len(PHASE_MS) and all(r['budget_pass'] for r in rows):
                eligible.append(dict(id=ident,index=index,response=response,worst_excess_mm=max(r['deficit_mm']-r['budget_mm'] for r in rows),
                                     traces=[r['trace'] for r in rows]))
        eligible.sort(key=lambda r:r['worst_excess_mm'])
        selected.append(dict(id=ident,eligible_count=len(eligible),selected=eligible[0] if eligible else None))
    write(stage/'result.json',dict(selected=selected,wall_s=time.perf_counter()-started,new_ppo_updates=0,
        new_physics_ticks=sum(r['new_physics_ticks'] for r in results),scope='Local teacher targets only;not full-task success.',goal_complete=False,method_frozen=False))
    write(stage/'receipt.json',dict(status='finished',wall_s=time.perf_counter()-started,sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':main()
