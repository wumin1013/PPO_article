"""Actual-arrival local feedback feasibility across physical dt and hold phases."""
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
from concurrent.futures import ProcessPoolExecutor
from evaluate_historical_dense import ROOT,BASE,write,sha,verify_bound_inputs

SOURCE=ROOT/'direct14_localized_preview_fit_round49'
RESPONSES=list(itertools.product([0.,-.002,.002],[-.04,0.,.04],[.9,1.,1.1]))
PHASES=[0,4,8,12,16]


def worker(dt,stage,deadline,source=SOURCE,checkpoint_index=2,corner_ids=('corner_10','corner_14'),extra_budgets=None,evaluation_checkpoint=None):
    import numpy as np
    import torch
    import joint_runtime
    from run_refined_turn_candidate import configure_resolution
    from compact_farvector_runtime import FarVectorRuntime
    from research_policy_loader import from_saved
    from actual_execution_contexts import reconstruct
    from corner_deficit_objective import CornerDeficitCost
    from train_mixed_resolution_ppo import HALF_METRICS
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    from observable_feedback_gate import modify,strength
    if dt!=.001:configure_resolution(dt)
    torch.set_num_threads(1);key='1ms' if dt==.001 else 'half_ms';folder=stage/key;folder.mkdir()
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    trace=source/f'eval_{checkpoint_index:04d}_original_{key}.npz';gates=source/f'eval_{checkpoint_index:04d}_original_{key}_gates.json'
    ref,expected,contexts,audit=reconstruct(trace,gates,geometry,corner_ids,dt)
    policy=from_saved(torch.load(source/f'checkpoint_{checkpoint_index:04d}.pt',map_location='cpu',weights_only=False))
    metric_paths=HALF_METRICS if dt==.0005 else None
    if extra_budgets:
        extra=json.loads(extra_budgets.read_text(encoding='utf-8'))
        if metric_paths is None:
            protocol=json.loads((ROOT/'joint_lowdim_acceptance_20260922.json').read_text(encoding='utf-8'))
            metric_paths=[Path(v['path']) for v in protocol['original']['comparators'].values()]
        metric_paths=list(metric_paths)+[Path(v) for v in extra[key]]
    cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=metric_paths)
    audit['budget_source_sha256']=cost.source_sha256
    errors=[];ticks=0;matrix=[]
    for context in contexts:
        env=FarVectorRuntime(ref);env.restore(context['state']);chunks=[]
        for _ in range(8):
            _,_,done,info=env.step(policy.act(env.observation(),True)[0]);chunks.append(info['rows']);assert not done
        rows=np.vstack(chunks);ticks+=len(rows);error=float(abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4]).max())
        assert error<2e-5,(key,context['id'],error);errors.append(error)
    audit['eight_decision_replay_xyv_errors']=errors
    if evaluation_checkpoint is not None:
        audit['evaluated_checkpoint']=str(evaluation_checkpoint)
        audit['evaluated_checkpoint_sha256']=sha(evaluation_checkpoint)
        policy=from_saved(torch.load(evaluation_checkpoint,map_location='cpu',weights_only=False))
    write(folder/'context_audit.json',audit)
    for context in contexts:
        ident=context['id'];corner=next(v for v in geometry['sharp_corners'] if v['id']==ident)
        for index,(steer,feed,gain) in enumerate(RESPONSES):
            for phase in PHASES:
                if time.perf_counter()>deadline-10:raise TimeoutError('Registered actual-arrival probe deadline')
                env=FarVectorRuntime(ref);env.restore(context['state']);error=None
                if phase:
                    z=env.held.copy()/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
                    old=joint_runtime.MACRO_TICKS
                    try:
                        joint_runtime.MACRO_TICKS=round(phase*.001/dt)
                        _,_,done,info=env.step(z);ticks+=len(info['rows']);assert not done
                    finally:joint_runtime.MACRO_TICKS=old
                start=env.env.state();chunks=[];xs=[];zs=[]
                try:
                    for _ in range(64):
                        x=env.observation();z=modify(policy.act(x,True)[0],x,feed,gain);z[0]+=strength(x)*steer
                        _,_,done,info=env.step(z);chunks.append(info['rows']);xs.append(x);zs.append(z)
                        if done:break
                except ValueError as exc:error=str(exc)
                rows=np.vstack(chunks) if chunks else np.empty((0,13));ticks+=len(rows)
                valid=False;complete=False;deficit=None;peaks=[];analytic={}
                if len(rows):
                    progress=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10;hi=corner['reference_s_mm']+10
                    complete=bool(progress.min()<lo and progress.max()>hi);mask=(progress>=lo)&(progress<=hi)
                    acc=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt;jerk=np.diff(np.vstack([start['acceleration'],acc]),axis=0)/dt
                    peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(acc,axis=1).max()),float(np.linalg.norm(jerk,axis=1).max())]
                    analytic=analytic_command_audit(start['position'],rows,geometry)
                    valid=bool(error is None and not env.done and complete and analytic['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
                    deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0).sum()*dt)
                output=folder/f'{ident}_r{index:02d}_p{phase:02d}.npz'
                np.savez_compressed(output,observations=np.array(xs),latents=np.array(zs),rows=rows,
                    start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
                item=dict(id=ident,index=index,response=[steer,feed,gain],dt_s=dt,phase_ms=phase,valid=valid,
                    budget_pass=bool(valid and deficit<=cost.budgets[ident]),deficit_mm=deficit,budget_mm=cost.budgets[ident],
                    window_complete=complete,reason=env.reason,error=error,peaks=peaks,analytic=analytic,trace=str(output),trace_sha256=sha(output))
                matrix.append(item)
                if not item['budget_pass']:break
            write(folder/'matrix.json',matrix)
        print('ACTUAL_ARRIVAL',key,ident,'records',len(matrix),flush=True)
    return dict(matrix=matrix,physics_ticks=ticks)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--index',type=int,default=2)
    p.add_argument('--corners',nargs='+',default=['corner_10','corner_14'])
    p.add_argument('--wall-budget',type=float,default=1800.)
    p.add_argument('--extra-budgets',type=Path);args=p.parse_args()
    assert 60<=args.wall_budget<=1800
    assert Path(args.name).name==args.name;verify_bound_inputs()
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir();started=time.perf_counter();deadline=started+args.wall_budget-30.
    sources=[Path(__file__)]+[ROOT/n for n in ['actual_execution_contexts.py','research_policy_loader.py','localized_joint_correction.py','localized_preview_head.py','observable_feedback_gate.py','corner_deficit_objective.py']]
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Actual-arrival local action feasibility probe;notPPO',
        checkpoint=str(args.source/f'checkpoint_{args.index:04d}.pt'),checkpoint_sha256=sha(args.source/f'checkpoint_{args.index:04d}.pt'),source_sha256=hashes,
        responses=RESPONSES,phases_ms=PHASES,corners=args.corners,dt_s=[.001,.0005],wall_budget_s=args.wall_budget,deadline_s=args.wall_budget-30.,
        extra_budget_file=str(args.extra_budgets) if args.extra_budgets else None,
        extra_budget_sha256=sha(args.extra_budgets) if args.extra_budgets else None,
        evidence_gap='Old local-start screening can pass while actual full execution fails a corner physical or deficit gate.',
        procedure='Reconstruct registered actual complete or explicitly failed-prefix states,verify8decisionunmodifiedreplay,probe small steering/feed/preview feedback across2dt x5phases.',
        adoption='Local targets only:every phase and dt must pass physical+complete-window+deficit. Never count failed prefix as complete success or a set of local rules as one full policy.',
        stop='Finite27responses per selected corner or deadline;skip remaining phases once a response fails.',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,dt,stage,deadline,args.source,args.index,args.corners,args.extra_budgets) for dt in [.001,.0005]];results=[f.result() for f in futures]
    matrix=[r for result in results for r in result['matrix']];selected=[]
    for ident in args.corners:
        good=[]
        for index,response in enumerate(RESPONSES):
            items=[r for r in matrix if r['id']==ident and r['index']==index]
            if len(items)==10 and all(r['budget_pass'] for r in items):good.append(dict(index=index,response=response,worst_excess_mm=max(r['deficit_mm']-r['budget_mm'] for r in items),traces=[r['trace'] for r in items]))
        good.sort(key=lambda r:r['worst_excess_mm']);selected.append(dict(id=ident,eligible_count=len(good),selected=good[0] if good else None))
    write(stage/'result.json',dict(selected=selected,physics_ticks=sum(r['physics_ticks'] for r in results),wall_s=time.perf_counter()-started,new_ppo_updates=0,goal_complete=False,method_frozen=False))
    write(stage/'receipt.json',dict(status='finished',sources_unchanged=all(sha(p)==h for p,h in hashes.items()),wall_s=time.perf_counter()-started));print(json.dumps(selected),flush=True)


if __name__=='__main__':main()
