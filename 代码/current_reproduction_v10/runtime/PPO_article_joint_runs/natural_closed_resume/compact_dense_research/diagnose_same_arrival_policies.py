"""Compare actual18arrival under preserved controllers;zero training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from evaluate_historical_dense import ROOT,BASE,write,sha

CHECKPOINTS=[ROOT/'direct14_farvector_local_feedback_fit_round21/fit_08000.pt',
    ROOT/'direct14_head_bias_search_round29/candidate_08/model.pt',
    ROOT/'direct14_localized_preview_fit_round49/checkpoint_0002.pt',
    ROOT/'direct14_actual_arrival_fit_round52_initialization/checkpoint_0000.pt',
    ROOT/'direct14_actual_arrival_fit_round52_initialization/checkpoint_0001.pt']


def worker(dt,stage,deadline):
    import numpy as np
    import torch
    from run_refined_turn_candidate import configure_resolution
    from actual_execution_contexts import reconstruct
    from compact_farvector_runtime import FarVectorRuntime
    from research_policy_loader import from_saved
    from corner_deficit_objective import CornerDeficitCost
    from train_mixed_resolution_ppo import HALF_METRICS
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import ordered_projection
    if dt!=.001:configure_resolution(dt)
    torch.set_num_threads(1);key='1ms' if dt==.001 else 'half_ms';folder=stage/key;folder.mkdir()
    source=ROOT/'direct14_actual_arrival_fit_round52';trace=source/f'eval_0000_original_{key}.npz';gates=source/f'eval_0000_original_{key}_gates.json'
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    ref,expected,contexts,audit=reconstruct(trace,gates,geometry,['corner_18'],dt);context=contexts[0]
    policy=from_saved(torch.load(CHECKPOINTS[3],map_location='cpu',weights_only=False));env=FarVectorRuntime(ref);env.restore(context['state']);chunks=[]
    for _ in range(8):
        _,_,done,info=env.step(policy.act(env.observation(),True)[0]);chunks.append(info['rows']);assert not done
    rows=np.vstack(chunks);error=float(abs(rows[:,:4]-expected[context['tick']:context['tick']+len(rows),:4]).max());assert error<2e-5
    audit['eight_decision_replay_error']=error;write(folder/'context_audit.json',audit)
    corner=next(c for c in geometry['sharp_corners'] if c['id']=='corner_18');cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=HALF_METRICS if dt==.0005 else None)
    results=[]
    for i,path in enumerate(CHECKPOINTS):
        if time.perf_counter()>deadline-10:raise TimeoutError('Registered diagnosis limit')
        policy=from_saved(torch.load(path,map_location='cpu',weights_only=False));env=FarVectorRuntime(ref);env.restore(context['state']);start=env.env.state();chunks=[];error=None
        try:
            for _ in range(64):
                _,_,done,info=env.step(policy.act(env.observation(),True)[0]);chunks.append(info['rows'])
                if done:break
        except ValueError as exc:error=str(exc)
        rows=np.vstack(chunks);s=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10;hi=corner['reference_s_mm']+10;mask=(s>=lo)&(s<=hi)
        complete=bool(s.min()<lo and s.max()>hi);analytic=analytic_command_audit(start['position'],rows,geometry)
        a=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt;j=np.diff(np.vstack([start['acceleration'],a]),axis=0)/dt
        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(a,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
        valid=bool(error is None and not env.done and complete and analytic['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
        deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0).sum()*dt)
        out=folder/f'policy_{i}.npz';np.savez_compressed(out,rows=rows,start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
        result=dict(index=i,checkpoint=str(path),checkpoint_sha256=sha(path),valid=valid,window_complete=complete,deficit_mm=deficit,
            budget_mm=cost.budgets['corner_18'],budget_pass=bool(valid and deficit<=cost.budgets['corner_18']),analytic=analytic,peaks=peaks,error=error,reason=env.reason,trace_sha256=sha(out))
        results.append(result)
    write(folder/'result.json',results);return dict(dt_s=dt,results=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args();assert Path(args.name).name==args.name
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir();started=time.perf_counter();deadline=started+1600.
    sources=[Path(__file__),ROOT/'actual_execution_contexts.py',ROOT/'research_policy_loader.py'];hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Same actual-arrival causal diagnostic;zeroPPO',
        checkpoint_sha256={str(p):sha(p) for p in CHECKPOINTS},source_sha256=hashes,wall_budget_s=1800,deadline_s=1600,
        evidence_gap='52candidate0failsat18afterpassing14;separate added correction effects from changed arrival state.',
        procedure='Five policies share each witnessed failure-run18arrival at bothdt;8decision identity replay before comparison.',
        adoption='Diagnostic only,not a new full policy or efficiency claim',stop='Ten local executions or deadline',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,dt,stage,deadline) for dt in [.001,.0005]];results=[f.result() for f in futures]
    write(stage/'result.json',dict(results=results,wall_s=time.perf_counter()-started,new_ppo_updates=0,goal_complete=False))
    write(stage/'receipt.json',dict(sources_unchanged=all(sha(p)==h for p,h in hashes.items()),wall_s=time.perf_counter()-started,status='finished'))
    for pair in results:print('SAME_ARRIVAL',pair['dt_s'],[(r['index'],r['valid'],r['budget_pass'],r['deficit_mm']) for r in pair['results']],flush=True)


if __name__=='__main__':main()
