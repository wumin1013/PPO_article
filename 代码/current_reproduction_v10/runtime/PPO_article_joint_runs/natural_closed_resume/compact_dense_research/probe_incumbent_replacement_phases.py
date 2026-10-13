"""Check the preserved incumbent as an actual18arrival replacement teacher."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from evaluate_historical_dense import ROOT,BASE,write,sha

PARENT=ROOT/'direct14_actual_arrival_fit_round52'
TEACHER=ROOT/'direct14_farvector_local_feedback_fit_round21/fit_08000.pt'


def worker(dt,stage,deadline):
    import numpy as np
    import torch
    import joint_runtime
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
    geometry=json.loads((BASE/'paper_evidence_review/natural_closed_case/geometry.json').read_text(encoding='utf-8'))
    trace=PARENT/f'eval_0000_original_{key}.npz';gates=PARENT/f'eval_0000_original_{key}_gates.json'
    ref,_,contexts,audit=reconstruct(trace,gates,geometry,['corner_18'],dt);context=contexts[0]
    prior=ROOT/'direct14_same_arrival_diagnosis_round53'/key/'context_audit.json';verified=json.loads(prior.read_text(encoding='utf-8'))
    assert verified['trace_sha256']==audit['trace_sha256'] and verified['eight_decision_replay_error']==0.
    audit['prior_exact_replay_audit_sha256']=sha(prior);write(folder/'context_audit.json',audit)
    teacher=from_saved(torch.load(TEACHER,map_location='cpu',weights_only=False));cost=CornerDeficitCost(ref,geometry,dt=dt,metric_paths=HALF_METRICS if dt==.0005 else None)
    corner=next(c for c in geometry['sharp_corners'] if c['id']=='corner_18');records=[]
    for phase in [0,4,8,12,16]:
        if time.perf_counter()>deadline-10:raise TimeoutError('Registered replacement teacher limit')
        env=FarVectorRuntime(ref);env.restore(context['state'])
        if phase:
            z=env.held.copy()/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6]);old=joint_runtime.MACRO_TICKS
            try:
                joint_runtime.MACRO_TICKS=round(phase*.001/dt);_,_,done,_=env.step(z);assert not done
            finally:joint_runtime.MACRO_TICKS=old
        start=env.env.state();chunks=[];xs=[];zs=[];error=None
        try:
            for _ in range(64):
                x=env.observation();z=teacher.act(x,True)[0]
                _,_,done,info=env.step(z);chunks.append(info['rows']);xs.append(x);zs.append(z)
                if done:break
        except ValueError as exc:error=str(exc)
        rows=np.vstack(chunks);s=ordered_projection(ref,rows[:,:2]);lo=corner['reference_s_mm']-10;hi=corner['reference_s_mm']+10
        mask=(s>=lo)&(s<=hi);complete=bool(s.min()<lo and s.max()>hi)
        acc=np.diff(np.vstack([start['velocity'],rows[:,2:4]]),axis=0)/dt;j=np.diff(np.vstack([start['acceleration'],acc]),axis=0)/dt
        peaks=[float(np.linalg.norm(rows[:,2:4],axis=1).max()),float(np.linalg.norm(acc,axis=1).max()),float(np.linalg.norm(j,axis=1).max())]
        analytic=analytic_command_audit(start['position'],rows,geometry)
        valid=bool(error is None and not env.done and complete and analytic['analytic_corridor_pass'] and np.all(np.array(peaks)<=[100.000001,2000.001,62500.02]))
        deficit=float(np.maximum(100-np.linalg.norm(rows[mask,2:4],axis=1),0).sum()*dt)
        output=folder/f'corner_18_p{phase:02d}.npz';np.savez_compressed(output,observations=np.array(xs),latents=np.array(zs),rows=rows,
            start_position=start['position'],initial_velocity=start['velocity'],initial_acceleration=start['acceleration'],dt_s=dt)
        records.append(dict(id='corner_18',phase_ms=phase,dt_s=dt,valid=valid,budget_pass=bool(valid and deficit<=cost.budgets['corner_18']),
            deficit_mm=deficit,budget_mm=cost.budgets['corner_18'],window_complete=complete,analytic=analytic,peaks=peaks,
            reason=env.reason,error=error,trace=str(output),trace_sha256=sha(output)))
    write(folder/'result.json',records);return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args();assert Path(args.name).name==args.name
    stage=ROOT/args.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir();started=time.perf_counter();deadline=started+1600.
    sources=[Path(__file__),ROOT/'actual_execution_contexts.py',ROOT/'research_policy_loader.py'];hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Actual18arrival incumbent teacher phase validation;noPPO',
        teacher=str(TEACHER),teacher_sha256=sha(TEACHER),source_sha256=hashes,wall_budget_s=1800,deadline_s=1600,
        evidence_gap='Same-arrival diagnosis21passes18atbothdt;need fiveholdphasechecks before using replacement labels.',
        procedure='Original21actions on witnessed52failurearrival,2dt x phases0/4/8/12/16;unchanged physical executor.',
        adoption='Local teacher only ifall10checksvalid+budget;never a deployed route-specific switch or fullsuccess.',
        stop='Ten localchecks or deadline',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,dt,stage,deadline) for dt in [.001,.0005]];groups=[f.result() for f in futures]
    records=[r for g in groups for r in g];passed=all(r['budget_pass'] for r in records)
    write(stage/'result.json',dict(records=records,all_phases_both_dt_pass=passed,new_ppo_updates=0,wall_s=time.perf_counter()-started,goal_complete=False,method_frozen=False))
    write(stage/'receipt.json',dict(status='finished',sources_unchanged=all(sha(p)==h for p,h in hashes.items()),wall_s=time.perf_counter()-started))
    print('REPLACEMENT_TEACHER',passed,[(r['dt_s'],r['phase_ms'],r['valid'],r['deficit_mm']-r['budget_mm']) for r in records],flush=True)


if __name__=='__main__':main()
