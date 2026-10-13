"""Check fitted actors on replay-verified incumbent arrivals, without feedback."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
from evaluate_historical_dense import ROOT,write,sha


def worker(index,dt,source,stage,deadline,arrival_source,arrival_index,corner):
    import probe_actual_arrival_repairs as probe
    probe.RESPONSES=[(0.,0.,1.)]
    result=probe.worker(dt,stage/f'candidate_{index}',deadline,
        source=arrival_source,checkpoint_index=arrival_index,corner_ids=(corner,),
        extra_budgets=ROOT/'all34_two_action_comparator_addendum_20260923.json',
        evaluation_checkpoint=source/f'checkpoint_{index:04d}.pt')
    return dict(index=index,dt_s=dt,records=result['matrix'],physics_ticks=result['physics_ticks'],
        all_five_phases_pass=len(result['matrix'])==5 and all(r['budget_pass'] for r in result['matrix']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--arrival-source',type=Path,default=ROOT/'direct14_actual_arrival_fit_round55')
    p.add_argument('--arrival-index',type=int,default=1);p.add_argument('--corner',default='corner_30');a=p.parse_args()
    assert Path(a.name).name==a.name
    started=time.perf_counter();deadline=started+270.;stage=ROOT/a.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    sources=[Path(__file__)]+[ROOT/n for n in ['probe_actual_arrival_repairs.py','actual_execution_contexts.py',
        'research_policy_loader.py','localized_joint_correction.py','localized_preview_head.py','observable_feedback_gate.py']]
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    for i in range(3):(stage/f'candidate_{i}').mkdir()
    checkpoints={str((a.source/f'checkpoint_{i:04d}.pt').resolve()):sha(a.source/f'checkpoint_{i:04d}.pt') for i in range(3)}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Fitted actor phase diagnostic,zeroPPO',
        source_sha256=hashes,checkpoints=checkpoints,wall_budget_s=300,deadline_s=270,
        evidence_gap='A verified local feedback teacher does not establish that fitted actors preserve its response across arrival phases.',
        arrival_source=str(a.arrival_source.resolve()),arrival_index=a.arrival_index,corner=a.corner,
        arrival_checkpoint_sha256=sha(a.arrival_source/f'checkpoint_{a.arrival_index:04d}.pt'),
        procedure='Replay registered source actual selected-corner arrival8decisions exactly,then test fitted actors without feedback at2physicaldt and5holdphases. Separate process per model/dt. After a failed phase skip later phases and report incomplete phase coverage.',
        adoption='Diagnostic only;complete-path all34andtime gates remain required.',stop='At most30local executions or5minute deadline.',
        new_ppo_updates=0,goal_complete=False,method_frozen=False))
    results=[]
    with mp.get_context('spawn').Pool(processes=2,maxtasksperchild=1) as pool:
        futures=[pool.apply_async(worker,(i,dt,a.source,stage,deadline,a.arrival_source,a.arrival_index,a.corner)) for i in range(3) for dt in [.001,.0005]]
        for f in futures:
            result=f.get(timeout=max(1.,deadline-time.perf_counter()));results.append(result)
            print('ACTUAL_FITTED_PHASE',result['index'],result['dt_s'],result['all_five_phases_pass'],
                [(r['phase_ms'],r['valid'],r['deficit_mm']-r['budget_mm']) for r in result['records']],flush=True)
    write(stage/'result.json',dict(results=results,wall_s=time.perf_counter()-started,new_ppo_updates=0,goal_complete=False))
    write(stage/'receipt.json',dict(status='finished',sources_unchanged=all(sha(p)==h for p,h in hashes.items()),wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
