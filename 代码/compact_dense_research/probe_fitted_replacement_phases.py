"""Reuse the verified actual18 contexts to test fitted policies, not teachers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import multiprocessing as mp
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
from evaluate_historical_dense import ROOT,write,sha


def worker(index,dt,source,stage,deadline):
    import probe_incumbent_replacement_phases as probe
    probe.TEACHER=source/f'checkpoint_{index:04d}.pt'
    # Physical contexts remain exactly those already replayed and verified in53.
    records=probe.worker(dt,stage/f'candidate_{index}',deadline)
    return dict(index=index,dt_s=dt,records=records,
                all_pass=all(r['budget_pass'] for r in records))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    assert Path(a.name).name==a.name
    stage=ROOT/a.name;stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    started=time.perf_counter();deadline=started+270.
    sources=[Path(__file__)]+[ROOT/n for n in ['probe_incumbent_replacement_phases.py',
        'actual_execution_contexts.py','research_policy_loader.py','localized_joint_correction.py','localized_preview_head.py']]
    hashes={str(s):sha(s) for s in sources}
    for s in sources:shutil.copy2(s,stage/'source'/s.name)
    for i in range(3):(stage/f'candidate_{i}').mkdir()
    checkpoints={str(a.source/f'checkpoint_{i:04d}.pt'):sha(a.source/f'checkpoint_{i:04d}.pt') for i in range(3)}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Fitted policy local phase check;noPPO',source_sha256=hashes,checkpoints=checkpoints,
        wall_budget_s=300,deadline_s=270,evidence_gap='Teacher18works but fitted correction may fail to preserve physical behavior.',
        procedure='Three fitted55policies on verified52candidate0 actual18arrivals;two dt x five hold phases. Fresh process for every model/dt pair because resolution setup changes module globals.',
        adoption='Diagnostic only;complete execution and all34/time gates remain necessary.',
        stop='30 local checks or five minute deadline',new_ppo_updates=0,goal_complete=False,method_frozen=False))
    with mp.get_context('spawn').Pool(processes=2,maxtasksperchild=1) as pool:
        futures=[pool.apply_async(worker,(i,dt,a.source,stage,deadline)) for i in range(3) for dt in [.001,.0005]]
        results=[]
        for future in futures:
            result=future.get(timeout=max(1.,deadline-time.perf_counter()));results.append(result)
            print('FITTED_PHASE',result['index'],result['dt_s'],result['all_pass'],
                [(r['phase_ms'],r['valid'],r['deficit_mm']-r['budget_mm']) for r in result['records']],flush=True)
    write(stage/'result.json',dict(results=results,wall_s=time.perf_counter()-started,new_ppo_updates=0,goal_complete=False))
    write(stage/'receipt.json',dict(status='finished',wall_s=time.perf_counter()-started,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items())))


if __name__=='__main__':main()
