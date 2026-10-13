"""Full S2 execution and independent audit for the unchanged natural case."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json
import time
import shutil
import argparse
import numpy as np
from study_io import BASE, ROOT, register, receipt, write_json, sha
from natural_s2_adapter import build
from natural_case_audit import analytic_command_audit
from geometry_spec import audit_commands
from validate_actions import difference_audit
from complete_motion_metrics import complete_motion_metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-enclosure',action='store_true');args=parser.parse_args()
    case=BASE/'paper_evidence_review/natural_closed_case'
    source=ROOT/'论文项目/revision_20260913_preview_efficiency_v2/conventional_ts_v1'
    if args.resume_enclosure:
        stage=case/'s2_complete'
        assert (stage/'registration.json').exists() and not (stage/'receipt.json').exists()
        write_json(stage/'initial_failure.json',dict(
            error='ValueError: Bounded curve-peak enclosure did not reach its declared tolerance',
            complete_command_generated=False,source_snapshot='source/1_natural_s2_adapter.py',
            repair='Use a 1e-3 enclosure tightness target; certified upper bounds still set every speed cap.',
            not_a_physical_trajectory_failure=True))
        for name in ('run_natural_s2.py','natural_s2_adapter.py'):
            shutil.copy2(BASE/name,stage/'source'/('enclosure_repair_'+name))
        write_json(stage/'enclosure_repair_registration.json',dict(
            before_command_generation=True,additional_wall_limit_s=600,
            sources={name:sha(BASE/name) for name in ('run_natural_s2.py','natural_s2_adapter.py')}))
    else:
        stage=register('paper_evidence_review/natural_closed_case/s2_complete',
        [__file__,BASE/'natural_s2_adapter.py',BASE/'natural_case_audit.py',case/'reference.csv',case/'geometry.json']+
        [source/n for n in ('geometry.py','scheduler.py','command.py')],
        dict(wall_limit_s=900,full_case_executions=1,training_updates=0,execution_ms=1),
        'Evaluate adapted S2 on the same analytic case. Whole-geometry reflection is an isometry. '
        'The small-angle empirical ratio extension is explicit, with independent geometry and '
        'Cartesian constraints. No per-angle fitted time parameters or hidden speed multiplier.')
    start=time.perf_counter();reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    assembled,meta=build(reference,geometry,sha(case/'reference.csv'))
    pos=assembled['pos']; rows=np.zeros((len(pos)-1,13));rows[:,:2]=pos[1:]
    rows[:,2:4]=np.diff(pos,axis=0)/.001
    rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/.001
    rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/.001
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,t=assembled['t'],sigma=assembled['sigma'])
    write_json(stage/'planner.json',meta)
    print('S2_ASSEMBLED',len(rows),'time',len(rows)*.001,flush=True)
    ordered,_=audit_commands(reference,pos,endpoint_tolerance=1.25,closed=True)
    physical=difference_audit(reference[0],rows)
    analytic=analytic_command_audit(reference[0],rows,geometry)
    success=ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass']
    result=dict(success=bool(success),complete_time_s=len(rows)*.001 if success else None,
        elapsed_prefix_or_complete_s=len(rows)*.001,geometry=ordered,audit=physical,analytic=analytic,
        motion=complete_motion_metrics(reference,rows),wall_s=time.perf_counter()-start,trace_sha256=sha(stage/'trace.npz'))
    write_json(stage/'result.json',result)
    receipt(stage,dict(status='completed_s2_audit',success=bool(success),wall_s=time.perf_counter()-start,
        result_sha256=sha(stage/'result.json'),physics_ticks=len(rows),ppo_updates=0,goal_complete=False))
    print('S2_COMPLETE',success,result['complete_time_s'],physical,analytic,flush=True)


if __name__=='__main__':main()
