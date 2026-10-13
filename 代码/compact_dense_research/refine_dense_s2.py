"""Sample the unchanged dense S2 schedule at0.5ms; no parameter optimization."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT,BASE,CASE,sha,write

STAGE=ROOT/'dense_s2_half_ms'


def worker():
    import numpy as np
    import natural_s2_adapter as adapter
    from geometry_spec import audit_commands
    from natural_case_audit import analytic_command_audit
    from run_natural_resolution import physical_audit
    from analyze_dense_trace import analyze
    dt=.0005
    loader=adapter.load_source
    def load(name,path,expected=None):
        module=loader(name,path,expected)
        if name=='natural_s2_command':
            assert module.DT_S==.001 and module.STATIONARY_HOLD_SAMPLES==25
            module.DT_S=dt
            module.STATIONARY_HOLD_SAMPLES=50
        return module
    adapter.load_source=load
    start=time.perf_counter()
    reference=np.loadtxt(CASE/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    assembled,planner=adapter.build(reference,geometry,sha(CASE/'reference.csv'))
    planner['actual_registered_case_id']='dense_closed_case'
    pos=assembled['pos'];rows=np.zeros((len(pos)-1,13));rows[:,:2]=pos[1:]
    rows[:,2:4]=np.diff(pos,axis=0)/dt
    rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/dt
    rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/dt
    ordered,events=audit_commands(reference,pos,endpoint_tolerance=1.25,closed=True)
    physical=physical_audit(reference,rows,dt)
    analytic=analytic_command_audit(reference[0],rows,geometry)
    success=bool(ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    np.savez_compressed(STAGE/'trace.npz',reference=reference,rows=rows,dt_s=dt,t=assembled['t'],sigma=assembled['sigma'])
    result=dict(success=success,complete_time_s=len(rows)*dt if success else None,
        elapsed_prefix_or_complete_s=len(rows)*dt,geometry=ordered,audit=physical,analytic=analytic,
        planner=planner,dt_s=dt,trace_sha256=sha(STAGE/'trace.npz'),wall_s=time.perf_counter()-start,
        goal_complete=False,ppo_updates=0,parameter_updates=0)
    write(STAGE/'result.json',result);write(STAGE/'gate_events.json',events)
    if success:write(STAGE/'all84_metrics.json',analyze(STAGE/'trace.npz',STAGE/'result.json',dt))
    print(json.dumps({k:v for k,v in result.items() if k!='planner'},ensure_ascii=False),flush=True)


def main():
    if '--worker' in sys.argv:worker();return
    STAGE.mkdir(exist_ok=False)
    files=[Path(__file__),BASE/'natural_s2_adapter.py',BASE/'run_natural_resolution.py']
    hashes={str(p):sha(p) for p in files}
    (STAGE/'source').mkdir()
    for p in files:shutil.copy2(p,STAGE/'source'/p.name)
    write(STAGE/'registration.json',dict(kind='Unchanged dense S2 schedule,half-ms sampling',
        wall_budget_s=300,child_timeout_s=270,dt_s=.0005,ppo_updates=0,parameter_updates=0,
        source_sha256=hashes,reference_sha256=sha(CASE/'reference.csv'),
        original_result_sha256=sha(ROOT/'dense_s2_initial/result.json'),
        unchanged='Geometry, curvature enclosures, feed caps, schedule and25ms stationary duration.',
        evidence_gap='Match physical audit resolution to the refined main-method diagnostic.',
        adoption='Require full corridor, ordered coverage, vector V/A/J and >=20ms rest.',
        stop='No planner adjustment or retuning; preserve failure or timeout.'))
    start=time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out,(STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=270)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-4000:])
    if code!=0:print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()
