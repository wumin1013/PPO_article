"""Bounded unchanged S2 evaluation on the already registered dense case."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(ROOT.parent)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def worker(stage):
    from natural_s2_adapter import build
    from geometry_spec import audit_commands
    from natural_case_audit import analytic_command_audit
    from validate_actions import difference_audit
    case=ROOT/'dense_closed_case'
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    started=time.perf_counter()
    assembled,planner=build(reference,geometry,sha(case/'reference.csv'))
    planner['actual_registered_case_id']='dense_closed_case'
    pos=assembled['pos'];rows=np.zeros((len(pos)-1,13));rows[:,:2]=pos[1:]
    rows[:,2:4]=np.diff(pos,axis=0)/.001
    rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/.001
    rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/.001
    ordered,events=audit_commands(reference,pos,endpoint_tolerance=1.25,closed=True)
    physical=difference_audit(reference[0],rows)
    analytic=analytic_command_audit(reference[0],rows,geometry)
    success=bool(ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,dt_s=.001,t=assembled['t'],sigma=assembled['sigma'])
    write(stage/'gate_events.json',events)
    result=dict(success=success,complete_time_s=len(rows)*.001 if success else None,
        elapsed_prefix_or_complete_s=len(rows)*.001,geometry=ordered,audit=physical,analytic=analytic,
        planner=planner,trace_sha256=sha(stage/'trace.npz'),wall_s=time.perf_counter()-started,goal_complete=False)
    write(stage/'result.json',result)
    print(json.dumps({k:result[k] for k in ['success','complete_time_s','wall_s','audit','analytic']},indent=2),flush=True)

def main():
    stage=ROOT/'dense_s2_initial'
    if '--worker' in sys.argv:
        worker(stage);return
    stage.mkdir(exist_ok=False)
    sources=[Path(__file__),BASE/'natural_s2_adapter.py',BASE/'geometry_spec.py',BASE/'validate_actions.py',ROOT/'dense_closed_case/registration.json']
    write(stage/'registration.json',dict(kind='unchanged S2 full dense-case execution',wall_budget_s=600,child_timeout_s=550,
        physics_dt_s=.001,ppo_updates=0,source_sha256={str(p):sha(p) for p in sources},
        adoption='Complete geometry, V/A/J, all ordered gates, brake and rest must pass; preserve failures.',
        stop='Terminate this child at550s; no baseline speed reduction or case change.'))
    started=time.perf_counter()
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'],stdout=out,stderr=err,timeout=550)
            status='finished';code=run.returncode
        except subprocess.TimeoutExpired:
            status='timeout';code=None
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-6000:])

if __name__=='__main__':main()
