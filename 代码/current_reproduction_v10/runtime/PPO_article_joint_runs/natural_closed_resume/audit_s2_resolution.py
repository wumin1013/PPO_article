"""Refine S2 command sampling without changing its geometry, schedule or hold time."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import natural_s2_adapter
from natural_case_audit import analytic_command_audit
from geometry_spec import audit_commands
from run_natural_resolution import physical_audit
from study_io import write_json,sha

def main():
    stage=OUT/'s2_resolution_half_ms';stage.mkdir(exist_ok=False)
    case=BASE/'paper_evidence_review/natural_closed_case';dt=.0005
    write_json(stage/'registration.json',dict(kind='S2_sampling_resolution',wall_budget_s=300,dt_s=dt,
        unchanged='All geometry, curvature enclosures, feed caps, scheduling and stationary25ms duration',
        training_updates=0,source_sha256=sha(Path(__file__)),adapter_sha256=sha(BASE/'natural_s2_adapter.py'),
        original_result_sha256=sha(case/'s2_complete/result.json')))
    original_loader=natural_s2_adapter.load_source
    def load(name,path,expected=None):
        module=original_loader(name,path,expected)
        if name=='natural_s2_command':
            assert module.DT_S==.001 and module.STATIONARY_HOLD_SAMPLES==25
            module.DT_S=dt;module.STATIONARY_HOLD_SAMPLES=50
        return module
    natural_s2_adapter.load_source=load
    start=time.perf_counter();reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    assembled,planner=natural_s2_adapter.build(reference,geometry,sha(case/'reference.csv'))
    assert time.perf_counter()-start<240,'S2 assembly budget exhausted'
    pos=assembled['pos'];rows=np.zeros((len(pos)-1,13));rows[:,:2]=pos[1:]
    rows[:,2:4]=np.diff(pos,axis=0)/dt
    rows[:,4:6]=np.diff(np.vstack([np.zeros(2),rows[:,2:4]]),axis=0)/dt
    rows[:,6:8]=np.diff(np.vstack([np.zeros(2),rows[:,4:6]]),axis=0)/dt
    ordered,_=audit_commands(reference,pos,endpoint_tolerance=1.25,closed=True)
    physical=physical_audit(reference,rows,dt);analytic=analytic_command_audit(reference[0],rows,geometry)
    success=bool(ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,dt_s=dt,t=assembled['t'],sigma=assembled['sigma'])
    result=dict(success=success,complete_time_s=len(rows)*dt if success else None,elapsed_prefix_or_complete_s=len(rows)*dt,
        geometry=ordered,audit=physical,analytic=analytic,dt_s=dt,wall_s=time.perf_counter()-start,
        trace_sha256=sha(stage/'trace.npz'),planner=planner)
    write_json(stage/'result.json',result)
    write_json(stage/'receipt.json',dict(status='complete_s2_resolution',goal_complete=False,
        wall_s=time.perf_counter()-start,result_sha256=sha(stage/'result.json'),training_updates=0))
    print('S2_RESOLUTION',success,result['complete_time_s'],physical,'wall',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
