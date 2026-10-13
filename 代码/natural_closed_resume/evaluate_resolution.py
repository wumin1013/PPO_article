"""Complete execution-resolution check of one immutable selected candidate."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,argparse,shutil
import numpy as np
import torch
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from run_natural_resolution import set_resolution,physical_audit
from natural_case_audit import analytic_command_audit
from natural_case_metrics import cruise_metrics
from study_io import sha,write_json
from arc_cruise_policy import build_policy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('--dt',type=float,default=.0005);args=ap.parse_args()
    candidate=Path(args.candidate);selection=json.loads((candidate/'selected_policy.json').read_text(encoding='utf-8'))
    registration=json.loads((candidate/'registration.json').read_text(encoding='utf-8'))
    stage=candidate/('resolution_'+str(args.dt));stage.mkdir(exist_ok=False)
    spec=selection['spec'];model=Path(selection['model']);assert sha(model)==selection['model_sha256']
    write_json(stage/'registration.json',dict(kind='immutable_candidate_execution_resolution',wall_budget_s=1800,
        dt_s=args.dt,decision_dt_s=.020,training_updates=0,selection_sha256=sha(candidate/'selected_policy.json'),
        model_sha256=sha(model),source_sha256=sha(Path(__file__)),
        policy_helper_sha256={p.name:sha(p) for p in (OUT/'cruise_residual_policy.py',OUT/'arc_cruise_policy.py',OUT/'turn_feedback_policy.py')},
        adoption='Complete geometry, vector V/A/J, >=20ms physical rest and analytic full-command corridor must pass. Compare against nominal full result; no policy tuning in this run.'))
    shutil.copy2(Path(__file__),stage/Path(__file__).name)
    for name in ('cruise_residual_policy.py','arc_cruise_policy.py','turn_feedback_policy.py'):
        shutil.copy2(OUT/name,stage/name)
    torch.set_num_threads(1);set_resolution(args.dt)
    case=BASE/'paper_evidence_review/natural_closed_case'
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    policy=build_policy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),spec)
    env=environment_factory(reference)
    if registration['arm']=='fixed':env.fixed_d=spec['fixed_d']
    policy.bind(env);start=time.perf_counter();last=start;chunks=[];observations=[];actions=[];decision_times=[];error=None
    try:
        while not env.done:
            if time.perf_counter()>start+1720:raise TimeoutError('Resolution round wall cap')
            x=env.observation();z=policy.act(x,True)[0]
            assert not np.any(env.pending_pulse)
            observations.append(x);decision_times.append(env.ticks*args.dt)
            _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
            if time.perf_counter()-last>40:
                last=time.perf_counter();print('RESOLUTION',args.dt,env.ticks*args.dt,env.tracker.frontier,flush=True)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks);physical=physical_audit(reference,rows,args.dt);ordered=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,geometry)
    success=bool(error is None and env.success and ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,observations=np.array(observations),
        actions=np.array(actions),decision_times_s=np.array(decision_times),dt_s=args.dt)
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=env.ticks*args.dt if success else None,
        elapsed_prefix_or_complete_s=env.ticks*args.dt,dt_s=args.dt,decision_dt_s=.020,geometry=ordered,audit=physical,analytic=analytic,
        cruise=cruise_metrics(np.linalg.norm(rows[:,2:4],axis=1),dt=args.dt),wall_s=time.perf_counter()-start,
        physics_ticks=len(rows),actor_calls=len(actions),trace_sha256=sha(stage/'trace.npz'),spec=spec)
    write_json(stage/'result.json',result);write_json(stage/'receipt.json',dict(status='complete_resolution_check',goal_complete=False,
        wall_s=time.perf_counter()-start,result_sha256=sha(stage/'result.json'),training_updates=0))
    print('RESOLUTION_COMPLETE',success,result['complete_time_s'],physical,analytic,flush=True)

if __name__=='__main__':main()
