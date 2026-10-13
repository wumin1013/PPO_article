"""Half-ms check of an explicit turn-aware candidate, with 20ms decisions."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,time,json,math,pickle,shutil,argparse
import numpy as np
import torch
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
import joint_runtime
import turn_aware_projector as turn
from candidate_policy_factory import candidate_policy
from arc_cruise_policy import build_policy
from deep_joint_credit_policy import load_credit_policy
from cache_corner_contexts import memory_state
from natural_case_metrics import ordered_projection,cruise_metrics
from natural_case_audit import analytic_command_audit
from run_natural_resolution import physical_audit

def configure_resolution(dt):
    joint_runtime.DT=dt;joint_runtime.MACRO_TICKS=round(.020/dt)
    turn.DT=dt;turn.ORIGINAL.DT=dt
    turn.ORIGINAL.MAX_RELEASE_TERMS=math.ceil(turn.ORIGINAL.AMAX_INTERNAL/(turn.ORIGINAL.JMAX_INTERNAL*dt))
    original_type=turn.TurnAwareProjector
    class RefinedProjector(original_type):
        def enforce_kinematic_constraints(self,v,a,j,dt_override=None):
            return super().enforce_kinematic_constraints(v,a,j,dt=dt)
    turn.TurnAwareProjector=RefinedProjector

def backup_check(dt,deadline):
    cases=0;maximum=np.zeros(3);max_steps=0
    for radius in (1.,10.,50.,80.,100.-2e-8):
        for fraction in (0.,.01,.25,.5,.9,.999):
            magnitude=min(1997.,math.sqrt(62375.*radius)*fraction);delta=2*math.asin(magnitude*dt/(2*radius))
            for angle in (0.,.4,2.,-1.7):
                for sign in (-1,1):
                    v=radius*np.array([math.cos(angle),math.sin(angle)])
                    previous=radius*np.array([math.cos(angle-sign*delta),math.sin(angle-sign*delta)]);a=(v-previous)/dt
                    assert turn.turn_certificate(v,a);steps=0
                    while np.linalg.norm(a)>1e-8:
                        if time.perf_counter()>deadline:raise TimeoutError('Refined backup-check budget')
                        vv,aa,j=turn.turn_backup(v,a);raw_a=(vv-v)/dt;raw_j=(raw_a-a)/dt
                        maximum=np.maximum(maximum,[np.linalg.norm(vv),np.linalg.norm(raw_a),np.linalg.norm(raw_j)])
                        assert np.all(maximum<=[100.000001,2000.00001,62500.01]);assert np.linalg.norm(aa)<=np.linalg.norm(a)+1e-7
                        v,a=vv,aa;steps+=1;assert steps<20000
                    cases+=1;max_steps=max(max_steps,steps)
    return dict(cases=cases,peaks=maximum.tolist(),maximum_backup_steps=max_steps,dt_s=dt,passed=True)

def corner_stats(reference,rows,geometry,dt):
    progress=ordered_projection(reference,rows[:,:2]);speed=np.linalg.norm(rows[:,2:4],axis=1);records=[]
    for c in geometry['sharp_corners']:
        low=c['reference_s_mm']-c['window_before_mm'];high=c['reference_s_mm']+c['window_after_mm']
        mask=(progress>=low)&(progress<=high);ids=np.flatnonzero(mask)
        complete=bool(len(ids) and progress.min()<low and progress.max()>high)
        records.append(dict(id=c['id'],included_angle_deg=c['interior_angle_deg'],complete_window=complete,
            speed_deficit_mm=float(np.maximum(100.-speed[mask],0.).sum()*dt) if len(ids) else None,
            minimum_speed_mm_s=float(speed[mask].min()) if len(ids) else None,
            duration_s=float(mask.sum()*dt),first_index=int(ids[0]) if len(ids) else None,last_index=int(ids[-1]) if len(ids) else None))
    return dict(corners=records,all34_windows_complete=all(r['complete_window'] for r in records),
        projection_raw_backward_steps=int((np.diff(progress)<-1e-7).sum()),cruise=cruise_metrics(speed,dt),dt_s=dt)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate_folder');args=ap.parse_args();candidate=Path(args.candidate_folder)
    qualified=candidate/'complete_contract_audit.json'
    if not qualified.exists():qualified=candidate/'full_result.json'
    source_result=json.loads(qualified.read_text(encoding='utf-8'));assert source_result['success']
    selection=json.loads((candidate/'selected_policy.json').read_text(encoding='utf-8'))
    torch.set_num_threads(1);stage=candidate/'half_ms';stage.mkdir(exist_ok=False);dt=.0005
    files=[Path(__file__),OUT/'turn_aware_projector.py',OUT/'coordinated_cruise_policy.py',OUT/'candidate_policy_factory.py',OUT/'bend_conditioned_policy.py',OUT/'arc_cruise_policy.py',BASE/'run_natural_resolution.py']
    (stage/'source').mkdir()
    for p in files:shutil.copy2(p,stage/'source'/p.name)
    write_json(stage/'registration.json',dict(kind='half_ms_turn_aware_full_validation',wall_budget_s=1800,execution_deadline_s=1710,
        dt_s=dt,decision_dt_s=.020,ppo_updates=0,source_sha256={str(p):sha(p) for p in files},
        selected_policy_sha256=sha(candidate/'selected_policy.json'),nominal_full_result_sha256=sha(candidate/'full_result.json'),
        nominal_qualification_path=str(qualified),nominal_qualification_sha256=sha(qualified),
        stop='Stop candidate on refined certificate check failure or execution failure; preserve all recorded rows. Coverage uses witnessed ordered gates, projection reversals remain diagnostic. No complete time for an incomplete prefix.'))
    start=time.perf_counter();configure_resolution(dt)
    safety=backup_check(dt,start+120);write_json(stage/'backup_checks.json',safety);print('REFINED_BACKUP_PASS',safety,flush=True)
    case=BASE/'paper_evidence_review/natural_closed_case';reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'));assert sha(Path(selection['model']))==selection['model_sha256']
    policy=candidate_policy(torch.load(selection['model'],map_location='cpu',weights_only=False),selection)
    env=turn.environment_factory(reference)
    if selection['fixed_d'] is not None:env.fixed_d=selection['fixed_d']
    policy.bind(env);chunks=[];observations=[];actions=[];latents=[];times=[];error=None;last=time.perf_counter()
    try:
        while not env.done:
            if time.perf_counter()>start+1710:raise TimeoutError('Refined full-candidate wall budget')
            x=env.observation();z=policy.act(x,True)[0];assert not np.any(env.pending_pulse)
            times.append(env.ticks*dt);observations.append(x);latents.append(z)
            _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
            if time.perf_counter()-last>40:
                print('TURN_REFINED',env.ticks*dt,env.tracker.frontier,flush=True);last=time.perf_counter()
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks) if chunks else np.empty((0,13))
    np.savez_compressed(stage/'full_trace.npz',reference=reference,rows=rows,observations=np.array(observations),actions=np.array(actions),latents=np.array(latents),decision_times_s=np.array(times),dt_s=dt)
    physical=physical_audit(reference,rows,dt);ordered=env.tracker.snapshot(1.25);analytic=analytic_command_audit(reference[0],rows,geometry)
    stats=corner_stats(reference,rows,geometry,dt);write_json(stage/'corner_metrics.json',stats)
    success=bool(error is None and env.success and ordered['geometry_complete'] and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'] and stats['all34_windows_complete'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=env.ticks*dt if success else None,
        elapsed_prefix_or_complete_s=env.ticks*dt,dt_s=dt,decision_dt_s=.020,geometry=ordered,audit=physical,analytic=analytic,
        trace_sha256=sha(stage/'full_trace.npz'),cruise=stats['cruise'],goal_complete=False,experimental_projector_counts=env.experimental_projector.counts)
    result['projection_backward_steps_diagnostic']=stats['projection_raw_backward_steps']
    result['coverage_criterion']='Witnessed ordered gate crossings; nearest-polyline projection monotonicity is not a physical completion condition.'
    write_json(stage/'ordered_gate_events.json',env.tracker.events)
    write_json(stage/'full_result.json',result)
    if error and 'TimeoutError' in error:
        with (stage/'exact_checkpoint.pkl').open('wb') as f:pickle.dump(dict(state=env.state(),memory=memory_state(policy),selection=selection,dt_s=dt),f,protocol=5)
    write_json(stage/'receipt.json',dict(goal_complete=False,status='refined_validation_finished',success=success,wall_s=time.perf_counter()-start,ppo_updates=0))
    print('REFINED_TURN_RESULT',success,result['complete_time_s'],error,time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
