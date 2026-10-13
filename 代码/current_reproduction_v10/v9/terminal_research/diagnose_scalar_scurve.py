"""Diagnose a generic scalar S-curve/normal-cubic terminal family on old states.

No policy is executed and no frozen source or run is edited. Every candidate is
retained. The final rest window is exactly the original qualified rest duration.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONUTF8']='1'
os.environ['PYTHONIOENCODING']='utf-8'
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

OLD=Path('D:/PPO_article_work/JNN_dominance_20261011_v5')
HERE=Path(__file__).resolve().parent
OUT=HERE/'scalar_scurve_diagnostic_v1'
sys.path[:0]=[str(OLD/'src'),str(OLD/'literature/ruckig_vendor')]
from common import sha,write_json
from runtime_bootstrap import setup
setup()
import joint_runtime
from metrics_exact_index_v5 import metrics
from ruckig import Ruckig,InputParameter,Trajectory,ControlInterface,Result

LIMITS=np.array([100.,2000.,62500.])
SCALAR_FRACTION=.98
MAX_PREFIX_S=.030


def scalar_stop(v,a):
    inp=InputParameter(1)
    inp.control_interface=ControlInterface.Velocity
    inp.current_position=[0.]
    inp.current_velocity=[float(v)]
    inp.current_acceleration=[float(a)]
    inp.target_velocity=[0.];inp.target_acceleration=[0.]
    inp.max_velocity=[100.]
    inp.max_acceleration=[SCALAR_FRACTION*2000.]
    inp.max_jerk=[SCALAR_FRACTION*62500.]
    trajectory=Trajectory(1)
    status=Ruckig(1).calculate(inp,trajectory)
    if status not in (Result.Working,Result.Finished):
        raise ValueError('Scalar Ruckig status: '+str(status))
    return trajectory


def differences(velocity,initial_acceleration,dt):
    acceleration=np.vstack((initial_acceleration,np.diff(velocity,axis=0)/dt))
    jerk=np.diff(acceleration,axis=0)/dt
    peaks=[np.linalg.norm(velocity[1:],axis=1).max(),
           np.linalg.norm(acceleration[1:],axis=1).max(),np.linalg.norm(jerk,axis=1).max()]
    return acceleration,jerk,np.asarray(peaks)


def candidate(position,velocity,acceleration,dt,prefix_ticks,rest_ticks):
    speed=float(np.linalg.norm(velocity))
    if speed<=1e-8:raise ValueError('Diagnostic family requires a nonzero initial velocity')
    tangent=velocity/speed;normal=np.array([-tangent[1],tangent[0]])
    frame=np.column_stack((tangent,normal))
    local_v=velocity@frame;local_a=acceleration@frame
    prefix=np.asarray([local_v+k*dt*local_a for k in range(prefix_ticks+1)])
    scalar=scalar_stop(prefix[-1,0],local_a[0])
    count=max(2,math.ceil(scalar.duration/dt))
    tangent_speed=np.asarray([scalar.at_time(min(k*dt,scalar.duration))[1][0]
                               for k in range(count+1)])
    normal_speed=joint_runtime.terminal.discrete_planar_stop_profile(
        [prefix[-1,1],0.],[local_a[1],0.],dt,count)[0][:,0]
    local=np.vstack((prefix,np.column_stack((tangent_speed,normal_speed))[1:]))
    global_velocity=local@frame.T
    global_velocity[0]=velocity
    global_velocity[-1]=0.
    # Count the same threshold-qualified final rest as the old trace; do not
    # gain efficiency by changing its hold padding or completion definition.
    acc,_,_=differences(global_velocity,acceleration,dt)
    resting=(np.linalg.norm(global_velocity[1:],axis=1)<=.001)&(np.linalg.norm(acc[1:],axis=1)<=.01)
    nonrest=np.flatnonzero(~resting)
    retained=len(resting)-(int(nonrest[-1])+1 if len(nonrest) else 0)
    pad=rest_ticks-retained
    if pad<0:raise ValueError('Candidate already exceeds the requested fixed rest duration')
    global_velocity=np.vstack((global_velocity,np.zeros((pad,2))))
    acc,jerk,peaks=differences(global_velocity,acceleration,dt)
    positions=np.vstack((position,position+np.cumsum(global_velocity[1:]*dt,axis=0)))
    return positions,global_velocity,dict(prefix_ticks=prefix_ticks,prefix_s=prefix_ticks*dt,
        scalar_continuous_duration_s=scalar.duration,scalar_sampled_ticks=count,
        normal_cubic_same_sample_count=True,rest_ticks=rest_ticks,
        total_tail_s=(len(positions)-1)*dt,vector_peaks=peaks.tolist(),
        vector_limit_ratios=(peaks/LIMITS).tolist(),
        vector_pass=bool(np.all(peaks<=LIMITS*(1+1e-10))),
        nonreversing=bool(np.all(global_velocity@tangent>=-1e-10)),
        frame_tangent=tangent.tolist(),frame_normal=normal.tolist(),
        local_initial_velocity=local_v.tolist(),local_initial_acceleration=local_a.tolist())


def run(period_us):
    dt=period_us*1e-6
    name=f'piecewise_curvature_coupon_scale10_JNN_dt{period_us}us'
    source=OLD/'results/representative_scaled_v8/learned/main'/(name+'.npz')
    record=json.loads(source.with_suffix('.json').read_text('utf-8'))
    assert sha(source)==record['trace_sha256']
    with np.load(source) as bundle:
        reference,old_positions,rows=[bundle[key] for key in ('reference','positions','rows')]
    start=int(np.flatnonzero(rows[:,12])[0])
    position=old_positions[start];velocity=rows[start-1,2:4];acceleration=rows[start-1,4:6]
    rest_ticks=round(record['terminal_rest_s']/dt)
    old_tail=(len(old_positions)-1-start)*dt
    folder=OUT/f'dt{period_us}us';folder.mkdir(parents=True)
    candidates=[];accepted=[]
    for prefix in range(round(MAX_PREFIX_S/dt)+1):
        candidate_id=f'prefix{prefix:03d}'
        path=folder/(candidate_id+'.npz')
        started=time.perf_counter()
        points,vectors,summary=candidate(position,velocity,acceleration,dt,prefix,rest_ticks)
        np.savez_compressed(path,positions=points,velocities=vectors,dt_s=dt)
        error=float(np.linalg.norm(points[-1]-reference[-1]))
        summary.update(id=candidate_id,tail_trace=path.relative_to(HERE).as_posix(),
            tail_trace_sha256=sha(path),endpoint_error_mm=error,same_one_mm_trigger=error<=1.,
            formal_new_experiment=False,complete_joined_audit=None)
        if summary['vector_pass'] and summary['nonreversing'] and error<=1.:
            joined=np.vstack((old_positions[:start+1],points[1:]))
            audited=metrics(reference,joined,dt=dt,tolerance=record['tolerance_mm'])
            summary['complete_joined_audit']=audited
            assert abs(audited['terminal_rest_s']-record['terminal_rest_s'])<dt/100
            if audited['success']:
                accepted.append((summary,joined))
        summary['diagnostic_wall_s']=time.perf_counter()-started
        candidates.append(summary)
        write_json(folder/(candidate_id+'.json'),summary)
    selected=min(accepted,key=lambda pair:pair[0]['total_tail_s']) if accepted else None
    use_new=selected is not None and selected[0]['total_tail_s']<old_tail-1e-12
    if use_new:
        selected_record,joined=selected
        np.savez_compressed(folder/'selected_full_trace.npz',reference=reference,positions=joined,dt_s=dt)
        selected_id=selected_record['id'];selected_total=selected_record['complete_joined_audit']['complete_time_s']
    else:
        selected_id='unchanged_cubic_fallback';selected_total=record['complete_time_s']
    return dict(case=record['case'],dt_s=dt,scope='Old-state terminal diagnostic only; no policy rollout.',
        source_trace=str(source),source_trace_sha256=sha(source),source_record_sha256=sha(source.with_suffix('.json')),
        splice_time_s=start*dt,initial_position=position.tolist(),initial_velocity=velocity.tolist(),
        initial_acceleration=acceleration.tolist(),original_tail_s=old_tail,original_complete_s=record['complete_time_s'],
        original_rest_ticks=rest_ticks,original_rest_s=record['terminal_rest_s'],candidate_count=len(candidates),
        complete_accepted_candidates=len(accepted),selected_id=selected_id,old_cubic_fallback_preserved=True,
        selected_complete_s=selected_total,local_splice_saving_s=record['complete_time_s']-selected_total,
        formal_new_experiment=False,candidates=candidates)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    registration=dict(scope='Two archived coupon terminal states; fixed scalar family, fixed0–30ms prefix search, same rest duration and full original acceptance. Not a new learned-controller experiment.',
        periods_us=[1000,500],scalar_limit_fraction=SCALAR_FRACTION,max_constant_acceleration_prefix_s=MAX_PREFIX_S,
        source_sha256=sha(__file__),frozen_files_modified=False,new_training=False,new_policy_rollouts=False,
        endpoint_trigger_mm=1.,hold_policy='Exactly preserve each original trace qualified terminal rest ticks.',
        fallback='The complete original cubic tail is retained whenever no shorter full-audit-passing candidate exists.',
        dependency='Unchanged Ruckig0.12.2 from the frozen local vendor, in the same PPO interpreter.')
    write_json(OUT/'registration.json',registration)
    rows=[run(us) for us in registration['periods_us']]
    write_json(OUT/'summary.json',dict(registration_sha256=sha(OUT/'registration.json'),rows=rows))
    print(json.dumps([{key:row[key] for key in ('dt_s','candidate_count','complete_accepted_candidates','selected_id',
        'original_complete_s','selected_complete_s','local_splice_saving_s','original_rest_s')} for row in rows],indent=2))


if __name__=='__main__':main()
