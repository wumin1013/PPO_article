"""Fixed eight-cell diagnostic of Ruckig state-to-state launch connectors.

This script does not invoke actors or modify original evidence.  The matched
candidate has one fixed extra tick beyond the native minimum duration, with a
bounded endpoint shooting solve.  It is an offline trajectory diagnostic only.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import sys
import time
from datetime import datetime,timezone
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\JNN_dominance_20261011_v5')
sys.path.insert(0,str(BASE/'src'))
sys.path.insert(0,str(BASE/'literature/ruckig_vendor'))
import ruckig
from metrics_exact_index_v5 import metrics
from runtime_bootstrap import setup
setup()
from geometry_spec import Corridor,GateTracker

CASE='piecewise_curvature_coupon_scale10'
LIMITS=np.array([100.,2000.,62500.])
NORMAL_FRACTION=.01
INTERIOR=.999
MATCH_TOL_MM=2e-11


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def source(method,dt):
    folder='main' if method=='JNN' else 'fixed'
    return BASE/'results/representative_scaled_v8/learned'/folder/f'{CASE}_{method}_dt{round(dt*1e6)}us.npz'


def differential(p,dt):
    v=np.diff(p,axis=0)/dt
    a=np.diff(np.vstack((np.zeros(2),v)),axis=0)/dt
    j=np.diff(np.vstack((np.zeros(2),a)),axis=0)/dt
    return v,a,j


def tracker_state(tracker):
    return dict(position=tracker.position.tolist(),inside=tracker.inside,
                frontier=tracker.frontier,pending_negative=tracker.pending_negative)


def launch(method,dt,join_s):
    run_id=f'{method}_dt{round(dt*1e6)}us_join{round(join_s*1000)}ms'
    out=HERE/'ruckig_results'/run_id
    out.mkdir(parents=True,exist_ok=False)
    oldfile=source(method,dt)
    record=json.loads(oldfile.with_suffix('.json').read_text('utf-8'))
    assert record['success'] and sha(oldfile)==record['trace_sha256']
    with np.load(oldfile) as f:reference=f['reference'];old=f['positions']
    join=round(join_s/dt)
    xaxis=reference[1]-reference[0];xaxis=xaxis/np.linalg.norm(xaxis)
    matrix=np.stack((xaxis,np.array([-xaxis[1],xaxis[0]])))
    offset=old[0]
    local=(old-offset)@matrix.T
    ov,oa,oj=differential(old,dt)
    targetp=local[join]
    targetv=ov[join-1]@matrix.T
    targeta=oa[join-1]@matrix.T
    factors=np.array([np.sqrt(1-NORMAL_FRACTION**2),NORMAL_FRACTION])
    vmax=LIMITS[0]*factors
    amax=LIMITS[1]*INTERIOR*factors
    jmax=LIMITS[2]*INTERIOR*factors
    solve_calls=[]
    count=0

    def calculate(velocity,acceleration,minimum=None,save_eval=False):
        nonlocal count
        inp=ruckig.InputParameter(2)
        inp.control_interface=ruckig.ControlInterface.Position
        inp.current_position=[0.,0.]
        inp.current_velocity=[0.,0.]
        inp.current_acceleration=[0.,0.]
        inp.target_position=targetp.tolist()
        inp.target_velocity=velocity.tolist()
        inp.target_acceleration=acceleration.tolist()
        inp.max_velocity=vmax.tolist()
        inp.max_acceleration=amax.tolist()
        inp.max_jerk=jmax.tolist()
        inp.duration_discretization=ruckig.DurationDiscretization.Discrete
        inp.synchronization=ruckig.Synchronization.Time
        if minimum is not None:inp.minimum_duration=minimum
        tr=ruckig.Trajectory(2)
        value=ruckig.Ruckig(2,dt).calculate(inp,tr)
        if value not in (ruckig.Result.Working,ruckig.Result.Finished):
            raise RuntimeError('Ruckig result '+str(value))
        ticks=round(tr.duration/dt)
        positions=np.array([tr.at_time(float(k*dt))[0] for k in range(ticks+1)])
        residual=(positions[-3:-1]-local[join-2:join]).reshape(-1)
        if save_eval:
            file=out/f'solver_eval_{count:03d}.npz'
            np.savez_compressed(file,local_prefix=positions)
            solve_calls.append(dict(index=count,velocity=velocity.tolist(),acceleration=acceleration.tolist(),
                duration_s=tr.duration,residual_mm=residual.tolist(),trace_sha256=sha(file),trace=str(file)))
            count+=1
        return tr,positions,residual

    def assess(label,local_prefix,raw_duration,error=None):
        prefix=offset+local_prefix@matrix
        prefix[0]=old[0]
        full=np.vstack((prefix,old[join+1:]))
        prefix_file=out/f'{label}_prefix.npz'
        full_file=out/f'{label}_full.npz'
        np.savez_compressed(prefix_file,positions=prefix,dt_s=dt)
        np.savez_compressed(full_file,positions=full,reference=reference,dt_s=dt)
        audit=metrics(reference,full,dt=dt,tolerance=record['tolerance_mm'])
        new_join=len(prefix)-1
        differential_values=differential(full,dt)
        seam_peaks=[float(np.linalg.norm(value[max(0,new_join-3):new_join+4],axis=1).max())
                    for value in differential_values]
        corridor=Corridor(reference,epsilon=record['tolerance_mm'],
                          closed=np.linalg.norm(reference[0]-reference[-1])<1e-9)
        original=GateTracker(corridor,old[0]);trial=GateTracker(corridor,prefix[0])
        for p in old[1:join+1]:original.update(p)
        for p in prefix[1:]:trial.update(p)
        outrow=dict(label=label,connector_s=raw_duration,saved_s=join_s-raw_duration,
            original_complete_s=record['complete_time_s'],metrics=audit,
            prefix_trace=str(prefix_file),prefix_sha256=sha(prefix_file),
            full_trace=str(full_file),full_sha256=sha(full_file),
            exact_last_three=np.array_equal(prefix[-3:],old[join-2:join+1]),
            unchanged_body_exact=np.array_equal(full[new_join+1:],old[join+1:]),
            suffix_difference_v_a_j_exact=all(np.array_equal(new[new_join:],original_diff[join:])
                for new,original_diff in zip(differential_values,(ov,oa,oj))),
            original_tracker=tracker_state(original),new_tracker=tracker_state(trial),
            seam_vector_peaks=seam_peaks,error=error)
        write(out/f'{label}.json',outrow)
        return outrow

    start=time.perf_counter();attempts=[]
    try:
        tr,positions,residual=calculate(targetv,targeta)
        attempts.append(assess('native_continuous_target',positions,tr.duration))
        # Endpoint shooting is fixed at a single duration with one slack tick.
        fixed_duration=(round(tr.duration/dt)+1)*dt
        scales=np.r_[vmax,amax]
        initial=np.r_[targetv,targeta]/scales
        def objective(scaled):
            velocity=scaled[:2]*vmax;acceleration=scaled[2:]*amax
            trajectory,p,err=calculate(velocity,acceleration,fixed_duration,save_eval=True)
            # Ruckig may extend an infeasible requested duration. Such a solve
            # is retained, but not accepted as the fixed-duration candidate.
            return err
        fit=least_squares(objective,initial,bounds=(-np.ones(4)*.999999999,np.ones(4)*.999999999),
            xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=60,diff_step=1e-5)
        tr,matched,residual=calculate(fit.x[:2]*vmax,fit.x[2:]*amax,fixed_duration)
        matched_ok=bool(np.max(np.abs(residual))<=MATCH_TOL_MM and
                        abs(tr.duration-fixed_duration)<=1e-12)
        shooting=dict(success=bool(fit.success),status=int(fit.status),message=str(fit.message),
            nfev=int(fit.nfev),calls=len(solve_calls),fixed_duration_s=fixed_duration,
            final_duration_s=tr.duration,residual_mm=residual.tolist(),matching_accepted=matched_ok,
            final_velocity=(fit.x[:2]*vmax).tolist(),final_acceleration=(fit.x[2:]*amax).tolist())
        if matched_ok:
            # This exact pin is permitted only below the registered tiny
            # residual tolerance. Reconstruct world prefix, pin there, and
            # audit it directly; no hidden endpoint repair is accepted.
            matched_world=offset+matched@matrix
            matched_world[-3:]=old[join-2:join+1]
            matched=(matched_world-offset)@matrix.T
            # The common local frame is exactly XY for this registered path.
            if not np.array_equal(offset+matched[-3:]@matrix,old[join-2:join+1]):
                raise RuntimeError('Local transform did not preserve exact endpoint pin')
            attempts.append(assess('matched_discrete_target',matched,tr.duration))
        else:
            attempts.append(assess('unmatched_shooting_result',matched,tr.duration,
                                  error='Fixed-duration endpoint matching did not meet registered tolerance'))
    except Exception as exc:
        shooting=dict(error=type(exc).__name__+': '+str(exc),matching_accepted=False)
    result=dict(run_id=run_id,case=CASE,method=method,dt_s=dt,join_s=join_s,
        source=str(oldfile),source_sha256=sha(oldfile),target_local_position=targetp.tolist(),
        target_discrete_velocity=targetv.tolist(),target_discrete_acceleration=targeta.tolist(),
        frame=matrix.tolist(),max_velocity_axes=vmax.tolist(),max_acceleration_axes=amax.tolist(),
        max_jerk_axes=jmax.tolist(),attempts=attempts,shooting=shooting,
        solve_calls=solve_calls,wall_s=time.perf_counter()-start,
        formal_new_experiment=False,
        scope='Offline boundary-connector diagnostic on archived planned commands. No actor calls, new closed-loop observation stream, or claim of independent final testing.')
    write(out/'result.json',result)
    print(run_id,[(x['label'],x['metrics']['success'],x['saved_s']) for x in attempts],flush=True)
    return result


def main():
    registration=HERE/'ruckig_registration.json'
    if registration.exists():raise FileExistsError('Preserve existing registration and runs')
    files=[source(method,dt) for method in ('JNN','FIXED') for dt in (.001,.0005)]
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        vendor_sha256=sha(ruckig.__file__),vendor_version=ruckig.__version__,
        input_sha256={str(p):sha(p) for p in files},case=CASE,methods=['JNN','FIXED'],
        resolutions_s=[.001,.0005],join_times_s=[.2,.3],cell_count=8,
        formal_new_experiment=False,normal_fraction=NORMAL_FRACTION,acceleration_jerk_interior=INTERIOR,
        protocol='Native 2D Position Ruckig at discrete minimum duration, then one endpoint shooting solve with one extra tick, max_nfev=60, no duration search. All solver-evaluation prefixes and failed full candidates retained. Exact last-three pin allowed only when residual <=2e-11 mm. All full candidates independently audited with unchanged finite differences, limits, all command intervals, ordered gates and final rest. No automatic adoption.',
        no_actor_state_claim='A future online connector needs a same-model preplanned prefix and complete actual join/executor state restoration; archived observations are not observations of the connector.')
    write(registration,reg)
    rows=[launch(method,dt,join) for method in reg['methods'] for dt in reg['resolutions_s'] for join in reg['join_times_s']]
    write(HERE/'ruckig_summary.json',dict(registration_sha256=sha(registration),rows=rows))


if __name__=='__main__':main()
