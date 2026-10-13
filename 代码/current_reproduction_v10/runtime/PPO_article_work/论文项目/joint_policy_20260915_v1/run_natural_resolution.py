"""Half-step physical integration check with unchanged 20 ms policy decisions."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import math,time,json
import numpy as np
import torch
from study_io import BASE,ROOT,register,receipt,write_json,sha
import joint_runtime as runtime
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from run_natural_support_search import LocalSupportPolicy
from natural_case_audit import analytic_command_audit


def set_resolution(dt):
    """All changes are process-local; frozen source files are never overwritten."""
    runtime.DT=dt
    runtime.MACRO_TICKS=int(round(.020/dt))
    projector=runtime.projector
    projector.DT=dt
    projector.MAX_RELEASE_TERMS=int(math.ceil(projector.AMAX_INTERNAL/(projector.JMAX_INTERNAL*dt)))
    original=projector.enforce_kinematic_constraints
    projector.enforce_kinematic_constraints=lambda v,a,j:original(v,a,j,dt=dt)


def physical_audit(reference,rows,dt):
    xy=np.vstack([reference[0],rows[:,:2]])
    v=np.diff(xy,axis=0)/dt
    a=np.diff(np.vstack([np.zeros(2),v]),axis=0)/dt
    j=np.diff(np.vstack([np.zeros(2),a]),axis=0)/dt
    peaks=[float(np.linalg.norm(z,axis=1).max()) for z in (v,a,j)]
    count=int(round(.020/dt))+1
    rest=len(rows)>=count and np.max(np.linalg.norm(v[-count:],axis=1))<=.001 and np.max(np.linalg.norm(a[-count:],axis=1))<=.01
    return dict(peaks=peaks,kinematic_pass=bool(np.all(np.array(peaks)<=np.array([100.,2000.,62500.])*1.00001)),
        rest_pass=bool(rest),rest_audit_sample_count=count,rest_audit_span_s=(count-1)*dt,
        scope='independent raw-XY vector differences on refined physical integration; not a servo guarantee')


def main():
    torch.set_num_threads(1);dt=.0005
    case=BASE/'paper_evidence_review/natural_closed_case'
    prior=json.loads((case/'support_calibration/results.json').read_text(encoding='utf-8'))
    selected=next(x for x in prior if x['success'])
    model=BASE/'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    stage=register('paper_evidence_review/natural_closed_case/resolution_half_ms',
        [__file__,model,case/'reference.csv',case/'geometry.json',case/'support_calibration/receipt.json',
         BASE/'run_natural_support_search.py',BASE/'joint_runtime.py',
         ROOT/'论文项目/revision_20260913_preview_efficiency_v2/recursive_kcm_roundoff_v1/projector.py'],
        dict(total_wall_s=1800,complete_runs=1,decision_ms=20,execution_ms=.5,training_updates=0),
        'Independent time-resolution sensitivity check, same checkpoint and support transform. '
        'Decision duration and memory decay per decision remain20ms; heading filter tau remains12ms. '
        'Execution dt and constraint disks use0.5ms; finite acceleration release horizon doubles '
        'from32 to64 terms, preserving32ms physical duration. Episode horizon/reward integrals '
        'and braking use physical dt; stationary audit spans at least20ms. No learning occurs. '
        'A changed feasibility outcome is reported and blocks a resolution-stable success claim.')
    set_resolution(dt)
    reference=np.loadtxt(case/'reference.csv',delimiter=',',skiprows=1)
    spec=selected['spec'];policy=LocalSupportPolicy(load_credit_policy(torch.load(model,map_location='cpu',weights_only=False)),spec['upper'],spec['median'])
    env=environment_factory(reference);policy.bind(env)
    started=time.perf_counter();last=started;chunks=[];obs=[];actions=[];times=[];error=None
    try:
        while not env.done:
            if time.perf_counter()>started+1760:raise TimeoutError('Registered refinement wall budget')
            x=env.observation();z=policy.act(x,True)[0]
            times.append(env.ticks*dt);obs.append(x)
            assert not np.any(env.pending_pulse)
            _,_,_,info=env.step(z);chunks.append(info['rows']);actions.append(info['held_action'])
            if time.perf_counter()-last>40:
                last=time.perf_counter();print('REFINED',env.ticks*dt,env.tracker.frontier,flush=True)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    rows=np.vstack(chunks)
    np.savez_compressed(stage/'trace.npz',reference=reference,rows=rows,observations=np.asarray(obs),
        actions=np.asarray(actions),decision_times_s=np.asarray(times),dt_s=dt)
    physical=physical_audit(reference,rows,dt)
    geometry=env.tracker.snapshot(1.25)
    analytic=analytic_command_audit(reference[0],rows,json.loads((case/'geometry.json').read_text(encoding='utf-8')))
    success=bool(error is None and env.success and physical['kinematic_pass'] and physical['rest_pass'] and analytic['analytic_corridor_pass'])
    result=dict(success=success,reason=env.reason,error=error,complete_time_s=env.ticks*dt if success else None,
        elapsed_prefix_or_complete_s=env.ticks*dt,geometry=geometry,audit=physical,analytic=analytic,
        dt_s=dt,decision_dt_s=.020,actor_calls=len(actions),physics_ticks=len(rows),wall_s=time.perf_counter()-started,
        source_spec=spec,trace_sha256=sha(stage/'trace.npz'),nominal_complete_time_s=selected['complete_time_s'])
    write_json(stage/'result.json',result)
    receipt(stage,dict(status='completed_resolution_check',success=success,ppo_updates=0,
        actor_calls=len(actions),physics_ticks=len(rows),wall_s=time.perf_counter()-started,
        result_sha256=sha(stage/'result.json'),goal_complete=False))
    print('RESOLUTION_COMPLETE',success,result['elapsed_prefix_or_complete_s'],physical,analytic,flush=True)


if __name__=='__main__':main()
