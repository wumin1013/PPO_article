"""Exact vector-cubic explanation of three observed brake-tail heading regressions."""
import time
import numpy as np
from joint_runtime import terminal
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def direction_witness(state, count, dt=.001):
    """Extreme ray of the existing brake family, for offline analysis only.

    For k=0..n-2, v[k] is a nonnegative scalar times
    v0 + lambda[k]*a0; lambda increases from0 to(n-2)*dt/3.
    Together with v0 this witness bounds the direction cone of all nonzero
    brake velocities. It does not choose count, change a splice or certify
    positional containment. The original planner still selects count.
    """
    jacobian=np.zeros((2,6));jacobian[:,2:4]=np.eye(2)
    jacobian[:,4:6]=(count-2)*dt/3*np.eye(2)
    return jacobian@state,jacobian


def main():
    feedback=BASE/'canonical_gate_band_ordinary_joint_feedback_20260918'
    sources=BASE/'canonical_gate_band_ordinary_joint_error_sources_20260918'
    data=read(feedback/'results.json');errors=read(sources/'results.json')
    cases=[r['path'] for r in errors['summary']['full_scale_regressions']
           if not r['already_worse_in_ordinary_motion']]
    assert cases==[4,7,13]
    stage=register('canonical_gate_band_terminal_heading_continuation_20260918',
        [__file__,feedback/'results.json',feedback/'receipt.json',sources/'results.json',
         sources/'receipt.json',BASE.parent/'revision_20260913_experiments_v1/planar_metrics.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,paths=cases),
        'Explain the three tail-only heading regressions with the unchanged '
        'actual cubic brake and actual splice v/a. Prove by numeric identity '
        'that every nonzero velocity is a positive multiple of v0+lambda*a0, '
        'with lambda from0 to(n-2)dt/3. Verify the measured >5mm/s heading '
        'peak directly against this analytic profile. Expose a six-state '
        'linear direction witness for future count-aware offline constraints, '
        'without changing the runtime or pretending that angular agreement '
        'proves position, duration selection or physical feasibility. No '
        'acceleration reset, deployment clamp, ideal template or new physics.')
    started=time.perf_counter();records=[]
    candidate=next(r for r in data['candidates'] if r['step']==1.)
    for path in cases:
        for label,pool in (('retained399a',data['baseline']),('unadopted_joint_candidate',candidate)):
            j=next(j for j,r in enumerate(pool['results']) if r['original_path']==path)
            with np.load(BASE/pool['traces'][j]) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
            stop=int(np.flatnonzero(rows[:,12])[0]);n=len(rows)-stop-23
            state=rows[stop-1,:6];v0,a0=state[2:4],state[4:6]
            profile=terminal.discrete_planar_stop_profile(v0,a0,.001,n)[0]
            physical_error=float(abs(profile[1:]-rows[stop:stop+n,2:4]).max())
            assert physical_error<1e-9
            k=np.arange(n-1,dtype=float)
            lam=k*.001*(n-1)/(n+1+2*k)
            factors=(n-k)*(n-1-k)*(n+1+2*k)/(n*(n-1)*(n+1))
            analytic=factors[:,None]*(v0+lam[:,None]*a0)
            identity_error=float(abs(analytic-profile[:n-1]).max());assert identity_error<1e-10
            witness,jac=direction_witness(state,n)
            assert np.max(abs(witness-(v0+lam[-1]*a0)))<1e-10
            u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
            sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
            turn=abs(np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)))
            def heading(vel):return np.degrees(np.arctan2(vel@normal,vel@u))
            use=np.linalg.norm(analytic,axis=1)>5.
            peak=float(heading(analytic[use]).max());initial=float(heading(v0))
            signed_cross=float(sign*(v0[0]*a0[1]-v0[1]*a0[0]))
            actual_tail=rows[stop:stop+n,2:4]
            actual_peak=float(heading(actual_tail[np.linalg.norm(actual_tail,axis=1)>5.]).max())
            assert abs(peak-actual_peak)<1e-10
            assert signed_cross>0. and peak>initial
            records.append(dict(path=path,label=label,initiation_tick=stop,braking_steps=n,
                initial_heading_deg=initial,tail_peak_heading_deg=peak,turn_deg=float(turn),
                tail_heading_increase_deg=peak-initial,tail_overshoot_deg=max(0.,peak-turn),
                signed_velocity_acceleration_cross=signed_cross,
                direction_witness=witness.tolist(),direction_witness_jacobian=jac.tolist(),
                physical_profile_identity_error=physical_error,analytic_profile_identity_error=identity_error))
    write_json(stage/'results.json',dict(records=records,
        interpretation='All six actual brake splices have acceleration rotating velocity farther into the turn. The exact unchanged cubic family therefore continues rotating during the brake. Learning must coordinate its arrival v/a with the outgoing tangent; resetting acceleration would violate the common splice and is not used.',
        witness_scope='Exact conditional on original planner-selected duration; no positional or unknown-state feasibility claim.',
        goal_complete=False))
    receipt(stage,dict(status='verified_exact_terminal_heading_continuation',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-started,goal_complete=False))
    state=read(BASE/'当前执行状态.json')
    state['current_ordinary_joint_round']['terminal_heading_cause']=stage.name
    state['current_ordinary_joint_round']['next']='Use measured ordinary model errors and duration-conditioned velocity/acceleration tail constraints; validate changing terminal modes and full XY physically before retention.'
    write_json(BASE/'当前执行状态.json',state)
    print('EXACT_BRAKE_HEADING',[(r['path'],r['label'],r['tail_heading_increase_deg'],r['tail_overshoot_deg']) for r in records],flush=True)


if __name__=='__main__':main()
