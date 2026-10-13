"""Separate ordinary feedback error from discrete terminal-duration credit error."""
import time
import numpy as np
from joint_runtime import terminal
from probe_closed_loop_tangent_credit import STATE_SCALE
from transition_waveform_model import chain_responses
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha
from terminal_duration_credit import fixed_duration_rows, fixed_duration_jacobian


def main():
    source = BASE/'canonical_gate_band_retained_center_credit_completed_20260918'
    reference = BASE/'canonical_gate_band_retained_center_reference_20260918'
    locality = BASE/'canonical_gate_band_retained_center_locality_20260918'
    data = read(source/'results.json')
    stage = register('canonical_gate_band_terminal_duration_credit_audit_20260918',
        [__file__, source/'results.json', source/'receipt.json', reference/'basis.npz',
         locality/'results.json', locality/'receipt.json', BASE/'closed_loop_transition_credit.py',
         BASE/'joint_runtime.py', terminal.__file__],
        dict(paths=[31,41], new_actor_decisions=0, new_physics_ticks=0,
             terminal_planner_queries=200, wall_s=60),
        'Path41 signed0.125 shared-parameter response has an incorrect first-order XY direction. '
        'Separate ordinary and terminal waveform errors using saved complete traces. '
        'Feed each actual braking initial state into the old terminal Jacobian, inspect '
        'selected braking sample counts under the actual variants and state-coordinate '
        'difference queries, and compare with the analytic fixed-duration cubic response. '
        'Support the duration-credit hypothesis only if a differentiated query changes '
        'duration, the old terminal-only prediction is wrong, and a same-duration exact '
        'response closes on unchanged-duration actual tails. If it fails, return to '
        'ordinary feedback rather than tune error reserves. Planner/executor and all '
        'acceptance boundaries remain frozen; no policy update or claimed shape success.')
    start = time.perf_counter(); costs = dict(terminal_planner_queries=0,new_actor_decisions=0,new_physics_ticks=0)
    with np.load(reference/'basis.npz') as z:
        direction = z['basis'][1]*z['amplitudes'][1]
    def plan(state):
        assert costs['terminal_planner_queries'] < 200 and time.perf_counter()-start < 60
        costs['terminal_planner_queries'] += 1
        return terminal.plan_planar_terminal_brake(state[:2],state[2:4],state[4:6],dt=.001,limits=terminal.LIMITS)
    records=[]
    for path, index in ((31,0),(41,1)):
        record=next(r for r in data['records'] if r['path']==path)
        with np.load(source/record['file']) as z: local={k:z[k].copy() for k in z.files}
        rows=local['rows']; brake=int(np.flatnonzero(rows[:,12])[0]); state=rows[brake-1,:6]
        nominal=plan(state); count=nominal['braking_steps']; tail=rows[brake:,:6]
        exact=fixed_duration_jacobian(count,len(tail)); old=local['row_jacobian'][brake:]
        assert np.max(np.abs(fixed_duration_rows(state,count,len(tail))-tail))<1e-7
        gradient=chain_responses(local,direction[None])[:,:,0]
        queries=[]
        for epsilon in (1e-4,1e-5,1e-6):
            for axis in range(6):
                counts=[]
                for sign in (-1,1):
                    perturbed=state.copy();perturbed[axis]+=sign*epsilon*STATE_SCALE[axis]
                    try:counts.append(plan(perturbed)['braking_steps'])
                    except ValueError:counts.append(None)
                queries.append(dict(epsilon=epsilon,axis=axis,counts=counts,
                                    crosses_duration=any(c!=count for c in counts)))
        errors=[]
        for scale in (-.5,-.125,.125,.5):
            with np.load(locality/f'scale{scale:g}/{index}.npz') as z: actual=z['rows'].copy()
            actual_brake=int(np.flatnonzero(actual[:,12])[0]);assert actual_brake==brake
            actual_state=actual[brake-1,:6]; actual_plan=plan(actual_state)
            assert np.max(np.abs(fixed_duration_rows(actual_state,actual_plan['braking_steps'],len(actual)-brake)-actual[brake:,:6]))<1e-7
            if len(actual)<len(rows):
                assert np.linalg.norm(actual[-1,2:6])<.001
                actual=np.vstack([actual,np.repeat(actual[-1:],len(rows)-len(actual),axis=0)])
            actual=actual[:len(rows),:6]
            delta=np.r_[(actual_state-state)/STATE_SCALE[:6],0.,0.,0.,0.]
            old_from_actual=tail+np.einsum('tcq,q->tc',old,delta)
            exact_from_actual=tail+np.einsum('tcq,q->tc',exact,delta)
            err=np.linalg.norm(actual[:,:2]-rows[:,:2]-scale*gradient[:,:2],axis=1)
            err_old=np.linalg.norm(actual[brake:,:2]-old_from_actual[:,:2],axis=1)
            err_exact=np.linalg.norm(actual[brake:,:2]-exact_from_actual[:,:2],axis=1)
            errors.append(dict(scale=scale,braking_steps=actual_plan['braking_steps'],
                ordinary_xy_error_mm=float(err[:brake].max()),full_xy_error_mm=float(err.max()),
                maximum_error_tick=int(err.argmax()),terminal_initial_state_error=(actual_state-state-scale*gradient[brake-1]).tolist(),
                terminal_from_actual_initial_old_error_mm=float(err_old.max()),
                terminal_from_actual_initial_fixed_duration_error_mm=float(err_exact.max())))
        item=dict(path=path,brake_tick=brake,braking_steps=count,total_ticks=len(rows),
                  terminal_jacobian_xy_column_errors=np.max(np.abs(old[:,:2]-exact[:,:2]),axis=(0,1)).tolist(),
                  duration_queries=queries,errors=errors)
        records.append(item)
        print('TERMINAL_DURATION',path,'brake',brake,'count',count,'switches',
              [q for q in queries if q['crosses_duration']],'errors',errors,flush=True)
    target=records[1]
    supported=any(q['crosses_duration'] and q['epsilon']==1e-4 for q in target['duration_queries']) and all(
        e['braking_steps']==target['braking_steps'] and e['terminal_from_actual_initial_fixed_duration_error_mm']<1e-8
        and e['terminal_from_actual_initial_old_error_mm']>1e-3 for e in target['errors'] if abs(e['scale'])==.125)
    write_json(stage/'results.json',dict(parent_sha256=data['parent_sha256'],records=records,costs=costs,
        duration_credit_hypothesis_supported=supported,frozen_planner_sha256=sha(terminal.__file__),goal_complete=False))
    receipt(stage,dict(status='completed_terminal_duration_credit_audit',hypothesis_supported=supported,
        costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
