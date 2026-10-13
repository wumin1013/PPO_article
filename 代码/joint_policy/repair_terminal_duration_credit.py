"""Reuse all ordinary credit, replacing only contaminated terminal derivatives."""
import time
import numpy as np
from joint_runtime import terminal
from terminal_duration_credit import fixed_duration_rows, fixed_duration_jacobian, TerminalDurationGuard
from transition_waveform_model import chain_responses
from model_terminal_trigger_boundaries import state_responses
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    source=BASE/'canonical_gate_band_retained_center_credit_completed_20260918'
    reference=BASE/'canonical_gate_band_retained_center_reference_20260918'
    data=read(source/'results.json');physical=read(reference/'results.json')
    assert data['parent_sha256']==physical['parent_sha256']
    stage=register('canonical_gate_band_duration_corrected_credit_20260918',
        [__file__,source/'results.json',source/'receipt.json',reference/'results.json',reference/'basis.npz',
         BASE/'canonical_gate_band_terminal_duration_credit_audit_20260918/results.json',
         BASE/'terminal_duration_credit.py',BASE/'closed_loop_transition_credit.py',terminal.__file__],
        dict(paths=29,new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=1000,wall_s=180),
        'Repair only the proven terminal finite-difference branch contamination. Reuse every '
        'ordinary Jacobian, observation, action, feature and physical trace byte-for-byte '
        'at4067. Replace terminal row derivatives by exact linear cubic-family derivatives '
        'at the frozen planner selected count. Bind the selected count and initial-state '
        'response, and test explicit duration guards on all saved complete comparison '
        'policies. Record changed counts separately from continuous prediction errors. '
        'Validate guard derivatives and any guard-accepted test against the real planner. '
        'This fixes credit, not the executor or policy; no shape acceptance or training.')
    start=time.perf_counter();costs=dict(terminal_planner_queries=0,new_actor_decisions=0,new_physics_ticks=0)
    def plan(state):
        assert time.perf_counter()-start<175 and costs['terminal_planner_queries']<1000
        costs['terminal_planner_queries']+=1
        return terminal.plan_planar_terminal_brake(state[:2],state[2:4],state[4:6],dt=.001,limits=terminal.LIMITS)
    with np.load(reference/'basis.npz') as z:directions=z['basis']*z['amplitudes'][:,None]
    records=[];audits=[];guard_data={};rng=np.random.default_rng(718)
    for record in data['records']:
        path=record['path'];j=next(j for j,r in enumerate(physical['baseline']['results']) if r['original_path']==path)
        original=source/record['file']
        with np.load(original) as z:local={k:z[k].copy() for k in z.files}
        rows=local['rows'];brake=int(np.flatnonzero(rows[:,12])[0]);state=rows[brake-1,:6]
        count=plan(state)['braking_steps'];tail=rows[brake:,:6]
        assert np.max(np.abs(fixed_duration_rows(state,count,len(tail))-tail))<1e-7
        old=chain_responses(local,directions)
        old_terminal=local['row_jacobian'][brake:].copy()
        exact=fixed_duration_jacobian(count,len(tail))
        local['row_jacobian'][brake:]=exact
        local['state_jacobian'][-1]=0.
        response=state_responses(local)[-1]
        guard=TerminalDurationGuard(state,response,count)
        new=chain_responses(local,directions)
        assert np.array_equal(old[:brake],new[:brake])
        baseline_guard=guard.evaluate(np.zeros(768));assert max(r[0] for r in baseline_guard)<1e-8
        checks=[]
        for c in physical['candidates']:
            with np.load(BASE/c['traces'][j]) as z:actual=z['rows'].copy()
            actual_brake=int(np.flatnonzero(actual[:,12])[0]);actual_count=plan(actual[actual_brake-1,:6])['braking_steps']
            d=c['sign']*directions[c['direction']]
            predicted_state=state+(response@d)[:6]*np.array([1.25,1.25,100.,100.,2000.,2000.])
            guard_ok=all(r[0]<=1e-10 for r in guard.evaluate(d))
            predicted_count=plan(predicted_state)['braking_steps']
            assert not guard_ok or predicted_count==count
            if len(actual)<len(rows):
                assert np.linalg.norm(actual[-1,2:6])<.001
                actual=np.vstack([actual,np.repeat(actual[-1:],len(rows)-len(actual),axis=0)])
            truth=actual[:len(rows),:2]
            olderr=np.linalg.norm(truth-rows[:,:2]-c['sign']*old[:,:2,c['direction']],axis=1)
            newerr=np.linalg.norm(truth-rows[:,:2]-c['sign']*new[:,:2,c['direction']],axis=1)
            checks.append(dict(direction=c['direction'],brake_tick=actual_brake,braking_steps=actual_count,
                duration_changed=actual_count!=count,trigger_changed=actual_brake!=brake,
                guard_accepts_prediction=guard_ok,predicted_braking_steps=predicted_count,
                old_xy_error_mm=float(olderr.max()),corrected_xy_error_mm=float(newerr.max()),
                ordinary_xy_error_mm=float(newerr[:brake].max())))
        # Finite derivative checks exercise nonzero state perturbations.
        sample=directions[1]*.05;probe=rng.normal(size=768);probe/=np.linalg.norm(probe)
        eps=1e-6
        center=guard.evaluate(sample);plus=guard.evaluate(sample+eps*probe);minus=guard.evaluate(sample-eps*probe)
        derivative_error=max(abs((a[0]-b[0])/(2*eps)-r[1]@probe) for r,a,b in zip(center,plus,minus))
        assert derivative_error<1e-5,(path,derivative_error)
        name=f'path{path}_local_model.npz';np.savez_compressed(stage/name,**local)
        records.append(dict(record,file=name,source_local_model_sha256=sha(original),
            terminal_derivative='exact_fixed_duration',terminal_braking_steps=count))
        guard_data[f'path{path}_state']=state;guard_data[f'path{path}_response']=response
        guard_data[f'path{path}_count']=count
        audits.append(dict(path=path,brake_tick=brake,braking_steps=count,
            maximum_terminal_jacobian_xy_correction=float(abs(old_terminal[:,:2]-exact[:,:2]).max()),
            guard_derivative_error=derivative_error,checks=checks))
        print('DURATION_REPAIR',path,'count',count,'changed',sum(c['duration_changed'] for c in checks),
              'old/new',max(c['old_xy_error_mm'] for c in checks),max(c['corrected_xy_error_mm'] for c in checks),flush=True)
        write_json(stage/'results.json',dict(parent_model=data['parent_model'],parent_sha256=data['parent_sha256'],
            records=records,costs=costs,goal_complete=False))
    np.savez_compressed(stage/'duration_guards.npz',**guard_data)
    write_json(stage/'audit.json',dict(records=audits,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_current_complete_transition_credit',paths=len(records),costs=costs,
        source_ordinary_models_preserved=True,terminal_derivative='exact_fixed_duration',
        wall_s=time.perf_counter()-start,policy_updates=0,goal_complete=False))


if __name__=='__main__':main()
