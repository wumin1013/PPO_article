"""Check the new offline terminal witness against the unchanged physical formulas."""
import time
import numpy as np
from joint_runtime import terminal
from terminal_completion_witness import TerminalCompletionWitness
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    data=read(source/'results.json');selected=data['chosen']
    stage=register('canonical_gate_band_completion_witness_audit_20260918',
        [__file__,BASE/'terminal_completion_witness.py',BASE/'terminal_duration_credit.py',
         source/'results.json',source/'receipt.json'],
        dict(saved_paths=29,new_actor_decisions=0,new_physics_ticks=0,wall_s=60),
        'Verify nominal feasibility, exact profile/endpoint reconstruction, and '
        'central directional derivatives of the new sufficient terminal witness '
        'against the unchanged cubic formulas on all29 saved parent splice states. '
        'No actual path or shorter-count planner equivalence is claimed.')
    start=time.perf_counter();rng=np.random.default_rng(91817);records=[]
    for r,t in zip(selected['results'],selected['traces']):
        with np.load(BASE/t) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        stop=int(np.flatnonzero(rows[:,12])[0]);count=len(rows)-stop-23;state=rows[stop-1,:6]
        witness=TerminalCompletionWitness(state,np.eye(6),count,ref[-1])
        nominal=witness.evaluate(np.zeros(6))
        assert max(v/s for v,g,s,l in nominal)<=1e-7
        trial=rng.normal(size=6)*np.array([.0001,.0001,.001,.001,.01,.01])
        point=state+trial;v,a,j=terminal.discrete_planar_stop_profile(point[2:4],point[4:6],.001,count)
        exact=np.array([v[1:]/100.,a[1:]/2000.,j[1:]/(62500.*terminal.PlanarBrakingPolicy().limit_fraction)])
        normalized=point[2:6]/np.array([100.,100.,2000.,2000.])
        predicted=np.einsum('ctij,j->cti',witness.maps,normalized)
        profile_error=float(abs(predicted-exact).max())
        end=point[:2]+v[1:].sum(axis=0)*.001-ref[-1]
        endpoint_error=float(abs(end-witness.end0-witness.endg@trial).max())
        direction=rng.normal(size=6)*np.array([1.,1.,100.,100.,2000.,2000.])
        eps=1e-7
        left=witness.evaluate(trial-eps*direction);right=witness.evaluate(trial+eps*direction)
        center=witness.evaluate(trial)
        derivative_error=max(abs((rp[0]-lp[0])/(2*eps)-cp[1]@direction)/
                             max(1.,abs(cp[1]@direction)) for lp,rp,cp in zip(left,right,center))
        assert profile_error<1e-9 and endpoint_error<1e-10 and derivative_error<2e-5
        records.append(dict(path=r['original_path'],profile_error=profile_error,
            endpoint_error=endpoint_error,relative_directional_derivative_error=float(derivative_error)))
    assert time.perf_counter()-start<60
    write_json(stage/'results.json',dict(records=records,all_passed=True,goal_complete=False))
    receipt(stage,dict(status='verified_terminal_completion_witness',all29_passed=True,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    print('COMPLETION_WITNESS_AUDIT',len(records),max(r['relative_directional_derivative_error'] for r in records))


if __name__=='__main__':main()
