"""Check unchanged objectives and demonstrate retrospective-credit dependence."""
import time
import numpy as np
from bounded_shape_return import bounded_shape_cost,terminal_shape_cost
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'bounded_shape_full_feedback_ppo_20260916'
    stage=register('shape_credit_causality_audit_20260916',[__file__,BASE/'bounded_shape_return.py',source/'receipt.json'],
        dict(physics_ticks=0,training_decisions=0,wall_s=120),
        'Verify every saved episode keeps the identical total cost and show algebraically why globally normalized past spatial charges depend on future quality. Diagnose a reward-to-go accounting defect without claiming it caused initial reverse commands.')
    started=time.perf_counter();maxerr=0.;count=0
    for r in read(source/'episode_results.json')['episodes']:
        with np.load(source/'episodes'/r['trace']) as z:
            old,details=bounded_shape_cost(z['reference'],z['rows'],z['actions'])
            new,new_details=terminal_shape_cost(z['reference'],z['rows'],z['actions'])
            maxerr=max(maxerr,float(abs(new.sum()-z['shape_cost'].sum())))
            assert np.allclose(old,z['shape_cost'],rtol=0,atol=1e-12)
            assert np.count_nonzero(new[:-1])==0 and abs(new.sum()-old.sum())<1e-12
            count+=1
    # Two time bins q0=1, q1 variable: C0=q0*(1-exp(-Q))/Q.
    q0=1.;q1=1.;h=1e-5
    past=lambda q:q0*(-np.expm1(-(q0+q)))/(q0+q)
    omitted=(past(q1+h)-past(q1-h))/(2*h)
    full_derivative=np.exp(-(q0+q1))
    future=lambda q:q*(-np.expm1(-(q0+q)))/(q0+q)
    truncated=(future(q1+h)-future(q1-h))/(2*h)
    assert abs(omitted)>0.1 and abs(full_derivative-(truncated+omitted))<1e-8
    result=dict(saved_episodes=count,total_cost_max_error=maxerr,
        example=dict(past_energy=q0,future_energy=q1,true_total_cost_derivative=full_derivative,
            derivative_using_only_future_spatial_charge=truncated,omitted_past_charge_derivative=omitted),
        conclusion='Global cost normalization and retrospective apex window violate reward causality under per-bin reward-to-go. Terminal charging keeps the objective and ensures every action sees the entire bounded cost. This accounting defect is separate from why original policies learned reverse commands.',
        failure_ordering='Same total cost0<=C<=1 and unchanged Tmax+1+remaining/20 failure penalty; all prior success-vs-failure ordering remains identical.',physics_ticks=0)
    write_json(stage/'results.json',result);receipt(stage,dict(status='verified_objective_preserving_causal_credit_fix',saved_episodes=count,total_cost_max_error=maxerr,physics_ticks=0,wall_s=time.perf_counter()-started))
    print(result)


if __name__=='__main__':main()
