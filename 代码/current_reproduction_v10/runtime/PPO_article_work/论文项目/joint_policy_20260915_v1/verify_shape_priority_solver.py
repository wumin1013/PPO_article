"""Analytic epigraph checks and preregistration of the matched objective study."""
import time
import argparse
import numpy as np
from convex_cut_solver import solve_convex_cuts
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='canonical_gate_band_shape_priority_round_20260917')
    args=parser.parse_args()
    stage=register(args.stage,
        [__file__,BASE/'convex_cut_solver.py',BASE/'solve_transition_shape_step.py',
         BASE/'canonical_gate_band_shape_cruise_statewise_solve_20260917/optimizer.json',
         BASE/'canonical_gate_band_shape_cruise_statewise_half_full_20260917/receipt.json'],
        dict(analytic_checks=3,new_solver_runs=1,solve_wall_s=180,exchange_rounds=40,
            small_actor_calls=6000,small_physics_ticks=140000,small_wall_s=120,
            large_actor_calls=16000,large_physics_ticks=400000,large_wall_s=300,
            large_scales=[.5,1.],maximum_remaining_full_routes=35),
        'Compare shape-priority learning with the completed statewise-cruise '
        'energy arm at the SAME730d8352 parent. Change only the optimization '
        'objective: minimize the largest residual fraction of the existing '
        'positive shape totals. Keep repaired path5 credit, the empirical error '
        'model, physical metric/radius, statewise cruise constraints, all shape '
        'and energy ceilings, corridor and terminal constraints unchanged. '
        'Reuse all existing models, credit and control results; do not claim '
        'recentered4067 credit. Validate exact direction at+/-0.125; stop if '
        'complete success or per-corner nonregression fails. If supported, '
        'evaluate0.5/1.0 and select at most one by worst actual shape fraction '
        'among eligible steps. Same-scale comparisons isolate the objective; '
        'selection differs explicitly from the earlier energy arm. Acceptance '
        'requires all64 complete, per-corner shape/curvature/energy nonregression '
        'from the shared anchor, at least10% reduction in every positive shape '
        'total, zero entry excursion preserved, full time nonincreasing and '
        'cruise99 nondecreasing. Compare additionally with retained4067 and '
        'retain a Pareto alternative only on documented shape/time/cruise gain. '
        'No repeated inflation of empirical errors or larger radius. Failure '
        'returns to the measured mechanism; no same-recipe continuation, '
        'development/sealed access or final-goal claim.')
    records=[]
    oracle=lambda x:(np.array([-1.]),np.zeros((1,len(x))))
    for name in ('quadratic_control','linear_epigraph','two_quadratic_epigraph'):
        hessian=np.eye(2) if name=='quadratic_control' else np.zeros((2,2))
        linear=np.array([-.3,0.]) if name=='quadratic_control' else np.zeros(2)
        if name=='linear_epigraph':
            objective=lambda x:(np.array([.3-x[0]]),np.array([[-1.,0.]]))
            target=np.array([.1,0.])
        elif name=='two_quadratic_epigraph':
            objective=lambda x:(np.array([(x[0]-.04)**2+x[1]**2+.1,(x[0]+.04)**2+x[1]**2+.1]),
                                np.array([[2*(x[0]-.04),2*x[1]],[2*(x[0]+.04),2*x[1]]]))
            target=np.zeros(2)
        else:objective=None;target=np.array([.1,0.])
        actual,details=solve_convex_cuts(hessian,linear,np.array([.08,0.]),oracle,.1,
            time.perf_counter()+30,lambda x:None,objective_facets=objective)
        error=float(np.linalg.norm(actual-target));assert error<1e-5
        assert np.linalg.norm(actual)<=.1+1e-12
        records.append(dict(check=name,actual=actual.tolist(),analytic=target.tolist(),error=error,details=details))
    write_json(stage/'numerical_checks.json',dict(records=records,passed=True))
    receipt(stage,dict(status='shape_priority_round_registered_and_solver_verified',analytic_checks_passed=True,
        matched_parent_sha256='730d835277a7cdcae7941e5ab799b2e664f728cb3ff5b98bc078873f75921f30',goal_complete=False))
    print('SHAPE_PRIORITY_SOLVER_VERIFIED',flush=True)


if __name__=='__main__':main()
