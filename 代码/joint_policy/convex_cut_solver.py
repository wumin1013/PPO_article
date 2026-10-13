"""Accumulate supporting planes for a bounded convex learning subproblem."""
import sys
import time
from pathlib import Path
import numpy as np
from scipy import sparse


def solve_convex_cuts(hessian, linear, initial, oracle, radius, deadline,
                      callback, max_rounds=40, objective_facets=None):
    """Use an exact radius cone and audit every proposal with the full oracle.

    The oracle returns convex constraint values and subgradients in the same
    whitened coordinates. Cuts are necessary conditions, never certificates
    of the nonlinear constraints. Final feasibility belongs to the caller.
    """
    sys.path.insert(0,str(Path(__file__).resolve().parent/'.runtime_dependencies'))
    import clarabel
    physical_dimension=len(initial)
    dimension=physical_dimension+int(objective_facets is not None)
    if objective_facets is not None:
        # An auxiliary epigraph variable represents a convex shape objective.
        # Only physical parameter coordinates belong to the update-radius cone.
        hessian=np.pad(hessian,((0,1),(0,1)))
        linear=np.r_[linear,1.]
    hessian=.5*(hessian+hessian.T)
    eigenvalue=float(np.linalg.eigvalsh(hessian)[0])
    assert eigenvalue>=-1e-8*max(1.,np.linalg.norm(hessian,2)),eigenvalue
    # Remove roundoff-only negative curvature; record the modification.
    ridge=max(0.,-eigenvalue)
    hessian=hessian+ridge*np.eye(dimension)
    p=sparse.csc_matrix(np.triu(hessian))
    rows=[];bounds=[];known=set();history=[];y=initial.copy()
    if objective_facets is not None:y=np.r_[y,max(objective_facets(y)[0])]
    def all_constraints(point):
        physical=point[:physical_dimension]
        values,jacobian=oracle(physical)
        if objective_facets is None:return values,jacobian
        objective,derivatives=objective_facets(physical)
        return (np.r_[values,objective-point[-1]],
                np.vstack([np.pad(jacobian,((0,0),(0,1))),
                           np.c_[derivatives,-np.ones(len(objective))]]))
    cone_radius=radius*(1.-1e-7)
    for iteration in range(max_rounds):
        if time.perf_counter()>=deadline:break
        values,jacobian=all_constraints(y)
        added=0
        for value,row in zip(values,jacobian):
            norm=np.linalg.norm(row)
            if norm<1e-14:
                assert value<=1e-7,'Positive constant constraint'
                continue
            bound=float(row@y-value)
            key=np.round(np.r_[row/norm,bound/norm],11).tobytes()
            if key in known:continue
            # Retain the initial complete set and subsequently active cuts.
            if iteration and value < -1e-4:continue
            known.add(key);rows.append(row.copy());bounds.append(bound);added+=1
        if not added and iteration:break
        a=sparse.vstack([sparse.csc_matrix(np.asarray(rows).reshape(-1,dimension)),
            sparse.csc_matrix((1,dimension)),
            -sparse.eye(physical_dimension,dimension,format='csc')],format='csc')
        b=np.r_[bounds,cone_radius,np.zeros(physical_dimension)]
        cones=[clarabel.NonnegativeConeT(len(rows)),clarabel.SecondOrderConeT(physical_dimension+1)]
        settings=clarabel.DefaultSettings()
        settings.verbose=False;settings.max_iter=100
        settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-10
        settings.max_threads=1
        settings.time_limit=max(.05,min(20.,deadline-time.perf_counter()))
        fit=clarabel.DefaultSolver(p,linear,a,b,cones,settings).solve()
        trial=np.asarray(fit.x)
        assert np.all(np.isfinite(trial))
        y=trial
        value,_=all_constraints(y)
        entry=dict(round=iteration+1,cuts=len(rows),new_cuts=added,
            conic_status=str(fit.status),maximum_scaled_violation=float(max(value)),
            metric_norm=float(np.linalg.norm(y[:physical_dimension])),conic_iterations=fit.iterations,
            relaxation_objective=float(.5*y@hessian@y+linear@y))
        if objective_facets is not None:
            entry['actual_shape_objective']=float(max(objective_facets(y[:physical_dimension])[0]))
            entry['objective_epigraph']=float(y[-1])
        history.append(entry);callback(y[:physical_dimension])
        print('CONVEX_CUT_ROUND',entry,flush=True)
        if max(value)<=1e-8 and np.linalg.norm(y[:physical_dimension])<=radius+1e-12:break
        if str(fit.status) not in ('Solved','AlmostSolved'):break
    return y[:physical_dimension],dict(method='convex_supporting_plane_exchange_with_exact_radius_cone',
        clarabel_version=clarabel.__version__,hessian_roundoff_ridge=ridge,
        cone_radius=cone_radius,convex_objective_epigraph=objective_facets is not None,history=history)
