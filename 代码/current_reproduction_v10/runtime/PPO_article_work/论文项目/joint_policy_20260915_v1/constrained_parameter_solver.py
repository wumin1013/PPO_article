"""Metric projection with explicit checks of every saved-state constraint."""
import numpy as np
import time
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint,minimize


def minimum_metric_step(metric,a,b,feasible_start,tolerance=1e-7,max_rounds=12,deadline=None):
    """Solve min .5*x'Mx with Ax<=b using whitening and constraint generation.

    The caller supplies a feasible LP point. Restricting the QP to an active
    subset reduces cost; every omitted constraint is checked after each solve.
    A feasible subset optimum is also an optimum for the full constraint set.
    No physical safety is implied by these empirical linear inequalities.
    """
    metric=np.asarray(metric,float);a=np.asarray(a,float);b=np.asarray(b,float)
    def check_deadline():
        if deadline is not None and time.perf_counter()>deadline:
            raise TimeoutError('Registered parameter-projection deadline')
    check_deadline()
    initial=np.asarray(feasible_start,float)
    if np.max(a@initial-b)>1e-6:raise ValueError('QP initialization is not feasible')
    chol=np.linalg.cholesky(metric)
    c=solve_triangular(chol,a.T,lower=True,check_finite=False).T
    scale=np.linalg.norm(c,axis=1)
    nonzero=scale>1e-12
    if np.any(b[~nonzero]<-tolerance):raise ValueError('Infeasible constant inequality')
    c=c[nonzero]/scale[nonzero,None];d=b[nonzero]/scale[nonzero]
    y=chol.T@initial
    # Negative right-hand sides are the requirements that exclude no update.
    active=set(np.flatnonzero((d<0)|(d-c@y<1e-8)).tolist())
    audit=[]
    for iteration in range(max_rounds):
        check_deadline()
        ids=np.asarray(sorted(active),int)
        fit=minimize(lambda value:.5*value@value,y,jac=lambda value:value,
            constraints=[LinearConstraint(c[ids],-np.inf,d[ids])] if len(ids) else [],
            method='SLSQP',options=dict(ftol=1e-11,maxiter=1200),
            callback=(lambda _:check_deadline()) if deadline is not None else None)
        y=fit.x;violation=c@y-d
        missing=np.flatnonzero(violation>tolerance)
        audit.append(dict(round=iteration,active_constraints=len(ids),iterations=int(fit.nit),
            solver_success=bool(fit.success),message=str(fit.message),
            maximum_normalized_violation=float(max(0.,violation.max()))))
        if not fit.success:raise RuntimeError(f'Metric QP failed: {audit[-1]}')
        if not len(missing):
            x=solve_triangular(chol.T,y,lower=False,check_finite=False)
            raw=float(max(0.,np.max(a@x-b)))
            if raw>1e-6:raise RuntimeError(f'Unscaled QP constraint violation: {raw}')
            return x,dict(rounds=audit,raw_constraint_violation=raw,metric_norm=float(np.sqrt(y@y)))
        before=len(active)
        active.update(missing[np.argsort(violation[missing])[-32:]].tolist())
        if len(active)==before:raise RuntimeError('Active QP violates an included constraint')
    raise RuntimeError('Metric QP constraint generation did not converge within its budget')
