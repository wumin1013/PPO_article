"""Bounded joint optimization with full-grid feasibility and saved incumbents."""
import time
import numpy as np
from scipy.optimize import minimize
from dense_trace_credit_model import values, objective
from study_io import write_json


def guard_indices(model):
    offset=0;indices=[]
    for r in model['baseline']['records']:
        if 'shape' in r:
            indices.extend(offset+k for k in range(2,7));offset+=9
        else:offset+=2
    return np.array(indices)


def solve_guarded(model, initial, scale, checkpoint, deadline, buffer=None):
    """Generate trust-region cuts, but accept only full-support feasible iterates."""
    n=len(initial);zero=np.zeros(n);anchor=model['anchors'];idx=guard_indices(model)
    upper=np.maximum(model['base'][idx],0.)
    if buffer is not None:upper-=np.asarray(buffer)[idx]
    normalization=max(objective(model,zero)[0],1.)
    jac=values(model,zero)[1]
    diagonal=np.maximum(np.sqrt((jac*jac).sum(0)/normalization+.04*(anchor*anchor).mean(0)),1e-4)
    trust=anchor/diagonal
    active=model['margins']<.05
    mg=model['margin_jac'][active]/diagonal;mb=model['margins'][active]
    floor=np.minimum(mb,.02)
    # Start with informative support rows; every omitted row is checked before
    # accepting a parameter vector and added whenever it is violated.
    cuts=set(map(int,np.argmax(abs(trust),axis=0)))
    cuts.update(map(int,np.argsort(-np.linalg.norm(trust,axis=1))[:128]))
    cuts.update(map(int,np.flatnonzero(abs(anchor@initial)>.7*scale)))
    best=None;best_value=np.inf;calls=0;history=[];last_y=initial*diagonal

    def fun(y):
        z=y/diagonal;value,gradient=objective(model,z);a=anchor@z
        return value/normalization+.02*float(np.mean(a*a)),(gradient/normalization+.04*anchor.T@a/len(a))/diagonal

    def guard(y):return upper-values(model,y/diagonal)[0][idx]
    def guard_jac(y):return -values(model,y/diagonal)[1][idx]/diagonal

    def margins(z):
        return np.array([float(scale-abs(anchor@z).max()),
            float(((mb+model['margin_jac'][active]@z-floor)/.01).min()),
            float((upper-values(model,z)[0][idx]).min())])

    def remember(y):
        nonlocal best,best_value,calls,last_y
        calls+=1;last_y=y.copy();z=y/diagonal;feas=margins(z)
        value=objective(model,z)[0]
        if feas.min()>=-1e-7 and value<best_value-1e-8:
            best=z.copy();best_value=value
            np.savez_compressed(checkpoint.with_suffix('.npz'),coordinates=best)
            write_json(checkpoint.with_suffix('.json'),dict(status='feasible_iterate_not_convergence_claim',
                callback=calls,predicted_objective=best_value,full_support_margins=feas.tolist()))
        if time.perf_counter()>=deadline:raise TimeoutError('Bounded solver; feasible incumbent preserved')

    remember(np.zeros(n))
    remember(initial*diagonal)
    y=initial*diagonal
    for sweep in range(4):
        ids=np.array(sorted(cuts));a=trust[ids]
        expired=False
        try:
            solution=minimize(fun,y,jac=True,method='SLSQP',
                constraints=[dict(type='ineq',fun=lambda y:np.r_[scale-a@y,scale+a@y],jac=lambda y:np.vstack([-a,a])),
                    dict(type='ineq',fun=lambda y:(mb+mg@y-floor)/.01,jac=lambda y:mg/.01),
                    dict(type='ineq',fun=guard,jac=guard_jac)],
                callback=remember,options=dict(maxiter=180,ftol=1e-9))
            y=solution.x;remember(y)
            status=dict(converged=bool(solution.success),message=solution.message,iterations=int(solution.nit))
        except TimeoutError as exc:
            y=last_y;expired=True
            status=dict(converged=False,message=str(exc),iterations=None)
        z=y/diagonal
        violations=np.flatnonzero(abs(anchor@z)>scale+1e-8)
        added=[int(i) for i in violations if int(i) not in cuts]
        cuts.update(added)
        status.update(sweep=sweep,trust_rows=len(ids),added_rows=len(added),
            exact_margins=margins(z).tolist(),predicted_objective=objective(model,z)[0])
        history.append(status)
        print('GUARDED_TRACE_SOLVE',scale,status,flush=True)
        if expired or not added:break
    selected=None if best is None or best_value>=objective(model,zero)[0]-1e-5 else best
    return selected,dict(history=history,callback_count=calls,
        selected_predicted_objective=None if selected is None else best_value,
        selected_full_support_margins=None if selected is None else margins(selected).tolist(),
        solver_converged=bool(status['converged'] and not added and selected is not None
                              and np.max(abs(selected-z))<1e-8))
