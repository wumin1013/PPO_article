"""Training-only continuous completion margins; deployment and guards are unchanged."""
import numpy as np
from scipy.optimize import linprog
from audit_coupled_trust_model import labels
from audit_refreshed_terminal_events import terminal_event
from learn_verified_trust_joint import positive_sum


def event_values(candidate, specs):
    by={r['path']:r for r in candidate['records']};values=[];events=[]
    for spec in specs:
        q=terminal_event(str(by[spec['path']]['trace']),spec['tick'])
        if q['already_braking']:
            raise ValueError('Earlier completion branch: re-audit observed support before fitting')
        # This local response model is conditional on these observed predicates.
        # Do not pretend an unobserved/later state or a changed duration is supported.
        assert q['covered_gates']==q['required_gates'] and q['remaining_mm']<=12.
        assert q['planned_total_ticks']==spec['total_ticks'],('completion_branch_changed',spec,q)
        assert (q['endpoint_margin_mm']>=0)==(q['total_ticks']<=spec['total_ticks'])
        values.append(q['endpoint_margin_mm']);events.append(q)
    candidate['completion_events']=events
    candidate['completion_event_margins_mm']=values
    return np.asarray(values)


def fit(history,anchor,fresh):
    a=np.array(anchor['subspace_coordinates']);values=np.array(anchor['augmented_learning_inequalities'])
    events=np.array(anchor['completion_event_margins_mm']);k=len(a)
    near=[c for c in history if c['physical_pass'] and c['tag']!=anchor['tag'] and
          np.max(abs(np.array(c['subspace_coordinates'])-a))<=4.]
    x=np.array([np.array(c['subspace_coordinates'])-a for c in near])
    assert np.linalg.matrix_rank(x,tol=1e-10)==k
    y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                     c['training_score']-anchor['training_score']] for c in near])
    ye=np.array([np.array(c['completion_event_margins_mm'])-events for c in near])
    weight=1./np.maximum(np.linalg.norm(x,axis=1),.25);gram=(x*weight[:,None]).T@x
    model=np.linalg.solve(gram+.01*np.eye(k),(x*weight[:,None]).T@y+.01*fresh)
    event_model=np.linalg.solve(gram+1e-8*np.eye(k),(x*weight[:,None]).T@ye)
    # Retrospective residuals are a buffer heuristic, not a probability guarantee.
    residual=np.abs(ye-x@event_model)
    buffer=np.maximum(.001,np.quantile(residual,.9,axis=0))
    return model,event_model,buffer


def propose(anchor,transform,model,event_model,buffer,radius,normalizer,base,specs,*,l1_budget=None):
    a=np.array(anchor['subspace_coordinates']);u=base+transform@a;k=len(a)
    values=np.array(anchor['augmented_learning_inequalities']);m=len(values)
    jac,gradient=model[:,:-1].T.copy(),model[:,-1]
    names=labels(anchor['records']);indices=[names.index(f"path{s['path']}.complete_time_s") for s in specs]
    # A valid endpoint at this observed decision implies this known completion
    # time; remove the fitted staircase slope and solve for positive event margin.
    original_values=values.copy()
    values[indices]=0.;jac[indices]=0.
    e=np.array(anchor['completion_event_margins_mm'])
    matrix=np.vstack([np.hstack([-jac,-np.eye(m)]),
        np.hstack([transform,np.zeros((len(u),m))]),np.hstack([-transform,np.zeros((len(u),m))]),
        np.hstack([-event_model.T,np.zeros((len(specs),m))])])
    bounds=[(max(-radius,-2-a[i]),min(radius,2-a[i])) for i in range(k)]+[(0,None)]*m
    rhs=np.r_[values,3-u,3+u,e-buffer]
    objective=np.r_[gradient/normalizer,np.full(m,.1)]
    if l1_budget is not None:
        matrix=np.vstack([np.column_stack([matrix,np.zeros((len(matrix),k))]),
            np.hstack([np.eye(k),np.zeros((k,m)),-np.eye(k)]),
            np.hstack([-np.eye(k),np.zeros((k,m)),-np.eye(k)]),
            np.r_[np.zeros(k+m),np.ones(k)][None,:]])
        rhs=np.r_[rhs,np.zeros(2*k),l1_budget]
        objective=np.r_[objective,np.zeros(k)];bounds += [(0,None)]*k
    solved=linprog(objective,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs')
    if not solved.success:return None
    delta=solved.x[:k];post=values+jac@delta
    predicted=.1*(positive_sum(original_values)-positive_sum(post))-gradient@delta/normalizer
    return delta,float(predicted),e+delta@event_model
