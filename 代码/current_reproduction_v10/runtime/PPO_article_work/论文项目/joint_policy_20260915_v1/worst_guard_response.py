"""Offline minimax restoration of existing learning guards, with full guards retained."""
import numpy as np
from scipy.optimize import linprog
from audit_coupled_trust_model import labels
from learn_verified_trust_joint import positive_sum


def worst_merit(candidate,normalizer):
    g=np.array(candidate['augmented_learning_inequalities'])
    return float(max(0.,-g.min())+.001*(candidate['training_score']/normalizer+.1*positive_sum(g)))


def solve(anchor,transform,model,event_model,buffer,radius,normalizer,base,specs):
    a=np.array(anchor['subspace_coordinates']);u=base+transform@a;k=len(a)
    g=np.array(anchor['augmented_learning_inequalities']);m=len(g);original=g.copy()
    jac,gradient=model[:,:-1].T.copy(),model[:,-1]
    names=labels(anchor['records'])
    for spec in specs:
        index=names.index(f"path{spec['path']}.complete_time_s");g[index]=0.;jac[index]=0.
    # Variables: direction, individual violation slacks, worst slack, absolute direction.
    matrix=np.vstack([
        np.hstack([-jac,-np.eye(m),np.zeros((m,1+k))]),
        np.hstack([-jac,np.zeros((m,m)),-np.ones((m,1)),np.zeros((m,k))]),
        np.hstack([transform,np.zeros((len(u),m+1+k))]),
        np.hstack([-transform,np.zeros((len(u),m+1+k))]),
        np.hstack([-event_model.T,np.zeros((len(specs),m+1+k))]),
        np.hstack([np.eye(k),np.zeros((k,m+1)),-np.eye(k)]),
        np.hstack([-np.eye(k),np.zeros((k,m+1)),-np.eye(k)]),
        np.r_[np.zeros(k+m+1),np.ones(k)][None,:]])
    rhs=np.r_[g,g,3-u,3+u,np.array(anchor['completion_event_margins_mm'])-buffer,
              np.zeros(2*k),12.*radius]
    bounds=[(max(-radius,-2-a[i]),min(radius,2-a[i])) for i in range(k)]+[(0,None)]*(m+1+k)
    primary=np.zeros(k+m+1+k);primary[k+m]=1.
    first=linprog(primary,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs')
    if not first.success:return None
    initial_worst=max(0.,-original.min());optimum=first.x[k+m]
    ceiling=optimum+.1*max(initial_worst-optimum,0.)
    bounds[k+m]=(0,ceiling+1e-10)
    secondary=np.r_[gradient/normalizer,np.full(m,.1),np.zeros(1+k)]
    final=linprog(secondary,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs')
    if not final.success:return None
    delta=final.x[:k];post=g+jac@delta
    predicted=initial_worst-max(0.,-post.min())+.001*(
        .1*(positive_sum(original)-positive_sum(post))-gradient@delta/normalizer)
    return (delta,float(predicted),np.array(anchor['completion_event_margins_mm'])+delta@event_model,
        dict(initial_worst=float(initial_worst),minimum_predicted_worst=float(optimum),
             chosen_predicted_worst=float(max(0.,-post.min())),worst_ceiling=float(ceiling)))
