"""Local complete-return response fitting with explicit information diagnostics."""
import numpy as np


def fit_local_response(history,anchor,steps,prior,distance=4.,regularizer=.05):
    u=np.array(anchor['coefficients']);values=np.array(anchor['augmented_learning_inequalities'])
    near=[c for c in history if c['physical_pass'] and c['tag']!=anchor['tag'] and
          np.max(abs((np.array(c['coefficients'])-u)/steps))<=distance]
    if not near:return prior.copy(),dict(samples=0,rank=0,prior_effective_dimensions=len(u))
    x=np.array([(np.array(c['coefficients'])-u)/steps for c in near])
    y=np.array([np.r_[np.array(c['augmented_learning_inequalities'])-values,
                            c['training_score']-anchor['training_score']] for c in near])
    weight=1./np.maximum(np.linalg.norm(x,axis=1),.25);gram=(x*weight[:,None]).T@x
    model=np.linalg.solve(gram+regularizer*np.eye(len(u)),
                          (x*weight[:,None]).T@y+regularizer*prior)
    eigen=np.maximum(np.linalg.eigvalsh(gram),0.)
    return model,dict(samples=len(near),rank=int(np.linalg.matrix_rank(x,tol=1e-10)),
        prior_effective_dimensions=float(np.sum(regularizer/(eigen+regularizer))))
