"""Audit measured local credit and feasibility conflicts; no new rollout."""
import argparse
import numpy as np
from scipy.optimize import linprog
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    file=stage/'innovation_coordinates/results.json';history=read(file)['candidates']
    zero=next(c for c in history if not np.any(c['coefficients']))
    valid=[c for c in history if np.any(c['coefficients']) and c['score']<1e4
           and all(r['success'] for r in c['records'])]
    x=np.asarray([c['coefficients'] for c in valid])
    g0=np.asarray(zero['measured_inequalities']);score0=zero['score']
    y=np.asarray([[c['score']-score0,* (np.asarray(c['measured_inequalities'])-g0)] for c in valid])
    norms=np.linalg.norm(x,axis=1);weights=1./np.maximum(norms,.01)**2
    def fit(indices):
        a=x[indices]*np.sqrt(weights[indices,None]);b=y[indices]*np.sqrt(weights[indices,None])
        return np.linalg.lstsq(a,b,rcond=1e-8)
    coeff,residual,rank,sv=fit(np.arange(len(x)))
    held=np.arange(len(x))%4==0
    test_coeff,_,test_rank,_=fit(np.flatnonzero(~held))
    error=x[held]@test_coeff-y[held]
    active=np.flatnonzero(abs(g0)<1e-8)
    scale=np.maximum(np.max(abs(y[held]),axis=0),1e-8)
    normalized_rmse=np.sqrt(np.mean(error**2,axis=0))/scale
    epsilon=.0025
    lp=linprog(coeff[:,0],A_ub=-coeff[:,1:].T,b_ub=g0+1e-10,
               bounds=[(-epsilon,epsilon)]*x.shape[1],method='highs')
    proposal=lp.x if lp.success else np.zeros(x.shape[1])
    delta=proposal@coeff
    report=dict(new_rollouts=0,source_results_sha256=sha(file),total_saved_candidates=len(history),
        successful_continuous_score_candidates=len(valid),dimensions=x.shape[1],rank=int(rank),
        singular_values=sv.tolist(),minimum_measured_nonzero_parameter_norm=float(norms.min()),
        median_measured_parameter_norm=float(np.median(norms)),
        heldout_rank=int(test_rank),heldout_score_normalized_rmse=float(normalized_rmse[0]),
        active_guard_indices=active.tolist(),
        heldout_active_guard_normalized_rmse=normalized_rmse[1:][active].tolist(),
        linear_proposal=proposal.tolist(),linear_predicted_score_change=float(delta[0]),
        linear_predicted_minimum_guard=float(np.min(g0+delta[1:])),
        proposal_evaluated=False,
        interpretation='Weighted linear fit of saved complete-return responses, with every fourth sample held out. '
            'A feasible linear cone or its absence is only a surrogate diagnostic; finite-scale data and '
            'closed-loop nonlinearity preclude claiming physical feasibility or a causal state limitation.')
    write_json(stage/'shared_constraint_credit_audit.json',report)
    print('SHARED_CONSTRAINT_CREDIT',report,flush=True)


if __name__=='__main__':main()
