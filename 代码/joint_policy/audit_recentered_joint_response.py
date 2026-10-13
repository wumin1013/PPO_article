"""Compare measured local credit with the reused prior and diagnose model conflicts."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import numpy as np
from scipy.optimize import linprog
from audit_remaining_joint_credit import names_for
from study_io import BASE, write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_recentered_joint_subspace_learning_20260920')
    args=parser.parse_args();stage=BASE/args.stage
    model=read(stage/'measured_local_credit.json');subspace=read(stage/'subspace.json')
    history=read(stage/'state_joint/results.json')['candidates'];indexed={c['tag']:c for c in history}
    anchor=(indexed[subspace['anchor']] if subspace['anchor'] in indexed else
            next(c for c in history if c.get('kind')=='changed_observation_metric'))
    values=np.asarray(anchor['learning_inequalities'])
    names=names_for(anchor['records']);jac=np.asarray(model['jacobian'])
    prior=np.asarray(read(BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920/final_secant_credit.json')['jacobian'])@np.asarray(subspace['basis'])
    failures=np.asarray(anchor['measured_inequalities']) < -1e-8
    comparison=[]
    for pair in model['pairs']:
        j=pair['direction']
        p,m=[np.asarray(indexed[pair[k]]['learning_inequalities']) for k in ('plus','minus')]
        prior_error=np.r_[p-values-prior[:,j]*.5,m-values+prior[:,j]*.5]
        measured_error=np.r_[p-values-jac[:,j]*.5,m-values+jac[:,j]*.5]
        comparison.append(dict(direction=j,prior_error_norm=float(np.linalg.norm(prior_error)),
            measured_error_norm=float(np.linalg.norm(measured_error)),
            failed_row_derivative_sign_changes=int(((prior[:,j]*jac[:,j]<0.)&failures).sum()),
            failed_row_prior_error=float(np.linalg.norm((jac-prior)[failures,j])),
            even_error_max=float(abs(measured_error).max())))
    elastic=[]
    for radius in (.25,1.,4.):
        # Model slack is a diagnostic only, never an actual acceptance allowance.
        scaling=np.maximum(np.maximum(np.max(abs(jac),axis=1),abs(values)),1e-10)
        a=np.c_[-jac/scaling[:,None],-np.ones(len(values))]
        solved=linprog(np.r_[np.zeros(jac.shape[1]),1.],A_ub=a,b_ub=values/scaling,
            bounds=[(-radius,radius)]*jac.shape[1]+[(0.,None)],method='highs')
        row=dict(radius=radius,success=bool(solved.success))
        if solved.success:
            margins=values+jac@solved.x[:-1]
            dual=abs(solved.ineqlin.marginals)
            row.update(relative_model_slack=float(solved.x[-1]),
                minimum_predicted_guard=float(margins.min()),step=solved.x[:-1].tolist(),
                conflicts=[dict(name=names[i],dual=float(dual[i]),guard=float(values[i]))
                           for i in np.flatnonzero(dual>1e-6)])
        elastic.append(row)
    output=dict(anchor=anchor['tag'],comparison=comparison,elastic_diagnostic=elastic,
        total_prior_error=float(sum(c['prior_error_norm'] for c in comparison)),
        total_local_even_error=float(sum(c['measured_error_norm'] for c in comparison)),
        subspace_explained_energy=float(sum(np.asarray(subspace['singular_values'])[:jac.shape[1]]**2)/
                                      sum(np.asarray(subspace['singular_values'])**2)),
        new_physical_runs=0,actual_acceptance_unchanged=True)
    write_json(stage/'measured_credit_quality.json',output)
    print('RESPONSE_ERROR',output['total_prior_error'],output['total_local_even_error'],
          'SUBSPACE_ENERGY',output['subspace_explained_energy'],flush=True)
    for row in elastic:print('MODEL_CONFLICT',row['radius'],row.get('relative_model_slack'),row.get('conflicts'),flush=True)


if __name__=='__main__':main()
