"""Separate shared-context capacity from per-route local repair feasibility."""
import argparse
import numpy as np
from scipy.optimize import linprog
from duration_joint_metrics import guard_names
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_approach_conditioned_recovery_learning_20260920')
    parser.add_argument('--model',default='conditioned_credit_models.json')
    args=parser.parse_args();stage=BASE/args.stage
    assert read(stage/'receipt.json')['status'].startswith('completed')
    model=read(stage/args.model)
    paths=read(stage/'support_audit.json')['selected']['paths']
    dimensions=np.asarray(model['base_loss_gradient'])[0].size
    jac=np.asarray(model['base_constraints']).reshape(-1,dimensions)
    gradient=np.asarray(model['base_loss_gradient']).reshape(len(paths),dimensions)
    c0=np.asarray(model['constraint_intercept'])
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
    names=[f'{p}:{k}' for p in paths for k in ('incoming','outgoing','reverse','energy','peak','apex_speed','time_old','cruise_old','first_turn','corridor','lateral')]
    names += [f'{p}:{k}' for p in paths for k in guard_names()]
    rows=[]
    for i,p in enumerate(paths):
        mask=owners==i
        if c0[mask].min()>=-1e-8:
            continue
        for radius in (1.,4.):
            a=-jac[mask];b=c0[mask]
            scales=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
            result=linprog(gradient[i],A_ub=a/scales[:,None],b_ub=b/scales,bounds=[(-radius,radius)]*dimensions,
                method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(path=p,radius=radius,success=bool(result.success),status=str(result.message),
                predicted_change=float(result.fun) if result.success else None,
                scaled_direction=result.x.tolist() if result.success else None)
            if not result.success and radius==4.:
                relaxed=linprog(np.r_[np.zeros(dimensions),1.],
                    A_ub=np.c_[a/scales[:,None],-np.ones(len(b))],b_ub=b/scales,
                    bounds=[(-radius,radius)]*dimensions+[(0.,None)],method='highs')
                if relaxed.success:
                    residual=(b-a@relaxed.x[:dimensions])/scales
                    labels=np.asarray(names)[mask]
                    row['minimum_relative_violation']=float(relaxed.x[-1])
                    row['blocking_rows']=[dict(name=str(labels[q]),scaled_slack=float(residual[q])) for q in np.argsort(residual)[:5]]
            rows.append(row);print('CONDITIONED_ROUTE_CAPACITY',{k:v for k,v in row.items() if k!='scaled_direction'},flush=True)
            if result.success:break
    write_json(stage/'conditioned_route_capacity.json',dict(models=rows,new_physical_rollouts=0,
        source_sha256=sha(stage/args.model),runtime_route_controls_added=False,
        interpretation='Independent route coefficients and fourfold boxes are offline capacity/scale controls only. '
                       'They neither define a deployable policy nor prove global impossibility when infeasible.'))


if __name__=='__main__':main()
