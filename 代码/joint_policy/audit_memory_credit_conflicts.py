"""Audit the measured local credit cone and observable state distinctions."""
import numpy as np
from scipy.optimize import linprog
from audit_early_constraint_activity import names
from study_io import BASE,write_json,sha
from search_complete_return import read
from learn_complete_joint_teachers import teacher_score
from multi_corner_raw_metrics import objective


def main():
    stage=BASE/'canonical_gate_band_memory_calibrated_joint_credit_learning_20260920'
    model=read(stage/'local_credit_model.json');rows=read(stage/'state_joint/results.json')['candidates']
    zero=rows[0];labels=names(zero['records']);jac=np.asarray(model['central_constraints']);g=np.asarray(model['central_loss'])
    c0=np.asarray(zero['measured_inequalities']);solution=linprog(g,A_ub=-jac,b_ub=c0,bounds=[(-1.,1.)]*len(g),method='highs')
    assert solution.success
    marginal=np.asarray(solution.ineqlin.marginals)
    dual=[dict(labels[i],multiplier=float(marginal[i])) for i in np.argsort(marginal) if marginal[i]<-1e-6]
    subset=[]
    for path in sorted({r['path'] for r in labels}):
        mask=np.array([r['path']!=path for r in labels])
        r=linprog(g,A_ub=-jac[mask],b_ub=c0[mask],bounds=[(-1.,1.)]*len(g),method='highs')
        subset.append(dict(path_omitted_from_model_only=path,predicted_loss_change=float(r.fun) if r.success else None))
    individual=[];data=read(stage/'baseline.json');old={r['path']:r['baseline_shape'] for r in data['retained'] if 'baseline_shape' in r}
    for b in zero['records']:
        path=b['path'];mask=np.array([r['path']==path for r in labels]);gp=[]
        def local_score(r):
            return teacher_score(r,b,old[path],barrier=False) if 'shape' in b else objective(r,b)
        for pair in model['pairs']:
            a=next(r for c in rows if c['tag']==pair['plus'] for r in c['records'] if r['path']==path)
            z=next(r for c in rows if c['tag']==pair['minus'] for r in c['records'] if r['path']==path)
            gp.append(.5*(local_score(a)-local_score(z)))
        single=linprog(gp,A_ub=-jac[mask],b_ub=c0[mask],bounds=[(-1.,1.)]*len(g),method='highs')
        individual.append(dict(path=path,own_loss_predicted_change=float(single.fun) if single.success else None))
    pairs=[]
    for pair in model['pairs']:
        if pair['index'] not in (3,9,15):continue
        minus=next(c for c in rows if c['tag']==pair['minus'])
        pairs.append(dict(index=pair['index'],step=-pair['step'],tag=minus['tag'],changes=[dict(path=r['path'],
            reverse_change_deg=r['shape']['full_physical_reverse_deg']-b['shape']['full_physical_reverse_deg'],
            lateral_change_mm=r['shape']['reverse_lateral_total_mm']-b['shape']['reverse_lateral_total_mm'],
            exit_change_mm=r['shape']['out_edge_outward_mm']-b['shape']['out_edge_outward_mm'])
            for r,b in zip(minus['records'],zero['records']) if r['path'] in (14,104,204,304)]))
    write_json(stage/'local_credit_conflict_audit.json',dict(model_sha256=sha(stage/'local_credit_model.json'),
        full_local_predicted_loss_change=float(solution.fun),binding_dual_constraints=dual,
        individual_route_local_models=individual,
        leave_one_route_model_only=subset,measured_negative_steering_responses=pairs,
        physical_reruns=0,acceptance_constraints_changed=False,
        interpretation='LP dual and leave-one-route calculations diagnose only the current '
        'finite-difference local model. No route is removed from actual training or selection. '
        'The cone result is not global impossibility and may reflect model error as well '
        'as limited shared state representation. Recorded physical pairs retain all routes.'))
    print('LOCAL_DUAL',dual[:12],flush=True)
    print('MODEL_ONLY_ROUTE_EFFECT',sorted(subset,key=lambda x:x['predicted_loss_change'] or 0.)[:6],flush=True)
    print('ACTUAL_NEGATIVE_STEERING',pairs,flush=True)
    print('INDIVIDUAL_ROUTE_CONES',individual,flush=True)


if __name__=='__main__':main()
