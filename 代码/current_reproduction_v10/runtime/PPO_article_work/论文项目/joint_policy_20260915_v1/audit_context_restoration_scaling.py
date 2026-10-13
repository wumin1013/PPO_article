"""Check numerical conditioning and optional optimization margins, without new rollouts."""
import numpy as np
from scipy.optimize import linprog
from study_io import BASE,write_json
from search_complete_return import read


def solve_correction(jac,g,values,loss_allowance,*,margin=0.,equilibrate=True,radius=.5):
    n=len(g);variable=np.max(abs(jac),axis=1)>1e-12
    required=np.where(variable,margin,0.)
    a=np.vstack([np.c_[-jac,np.zeros((len(values),n))],np.c_[np.eye(n),-np.eye(n)],
                 np.c_[-np.eye(n),-np.eye(n)],np.r_[g,np.zeros(n)][None]])
    b=np.r_[values-required,np.zeros(2*n),loss_allowance]
    if equilibrate:
        scales=np.maximum(np.maximum(np.max(abs(a),axis=1),abs(b)),1e-12)
        a=a/scales[:,None];b=b/scales
    result=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=a,b_ub=b,
        bounds=[(-radius,radius)]*n+[(0.,None)]*n,method='highs',
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    step=result.x[:n] if result.success else None
    return dict(success=bool(result.success),status=str(result.message),
        scaled_parameter_step=step.tolist() if step is not None else None,
        unscaled_predicted_guard_minimum=float(np.min(values+jac@step)) if step is not None else None,
        predicted_loss_change=float(g@step) if step is not None else None,
        lp_residual_minimum=float((b-a@result.x).min()) if result.success else None)


def main():
    stage=BASE/'canonical_gate_band_recentered_context_memory_credit_20260920'
    assert read(stage/'receipt.json')['status'].startswith('completed')
    result=read(stage/'state_joint/results.json');near=next(c for c in result['candidates'] if c['tag']==result['nearest'])
    model=read(stage/'recentered_credit_model.json')
    jac=np.array(model['central_constraints']);g=np.array(model['central_loss']);values=np.array(near['measured_inequalities'])
    # Never let this diagnostic change the actual acceptance conditions.
    rows=[]
    for margin in (2e-7,0.):
        for equilibrate in (False,True):
            r=solve_correction(jac,g,values,0.,margin=margin,equilibrate=equilibrate)
            rows.append(dict(margin=margin,equilibrate=equilibrate,**r))
    write_json(stage/'restoration_scaling_audit.json',dict(anchor=near['tag'],models=rows,
        new_physical_rollouts=0,acceptance_tolerance_unchanged=1e-8,
        interpretation='Only optional positive optimization margin and numerical row scaling '
        'are varied. Zero margin still requires nonnegative predicted guards; no shape or '
        'physical acceptance bound is weakened. Every proposal requires physical validation.'))
    print('RESTORATION_SCALING',[(r['margin'],r['equilibrate'],r['success'],r['unscaled_predicted_guard_minimum']) for r in rows])


if __name__=='__main__':main()
