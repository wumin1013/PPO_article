"""Check whether shared recovery is limited by the local response box."""
import numpy as np
from scipy.optimize import linprog
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    model=read(stage/'joint_main_turn_credit_models.json');support=read(stage/'support_audit.json')
    history=read(stage/'state_joint/results.json')['candidates'];anchor=history[1]
    paths=support['selected']['paths'];context=np.asarray(support['physical_contexts'])
    jac=np.asarray(model['base_constraints']);g=np.asarray(model['base_loss_gradient']);c0=np.asarray(model['constraint_intercept'])
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)];results=[]
    for radius in (2.,4.):
        for count in (1,3,4,6):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-radius,radius)]*len(gradient),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(radius=radius,context_count=count,parameters=len(gradient),success=bool(solved.success),
                predicted_loss=float(anchor['score']+solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None)
            results.append(row);print('SHARED_TRUST_REGION',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
    write_json(stage/'shared_trust_region_audit.json',dict(models=results,new_physical_runs=0,
        source_sha256=sha(stage/'joint_main_turn_credit_models.json'),physical_limits_changed=False,
        interpretation='Larger response boxes only test scale limitation of this shared local model. '
                       'Extrapolated proposals require complete physical validation and retain original coefficient/acceptance bounds.'))


if __name__=='__main__':main()
