"""Check measured-intercept correction of the actual geometry-conditioned policy."""
import numpy as np
from audit_context_restoration_scaling import solve_correction
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_geometry_conditioned_joint_learning_20260920'
    source=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    model=read(source/'joint_main_turn_credit_models.json');audit=read(source/'geometry_context_capacity.json')
    history=read(stage/'state_joint/results.json')['candidates'];current=history[2]
    paths=read(source/'support_audit.json')['selected']['paths'];context=np.asarray(audit['context'])[:,:7]
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
    base=np.asarray(model['base_constraints']);loss=np.asarray(model['base_loss_gradient'])
    jac=np.stack([base*context[owners,None,c,:,None] for c in range(7)],axis=2).reshape(len(owners),-1)
    gradient=np.sum(np.stack([loss*context[:,None,c,:,None] for c in range(7)],axis=2),axis=0).reshape(-1)
    rows=[]
    for radius in (.25,1.,4.):
        row=solve_correction(jac,gradient,np.asarray(current['measured_inequalities']),
            history[1]['score']-current['score'],radius=radius)
        rows.append(dict(radius=radius,**row))
        print('GEOMETRY_RESTORATION',radius,{k:v for k,v in row.items() if k!='scaled_parameter_step'},flush=True)
    write_json(stage/'measured_intercept_restoration_audit.json',dict(anchor=current['tag'],models=rows,
        jacobian=jac.tolist(),loss_gradient=gradient.tolist(),source_sha256=sha(stage/'state_joint/results.json'),
        new_physical_runs=0,acceptance_bounds_unchanged=True))


if __name__=='__main__':main()
