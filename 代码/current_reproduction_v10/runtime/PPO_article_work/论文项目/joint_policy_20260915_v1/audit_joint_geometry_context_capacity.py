"""Test whether observable future bend separates conflicting shared motion credit."""
import numpy as np
import torch
from scipy.optimize import linprog
from joint_teacher_state_policy import features
from observable_memory_joint_policy import future_bend_context
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    model=read(stage/'joint_main_turn_credit_models.json');support=read(stage/'support_audit.json')
    paths=support['selected']['paths'];old_context=np.asarray(support['physical_contexts'])
    with np.load(stage/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations']);bend=future_bend_context(training['observations'])
    mean=[];scale=[];context=np.zeros((len(paths),10,4));context[:,:6]=old_context
    for k in range(5):
        mask=np.any(f[:,k]!=0.,axis=1);mean.append(float(bend[mask].mean()))
        scale.append(float(max(bend[mask].std(),.005)))
        if k==2:continue
        j=support['trainable_kernel_indices'].index(k)
        for i,p in enumerate(paths):
            ids=np.flatnonzero(mask&(training['path_ids']==p));assert len(ids)<=1
            if len(ids):
                geom=np.tanh((bend[ids[0]]-mean[k])/scale[k]);v,a=old_context[i,1:3,j]
                context[i,6:,j]=[geom,v*geom,a*geom,geom*geom]
    jac=np.asarray(model['base_constraints']);g=np.asarray(model['base_loss_gradient']);c0=np.asarray(model['constraint_intercept'])
    anchor=read(stage/'state_joint/results.json')['candidates'][1]
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)];models=[]
    for count in (7,9,10):
        for radius in (1.,4.):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-radius,radius)]*len(gradient),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(context_count=count,radius=radius,success=bool(solved.success),status=str(solved.message),
                predicted_loss=float(anchor['score']+solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None)
            models.append(row);print('GEOMETRY_CONTEXT_CAPACITY',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
    write_json(stage/'geometry_context_capacity.json',dict(models=models,context=context.tolist(),
        per_center_geometry_mean=mean,per_center_geometry_scale=scale,
        source_sha256=sha(stage/'joint_main_turn_credit_models.json'),new_physical_runs=0,
        interpretation='Observed reference-tangent bend is an input to a shared learned function, not an angle-speed table. '
                       'This is a conditional local response model; every proposal needs full physical validation.'))


if __name__=='__main__':main()
