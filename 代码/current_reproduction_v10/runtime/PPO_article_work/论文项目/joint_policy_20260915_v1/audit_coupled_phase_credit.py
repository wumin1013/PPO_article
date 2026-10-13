"""Matched one-phase controls and actual joint-composition errors on the same parent."""
import numpy as np
from scipy.optimize import linprog
from duration_joint_metrics import guard_names
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    model=read(stage/'joint_phase_credit_models.json');history=read(stage/'state_joint/results.json')['candidates']
    support=read(stage/'support_audit.json');paths=support['selected']['paths'];context=np.array(support['physical_contexts'])
    jac=np.array(model['base_constraints']);g=np.array(model['base_per_path_loss_gradient']);c0=np.array(history[0]['measured_inequalities'])
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
    controls=[]
    for count in (1,2,3):
        j=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
        grad=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
        for phase in (0,1):
            mask=np.zeros((3,count,2,3),bool);mask[:,:,phase,:]=True;mask=mask.reshape(-1)
            matrix=j[:,mask];gradient=grad[mask];scale=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            fit=linprog(gradient,A_ub=-matrix/scale[:,None],b_ub=c0/scale,bounds=[(-1.,1.)]*len(gradient),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            controls.append(dict(context_count=count,phase='prejump' if phase==0 else 'postpeak',
                success=bool(fit.success),predicted_change=float(fit.fun) if fit.success else None,
                scaled_parameters=fit.x.tolist() if fit.success else None))
    names=[str(p)+':'+k for p in paths for k in ('incoming','outgoing','reverse','energy','peak','apex_speed',
        'time_old','cruise_old','first_turn','corridor','lateral')]+[str(p)+':'+k for p in paths for k in guard_names()]
    compositions=[]
    for block in read(stage/'optimization_blocks.json'):
        if block.get('method')!='joint_phase_composition':continue
        c=next(c for c in history if c['tag']==block['tag']);u=np.array(c['coefficients']).reshape(3,3,2,3)
        compositions.append(dict(block,prejump_coefficient_norm=float(np.linalg.norm(u[:,:,0])),
            postpeak_coefficient_norm=float(np.linalg.norm(u[:,:,1])),
            failed_guards=[dict(name=n,value=v) for n,v in zip(names,c['measured_inequalities']) if v < -1e-8]))
    write_json(stage/'matched_phase_credit_audit.json',dict(matched_one_phase_controls=controls,
        compositions=compositions,new_physical_rollouts=0,
        same_frozen_parent_and_metric_reference=True,
        interpretation='One-phase control LPs use this exact parent and response batch, '
        'so earlier different-parent studies are not used as causal controls. A better '
        'joint model optimum is not itself proof of a physical interaction or successful learning.'))
    print('MATCHED_PHASE_MODELS',controls,flush=True)
    for c in compositions:print('JOINT_REALITY',c,flush=True)


if __name__=='__main__':main()
