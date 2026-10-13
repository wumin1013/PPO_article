"""Same local basis and labels; only add the six previously observed failures."""
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from learn_constrained_entry_credit import solve_halfspace_qp
from search_complete_return import read
from study_io import BASE, register, receipt, write_json
from train_pilot import save_torch


def main():
    torch.set_num_threads(1)
    prior=BASE/'local_basis_saved_data_fit_20260916'
    probe=BASE/'symmetric_entry_credit_probe_20260916'
    dataset=BASE/'matched_entry_interaction_feedback_20260916/full_training_data.npz'
    initial=prior/'kernel_width_0.5.pt'
    stage=register('excluded_local_basis_fit_20260916',
        [__file__,BASE/'kernel_local_feedback_policy.py',BASE/'diagnose_local_basis_fit.py',
         BASE/'learn_constrained_entry_credit.py',initial,prior/'receipt.json',dataset,
         probe/'results.json',probe/'receipt.json',
         BASE/'local_basis_closed_loop_feedback_20260916/receipt.json'],
        dict(new_decisions=0,new_physics_ticks=0,wall_s=90,coefficients=291,
             constraints=6,ridge=.001,width_multiplier=.5,steps_for_later_physics=[.5,1.]),
        'Matched local-basis regression with only observed failed-axis halfspaces added. '
        'Reuse the selected Gaussian width, centers, frozen hidden representation, exact '
        '4421-state labels/weights, ridge, 291 coefficients and support normalization. '
        'Six axial failures come from an earlier independent symmetric probe, not fitted '
        'from the new kernel failures. Solve strict convex per-head QPs with verified KKT '
        'conditions. Check fitted positive MSE and original/tail residual RMS against the '
        'same previous linear fit; retain only if all three offline gates still pass. '
        'If retained, separately register half/full-step physical evaluation; reuse all '
        'unconstrained physics. These halfspaces are heuristic local observations, not '
        'proof that mixtures or nearby states are safe. No altered physics or path rules.')
    started=time.perf_counter()
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    p=KernelLocalFeedbackPolicy();p.load_state_dict(saved['policy'])
    with np.load(dataset) as z:x,y,w=[z[k].copy() for k in ('observation','target_residual','weight')]
    f=feature_batches(p,x)
    failed=[r for r in read(probe/'results.json')['records'] if not r['success']]
    query=[]
    for r in failed:
        with np.load(BASE/r['trace']) as z:query.append(z['observation'][0])
    qf=feature_batches(p,np.asarray(query))
    coeff=[];audits=[]
    for head in range(3):
        a=f[:,head]
        fisher=a.T@(w[:,None]*a)+.001*np.eye(97)
        gradient=a.T@(w*y[:,head])
        c=[np.sign(r['delta'][head])*qf[i,head] for i,r in enumerate(failed) if r['axis']==head]
        direction,audit=solve_halfspace_qp(fisher,gradient,c)
        coeff.append(direction);audits.append(audit)
    coeff=np.asarray(coeff)
    pred=np.einsum('nhf,hf->nh',f,coeff)
    magnitude=max(1.,float(abs(pred).max()))
    coeff/=magnitude;pred/=magnitude
    with torch.no_grad():p.residual_coefficients.copy_(torch.tensor(coeff,dtype=torch.float32))
    after=np.einsum('nhf,hf->nh',qf,coeff)
    violations=[float(np.sign(r['delta'][r['axis']])*after[i,r['axis']]) for i,r in enumerate(failed)]
    assert max(violations)<1e-8
    original=read(prior/'results.json');n=original['original_states'];npos=original['positive_states']
    metrics={}
    for name,s in [('original',slice(0,n)),('positive',slice(n,n+npos)),('tail',slice(n+npos,None))]:
        metrics[name]=dict(mse=float(np.mean((pred[s]-y[s])**2)),
            rms_residual=float(np.sqrt(np.mean(pred[s]**2))),max_residual=float(abs(pred[s]).max()))
    linear=original['linear']
    qualifies=bool(metrics['positive']['mse']<linear['positive']['mse']
        and metrics['original']['rms_residual']<linear['original']['rms_residual']
        and metrics['tail']['rms_residual']<linear['tail']['rms_residual'])
    model='excluded_kernel.pt'
    save_torch(stage/model,dict(saved,policy=p.state_dict(),
        learning_method='same_local_basis_regression_with_observed_failure_exclusions'))
    write_json(stage/'results.json',dict(linear=linear,kernels=[dict(model=model,
        width_multiplier=.5,metrics=metrics,offline_eligible=qualifies)],kkt=audits,
        failed_direction_values=violations,support_normalization=magnitude,
        matched_unconstrained_model=str(initial),goal_complete=False))
    receipt(stage,dict(status='completed_matched_local_basis_exclusion_fit',
        new_decisions=0,new_physics_ticks=0,offline_eligible=qualifies,wall_s=time.perf_counter()-started))
    print('EXCLUDED_LOCAL_BASIS',metrics,'eligible',qualifies,flush=True)


if __name__=='__main__':main()
