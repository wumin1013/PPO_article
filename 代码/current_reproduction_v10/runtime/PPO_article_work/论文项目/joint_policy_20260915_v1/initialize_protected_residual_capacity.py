"""Add zero-initialized learned state capacity after a measured nullspace limit."""
import time
from datetime import datetime,timezone
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from learn_exit_local_credit import last_layer_features
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from probe_closed_loop_parameter_credit import METRICS
from search_complete_return import read
from train_pilot import save_torch
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'protected_complete_function_credit_learning_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    dense=BASE/'dense_corner_complete_credit_collection_20260917';sparse=BASE/'single_decision_complete_credit_probe_20260917'
    stage=register('protected_residual_capacity_initialization_20260917',
        [__file__,model,parent/'receipt.json',parent/'training_selection.json',
         BASE/'kernel_local_feedback_policy.py',BASE/'diagnose_local_basis_fit.py',
         BASE/'probe_coordinated_suffix.py',BASE/'learn_exit_local_credit.py',
         dense/'results.json',dense/'queries.npz',sparse/'results.json',
         BASE/'local_basis_closed_loop_feedback_20260916/receipt.json'],
        dict(new_physics_ticks=0,new_policy_decisions=0,wall_s=180,centers=256,
             additional_learnable_coefficients=768,protected_paths=[16,31],
             kernel_width_rule='median distance to fourth-nearest farthest-point center',
             ridge=.001),
        'Address a verified limitation of final-layer-only skill protection. At the latest '
        '64/64,12/12 candidate,137 states from paths16+31 span rank96 for odd steering '
        'and97 for both even heads: preserving both leaves no functional update. Reuse '
        'the existing reflection-equivariant Gaussian hidden-state residual implementation, '
        'with256 training-state centers and zero coefficients so initialization exactly '
        'preserves the learned policy. This is NOT a repeat of failed97-center target '
        'regression: no positive labels/MSE fit, no deployment of interventions; proposals '
        'will be trained from complete-function returns with explicit two-path action '
        'preservation. Choose centers/width from current64 training states only, using '
        'a single declared rule, not development tuning. Freeze base and kernel features; '
        'the768 coefficients remain joint learned feedback parameters. Build five proposal '
        'directions in the protected coefficient subspace using saved old-parent action '
        'effects only as proposals plus an average-feed direction. Actual new-policy '
        'full-function probes are required before learning/adoption. Capacity, feature '
        'basis,metric support and protected path set change together; no single-factor '
        'architecture superiority claim. Keep original72 observations and executor here '
        'without claiming minimal or continuous state. Stop if no useful nullspace or '
        'initial functional identity fails. No new physics/development/sealed tests.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0)
    state=torch.load(model,map_location='cpu',weights_only=False)['policy']
    base=LocalGeometryFeedbackPolicy(mask_endpoint=True);base.load_state_dict(state)
    p=KernelLocalFeedbackPolicy(centers=256);missing=p.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys and set(missing.missing_keys)=={
        'kernel_centers','kernel_scale','kernel_width','residual_scale','residual_coefficients'}
    observations=[];weights=[];path_ids=[];protected=[]
    for i,trace in enumerate(selection['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        local=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=local)
        local.append(env.observation());observations.extend(local);weights.extend([1/(64*len(local))]*len(local))
        path_ids.extend([i]*len(local))
        if i in (16,31):protected.extend(local)
    x=np.asarray(observations);w=np.asarray(weights);cx=np.asarray(protected);assert len(cx)==137
    with torch.no_grad():
        tensor=torch.tensor(x);a,b=p.hidden_pair(tensor);support=torch.cat([a,b]).numpy().astype(float)
    scale=np.maximum(support.std(0),.05);normalized=support/scale
    nearest=np.full(len(support),np.inf);indices=[]
    for _ in range(256):
        index=int(np.argmax(nearest));indices.append(index)
        nearest=np.minimum(nearest,np.mean((normalized-normalized[index])**2,axis=1))
    centers=support[indices]
    squared=np.stack([np.mean((normalized-c/scale)**2,axis=1) for c in centers],axis=1)
    width=float(np.median(np.sqrt(np.partition(squared,3,axis=1)[:,3])));assert width>0
    with torch.no_grad():
        p.kernel_centers.copy_(torch.tensor(centers));p.kernel_scale.copy_(torch.tensor(scale))
        p.kernel_width.fill_(width);p.residual_scale.copy_(torch.tensor(AMPLITUDE))
        initial_error=float(abs(p.distribution(tensor).mean-base.distribution(tensor).mean).max())
    assert initial_error==0.
    f=feature_batches(p,x);cf=feature_batches(p,cx)
    with np.load(dense/'queries.npz') as z:qx=z['observation'].copy()
    qf=feature_batches(p,qx);support_features=np.concatenate([f,qf])
    oldf=last_layer_features(base,cx);old_ranks=[int((np.linalg.svd(oldf[:,h],compute_uv=False)>1e-12).sum()) for h in range(3)]
    raw=read(dense/'results.json');original=read(sparse/'results.json')
    baselines={r['path']:r for r in original['baselines']}
    lookup={(r['query'],r['axis'],r['direction']):r for r in raw['records']}
    sensitivities={}
    for metric in METRICS:
        values=[]
        for q in raw['queries']:
            row=[]
            for h in range(3):
                lo,hi=[lookup[q['query'],h,s] for s in (-1,1)]
                if lo['success'] and hi['success']:g=(hi['shape'][metric]-lo['shape'][metric])/(2*AMPLITUDE[h])
                elif lo['success']:g=(lo['shape'][metric]-baselines[q['path']]['shape'][metric])/(-AMPLITUDE[h])
                elif hi['success']:g=(hi['shape'][metric]-baselines[q['path']]['shape'][metric])/AMPLITUDE[h]
                else:g=0.
                row.append(g)
            values.append(row)
        sensitivities[metric]=np.asarray(values)
    basis=np.zeros((5,3,256));ranks=[]
    for h in range(3):
        c=cf[:,h];_,sv,vt=np.linalg.svd(c,full_matrices=True);rank=int((sv>1e-10).sum());ranks.append(rank)
        z=vt[rank:].T;assert z.shape[1]>0
        fisher=f[:,h].T@(w[:,None]*f[:,h])+.001*np.eye(256);reduced=z.T@fisher@z
        for i,metric in enumerate(METRICS):
            g=-np.sum(qf[:,h]*(sensitivities[metric][:,h]*AMPLITUDE[h]/16)[:,None],axis=0)
            basis[i,h]=z@np.linalg.solve(reduced,z.T@g)
        if h==1:
            g=f[:,h].T@w;basis[4,h]=z@np.linalg.solve(reduced,z.T@g)
    audits=[]
    for i in range(5):
        size=float(abs(np.einsum('nhf,hf->nh',support_features,basis[i])).max());assert size>1e-10
        basis[i]*=(2. if i==4 else 1.)/size
        algebraic=float(abs(np.einsum('nhf,hf->nh',cf,basis[i])*AMPLITUDE).max());assert algebraic<1e-7
        audits.append(dict(index=i,max_coefficient=float(abs(basis[i]).max()),protected_latent_error=algebraic))
    names=list(METRICS)+['average_feed']
    np.savez_compressed(stage/'basis.npz',basis=basis,names=np.asarray(names),observation=x,weight=w,
        protected_observations=cx,protected_features=cf,path_id=np.asarray(path_ids))
    save_torch(stage/'initial.pt',dict(policy=p.state_dict(),architecture='kernel_local_feedback',centers=256,
        parent_sha256=sha(model),zero_residual_initialization=True,learning_method='pending_complete_function_credit'))
    write_json(stage/'results.json',dict(initial_mean_error=initial_error,states=len(x),protected_states=len(cx),
        original_rank_by_head=old_ranks,residual_rank_by_head=ranks,nullity_by_head=[256-r for r in ranks],
        width=width,coefficients=768,proposals=audits,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_zero_residual_capacity_initialization',new_physics_ticks=0,new_policy_decisions=0,
        states=len(x),initial_mean_error=initial_error,costs=costs,wall_s=time.perf_counter()-started))
    status_path=BASE/'当前执行状态.json';status=read(status_path)
    status['current'].update(state='protected_residual_capacity_ready',training_running=False,physical_diagnostic_running=False,
        working_candidate='protected_complete_function_credit_learning_20260917/candidate_0.5.pt',
        working_candidate_sha256=sha(model),candidate_final_accepted=False)
    status['completed'].append(dict(stage=stage.name,receipt_sha256=sha(stage/'receipt.json')))
    status['next']='Measure actual complete-function effects of the zero-initialized protected residual proposals; do not claim capacity or offline proposals solve shape.'
    status['updated_utc']=datetime.now(timezone.utc).isoformat();write_json(status_path,status)
    print('RESIDUAL_CAPACITY',len(x),old_ranks,ranks,width,audits,flush=True)


if __name__=='__main__':main()
