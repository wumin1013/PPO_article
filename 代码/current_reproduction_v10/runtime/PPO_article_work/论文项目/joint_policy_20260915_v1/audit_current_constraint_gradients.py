"""All-time policy gradients from current complete one-action consequences."""
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


SHAPE_METRICS=('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
               'full_normalized_energy','full_spatial_peak_curvature_mm_inv')
METRICS=SHAPE_METRICS+('segment_max_distance_mm','complete_time_s')


def values(record):
    shape=record['shape'];out=[shape[k] if shape is not None else 0. for k in SHAPE_METRICS]
    out.append(record['segment_distance']['max_distance_upper_mm'])
    out.append(record['full_time_s'])
    # Baseline time comes from the saved full trace, supplied by the caller.
    return np.asarray(out,float)


def assemble(data,observations,policy,spaces):
    """Return per-path gradients and failed-axis exclusions, never failed-prefix scores."""
    ids=[r['path'] for r in data['baselines']];index={p:i for i,p in enumerate(ids)}
    cuts=np.r_[0,np.cumsum([z.shape[1] for z in spaces])];dimension=int(cuts[-1])
    features=feature_batches(policy,observations)
    jac=np.zeros((len(observations),3,dimension))
    for h,z in enumerate(spaces):jac[:,h,cuts[h]:cuts[h+1]]=AMPLITUDE[h]*(features[:,h]@z)
    base={r['path']:r for r in data['baselines']}
    for r in base.values():
        with np.load(BASE/r['trace']) as z:r['full_time_s']=len(z['rows'])*.001
    gradient=np.zeros((len(ids),len(METRICS),dimension));failed=[];modes={}
    lookup={(r['query'],r['axis'],r['direction']):r for r in data['records']}
    for q in data['queries']:
        qid=q['query'];path=q['path'];b=values(base[path])
        for h in range(3):
            lo,hi=[lookup[qid,h,s] for s in (-1,1)]
            if lo['success'] and hi['success']:
                width=hi['latent_offset']-lo['latent_offset']
                assert width>0
                slope=(values(hi)-values(lo))/width;mode='central_complete'
            else:
                good=[r for r in (lo,hi) if r['success']]
                if good:slope=(values(good[0])-b)/good[0]['latent_offset'];mode='complete_one_sided'
                else:slope=np.zeros(len(METRICS));mode='both_failed_no_gradient'
            gradient[index[path]]+=slope[:,None]*jac[qid,h]
            modes[mode]=modes.get(mode,0)+1
            for r in (lo,hi):
                if not r['success']:
                    failed.append(dict(query=qid,path=path,anchor=q['anchor'],head=h,sign=r['direction']))
    exclusions=np.asarray([r['sign']*jac[r['query'],r['head']]/AMPLITUDE[r['head']] for r in failed]).reshape(-1,dimension)
    return ids,gradient,jac,exclusions,failed,modes


def main():
    torch.set_num_threads(1)
    data_dir=BASE/'current_dense_constraint_credit_collection_20260917'
    assert (data_dir/'receipt.json').exists(),'Wait for the complete registered physical collection'
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    function_dir=BASE/'constraint_normal_function_credit_probe_20260917'
    margin_dir=BASE/'constraint_normal_continuous_margin_credit_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    stage=register('current_direct_constraint_gradient_audit_20260917',
        [__file__,data_dir/'receipt.json',data_dir/'results.json',data_dir/'queries.npz',
         prepared/'support.npz',prepared/'receipt.json',model,function_dir/'basis.npz',
         function_dir/'results.json',margin_dir/'results.json',BASE/'diagnose_local_basis_fit.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=120),
        'Sum exact-time, current-parent finite single-action effects into every free residual '
        'parameter, separately for each path and existing shape/corridor/time metric. '
        'Failed-prefix scores are excluded; retain each failed axis as a local update '
        'exclusion. Verify the fixed-state parameter Jacobian against real network means. '
        'Compare its projection onto17 already measured complete-policy directions to '
        'their actual finite responses, including nonlinear and discrete event differences. '
        'These finite estimates are not exact infinitesimal derivatives or safety proofs. '
        'No actor update from this audit alone; require a feasible constrained proposal '
        'and a small whole-policy physical composition check before learning.')
    started=time.perf_counter();data=read(data_dir/'results.json')
    assert data['completed_paths']==17 and len(data['queries'])==832
    with np.load(data_dir/'queries.npz') as z:observations=z['observation'].copy()
    with np.load(prepared/'support.npz') as z:spaces=[z[f'nullspace_{h}'].copy() for h in range(3)]
    saved=torch.load(model,map_location='cpu',weights_only=False)['policy']
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(saved)
    ids,gradient,jac,exclusions,failed,modes=assemble(data,observations,p,spaces)
    rng=np.random.default_rng(917062);delta=rng.normal(size=jac.shape[-1]);delta*=.1/max(abs(np.einsum('nhd,d->nh',jac,delta)/AMPLITUDE).max(),1e-12)
    cuts=np.r_[0,np.cumsum([z.shape[1] for z in spaces])]
    with torch.no_grad():
        before=p.distribution(torch.tensor(observations)).mean.numpy()
        for h,z in enumerate(spaces):p.residual_coefficients[h].add_(torch.tensor(z@delta[cuts[h]:cuts[h+1]],dtype=torch.float32))
        actual=p.distribution(torch.tensor(observations)).mean.numpy()-before
    error=float(abs(actual-np.einsum('nhd,d->nh',jac,delta)).max());assert error<1e-6
    with np.load(function_dir/'basis.npz') as z:basis=z['basis'].copy()
    reduced=np.concatenate([basis[:,h]@spaces[h] for h in range(3)],axis=1)
    predicted=np.einsum('pmd,kd->pmk',gradient,reduced)
    function=read(function_dir/'results.json');margins=read(margin_dir/'results.json')
    actual=np.zeros_like(predicted)
    for k in range(len(basis)):
        pair=[]
        for sign in (-1,1):
            r=next(r for r in function['candidates'] if r['basis']==k and r['sign']==sign)
            m=next(r for r in margins['candidates'] if r['basis']==k and r['sign']==sign)
            shapes={s['path']:s for s in r['shapes']};runs={s['original_path']:s for s in r['results']}
            margin={i:s for i,s in zip(margins['original_path'],m['margins'])}
            pair.append(np.asarray([[shapes[i][metric] if i in shapes else 0. for metric in SHAPE_METRICS]+
                [margin[i]['max_distance_upper_mm'],runs[i]['completion_time_s']] for i in ids]))
        actual[:,:,k]=pair[1]-pair[0]
    comparison=[]
    for m,name in enumerate(METRICS):
        end=16 if name in SHAPE_METRICS else len(ids)
        a,b=predicted[:end,m].ravel(),actual[:end,m].ravel()
        actual_signal=abs(b)>1e-6;both=actual_signal&(abs(a)>1e-6)
        comparison.append(dict(metric=name,mean_absolute_difference=float(np.mean(abs(a-b))),
            mean_abs_actual=float(np.mean(abs(b))),actual_signal_pairs=int(actual_signal.sum()),
            both_resolved_pairs=int(both.sum()),sign_resolution_threshold=1e-6,
            missed_actual_signal_fraction=float(np.mean(abs(a[actual_signal])<=1e-6)) if actual_signal.any() else None,
            matching_sign_fraction_when_both_resolved=float(np.mean(a[both]*b[both]>0)) if both.any() else None))
    np.savez_compressed(stage/'gradients.npz',path_id=np.asarray(ids),metric_names=np.asarray(METRICS),
        gradient=gradient,failed_axis_constraints=exclusions,query_jacobian=jac,
        measured_basis=reduced,predicted_function_derivative=predicted,actual_function_derivative=actual)
    write_json(stage/'results.json',dict(modes=modes,failed_axes=failed,fixed_state_jacobian_error=error,
        function_comparison=comparison,free_parameters=jac.shape[-1],goal_complete=False))
    receipt(stage,dict(status='completed_current_direct_gradient_audit',modes=modes,
        fixed_state_jacobian_error=error,new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-started))
    print('CURRENT_GRADIENT',modes,error,comparison,flush=True)


if __name__=='__main__':main()
