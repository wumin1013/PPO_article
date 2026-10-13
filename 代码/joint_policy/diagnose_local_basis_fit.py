"""Offline matched-label fit diagnostic before any new closed-loop evaluations."""
import time
import numpy as np
import torch
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from learn_exit_local_credit import last_layer_features
from reflection_policy import reflect_observation
from search_complete_return import read
from study_io import BASE, register, receipt, write_json
from train_pilot import save_torch


@torch.no_grad()
def feature_batches(policy, observations):
    return np.concatenate([policy.residual_features(x).numpy().astype(float)
                           for x in torch.tensor(observations).split(128)])


def fit_and_measure(features, weights, target, original_count, positive_count):
    coefficients = []
    for h in range(3):
        f = features[:, h]
        coefficients.append(np.linalg.solve(f.T@(weights[:, None]*f)+.001*np.eye(f.shape[1]),
                                           f.T@(weights*target[:, h])))
    coefficients = np.asarray(coefficients)
    prediction = np.einsum('nhf,hf->nh', features, coefficients)
    magnitude = max(1., float(abs(prediction).max()))
    coefficients /= magnitude; prediction /= magnitude
    cuts = [slice(0,original_count), slice(original_count,original_count+positive_count),
            slice(original_count+positive_count,None)]
    metrics = {}
    for name, indices in zip(('original','positive','tail'), cuts):
        metrics[name] = dict(mse=float(np.mean((prediction[indices]-target[indices])**2)),
                             rms_residual=float(np.sqrt(np.mean(prediction[indices]**2))),
                             max_residual=float(abs(prediction[indices]).max()))
    metrics['weighted_mse'] = float(np.sum(weights[:,None]*(prediction-target)**2)/3)
    metrics['support_normalization'] = magnitude
    return coefficients, metrics


def main():
    torch.set_num_threads(1)
    parent = BASE/'exit_local_credit_parameter_learning_20260916'
    fit = BASE/'matched_entry_interaction_feedback_20260916'
    stage = register('local_basis_saved_data_fit_20260916',
        [__file__,BASE/'kernel_local_feedback_policy.py',BASE/'learn_exit_local_credit.py',
         parent/'candidate_1.pt',parent/'training_1/summary.json',fit/'full_training_data.npz',
         fit/'targets.json',fit/'receipt.json'],
        dict(new_decisions=0,new_physics_ticks=0,wall_s=90,coefficients=291,centers=97,
             ridge=.001,width_multipliers=[.5,1.,2.],steps_for_later_physics=[.5,1.]),
        'Offline diagnostic of parameter-update locality. Reuse exactly the4421 states, '
        'three-component labels, path/block weights and latent scales of the prior completed '
        'last-layer fit. The new residual uses97 Gaussian basis functions of the same frozen '
        '96-unit actor hidden states, with reflection averaging and291 coefficients. '
        'Centers include all positive hidden states and their reflected counterparts, then '
        'farthest-point support states up to97. Per-hidden-unit standard deviation (floor0.05) '
        'and the median8th-nearest support-state distance at positives set a data-based metric '
        'and nominal width. Three widths(.5,1,2) get the same ridge and maximum saved-support '
        'residual bound as the previous linear fit. Basis change also changes regularization '
        'geometry; this is not a proof of a unique cause. Advance only widths lowering positive '
        'target MSE and original/tail residual RMS relative to the recomputed linear fit. '
        'No new physics, path-specific runtime rules or development access in this diagnostic.')
    started=time.perf_counter()
    with np.load(fit/'full_training_data.npz') as z:
        x,y,w=[z[k].copy() for k in ('observation','target_residual','weight')]
    n=sum(r['decisions'] for r in read(parent/'training_1/summary.json')['results'])
    npos=read(fit/'targets.json')['full']['positive_states']
    policy=KernelLocalFeedbackPolicy()
    saved=torch.load(parent/'candidate_1.pt',map_location='cpu',weights_only=False)
    missing=policy.load_state_dict(saved['policy'],strict=False)
    assert not missing.unexpected_keys and set(missing.missing_keys)=={
        'kernel_centers','kernel_scale','kernel_width','residual_scale','residual_coefficients'}
    with torch.no_grad(): a,b=policy.hidden_pair(torch.tensor(x));a=a.numpy();b=b.numpy()
    support=np.concatenate([a,b]); scale=np.maximum(support.std(0),.05)
    positive=np.concatenate([a[n:n+npos],b[n:n+npos]])
    centers=list(np.unique(positive,axis=0))
    assert len(centers)<=97
    normalized=support/scale
    nearest=np.full(len(support),np.inf)
    for center in centers:
        nearest=np.minimum(nearest,np.mean((normalized-center/scale)**2,axis=1))
    while len(centers)<97:
        index=int(np.argmax(nearest));center=support[index];centers.append(center.copy())
        nearest=np.minimum(nearest,np.mean((normalized-center/scale)**2,axis=1))
    distances=np.stack([np.mean((normalized-p/scale)**2,axis=1) for p in positive])
    width=float(np.median(np.sqrt(np.partition(distances,8,axis=1)[:,8])))
    assert width>0
    with torch.no_grad():
        policy.kernel_centers.copy_(torch.tensor(np.asarray(centers)))
        policy.kernel_scale.copy_(torch.tensor(scale))
    _,linear=fit_and_measure(last_layer_features(policy,x),w,y,n,npos)
    records=[]
    for multiplier in (.5,1.,2.):
        policy.kernel_width.fill_(width*multiplier)
        features=feature_batches(policy,x)
        coeff,metrics=fit_and_measure(features,w,y,n,npos)
        with torch.no_grad():policy.residual_coefficients.copy_(torch.tensor(coeff,dtype=torch.float32))
        test=torch.tensor(x[::41])
        with torch.no_grad():
            symmetry=float(abs(policy.distribution(test).mean*torch.tensor([-1.,1.,1.])-
                               policy.distribution(reflect_observation(test)).mean).max())
        assert symmetry<1e-6
        qualifies=(metrics['positive']['mse']<linear['positive']['mse']
            and metrics['original']['rms_residual']<linear['original']['rms_residual']
            and metrics['tail']['rms_residual']<linear['tail']['rms_residual'])
        model=f'kernel_width_{multiplier:g}.pt'
        save_torch(stage/model,dict(policy=policy.state_dict(),architecture='kernel_local_feedback',
            centers=97,nominal_width=width,width_multiplier=multiplier,step=1.,
            learning_method='counterfactual_targets_with_local_hidden_state_basis'))
        records.append(dict(width_multiplier=multiplier,model=model,metrics=metrics,
                            reflection_error=symmetry,offline_eligible=qualifies))
        print('LOCAL_BASIS',multiplier,metrics,'eligible',qualifies,flush=True)
    write_json(stage/'results.json',dict(linear=linear,kernels=records,nominal_width=width,
        original_states=n,positive_states=npos,tail_states=len(x)-n-npos,goal_complete=False))
    receipt(stage,dict(status='completed_saved_data_local_basis_fit',new_decisions=0,
        new_physics_ticks=0,eligible_widths=[r['width_multiplier'] for r in records if r['offline_eligible']],
        wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
