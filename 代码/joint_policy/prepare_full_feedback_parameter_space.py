"""Prepare paired full-coefficient and continuous-geometry parameter spaces."""
import time
import numpy as np
import torch
from scipy.linalg import block_diag
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from geometry_homotopy_policy import GeometryHomotopyPolicy
from diagnose_local_basis_fit import feature_batches
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    geometry=BASE/'geometry_homotopy_exact_preparation_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    stage=register('full_feedback_parameter_space_preparation_20260917',
        [__file__,model,prepared/'support.npz',prepared/'receipt.json',
         geometry/'jacobians.npz',geometry/'receipt.json',BASE/'geometry_homotopy_policy.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'diagnose_local_basis_fit.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=90,
             coefficient_parameters=768,optional_geometry_parameter=1),
        'Prepare TWO paired mathematical parameter spaces for the next mechanism '
        'comparison: all768 existing kernel coefficients, and those same coefficients '
        'plus one continuous-geometry blend. Both initialize to the exact current '
        'parent, use the same4167 saved states and action/ridge metric. The old '
        '366-dimensional nullspace is embedded isometrically in the full metric; '
        'verify this rather than interpreting a coordinate rescale as added freedom. '
        'Release only the optimizer device of exactly freezing paths16/31, NOT '
        'their physical constraints or success requirement. Their CURRENT all-time '
        'physical credit is being collected separately before any proposal. No new '
        'actor hidden weights, runtime rules, trajectory templates, model promotion '
        'or physical rollout. The optional blend remains0 until separately validated.')
    started=time.perf_counter()
    with np.load(prepared/'support.npz') as z:
        x=z['observation'].copy();w=z['weight'].copy();ids=z['path_id'].copy()
        protected=z['protected_observations'].copy();old_metric=z['metric'].copy()
        spaces=[z[f'nullspace_{h}'].copy() for h in range(3)]
    with np.load(geometry/'jacobians.npz') as z:mix=z['support_mix_jacobian'].copy()/AMPLITUDE
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    features=feature_batches(p,x)
    metric=block_diag(*[features[:,h].T@(w[:,None]*features[:,h])+.001*np.eye(256) for h in range(3)])
    cross=np.concatenate([features[:,h].T@(w*mix[:,h]) for h in range(3)])
    extended=np.block([[metric,cross[:,None]],[cross[None,:],np.array([[np.sum(w[:,None]*mix**2)+.001]])]])
    embedding=block_diag(*spaces)
    identity_error=float(abs(embedding.T@metric@embedding-old_metric).max())
    assert identity_error<1e-10
    np.linalg.cholesky(metric);np.linalg.cholesky(extended)
    rng=np.random.default_rng(917092)
    delta=rng.normal(size=(3,256))
    change=np.einsum('nhf,hf->nh',features,delta)
    delta*=.05/max(abs(change).max(),1e-12)
    test=GeometryHomotopyPolicy(256)
    missing=test.load_state_dict(p.state_dict(),strict=False)
    assert missing.missing_keys==['geometry_mix']
    with torch.no_grad():
        before=np.concatenate([p.distribution(a).mean.numpy() for a in torch.tensor(x).split(128)])
        initial=np.concatenate([test.distribution(a).mean.numpy() for a in torch.tensor(x).split(128)])
        assert np.array_equal(before,initial)
        test.residual_coefficients.add_(torch.tensor(delta,dtype=torch.float32))
        after=np.concatenate([test.distribution(a).mean.numpy() for a in torch.tensor(x).split(128)])
    predicted=np.einsum('nhf,hf->nh',features,delta)*AMPLITUDE
    coefficient_error=float(abs(after-before-predicted).max());assert coefficient_error<1e-6
    np.savez_compressed(stage/'support.npz',observation=x,weight=w,path_id=ids,
        protected_observations=protected,features=features,normalized_mix_jacobian=mix,
        metric=metric,extended_metric=extended,old_embedding=embedding,amplitude=AMPLITUDE)
    result=dict(states=len(x),full_parameters=768,extended_parameters=769,
        old_metric_embedding_error=identity_error,fixed_state_coefficient_jacobian_error=coefficient_error,
        exact_parent_initialization=True,parent_sha256=sha(model),goal_complete=False,
        requires_current_credit_on_previously_protected_paths=True)
    write_json(stage/'results.json',result)
    assert time.perf_counter()-started<90
    receipt(stage,dict(status='completed_full_feedback_parameter_space_preparation',
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-started,**result))
    print('FULL_FEEDBACK_SPACE',result,flush=True)


if __name__=='__main__':main()
