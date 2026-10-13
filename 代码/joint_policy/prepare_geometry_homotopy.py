"""Exact initialization and fixed-state Jacobian for a representation repair."""
import time
import numpy as np
import torch
from geometry_homotopy_policy import GeometryHomotopyPolicy
from continuous_tangent_policy import interval_geometry
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from probe_one_decision_credit import AMPLITUDE
from train_pilot import save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def mix_jacobian(policy,observations):
    output=[]
    for x in torch.tensor(observations).split(128):
        delta=interval_geometry(x)-x
        _,jac=torch.autograd.functional.jvp(lambda a:policy.distribution(x+a*delta).mean,
            torch.tensor(0.),torch.tensor(1.))
        output.append(jac.detach().numpy())
    return np.concatenate(output)


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    query=BASE/'current_dense_constraint_credit_collection_20260917'
    selected=read(parent/'training_selection.json');model=parent/selected['model']
    stage=register('geometry_homotopy_exact_preparation_20260917',
        [__file__,BASE/'geometry_homotopy_policy.py',BASE/'continuous_tangent_policy.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'diagnose_local_basis_fit.py',
         model,prepared/'support.npz',query/'queries.npz',
         BASE/'retained_incoming_geometry_continuity_audit_20260917/receipt.json'],
        dict(new_physics_ticks=0,forward_validation_states=4167,wall_s=90,
             new_representation_parameters=1,initial_geometry_mix=0.),
        'Prepare a mechanism-directed representation repair without repeating failed '
        'teacher distillation. Add one GLOBAL continuous blend between point tangents '
        'and interval geometry. At mix0 every actor mean and decoded action must be '
        'exactly the current credit parent on all4167 saved training states; no rollout '
        'or candidate promotion. Compute fixed-state autograd Jacobians on support, '
        'query and protected states so current single-action consequences can later '
        'estimate a coupled representation/actor update. Audit the minimum-ridge '
        'coefficient correction needed to preserve the two protected paths in the '
        'linear model. Exact action preservation is an existing optimizer device, '
        'not a physical requirement; do not confuse its feasibility with goal '
        'reachability. A future learned nonzero blend must pass whole-policy tests. '
        'No trajectory templates, per-corner rules or changes to the executor.')
    started=time.perf_counter()
    with np.load(prepared/'support.npz') as z:
        x=z['observation'].copy();w=z['weight'].copy();protected=z['protected_observations'].copy()
        spaces=[z[f'nullspace_{h}'].copy() for h in range(3)]
    with np.load(query/'queries.npz') as z:qx=z['observation'].copy()
    saved=torch.load(model,map_location='cpu',weights_only=False)['policy']
    parent_policy=KernelLocalFeedbackPolicy(256);parent_policy.load_state_dict(saved)
    policy=GeometryHomotopyPolicy(256);missing=policy.load_state_dict(saved,strict=False)
    assert missing.missing_keys==['geometry_mix'] and not missing.unexpected_keys
    with torch.no_grad():
        before=[];after=[]
        for bx in torch.tensor(x).split(128):
            before.append(parent_policy.distribution(bx).mean.numpy());after.append(policy.distribution(bx).mean.numpy())
        before=np.concatenate(before);after=np.concatenate(after)
    assert np.array_equal(before,after)
    sx=mix_jacobian(parent_policy,x);qq=mix_jacobian(parent_policy,qx);px=mix_jacobian(parent_policy,protected)
    f=feature_batches(parent_policy,x);cf=feature_batches(parent_policy,protected)
    compensation=[];audits=[]
    for h,z in enumerate(spaces):
        matrix=f[:,h];c=cf[:,h];g=sx[:,h]/AMPLITUDE[h];target=-px[:,h]/AMPLITUDE[h]
        u,sv,vt=np.linalg.svd(c,full_matrices=False);keep=sv>1e-10
        part=vt[keep].T@((u[:,keep].T@target)/sv[keep])
        metric=matrix.T@(w[:,None]*matrix)+.001*np.eye(matrix.shape[1])
        linear=matrix.T@(w*g)
        reduced=np.linalg.solve(z.T@metric@z,-z.T@(metric@part+linear))
        coefficient=part+z@reduced;compensation.append(coefficient)
        actual=g+matrix@coefficient
        audits.append(dict(head=h,protected_normalized_error=float(abs(c@coefficient-target).max()),
            coefficient_max=float(abs(coefficient).max()),
            largest_normalized_support_change_per_unit_mix=float(abs(actual).max()),
            normalized_support_rms=float(np.sqrt(np.sum(w*actual**2))),
            ridge_coefficient_norm=float(np.sqrt(.001*(coefficient@coefficient)))))
    # Check a small representational step at fixed states, without moving the machine.
    eps=1e-4
    with torch.no_grad():
        policy.geometry_mix.fill_(eps)
        actual=np.concatenate([policy.distribution(bx).mean.numpy() for bx in torch.tensor(x).split(128)])-before
        policy.geometry_mix.zero_()
    check=dict(step=eps,maximum_absolute_latent_linearization_error=float(abs(actual-eps*sx).max()),
        maximum_actual_latent_change=float(abs(actual).max()))
    np.savez_compressed(stage/'jacobians.npz',support_mix_jacobian=sx,query_mix_jacobian=qq,
        protected_mix_jacobian=px,protected_linear_compensation=np.asarray(compensation))
    save_torch(stage/'initial.pt',dict(policy=policy.state_dict(),architecture='geometry_homotopy_kernel',
        centers=256,parent_sha256=sha(model),geometry_mix=0.,initialization_exact=True))
    result=dict(exact_parent_mean_identity=True,states=len(x),query_states=len(qx),
        fixed_state_linearization=check,protected_compensation=audits,
        goal_complete=False,training_performed=False,physical_rollouts_performed=False)
    write_json(stage/'results.json',result)
    assert time.perf_counter()-started<90
    receipt(stage,dict(status='completed_exact_geometry_homotopy_preparation',new_physics_ticks=0,
        wall_s=time.perf_counter()-started,**result))
    print('GEOMETRY_HOMOTOPY_PREPARATION',result,flush=True)


if __name__=='__main__':main()
