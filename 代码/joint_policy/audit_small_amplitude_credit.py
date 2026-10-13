"""Matched amplitude check of complete temporal-credit composition."""
import time
import numpy as np
import torch
from audit_current_constraint_gradients import assemble, METRICS
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'small_amplitude_constraint_credit_20260917'
    assert (source/'receipt.json').exists()
    previous=BASE/'current_direct_constraint_gradient_audit_20260917'
    prepared=BASE/'direct_constraint_parameter_preparation_20260917'
    proposal=BASE/'direct_full_subspace_constraint_solve_20260917'
    probe=BASE/'direct_constraint_composition_probe_20260917'
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    stage=register('small_amplitude_credit_composition_audit_20260917',
        [__file__,BASE/'audit_current_constraint_gradients.py',source/'receipt.json',
         source/'results.json',source/'queries.npz',previous/'gradients.npz',
         prepared/'support.npz',proposal/'direction.npz',probe/'comparison.json',model],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=90),
        'Compare all-time credit at the same current states and parent with single-action '
        'amplitude divided by8 on paths3,7,13. Project both gradients onto the SAME already '
        'measured full-policy direction; compare signs and magnitude to actual +/-0.125 '
        'closed-loop finite response. Save replacement local gradients and the matched '
        'amplitude evidence, without claiming exact differentiability, safety or model '
        'improvement. Failures remain exclusions and cannot supply complete-prefix credit.')
    started=time.perf_counter()
    with np.load(source/'queries.npz') as z:observations=z['observation'].copy()
    with np.load(prepared/'support.npz') as z:spaces=[z[f'nullspace_{h}'].copy() for h in range(3)]
    policy=KernelLocalFeedbackPolicy(256)
    policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    ids,gradient,jac,failed,failures,modes=assemble(read(source/'results.json'),observations,policy,spaces)
    with np.load(proposal/'direction.npz') as z:direction=z['reduced_direction'].copy()
    with np.load(previous/'gradients.npz') as z:
        old=dict(zip(z['path_id'].tolist(),z['gradient'].copy()))
    measured={r['path']:r for r in read(probe/'comparison.json')['paths']}
    result=[]
    for path,g in zip(ids,gradient):
        before=old[path]@direction;after=g@direction
        actual=np.asarray(measured[path]['actual_central_per_unit'])
        result.append(dict(path=path,coarse_credit=before.tolist(),small_credit=after.tolist(),
            actual_complete_direction=actual.tolist(),
            small_absolute_error=abs(after-actual).tolist(),coarse_absolute_error=abs(before-actual).tolist()))
        print('SMALL_CREDIT',path,dict(zip(METRICS,zip(before.tolist(),after.tolist(),actual.tolist()))),flush=True)
    np.savez_compressed(stage/'gradients.npz',path_id=ids,gradient=gradient,
        metric_names=np.asarray(METRICS),query_jacobian=jac,failed_axis_constraints=failed)
    write_json(stage/'results.json',dict(paths=result,modes=modes,failed_axes=failures,
        column_definition=['coarse_credit','small_credit','actual_full_policy'],goal_complete=False))
    receipt(stage,dict(status='completed_matched_amplitude_credit_audit',modes=modes,
        new_physics_ticks=0,new_actor_decisions=0,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
