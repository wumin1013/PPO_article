"""Matched reference-oriented support controls, followed by unchanged raw guards."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import copy
import time
import numpy as np
import torch
from gate_band_state import GateBandRuntime
from saved_memory_action_identity import SavedMemoryActionIdentity
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from learn_shared_temporal_joint_policy import evaluate_complete_parameters
from learn_early_memory_joint_policy import metric_functions
from duration_joint_metrics import guards
from learn_intrinsic_joint_recovery import intrinsic_guards
from study_io import BASE,register,write_json,receipt
from search_complete_return import read


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+85.
    source=BASE/'canonical_gate_band_postpeak_bounded_joint_recovery_20260920'
    parent=BASE/'canonical_gate_band_reversal_window_local_recovery_20260920'
    md=torch.load(source/'reference_supported_initial_policy.pt',map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');history=read(source/'state_joint/results.json')['candidates']
    selected=history[0]['records'];paths=[r['path'] for r in selected]
    with np.load(parent/'active_window_affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    proof=SavedMemoryActionIdentity(md,replay,GateBandRuntime.decode_action)
    stage=register('canonical_gate_band_reference_support_closed_loop_probe_20260920',
        [__file__,source/'receipt.json',source/'reference_supported_initial_policy.pt',
         source/'reference_feature_identity.json',source/'reference_oriented_support_audit.json',
         source/'measured_kernel_support_audit.json',source/'state_joint/results.json',
         BASE/'joint_teacher_state_policy.py',BASE/'observable_memory_joint_policy.py',BASE/'saved_memory_action_identity.py'],
        dict(wall_s=90,matched_parameter_points=['bounded_50','bounded_51'],new_full_points=2,
             preserved_original_kernel_count=5,decision_ms=20,execution_ms=1),
        'Actual half/full earlier actions reduce the340ms kernel activation0.5 to0.200/0.0108. '
        'Removing motion channels alone recovers only0.342/0.127 because geometry vectors '
        'also rotate with heading. Reference-oriented OBSERVED geometry gives0.484/0.443 '
        'on those same saved states while all five other kernels remain bitwise equal and '
        'reflection equivariance is exact. Compare the same two learned parameter vectors '
        'in complete closed loops, changing only this support representation. Reuse exact '
        'decoded-action-identical routes, never a small-feature approximation. Original '
        'physics, full coverage, stop, shape, speed and cruise checks unchanged. No '
        'adoption from a diagnostic and no claim about all failures or PPO-estimator bias.')
    write_json(stage/'baseline.json',data)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    results=[]
    try:
        for tag in ('bounded_50','bounded_51'):
            original=next(c for c in history if c['tag']==tag);u=np.array(original['coefficients'])
            unchanged=[p for p in paths if proof.unchanged(p,u)]
            actual=[r for r in selected if r['path'] not in unchanged]
            policy=ObservableMemoryJointPolicy(copy.deepcopy(md))
            measured=evaluate_complete_parameters(policy,u,actual,stage,tag,deadline)
            by={r['path']:r for r in measured};records=[by.get(r['path'],copy.deepcopy(r)) for r in selected]
            original_checks=np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
            checks=np.r_[original_checks,intrinsic_guards(records,selected)]
            result=dict(tag=tag,coefficients=u.tolist(),records=records,
                measured_inequalities=original_checks.tolist(),augmented_measured_inequalities=checks.tolist(),
                parent_inequality_min=min(original['augmented_measured_inequalities']),
                current_inequality_min=float(checks.min()),all_original_guards_pass=bool(checks.min()>=-1e-8),
                exact_decoded_identity_paths=unchanged,actual_rerun_paths=[r['path'] for r in actual])
            results.append(result)
            metadata=copy.deepcopy(md);metadata['coefficients']=u.reshape(-1,3)
            torch.save(metadata,stage/(tag+'.pt'))
            write_json(stage/'results.json',dict(controls=results,adopted=False))
            print('REFERENCE_SUPPORT_CONTROL',tag,result['parent_inequality_min'],result['current_inequality_min'],
                [(r['path'],r['shape']['out_edge_outward_mm'],r['shape']['full_spatial_peak_curvature_mm_inv']) for r in records],flush=True)
        receipt(stage,dict(status='completed_reference_support_closed_loop_diagnostic',new_controls=2,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        receipt(stage,dict(status='failed_reference_support_closed_loop_diagnostic',error=repr(exc),
            completed=len(results),adopted=False,goal_complete=False));raise


if __name__=='__main__':main()
