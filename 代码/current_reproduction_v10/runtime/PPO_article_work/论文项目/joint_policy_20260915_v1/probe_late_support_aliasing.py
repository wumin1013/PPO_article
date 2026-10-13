"""Matched full-loop radius controls; diagnostic only, not an adopted policy."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import copy
import time
import numpy as np
import torch
from scipy.optimize import linprog
from gate_band_state import GateBandRuntime
from saved_memory_action_identity import SavedMemoryActionIdentity
from observable_memory_joint_policy import ObservableMemoryJointPolicy
from learn_shared_temporal_joint_policy import evaluate_complete_parameters
from audit_early_geometry_outcomes import constraint_names
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+85.
    source=BASE/'canonical_gate_band_preturn_supported_joint_recovery_20260920'
    parent=BASE/'canonical_gate_band_reversal_window_local_recovery_20260920'
    model=read(source/'joint_74_local_credit.json')
    history=read(source/'state_joint/results.json')['candidates']
    origin=next(c for c in history if c['tag']=='joint_74');zero=history[0]
    names=constraint_names([r['path'] for r in zero['records']])+['lateral_gain']
    jac=np.array(model['jacobian']);gradient=np.array(model['loss_gradient'])
    values=np.r_[origin['augmented_learning_inequalities'],.25*(zero['lateral_mm']-origin['lateral_mm'])]
    n=jac.shape[1]
    solved=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack([np.c_[-jac,-np.ones(len(values))],np.r_[gradient,0.]]),
        b_ub=np.r_[values,.25*(zero['score']-origin['score'])],bounds=[(-64.,64.)]*n+[(0.,None)],method='highs')
    assert solved.success
    conflict=dict(minimum_normalized_slack=float(solved.fun),
        active=[dict(name=names[i],predicted_margin=float(values[i]+jac[i]@solved.x[:-1]),
                     dual=float(solved.ineqlin.marginals[i])) for i in range(len(values))
                if abs(solved.ineqlin.marginals[i])>1e-5],
        violated_flat_responses=[dict(name=names[i],margin=float(values[i]),gradient_max=float(max(abs(jac[i]))))
            for i in range(len(values)) if values[i]<-1e-8 and max(abs(jac[i]))<1e-8],
        interpretation='Local sampled-model conflict only; does not establish physical infeasibility.')
    write_json(source/'local_constraint_conflict_audit.json',conflict)
    stage=register('canonical_gate_band_late_support_aliasing_probe_20260920',
        [__file__,source/'receipt.json',source/'initial_policy.pt',source/'state_joint/results.json',
         source/'joint_74_local_credit.json',source/'local_constraint_conflict_audit.json',
         parent/'active_window_affected_replay.npz',BASE/'saved_memory_action_identity.py',
         BASE/'joint_teacher_state_policy.py',BASE/'observable_memory_joint_policy.py'],
        dict(wall_s=90,matched_radius_fractions=[.97,.95],new_complete_coefficient_points=2,
             unchanged_coefficient_anchor='joint_74',no_adoption_from_probe=True),
        'The actual joint_74 recovery has a local peak-curvature versus heading-overshoot-duration '
        'conflict, and307 exit/reverse regressions with zero response to the sampled small steps. '
        'Test the latter separately: keep ALL45 learned coefficients fixed and reduce only '
        'the compact300ms-training-state kernel radius by3 or5 percent. Radius is a shared '
        'observable-state function parameter, not a time/path switch. Prove any unchanged '
        'complete trajectory by exact decoded actions on saved parent states; otherwise run '
        'the full physical route. Do not ignore tiny features, adopt a probe, loosen guards, '
        'or claim one cause explains both failures. Use the measured result to design joint '
        'learning with independent postpeak action support.')
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    with np.load(parent/'active_window_affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    selected=zero['records'];paths=[r['path'] for r in selected];rows=[]
    write_json(stage/'baseline.json',data)
    try:
        for fraction in (.97,.95):
            modified=copy.deepcopy(md);modified['radii'][2]*=fraction
            proof=SavedMemoryActionIdentity(modified,replay,GateBandRuntime.decode_action)
            unchanged=[p for p in paths if proof.unchanged(p,origin['coefficients'])]
            actual=[r for r in selected if r['path'] not in unchanged]
            policy=ObservableMemoryJointPolicy(modified)
            tag=f'radius_{fraction}'
            measured=evaluate_complete_parameters(policy,origin['coefficients'],actual,stage,tag,deadline)
            by={r['path']:r for r in measured}
            records=[by.get(r['path'],copy.deepcopy(r)) for r in selected]
            modified['coefficients']=np.array(origin['coefficients']).reshape(-1,3)
            torch.save(modified,stage/(tag+'.pt'))
            record=dict(tag=tag,radius_fraction=fraction,records=records,
                exact_decoded_identity_paths=unchanged,actual_rerun_paths=[r['path'] for r in actual])
            rows.append(record);write_json(stage/'results.json',dict(controls=rows,adopted=False))
            print('LATE_RADIUS_CONTROL',fraction,unchanged,[(r['path'],r['shape']['out_edge_outward_mm'],
                r['shape']['full_spatial_peak_curvature_mm_inv']) for r in records],flush=True)
        receipt(stage,dict(status='completed_late_support_aliasing_diagnostic',new_controls=len(rows),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False,
            protected_parent_sha256=sha(parent/'research_candidate.pt')))
    except Exception as exc:
        receipt(stage,dict(status='failed_late_support_aliasing_diagnostic',error=repr(exc),
            completed_controls=len(rows),adopted=False,goal_complete=False));raise


if __name__=='__main__':main()
