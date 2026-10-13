"""Prepare the full learnable residual subspace without additional motion simulation."""
import time
import numpy as np
import torch
from scipy.linalg import block_diag
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    original=BASE/'protected_residual_capacity_initialization_20260917'
    stage=register('direct_constraint_parameter_preparation_20260917',
        [__file__,model,parent/'training_selection.json',original/'basis.npz',
         BASE/'kernel_local_feedback_policy.py',BASE/'probe_coordinated_suffix.py',
         BASE/'constrained_parameter_solver.py'],
        dict(new_physics_ticks=0,actor_verification_calls=5000,saved_geometry_ticks=100000,wall_s=120),
        'Prepare direct optimization over ALL free kernel coefficients, instead of choosing '
        'a few fixed function directions. Preserve the original path16/31 action nullspace; '
        'retain the same frozen256-kernel policy and exact parent. Reconstruct every saved '
        'decision state on all64 parent training trajectories and verify all decoded actor '
        'actions exactly. Use these current states for a path-balanced action-change metric '
        'with ridge0.001. No physics simulation, teacher fit, hidden-feature change or actor '
        'update. Current dense physical credit is collected separately; no gradient is '
        'invented from this state reconstruction or from old-parent observations.')
    started=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,actor_verification_calls=0)
    p=KernelLocalFeedbackPolicy(256);p.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    with np.load(original/'basis.npz') as z:cx=z['protected_observations'].copy()
    observations=[];weights=[];ids=[]
    for path,trace in enumerate(selection['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        local=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,observations=local)
        local.append(env.observation())
        for obs,held in zip(local,actions):
            latent=p.act(obs,True)[0];costs['actor_verification_calls']+=1
            assert np.array_equal(env.decode_action(latent),held)
        observations.extend(local);weights.extend([1/(64*len(local))]*len(local));ids.extend([path]*len(local))
        assert time.perf_counter()-started<120 and costs['actor_verification_calls']<=5000 and costs['saved_prefix_geometry_ticks']<=100000
    x=np.asarray(observations);w=np.asarray(weights);f=feature_batches(p,x);cf=feature_batches(p,cx)
    spaces=[];metrics=[];ranks=[];errors=[]
    for h in range(3):
        _,sv,vt=np.linalg.svd(cf[:,h],full_matrices=True);rank=int((sv>1e-10).sum());z=vt[rank:].T
        spaces.append(z);ranks.append(rank);errors.append(float(abs(cf[:,h]@z).max()))
        reduced=f[:,h]@z;metrics.append(reduced.T@(w[:,None]*reduced)+.001*np.eye(z.shape[1]))
    assert max(errors)<1e-9
    np.savez_compressed(stage/'support.npz',observation=x,weight=w,path_id=np.asarray(ids),
        protected_observations=cx,metric=block_diag(*metrics),amplitude=AMPLITUDE,
        **{f'nullspace_{h}':z for h,z in enumerate(spaces)})
    result=dict(states=len(x),rank_by_head=ranks,free_parameters_by_head=[z.shape[1] for z in spaces],
        free_parameters=sum(z.shape[1] for z in spaces),algebraic_nullspace_errors=errors,
        decoded_actions_all_exact=True,costs=costs,parent_sha256=sha(model),goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='completed_direct_parameter_preparation',**result,wall_s=time.perf_counter()-started))
    print('DIRECT_PARAMETER_SPACE',result,flush=True)


if __name__=='__main__':main()
