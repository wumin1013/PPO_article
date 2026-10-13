"""Reuse saved observations to test a direct-state representation control."""
import time
import numpy as np
import torch
from raw_state_skip_policy import RawStateSkipPolicy,raw_state_features,RAW_STEERING_INDICES
from canonical_gate_state import CanonicalGatePolicy
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    observed=BASE/'canonical_gate_band_current_actor_head_support_20260918'
    geometry=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    d=read(parent/'results.json');model=parent/d['chosen']['model']
    stage=register('canonical_gate_band_current_raw_state_support_20260918',
        [__file__,BASE/'raw_state_skip_policy.py',model,observed/'support.npz',observed/'receipt.json',
         geometry/'geometry.npz',geometry/'receipt.json',
         BASE/'canonical_gate_band_actor_head_command_capacity_20260918/results.json',
         BASE/'canonical_gate_band_shared_command_closed_loop_probe_20260918/results.json'],
        dict(new_physics_ticks=0,new_state_reconstruction_ticks=0,replayed_actor_calls=800,
             wall_s=45,added_steering_parameters=len(RAW_STEERING_INDICES)),
        'The bounded hidden-output fit is algebraically feasible but uses coefficient '
        'norms139.6/80.3 and fails actual closed-loop shape. Compare a direct linear '
        'skip from the SAME72 canonical observation values to the steering output, '
        'rather than adding more hidden output weights. Retain the exact history '
        'and endpoint masks, reflection symmetry, kernel features,0.025 latent unit '
        'and statewise half-degree bounds. Only the function basis changes; input '
        'information and runtime are unchanged. The parameter count and ridge '
        'geometry also differ, so norm values alone are not a unique encoding-cause '
        'proof. Feasibility and physical action changes are the matched comparisons. '
        'Reuse all current observations; no repeated motion or geometry replay. '
        'Check zero skip exactly matches the parent and coefficient linearity '
        'against the actual float32 policy. Future full-feedback probes remain '
        'mandatory; no runtime clamp or ideal trajectory is introduced.')
    start=time.perf_counter()
    with np.load(observed/'support.npz') as z:obs=z['observation'].copy();path=z['state_path'].copy()
    with np.load(geometry/'geometry.npz') as z:
        kernel=z['support'].reshape(-1,3,768)[:,0,:256].copy()
        metric=z['metric_physical'][:256,:256].copy();latent=z['latent'].copy();weights=z['state_weight'].copy()
        assert np.array_equal(path,z['state_path'])
    md=torch.load(model,map_location='cpu',weights_only=False)
    p=RawStateSkipPolicy(256);missing=p.load_state_dict(md['policy'],strict=False)
    assert missing.missing_keys==['raw_skip_coefficients'] and not missing.unexpected_keys
    old=CanonicalGatePolicy(256);old.load_state_dict(md['policy'])
    x=torch.tensor(obs)
    with torch.no_grad():
        features=(raw_state_features(x)[:,0,list(RAW_STEERING_INDICES)]*p.residual_scale[0]).numpy().astype(float)
        sample=x[::17];before=p.distribution(sample).mean.numpy()
        assert np.array_equal(before,old.distribution(sample).mean.numpy())
        delta=np.random.default_rng(9180622).normal(size=len(RAW_STEERING_INDICES))*.001
        p.raw_skip_coefficients[0,list(RAW_STEERING_INDICES)]=torch.tensor(delta,dtype=torch.float32)
        after=p.distribution(sample).mean.numpy()
    error=float(abs(after[:,0]-before[:,0]-features[::17]@delta).max())
    assert error<1e-6 and np.array_equal(before[:,1:],after[:,1:])
    support=np.column_stack([kernel,features]);derivative=90*(1.-np.tanh(latent[:,0])**2)
    physical=support*derivative[:,None]
    extended=physical.T@(weights[:,None]*physical)/3.+.001*np.eye(support.shape[1])
    assert np.max(abs(extended[:256,:256]-metric))<1e-10
    np.savez_compressed(stage/'support.npz',extra_steering_features=features,extended_metric=extended,
        state_path=path,extra_weight_scale=float(md['policy']['residual_scale'][0]),
        raw_steering_indices=RAW_STEERING_INDICES)
    result=dict(parent_sha256=sha(model),states=len(obs),extra_parameters=len(RAW_STEERING_INDICES),
        extra_feature_kind='raw_state_skip',exact_parent_initialization=True,
        existing_information_only=True,actor_output_affine_error=error,goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='prepared_current_raw_state_capacity_support',
        new_physics_ticks=0,new_state_reconstruction_ticks=0,wall_s=time.perf_counter()-start,**result))
    print('RAW_STATE_SUPPORT',result,flush=True)


if __name__=='__main__':main()
