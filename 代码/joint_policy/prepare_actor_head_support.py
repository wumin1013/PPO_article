"""Extract current hidden features for a matched output-layer capacity test."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918'
    geometry=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    data=read(parent/'results.json');model=parent/data['chosen']['model'];pool=data['full_candidate']
    stage=register('canonical_gate_band_current_actor_head_support_20260918',
        [__file__,model,parent/'results.json',parent/'receipt.json',geometry/'geometry.npz',
         geometry/'receipt.json',BASE/'canonical_gate_state.py',BASE/'frame_feedback_policy.py',
         BASE/'kernel_local_feedback_policy.py',BASE/'probe_coordinated_suffix.py'],
        dict(new_physics_ticks=0,replayed_actor_calls=6000,saved_prefix_geometry_ticks=100000,
             wall_s=120,added_steering_output_weights=96,actor_hidden_weights_changed=False),
        'The existing256-coefficient shared steering space cannot meet all saved '
        'command intervals under its half-degree support bounds. Extract the missing '
        'CURRENT96 hidden output features from the same64 saved paths; original '
        'geometry did not save these features or observations. Verify old actor '
        'actions and kernel features against the completed geometry, and preserve '
        'those arrays exactly. Extend only the steering affine function with the '
        'existing96 base-actor output weights, reflection-averaged and scaled by '
        'the same0.025 latent unit as the kernel coefficients. Hidden weights, '
        'kernel centers, physical rules and feed/preview outputs stay unchanged. '
        'Check affine prediction against the actual current policy after a small '
        'output-weight perturbation. This is a representability diagnostic, not '
        'adoption, new input information or a claim about joint-control impossibility.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,replayed_actor_calls=0)
    md=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(md['policy'])
    factory=checkpoint_environment_factory(md['architecture'])
    with np.load(geometry/'geometry.npz') as z:
        old_support=z['support'].reshape(-1,3,768)[:,0,:256].copy()
        old_latent=z['latent'].copy();old_paths=z['state_path'].copy()
        weights=z['state_weight'].copy();metric=z['metric_physical'][:256,:256].copy()
    observations=[];path_ids=[]
    for result,trace in zip(pool['results'],pool['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
            state_snapshots=states,env_factory=factory);states.append(env.state())
        for k,state in enumerate(states):
            env.restore(state);o=env.observation();a=p.act(o,True)[0];costs['replayed_actor_calls']+=1
            index=len(observations)
            assert np.array_equal(a,old_latent[index]) and np.array_equal(env.decode_action(a),actions[k])
            observations.append(o);path_ids.append(result['original_path'])
        assert time.perf_counter()-start<115
    observations=np.asarray(observations);assert np.array_equal(path_ids,old_paths)
    hidden=[];kernel=[]
    with torch.no_grad():
        for obs in torch.tensor(observations).split(128):
            a,b=p.hidden_pair(obs)
            hidden.append((.5*(a-b)*p.residual_scale[0]).numpy().astype(float))
            kernel.append((p.residual_features(obs)[:,0]*p.residual_scale[0]).numpy().astype(float))
    added=np.concatenate(hidden);replayed_kernel=np.concatenate(kernel)
    assert np.array_equal(replayed_kernel,old_support)
    support=np.column_stack([old_support,added])
    derivative=90.*(1.-np.tanh(old_latent[:,0])**2)
    physical=support*derivative[:,None]
    extended_metric=physical.T@(weights[:,None]*physical)/3.+.001*np.eye(352)
    embedding_error=float(abs(extended_metric[:256,:256]-metric).max());assert embedding_error<1e-10
    x=torch.tensor(observations[::31]);rng=np.random.default_rng(9180611);delta=rng.normal(size=96)*.001
    with torch.no_grad():
        before=p.distribution(x).mean.numpy().astype(float)
        p.actor[-1].weight[0].add_(torch.tensor(delta*p.residual_scale[0].item(),dtype=torch.float32))
        after=p.distribution(x).mean.numpy().astype(float)
    prediction=added[::31]@delta
    affine_error=float(abs(after[:,0]-before[:,0]-prediction).max())
    assert affine_error<1e-6 and np.array_equal(before[:,1:],after[:,1:])
    np.savez_compressed(stage/'support.npz',observation=observations,extra_steering_features=added,
        extended_metric=extended_metric,state_path=old_paths,extra_weight_scale=float(md['policy']['residual_scale'][0]))
    result=dict(parent_sha256=sha(model),states=len(observations),old_kernel_exact=True,
        metric_embedding_error=embedding_error,actor_output_affine_error=affine_error,
        extra_parameters=96,hidden_parameters_changed=False,costs=costs,goal_complete=False)
    write_json(stage/'results.json',result)
    assert costs['replayed_actor_calls']<=6000 and costs['saved_prefix_geometry_ticks']<=100000
    receipt(stage,dict(status='prepared_current_actor_head_capacity_support',
        wall_s=time.perf_counter()-start,**result))
    print('ACTOR_HEAD_SUPPORT',result,flush=True)


if __name__=='__main__':main()
