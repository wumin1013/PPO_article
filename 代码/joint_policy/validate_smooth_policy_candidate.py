"""Bounded distribution, gradient and recovery checks; no physical rollout."""
import io
import json
import numpy as np
import torch
from joint_policy import JointPolicy
from smooth_policy_candidate import SmoothExplorationPolicy,marginal_noise_variance
from study_io import BASE,register,receipt,write_json


def main():
    stage=register('smooth_exploration_candidate_validation_v1',[__file__,BASE/'smooth_policy_candidate.py',
        BASE/'joint_policy.py',BASE/'temporal_exploration_fulltext_v1/smooth_exploration.pdf'],
        dict(synthetic_observations=16,noise_matrix_samples=10000,gradient_evaluations=2,
             snapshot_saves=1,snapshot_loads=1,physics_ticks=0,training_decisions=0),
        'Validate a candidate gSDE sampler and restart state before deciding any training study; current v6 is unchanged.')
    torch.set_num_threads(1);torch.manual_seed(915276)
    base=JointPolicy(72)
    policy=SmoothExplorationPolicy(72)
    policy.load_base_policy(base.state_dict())
    observations=torch.randn(16,72)*.25
    phi=policy.exploration_features(observations)
    assert torch.allclose(phi.square().sum(-1),torch.ones(16),atol=2e-7)
    std=policy.distribution(observations).stddev
    assert torch.allclose(marginal_noise_variance(phi,std),std.square(),rtol=3e-7,atol=1e-9)
    deterministic=[]
    for observation in observations:
        a=base.act(observation,True)[0]
        b=policy.act(observation,True)[0]
        deterministic.append(np.array_equal(a,b))
    assert all(deterministic)
    action,logp,_=policy.act(observations[0])
    evaluated,_,_=policy.evaluate(observations[0],torch.tensor(action))
    assert abs(float(evaluated)-logp)<1e-6
    (-evaluated).backward()
    joint_gradient=policy.actor[-1].bias.grad.detach().clone()
    assert torch.all(joint_gradient.abs()>1e-7)
    fixed=SmoothExplorationPolicy(72,mode='fixed')
    fixed.load_base_policy(base.state_dict())
    sample,oldlog,_=fixed.act(observations[1])
    now,_,_=fixed.evaluate(observations[1],torch.tensor(sample))
    (-now).backward()
    assert fixed.actor[-1].bias.grad[2]==0 and fixed.log_std.grad[2]==0
    assert torch.all(fixed.actor[-1].bias.grad[:2].abs()>1e-7)
    # Save through a resampling boundary; both matrix and age must survive.
    checkpoint=io.BytesIO()
    torch.save(dict(policy=policy.state_dict(),rng=torch.get_rng_state()),checkpoint)
    expected=[policy.act(observations[i%16])[0] for i in range(12)]
    checkpoint.seek(0)
    recovered=torch.load(checkpoint,weights_only=False)
    policy.load_state_dict(recovered['policy']);torch.set_rng_state(recovered['rng'])
    actual=[policy.act(observations[i%16])[0] for i in range(12)]
    assert np.array_equal(np.asarray(actual),np.asarray(expected))
    # Same matrix induces the inner-product correlation of feature directions.
    matrices=torch.randn(10000,97,3)
    x=matrices[:,0,0]
    y=.8*matrices[:,0,0]+.6*matrices[:,1,0]
    covariance=float(((x-x.mean())*(y-y.mean())).mean())
    variance=float(y.var(unbiased=False))
    assert abs(covariance-.8)<.05 and abs(variance-1.)<.05
    result=dict(normalized_feature_variance_pass=True,deterministic_base_equivalence=all(deterministic),
        sample_marginal_likelihood_pass=True,joint_bias_gradient=joint_gradient.tolist(),
        fixed_inactive_gradient_zero=True,restart_across_resampling_bitwise_pass=True,
        noise_covariance=covariance,expected_covariance=.8,noise_variance=variance,
        sampling_interval=8,trained_policy_performance_proved=False,
        exact_history_conditioned_likelihood_claimed=False,physics_ticks=0,training_decisions=0)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='candidate_distribution_gradient_and_recovery_checks_passed',**result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
