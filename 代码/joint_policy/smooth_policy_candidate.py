"""Unexecuted training candidate: normalized-feature generalized SDE."""
import numpy as np
import torch
from joint_policy import JointPolicy


class SmoothExplorationPolicy(JointPolicy):
    def __init__(self,observation_dim,*,mode='joint',noise_interval=8):
        if mode not in ('joint','fixed') or noise_interval<1:
            raise ValueError('Joint/fixed mode and a positive noise interval are required')
        super().__init__(observation_dim,mode=mode)
        self.noise_interval=int(noise_interval)
        self.register_buffer('exploration_matrix',torch.zeros(97,3))
        self.register_buffer('exploration_age',torch.tensor(self.noise_interval,dtype=torch.int64))

    def exploration_features(self,observation):
        hidden=self.actor[:-1](observation)
        features=torch.cat([hidden,torch.ones_like(hidden[..., :1])],dim=-1)
        return features/torch.linalg.vector_norm(features,dim=-1,keepdim=True)

    @torch.no_grad()
    def reset_noise(self):
        self.exploration_matrix.normal_()
        self.exploration_age.zero_()

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        if deterministic:
            return super().act(observation,deterministic=True)
        observation=torch.as_tensor(observation,dtype=torch.float32)
        if observation.ndim!=1:
            raise ValueError('This candidate sampler handles one environment per policy')
        if int(self.exploration_age)>=self.noise_interval:
            self.reset_noise()
        distribution=self.distribution(observation)
        noise=self.exploration_features(observation)@self.exploration_matrix
        latent=distribution.mean+distribution.stddev*noise*self.active_mask
        self.exploration_age.add_(1)
        logp=(distribution.log_prob(latent)*self.active_mask).sum(-1)
        return latent.numpy(),float(logp),float(self.value(observation))

    def load_base_policy(self,state):
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={'exploration_matrix','exploration_age'}
        assert not result.unexpected_keys
        with torch.no_grad():
            self.active_mask.copy_(torch.tensor([1.,1.,1.] if self.mode=='joint' else [1.,1.,0.]))


def marginal_noise_variance(features,std):
    return features.square().sum(-1,keepdim=True)*std.square()
