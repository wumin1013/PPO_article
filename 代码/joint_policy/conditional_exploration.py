"""AR Gaussian exploration with its history-conditional likelihood."""
import math
import numpy as np
import torch
from joint_policy import JointPolicy


RHO = math.exp(-1/8)


class ConditionalExplorationPolicy(JointPolicy):
    def __init__(self, observation_dim, *, mode='joint'):
        super().__init__(observation_dim, mode=mode)
        self.reset_context()

    def reset_context(self):
        self.previous_observation = None
        self.previous_latent = None

    def context_state(self):
        return dict(previous_observation=self.previous_observation,
                    previous_latent=self.previous_latent)

    def conditional_distribution(self, observation, previous_observation, previous_latent, live):
        mean = self.actor(observation)
        previous_mean = self.actor(previous_observation)
        live = live.to(dtype=mean.dtype).unsqueeze(-1)
        center = mean + live * RHO * (previous_latent-previous_mean) * self.active_mask
        scale = torch.exp(self.log_std.clamp(-4.,-.3)) * torch.sqrt(1-live*RHO**2)
        return torch.distributions.Normal(center, scale)

    def evaluate_conditioned(self, observation, latent, previous_observation, previous_latent, live):
        distribution = self.conditional_distribution(observation, previous_observation, previous_latent, live)
        return ((distribution.log_prob(latent)*self.active_mask).sum(-1),
                (distribution.entropy()*self.active_mask).sum(-1), self.value(observation))

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        if deterministic:
            # Starting with zero residual and applying conditional means keeps residual zero.
            return super().act(observation, deterministic=True)
        obs = torch.as_tensor(observation,dtype=torch.float32)
        live = self.previous_observation is not None
        previous_obs = obs if not live else torch.as_tensor(self.previous_observation,dtype=torch.float32)
        previous_z = torch.zeros(3) if not live else torch.as_tensor(self.previous_latent,dtype=torch.float32)
        distribution = self.conditional_distribution(obs,previous_obs,previous_z,torch.tensor(live))
        latent = distribution.sample()
        latent = latent*self.active_mask+self.actor(obs)*(1-self.active_mask)
        logp = (distribution.log_prob(latent)*self.active_mask).sum(-1)
        self.last_context = dict(previous_observation=previous_obs.numpy().copy(),
                                 previous_latent=previous_z.numpy().copy(),context_live=live)
        self.last_distribution = dict(mean=distribution.mean.numpy().copy(),std=distribution.stddev.numpy().copy())
        self.previous_observation = np.asarray(observation).copy()
        self.previous_latent = latent.numpy().copy()
        return latent.numpy(),float(logp),float(self.value(obs))
