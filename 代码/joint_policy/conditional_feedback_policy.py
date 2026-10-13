"""AR8 exploration with exact conditional likelihood for the full actor."""
import math
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from frame_feedback_policy import FrameFeedbackPolicy
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy

RHO=math.exp(-1/8)
STD=(.015,.05,.15)


class ConditionalExplorationMixin:
    exploration_std=STD
    def __init__(self,observation_dim,**kwargs):
        super().__init__(observation_dim,**kwargs)
        self.reset_context()

    def reset_context(self):
        self.previous_observation=None;self.previous_latent=None

    def conditional(self,obs,previous_obs,previous_latent,live):
        mean=self.distribution(obs).mean;previous_mean=self.distribution(previous_obs).mean
        live=live.to(mean.dtype).unsqueeze(-1)
        center=mean+live*RHO*(previous_latent-previous_mean)
        scale=mean.new_tensor(self.exploration_std)*torch.sqrt(1-live*RHO**2)
        return torch.distributions.Normal(center,scale)

    def evaluate_conditioned(self,obs,latent,previous_obs,previous_latent,live):
        distribution=self.conditional(obs,previous_obs,previous_latent,live)
        return distribution.log_prob(latent).sum(-1),distribution.entropy().sum(-1),self.value(obs)

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        if deterministic:return super().act(observation,True)
        obs=torch.as_tensor(observation,dtype=torch.float32);live=self.previous_observation is not None
        previous_obs=obs if not live else torch.tensor(self.previous_observation)
        previous_z=torch.zeros(3) if not live else torch.tensor(self.previous_latent)
        distribution=self.conditional(obs,previous_obs,previous_z,torch.tensor(live))
        latent=distribution.sample();logp=distribution.log_prob(latent).sum()
        self.last_context=dict(previous_observation=previous_obs.numpy().copy(),previous_latent=previous_z.numpy().copy(),context_live=live)
        self.previous_observation=np.asarray(observation).copy();self.previous_latent=latent.numpy().copy()
        return latent.numpy(),float(logp),float(self.value(obs))

    def actor_parameters(self):
        parameters=list(self.actor.parameters())
        for name in ('dynamic_head','preview_head'):
            if hasattr(self,name):parameters+=list(getattr(self,name).parameters())
        return parameters


class ConditionalFeedbackPolicy(ConditionalExplorationMixin,PreviewResidualPolicy):
    pass


class ConditionalFramePolicy(ConditionalExplorationMixin,FrameFeedbackPolicy):
    pass


class ConditionalLocalGeometryPolicy(ConditionalExplorationMixin,LocalGeometryFeedbackPolicy):
    exploration_std=(.015,.15,.20)
