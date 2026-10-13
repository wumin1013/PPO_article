"""History-conditional multivariate Student innovations with matched covariance."""
import math
import numpy as np
import torch
from conditional_exploration import ConditionalExplorationPolicy,RHO


DF=5.


class StudentExplorationPolicy(ConditionalExplorationPolicy):
    def conditional_parameters(self,observation,previous_observation,previous_latent,live):
        mean=self.actor(observation);previous_mean=self.actor(previous_observation)
        live=live.to(dtype=mean.dtype).unsqueeze(-1)
        center=mean+live*RHO*(previous_latent-previous_mean)*self.active_mask
        scale=torch.exp(self.log_std.clamp(-4.,-.3))*torch.sqrt((1-live*RHO**2)*(DF-2)/DF)
        return center,scale

    def conditional_terms(self,observation,latent,previous_observation,previous_latent,live):
        center,scale=self.conditional_parameters(observation,previous_observation,previous_latent,live)
        k=int(self.active_mask.sum());v=torch.tensor(DF,dtype=center.dtype,device=center.device)
        square=(((latent-center)/scale).square()*self.active_mask).sum(-1)
        logdet=(scale.log()*self.active_mask).sum(-1)
        normalizer=torch.lgamma((v+k)/2)-torch.lgamma(v/2)-k/2*torch.log(v*math.pi)
        logp=normalizer-logdet-(v+k)/2*torch.log1p(square/v)
        entropy=-normalizer+logdet+(v+k)/2*(torch.digamma((v+k)/2)-torch.digamma(v/2))
        return logp,entropy

    def evaluate_conditioned(self,observation,latent,previous_observation,previous_latent,live):
        logp,entropy=self.conditional_terms(observation,latent,previous_observation,previous_latent,live)
        return logp,entropy,self.value(observation)

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        if deterministic:return super().act(observation,deterministic=True)
        obs=torch.as_tensor(observation,dtype=torch.float32);live=self.previous_observation is not None
        oldobs=obs if not live else torch.as_tensor(self.previous_observation,dtype=torch.float32)
        oldz=torch.zeros(3) if not live else torch.as_tensor(self.previous_latent,dtype=torch.float32)
        flag=torch.tensor(live);center,scale=self.conditional_parameters(obs,oldobs,oldz,flag)
        # One shared chi-square scale makes the action magnitudes dependent; directions remain symmetric.
        latent=center+scale*torch.randn_like(center)/torch.sqrt(torch.distributions.Chi2(DF).sample()/DF)
        latent=latent*self.active_mask+self.actor(obs)*(1-self.active_mask)
        logp,_=self.conditional_terms(obs,latent,oldobs,oldz,flag)
        self.last_context=dict(previous_observation=oldobs.numpy().copy(),previous_latent=oldz.numpy().copy(),context_live=live)
        self.last_distribution=dict(mean=center.numpy().copy(),scale=scale.numpy().copy(),df=DF)
        self.previous_observation=np.asarray(observation).copy();self.previous_latent=latent.numpy().copy()
        return latent.numpy(),float(logp),float(self.value(obs))
