"""PPO stores the sampled latent action and its joint Gaussian likelihood."""
import copy
import math
import numpy as np
import torch
from torch import nn


class JointPolicy(nn.Module):
    def __init__(self, observation_dim, *, mode='joint', reference_policy=None):
        super().__init__()
        if mode not in ('joint','fixed','d_only'):
            raise ValueError(mode)
        self.mode = mode
        def network(outputs):
            return nn.Sequential(nn.Linear(observation_dim,96),nn.Tanh(),nn.Linear(96,96),nn.Tanh(),nn.Linear(96,outputs))
        self.actor = network(3)
        self.critic = network(1)
        nn.init.zeros_(self.actor[-1].weight)
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor([0.,np.arctanh(-.4),np.arctanh(.3/1.9)],dtype=torch.float32))
        self.log_std = nn.Parameter(torch.tensor(np.log([.06,.12,.16]),dtype=torch.float32))
        self.register_buffer('active_mask',torch.tensor([0.,0.,1.] if mode=='d_only' else [1.,1.,0.] if mode=='fixed' else [1.,1.,1.]))
        self.reference_actor = None
        if mode=='d_only':
            if reference_policy is None:
                raise ValueError('Only-D requires the entire frozen steering/feed function')
            self.reference_actor = copy.deepcopy(reference_policy.actor).eval()
            for parameter in self.reference_actor.parameters():
                parameter.requires_grad_(False)

    def distribution(self, observation):
        mean = self.actor(observation)
        if self.reference_actor is not None:
            fixed = self.reference_actor(observation)
            mean = mean*self.active_mask+fixed*(1-self.active_mask)
        return torch.distributions.Normal(mean,torch.exp(self.log_std.clamp(-4.,-.3)))

    def value(self, observation):
        return self.critic(observation).squeeze(-1)

    def evaluate(self, observation, latent):
        distribution = self.distribution(observation)
        log_probability = (distribution.log_prob(latent)*self.active_mask).sum(-1)
        entropy = (distribution.entropy()*self.active_mask).sum(-1)
        return log_probability,entropy,self.value(observation)

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        observation = torch.as_tensor(observation,dtype=torch.float32)
        distribution = self.distribution(observation)
        latent = distribution.mean if deterministic else distribution.sample()
        latent = latent*self.active_mask+distribution.mean*(1-self.active_mask)
        log_probability = (distribution.log_prob(latent)*self.active_mask).sum(-1)
        return latent.numpy(),float(log_probability),float(self.value(observation))


def advantages(rewards, values, next_values, terminals, *, lam=.98):
    result = np.zeros(len(rewards),dtype=np.float32)
    carry = 0.
    for i in range(len(rewards)-1,-1,-1):
        live = 1.-float(terminals[i])
        delta = rewards[i]+live*next_values[i]-values[i]
        carry = delta+live*lam*carry
        result[i] = carry
    return result,result+np.asarray(values,dtype=np.float32)


def ppo_update(policy, optimizer, batch, *, epochs=4, minibatch=128, lam=.98):
    obs = torch.as_tensor(np.asarray(batch['observation']),dtype=torch.float32)
    latent = torch.as_tensor(np.asarray(batch['latent']),dtype=torch.float32)
    old_log = torch.as_tensor(batch['log_probability'],dtype=torch.float32)
    adv,returns = advantages(batch['reward'],batch['value'],batch['next_value'],batch['terminal'],lam=lam)
    adv = torch.as_tensor(adv)
    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    returns = torch.as_tensor(returns)
    statistics = []
    for epoch in range(epochs):
        for indices in torch.randperm(len(obs)).split(minibatch):
            logp,entropy,value = policy.evaluate(obs[indices],latent[indices])
            ratio = torch.exp(logp-old_log[indices])
            surrogate = torch.minimum(ratio*adv[indices],ratio.clamp(.8,1.2)*adv[indices])
            actor_loss = -surrogate.mean()
            value_loss = (value-returns[indices]).square().mean()
            loss = actor_loss+.5*value_loss-.001*entropy.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(nn.utils.clip_grad_norm_(policy.parameters(),.5))
            optimizer.step()
            statistics.append([float(actor_loss.detach()),float(value_loss.detach()),
                float(entropy.mean().detach()),gradient,float((old_log[indices]-logp).mean().detach())])
    return dict(zip(['actor_loss','value_loss','latent_entropy','gradient_norm','sample_kl'],
                    np.mean(statistics,axis=0).tolist()))
