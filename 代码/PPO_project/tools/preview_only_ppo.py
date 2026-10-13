"""Frozen feed policy with PPO sampling and optimizing only preview distance."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from preview_positive_feed_ppo import PositiveFeedPPO


def frozen_actor_digest(actor):
    """Bitwise identity of trunk and the two non-preview rows of every head."""
    digest=hashlib.sha256()
    for name,value in actor.state_dict().items():
        tensor=value[:2] if name.startswith(('mu_head.','std_head.')) else value
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str((array.shape,array.dtype)).encode());digest.update(array.tobytes())
    return digest.hexdigest()


class PreviewOnlyPPO(PositiveFeedPPO):
    """Keep the original network and loss, changing trainability and sampling.

    The original three-row head tensors keep checkpoint compatibility. Gradient
    hooks expose only their preview row to fresh zero-weight-decay Adam states;
    the trunk has requires_grad=False. Frozen bytes are checked before/after
    every update, never repaired after an unexpected optimizer mutation.
    """
    def __init__(self,*args,active_action_dims=(2,),initial_actor_state=None,
                 initial_critic_state=None,**kwargs):
        if tuple(active_action_dims)!=(2,):raise ValueError('Only preview dimension2 may be active')
        if (initial_actor_state is None)!=(initial_critic_state is None):
            raise ValueError('Actor and critic initializer must be supplied together')
        super().__init__(*args,active_action_dims=(1,2),**kwargs)
        if initial_actor_state is not None:
            self.actor.load_state_dict(initial_actor_state)
            self.critic.load_state_dict(initial_critic_state)
        self.active_action_dims=(2,)
        self.action_mask=np.asarray([0.,0.,1.],np.float32)
        trainable=[];self._preview_hooks=[]
        for name,parameter in self.actor.named_parameters():
            is_head=name.startswith(('mu_head.','std_head.'))
            parameter.requires_grad_(is_head)
            if is_head:
                mask=torch.zeros_like(parameter);mask[2]=1.
                self._preview_hooks.append(parameter.register_hook(lambda gradient,m=mask:gradient*m))
                trainable.append(parameter)
        rate=self.actor_optimizer.param_groups[0]['lr']
        self.actor_optimizer=torch.optim.Adam(trainable,lr=rate,weight_decay=0.)
        self.seal_frozen_parameters()
        # The unchanged trainer and evaluator load checkpoint states directly.
        # Check once per load, including zero-update resume/evaluation, without
        # hashing on every physical or macro tick and perturbing inference cost.
        self._load_guard=self.actor.register_load_state_dict_post_hook(
            lambda module,incompatible:self.assert_frozen_parameters())

    def seal_frozen_parameters(self):
        """Seal the initializer; later checkpoint loads must retain these bytes."""
        if hasattr(self,'_frozen_actor_sha256'):
            raise RuntimeError('Frozen initializer may only be sealed once')
        self._frozen_actor_sha256=frozen_actor_digest(self.actor)

    def assert_frozen_parameters(self):
        if frozen_actor_digest(self.actor)!=self._frozen_actor_sha256:
            raise RuntimeError('Frozen actor trunk/feed/angular bytes changed')

    @torch.no_grad()
    def take_action(self,state,return_info=False,deterministic=False):
        array=np.asarray(state,dtype=np.float32)
        if array.shape!=(self.state_dim,) or not np.isfinite(array).all():
            raise ValueError('Expected one finite state vector')
        tensor=torch.as_tensor(array[None],device=self.device)
        mu,std=self.actor(tensor)
        latent=mu.clone();latent[0,0]=0.
        if not deterministic:
            latent[0,2]=torch.distributions.Normal(mu[0,2],std[0,2]).sample()
        # Feed is the fixed policy's deterministic logit. The unchanged
        # environment applies sigmoid; feed is never sampled or scored here.
        action=latent[0].cpu().numpy().copy()
        if return_info:
            mask=torch.as_tensor(self.action_mask[None],device=self.device)
            return action,dict(action_mask=self.action_mask.copy(),
                old_log_prob=float(self.latent_log_prob(mu,std,latent,mask)[0,0]),
                mu=mu[0].cpu().numpy().copy(),std=std[0].cpu().numpy().copy(),
                deterministic=bool(deterministic),macro_period_s=self.macro_period_s,
                feed_sampling='none; current-state deterministic frozen-policy mean')
        return action

    def update(self,transition_dict):
        self.assert_frozen_parameters()
        count=len(transition_dict['states'])
        if count==0:raise ValueError('Cannot update from an empty rollout')
        values={}
        for name,shape in (('states',(count,self.state_dim)),('actions',(count,3))):
            array=np.asarray(transition_dict[name],dtype=np.float32)
            if array.shape!=shape or not np.isfinite(array).all():
                raise ValueError(f'{name} must be finite with shape {shape}')
            values[name]=torch.as_tensor(array,device=self.device)
        states,actions=values['states'],values['actions']
        with torch.no_grad():mu,_=self.actor(states)
        feed_error=float((actions[:,1]-mu[:,1]).abs().max())
        # Batched and single-row GEMM may differ at floating-point roundoff.
        if not torch.allclose(actions[:,1],mu[:,1],rtol=1e-6,atol=1e-6):
            raise ValueError('Rollout feed must equal the frozen policy mean; stochastic feed is not allowed')
        if not bool(torch.all(actions[:,0]==0)):
            raise ValueError('Angular residual is inactive')
        result=super().update(transition_dict)
        self.assert_frozen_parameters()
        self.last_update_diagnostics.update(physical_decision_count=0,
            feed_sampling='None: deterministic frozen-feed function',
            likelihood='Only raw preview latent Gaussian; mask[0,0,1]',
            entropy_coordinates='Preview latent Gaussian only',
            frozen_actor_sha256=self._frozen_actor_sha256,
            deterministic_feed_logit_max_absolute_error=feed_error,
            actor_optimization='Preview rows of mu/std heads only; all trunk/feed/angular bytes unchanged')
        return result
