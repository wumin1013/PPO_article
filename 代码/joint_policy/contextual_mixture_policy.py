"""PPO learns contextual weights between two frozen learned action functions.

The sampled PPO action is an unconstrained coefficient vector. Clipping it
to[0,1] and combining frozen network outputs is a deterministic action map;
likelihoods are always evaluated on the original sampled coefficients.
"""
import numpy as np
import torch
from conditional_feedback_policy import ConditionalExplorationMixin
from joint_policy import JointPolicy
from frame_feedback_policy import FrameFeedbackPolicy,heading_frame
from reflection_policy import reflect_observation


class ContextualMixturePolicy(ConditionalExplorationMixin,JointPolicy):
    exploration_std=(.25,.25,.25)

    def __init__(self,observation_dim,shape_state,cruise_state):
        super().__init__(observation_dim)
        self.shape_function=FrameFeedbackPolicy(frame='heading',history_independent=True)
        self.cruise_function=FrameFeedbackPolicy(frame='heading',history_independent=False)
        self.shape_function.load_state_dict(shape_state);self.cruise_function.load_state_dict(cruise_state)
        for module in (self.shape_function,self.cruise_function):
            module.eval()
            for parameter in module.parameters():parameter.requires_grad_(False)
        torch.nn.init.zeros_(self.actor[-1].weight);torch.nn.init.zeros_(self.actor[-1].bias)

    def distribution(self,observation):
        observation=observation.clone();observation[...,8:11]=observation.new_tensor([0.,.3,.25])
        # Mixing weights are reflection-even for all three heads.
        mean=.5*(self.actor(heading_frame(observation))+self.actor(heading_frame(reflect_observation(observation))))
        return torch.distributions.Normal(mean,mean.new_tensor(self.exploration_std))

    @torch.no_grad()
    def environment_latent(self,observation,coefficient_latent):
        obs=torch.as_tensor(observation,dtype=torch.float32)
        weight=torch.as_tensor(coefficient_latent,dtype=torch.float32).clamp(0.,1.)
        a=self.shape_function.distribution(obs).mean;b=self.cruise_function.distribution(obs).mean
        return (a+(b-a)*weight).numpy()


class DeployedContextualView:
    def __init__(self,policy):self.policy=policy

    def act(self,observation,deterministic=False):
        if not deterministic:raise ValueError('Evaluation adapter is deterministic')
        coefficient,logp,value=self.policy.act(observation,True)
        return self.policy.environment_latent(observation,coefficient),logp,value


def load_contextual_policy(saved):
    state=saved['policy']
    def part(prefix):return {key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
    p=ContextualMixturePolicy(saved['observation_dim'],part('shape_function.'),part('cruise_function.'))
    p.load_state_dict(state)
    return p
