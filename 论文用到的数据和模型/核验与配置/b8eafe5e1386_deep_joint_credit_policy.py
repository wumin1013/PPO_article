"""A shared learned residual for full-consequence deterministic actor updates."""
import numpy as np
import torch
from torch import nn
from conditioned_joint_policy import ConditionedJointPolicy
from raw_state_joint_policy import LOCAL_INDICES
from reflection_policy import reflect_observation


def load_credit_policy(metadata):
    if metadata['architecture']=='observable_memory_joint_radial':
        from observable_memory_joint_policy import ObservableMemoryJointPolicy
        return ObservableMemoryJointPolicy(metadata)
    if metadata['architecture']=='joint_expert_residual':
        from learned_action_pulse_policy import LearnedActionPulsePolicy
        return LearnedActionPulsePolicy(metadata['parent'],metadata['coefficients'],metadata['mode'],
                                       metadata['expert_state'],metadata.get('feature_spec'))
    if metadata['architecture']=='shared_joint_teacher_kernel':
        from shared_teacher_kernel_policy import SharedTeacherKernelPolicy
        return SharedTeacherKernelPolicy(metadata)
    if metadata['architecture']=='joint_teacher_state_radial':
        from joint_teacher_state_policy import JointTeacherStatePolicy
        return JointTeacherStatePolicy(metadata)
    if metadata['architecture']=='local_suffix_credit':
        from local_suffix_credit_policy import LocalSuffixCreditPolicy
        policy=LocalSuffixCreditPolicy(metadata['parent'],metadata['mean'],metadata['std'],metadata['centers'],metadata['radius'],metadata.get('precision','float32'))
        with torch.no_grad():
            policy.coefficients.copy_(torch.as_tensor(metadata['coefficients']))
            policy.step_scale.fill_(metadata['step_scale'])
        return policy
    if metadata['architecture'] in ('conditioned_raw_state','conditioned_joint_policy'):
        spec=metadata['policy_spec']
        policy=ConditionedJointPolicy(spec['parent'],spec['transform'])
        with torch.no_grad():policy.coefficients.copy_(torch.as_tensor(metadata['coefficients']))
        return policy
    if metadata['architecture']=='deep_joint_credit':
        policy=DeepJointCreditPolicy(metadata['parent'],metadata['mean'],metadata['std'])
        policy.residual.load_state_dict(metadata['residual'])
        with torch.no_grad():
            policy.output_mask.copy_(torch.as_tensor(metadata['output_mask']))
            policy.step_scale.fill_(metadata['step_scale'])
        return policy
    raise ValueError('Unsupported full-consequence parent: '+metadata['architecture'])


class DeepJointCreditPolicy(nn.Module):
    def __init__(self,parent,mean,std):
        super().__init__()
        self.base=load_credit_policy(parent)
        for parameter in self.base.parameters():parameter.requires_grad_(False)
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.register_buffer('output_scale',torch.tensor([.03,.2,.2]))
        self.register_buffer('output_mask',torch.ones(3))
        self.register_buffer('step_scale',torch.tensor(1.))
        self.residual=nn.Sequential(nn.Linear(len(LOCAL_INDICES),64),nn.Tanh(),
            nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def correction(self,observation):
        a=(observation[...,LOCAL_INDICES]-self.mean)/self.std
        b=(reflect_observation(observation)[...,LOCAL_INDICES]-self.mean)/self.std
        sign=observation.new_tensor([-1.,1.,1.])
        return .5*(self.residual(a)+sign*self.residual(b))*self.output_scale*self.output_mask*self.step_scale

    @torch.no_grad()
    def act(self,observation,deterministic=True):
        x=torch.as_tensor(observation,dtype=torch.float32)
        return self.base.act(observation,deterministic)[0]+self.correction(x).numpy(),0.,0.


def credit_normalization(observations):
    x=torch.as_tensor(observations,dtype=torch.float32)
    both=np.vstack([x.numpy()[:,LOCAL_INDICES],reflect_observation(x).numpy()[:,LOCAL_INDICES]])
    return both.mean(0),np.maximum(both.std(0),.05)
