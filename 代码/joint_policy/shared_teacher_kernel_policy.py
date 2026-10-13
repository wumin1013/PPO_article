"""Shared compact collocation kernel for all three learned action residuals."""
import numpy as np
from deep_joint_credit_policy import load_credit_policy
from joint_teacher_state_policy import state_pair
from decision_innovation_policy import innovation_features


def kernel(distance, width, dimensions):
    r=np.minimum(np.asarray(distance)/width,1.)
    power=dimensions//2+3
    return np.maximum(1.-r,0.)**power*(1.+power*r)


class SharedTeacherKernelPolicy:
    def __init__(self,metadata):
        self.metadata=metadata
        self.base=load_credit_policy(metadata['parent'])
        self.env=None
        self.expert=None
        if metadata.get('output_mode')=='expert_difference_weights':
            from canonical_gate_state import CanonicalGatePolicy
            self.expert=CanonicalGatePolicy(256)
            self.expert.load_state_dict(metadata['expert_state'])

    def bind(self,env):
        self.env=env
        if hasattr(self.base,'bind'):self.base.bind(env)
        return env

    def act(self,observation,deterministic=True):
        original=self.base.act(observation,deterministic)[0]
        _,inputs=innovation_features(self.env,original)
        a,b=state_pair(observation,inputs)
        m=self.metadata
        a,b=(a-m['mean'])/m['std'],(b-m['mean'])/m['std']
        ka=kernel(np.sqrt(np.mean((a-m['centers'])**2,axis=1)),m['width'],len(a))
        kb=kernel(np.sqrt(np.mean((b-m['centers'])**2,axis=1)),m['width'],len(a))
        values=np.column_stack([.5*(ka-kb),.5*(ka+kb),.5*(ka+kb)])
        if self.expert is not None:values[:,:]=(.5*(ka+kb))[:,None]
        correction=(values*m['coefficients']).sum(axis=0)
        if self.expert is not None:correction*=self.expert.act(observation,True)[0]-original
        return (original+correction).astype(original.dtype),0.,0.
