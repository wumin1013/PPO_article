"""Three global mixture coefficients over two learned feedback functions."""
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy


class LearnedHeadMixturePolicy(torch.nn.Module):
    def __init__(self,shape_state,cruise_state,coefficients=(0.,0.,0.)):
        super().__init__()
        self.shape_policy=FrameFeedbackPolicy(frame='heading',history_independent=True)
        self.cruise_policy=FrameFeedbackPolicy(frame='heading',history_independent=False)
        self.shape_policy.load_state_dict(shape_state);self.cruise_policy.load_state_dict(cruise_state)
        self.register_buffer('coefficients',torch.tensor(coefficients,dtype=torch.float32))

    def set_coefficients(self,coefficients):
        values=torch.tensor(coefficients,dtype=torch.float32)
        if values.shape!=(3,) or not torch.isfinite(values).all() or torch.any(values<-.5) or torch.any(values>1.5):raise ValueError('Three learned coefficients in[-0.5,1.5] required; decoded physical actions retain original tanh bounds')
        self.coefficients.copy_(values)

    def mean(self,observation):
        a=self.shape_policy.distribution(observation).mean;b=self.cruise_policy.distribution(observation).mean
        return a+(b-a)*self.coefficients

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        if not deterministic:raise ValueError('This parameter-learning pilot is deterministic')
        mean=self.mean(torch.as_tensor(observation,dtype=torch.float32))
        return mean.numpy(),0.,0.
