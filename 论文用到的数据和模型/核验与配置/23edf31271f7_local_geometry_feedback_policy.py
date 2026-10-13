"""Matched endpoint-bearing mask; ordered local geometry and distance remain."""
from frame_feedback_policy import FrameFeedbackPolicy


class LocalGeometryFeedbackPolicy(FrameFeedbackPolicy):
    def __init__(self,observation_dim=72,*,mask_endpoint=False):
        super().__init__(observation_dim,frame='heading',history_independent=True)
        self.mask_endpoint=mask_endpoint

    def distribution(self,observation):
        if self.mask_endpoint:
            observation=observation.clone();observation[...,13:15]=0.
        return super().distribution(observation)
