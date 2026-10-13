"""Continuous-geometry transition parameter in a learned shared-feedback actor."""
import torch
from continuous_tangent_policy import interval_geometry
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy


class GeometryHomotopyPolicy(KernelLocalFeedbackPolicy):
    """mix=0 is exactly the parent; mix=1 removes the point-tangent block.

    The scalar is a learned global representation parameter, not a per-path
    schedule. Intermediate values retain part of the original discontinuity.
    The optimizer must keep it within [0,1]; no hidden runtime clamp is used.
    """
    def __init__(self, centers=256):
        super().__init__(centers)
        self.geometry_mix=torch.nn.Parameter(torch.tensor(0.))

    def distribution(self, observation):
        continuous=interval_geometry(observation)
        blended=observation+self.geometry_mix*(continuous-observation)
        return super().distribution(blended)
