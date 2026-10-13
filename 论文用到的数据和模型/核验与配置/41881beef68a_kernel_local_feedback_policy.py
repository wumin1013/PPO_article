"""Learned local state-basis residual; no path, angle, tick or trajectory lookup."""
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from frame_feedback_policy import heading_frame
from reflection_policy import reflect_observation


class KernelLocalFeedbackPolicy(LocalGeometryFeedbackPolicy):
    def __init__(self, centers=97):
        super().__init__(mask_endpoint=True)
        self.register_buffer('kernel_centers', torch.zeros(centers, 96))
        self.register_buffer('kernel_scale', torch.ones(96))
        self.register_buffer('kernel_width', torch.tensor(1.))
        self.register_buffer('residual_scale', torch.tensor([.025, .3, .3]))
        self.residual_coefficients = torch.nn.Parameter(torch.zeros(3, centers))

    def hidden_pair(self, observation):
        x = observation.clone()
        x[..., 8:11] = x.new_tensor([0., .3, .25])
        x[..., 13:15] = 0.
        return (self.actor[:4](heading_frame(x)),
                self.actor[:4](heading_frame(reflect_observation(x))))

    def kernel(self, hidden):
        # Stable explicit distances avoid cancellation when a state is a center.
        difference = (hidden.unsqueeze(-2) - self.kernel_centers)/self.kernel_scale
        distance = difference.square().mean(-1)
        return torch.exp(-distance/(2*self.kernel_width.square()))

    def residual_features(self, observation):
        a, b = self.hidden_pair(observation)
        ka, kb = self.kernel(a), self.kernel(b)
        return torch.stack([.5*(ka-kb), .5*(ka+kb), .5*(ka+kb)], -2)

    def distribution(self, observation):
        base = super().distribution(observation)
        residual = (self.residual_features(observation)*self.residual_coefficients).sum(-1)
        return torch.distributions.Normal(base.mean + residual*self.residual_scale, base.scale)
