"""Learned actor with continuous interval geometry in place of point tangents."""
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from joint_runtime import SENSOR_DISTANCES


def interval_geometry(observation):
    """Mean reference displacement per interval, without unit normalization.

    Positions remain unchanged. These chord vectors are continuous when a sampled
    location crosses a vertex, including when an interval contains a sharp turn.
    The final interval is repeated to keep a matched actor architecture; this is
    not a claim that 72 inputs are minimal or that all gate/end events are smooth.
    """
    result = observation.clone()
    points = observation[..., 22:42].reshape(*observation.shape[:-1], 10, 2)
    widths = observation.new_tensor(SENSOR_DISTANCES[1:]-SENSOR_DISTANCES[:-1])
    chords = 30.*(points[..., 1:, :]-points[..., :-1, :])/widths[:, None]
    result[..., 42:62] = torch.cat([chords, chords[..., -1:, :]], dim=-2).flatten(-2)
    return result


class ContinuousTangentPolicy(LocalGeometryFeedbackPolicy):
    def distribution(self, observation):
        return super().distribution(interval_geometry(observation))
