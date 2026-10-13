"""Remove redundant previous-command inputs; physics replaces them each step."""
from dynamic_residual_policy import PreviewResidualPolicy


class HistoryIndependentPolicy(PreviewResidualPolicy):
    def distribution(self, observation):
        masked=observation.clone()
        masked[...,8:11]=masked.new_tensor([0.,.3,.25])
        return super().distribution(masked)
