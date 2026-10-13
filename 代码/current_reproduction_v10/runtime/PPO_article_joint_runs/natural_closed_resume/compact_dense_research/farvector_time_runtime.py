"""Candidate14feature encoder with the established physical-time PPO reward."""
from compact_farvector_runtime import FarVectorRuntime
from previewcue_time_runtime import PreviewCueTimeRuntime


class FarVectorTimeRuntime(FarVectorRuntime, PreviewCueTimeRuntime):
    pass
