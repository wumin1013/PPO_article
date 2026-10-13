"""Reuse the shared PPO costs with the validated14input preview-cue encoder."""
from compact_bounded_failure_runtime import BoundedFailureRuntime
from compact_previewcue_runtime import PreviewCueRuntime


class PreviewCueLearningRuntime(BoundedFailureRuntime,PreviewCueRuntime):
    pass
