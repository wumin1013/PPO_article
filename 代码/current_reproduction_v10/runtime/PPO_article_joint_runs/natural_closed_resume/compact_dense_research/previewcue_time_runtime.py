"""Use complete physical time with the existing bounded failure penalty."""
from previewcue_learning_runtime import PreviewCueLearningRuntime


class PreviewCueTimeRuntime(PreviewCueLearningRuntime):
    def step(self, action):
        x,reward,done,info=super().step(action)
        # Undo the extra deficit term, retaining the time cost, progress
        # potential, terminal semantics, and failure penalty of the parent.
        reward+=info['additional_speed_deficit_penalty_s']
        info['additional_speed_deficit_penalty_s']=0.
        return x,reward,done,info
