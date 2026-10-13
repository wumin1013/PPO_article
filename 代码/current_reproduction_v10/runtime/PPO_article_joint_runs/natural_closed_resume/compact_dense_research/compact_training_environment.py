"""Explicit training representation and potential, independent of action execution."""
import math
from compact_control import CompactRuntime, local_progress, feature_names

class TrainingRuntime(CompactRuntime):
    def __init__(self,reference,*,potential_speed=100.,remaining_encoding='terminal',**kwargs):
        self.potential_speed=float(potential_speed)
        self.remaining_encoding=remaining_encoding
        if self.potential_speed<=0 or remaining_encoding not in ('terminal','log'):raise ValueError('Training interface')
        super().__init__(reference,**kwargs)

    def potential(self):
        # Gamma=1, terminal potential=0: this telescopes to L/potential_speed-T
        # for every successful complete episode. No preferred path shape is added.
        return (self.progress()[0]-self.corridor.arcs[-1])/self.potential_speed

    def observation(self):
        result=super().observation()
        if self.remaining_encoding=='log':
            remaining=max(0.,self.corridor.arcs[-1]-local_progress(self))
            result[-1]=math.log1p(remaining/30.)/8.
        return result

    def feature_names(self):
        result=feature_names(self.dimension)
        if self.remaining_encoding=='log':result[-1]='log1p_remaining_distance_over30_div8'
        return result
