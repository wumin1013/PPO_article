"""Preserve13local features; replace global remaining distance by one far-turn cue."""
import numpy as np
from direct_compact_runtime import DirectCompactRuntime
from compact_control import local_progress


class PreviewCueRuntime(DirectCompactRuntime):
    def observation(self):
        result=super().observation()
        s=local_progress(self.env);total=self.env.corridor.arcs[-1]
        point,_=self.env.sample_reference(s)
        low,_=self.env.sample_reference(max(0.,s-1.))
        high,_=self.env.sample_reference(min(total,s+1.))
        tangent=high-low;tangent/=max(float(np.linalg.norm(tangent)),1e-12)
        normal=np.array([-tangent[1],tangent[0]])
        far,_=self.env.sample_reference(s+30.)
        result[-1]=(far-point)@normal/30.
        return result

    def feature_names(self):
        names=super().feature_names()
        names[-1]='far30_reference_normal_deviation_over30'
        return names
