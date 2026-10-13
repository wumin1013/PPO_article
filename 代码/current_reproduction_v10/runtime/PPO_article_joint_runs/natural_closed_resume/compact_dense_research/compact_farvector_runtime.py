"""A candidate14input encoder retaining local points and both30mm components."""
import numpy as np
from compact_previewcue_runtime import PreviewCueRuntime
from compact_control import local_progress


class FarVectorRuntime(PreviewCueRuntime):
    def observation(self):
        result=super().observation()
        s=local_progress(self.env);total=self.env.corridor.arcs[-1]
        point,_=self.env.sample_reference(s)
        low,_=self.env.sample_reference(max(0.,s-1.))
        high,_=self.env.sample_reference(min(total,s+1.))
        tangent=high-low;tangent/=max(float(np.linalg.norm(tangent)),1e-12)
        far,_=self.env.sample_reference(s+30.)
        result[5]=(far-point)@tangent/30.
        return result

    def feature_names(self):
        names=super().feature_names()
        names[5]='far30_reference_tangent_displacement_over30'
        return names
