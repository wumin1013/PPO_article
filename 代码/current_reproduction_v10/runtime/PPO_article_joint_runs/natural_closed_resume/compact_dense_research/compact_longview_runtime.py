"""Same14physical inputs; far geometric point30mm away, normalized by30mm."""
import math
import numpy as np
from direct_compact_runtime import DirectCompactRuntime
from compact_control import local_progress


class LongViewRuntime(DirectCompactRuntime):
    def observation(self):
        result=super().observation()
        s=local_progress(self.env)
        h=np.array([math.cos(self.env.heading),math.sin(self.env.heading)])
        frame=np.stack([h,[-h[1],h[0]]],axis=1)
        point,_=self.env.sample_reference(s+30.)
        result[11:13]=(point-self.env.position)@frame/30.
        return result

    def feature_names(self):
        names=super().feature_names()
        names[11:13]=['forward_30mm_x_over30','forward_30mm_y_over30']
        return names
