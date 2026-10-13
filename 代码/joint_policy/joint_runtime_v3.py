"""Time-scaled potential and an invertible remaining-distance observation."""
import math
from joint_runtime_v2 import JointRuntimeV2


class JointRuntimeV3(JointRuntimeV2):
    def potential(self):
        progress,_ = self.progress()
        return (progress-self.corridor.arcs[-1])/100.

    def observation(self):
        observation = super().observation()
        progress,_ = self.progress()
        remaining = max(0.,self.corridor.arcs[-1]-progress)
        observation[15] = math.log1p(remaining/30.)/8.
        return observation
