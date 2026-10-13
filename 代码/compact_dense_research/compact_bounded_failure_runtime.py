"""Cancel terminal potential on failure and apply one shared5-second penalty."""
from compact_speed_deficit_runtime import SpeedDeficitRuntime


class BoundedFailureRuntime(SpeedDeficitRuntime):
    def step(self,action):
        x,reward,done,info=super().step(action)
        if done and not self.env.success:
            remaining=max(0.,self.env.corridor.arcs[-1]-self.env.progress()[0])
            previous=self.env.max_ticks*.001+1.+remaining/20.
            replacement=5.+remaining/self.potential_speed
            reward+=previous-replacement
            info['failure_penalty_replacement_s']=replacement
            info['physical_failure_unchanged']=True
        return x,reward,done,info
