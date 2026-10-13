"""Dense geometric-prior cost during training; strict no-swing evaluation is separate.

The original corridor, physical limits, complete lap and terminal brake remain.
Training can observe and penalize a prior violation without truncating the lap;
such a trajectory is never marked as a successful no-swing completion.
"""
import numpy as np
from compact_training_environment import TrainingRuntime

class SoftPriorTrainingRuntime(TrainingRuntime):
    def __init__(self,reference,*,prior_penalty=50.,allow_swing=False,**kwargs):
        if allow_swing:raise ValueError('Soft prior is only a training aid for the no-swing group')
        self.prior_penalty=float(prior_penalty)
        self.outward_squared_integral=0.
        self.outward_violation_ticks=0
        self.maximum_outward_mm=0.
        self.physical_task_complete=False
        super().__init__(reference,allow_swing=True,**kwargs)
        original_update=self.tracker.update
        def update(position):
            count=original_update(position)
            violation=self.outward_displacement(position)
            self.outward_squared_integral+=violation**2*.001
            self.maximum_outward_mm=max(self.maximum_outward_mm,violation)
            self.outward_violation_ticks+=int(violation>1e-7)
            return count
        self.tracker.update=update

    def outward_displacement(self,position):
        j=int(np.searchsorted(self.corner_vertices,self.tracker.frontier,side='right')-1)
        if j<0:return 0.
        k=int(self.corner_vertices[j]);s,_=self.progress()
        next_s=(self.corridor.arcs[self.corner_vertices[j+1]] if j+1<len(self.corner_vertices) else self.corridor.arcs[-1])
        end=min(self.corridor.arcs[k]+50.,.5*(self.corridor.arcs[k]+next_s))
        if s>end+1e-9:return 0.
        outgoing=self.corridor.tangents[k]
        normal=self.corner_sign[j]*np.array([-outgoing[1],outgoing[0]])
        return max(0.,-float((np.asarray(position)-self.corridor.reference[k])@normal))

    def step(self,latent):
        previous=self.outward_squared_integral
        observation,reward,done,info=super().step(latent)
        penalty=self.prior_penalty*(self.outward_squared_integral-previous)
        reward-=penalty
        if done:
            self.physical_task_complete=bool(self.success)
            if self.success and self.outward_violation_ticks:
                self.success=False;self.reason='physical_lap_complete_with_outward_prior_violation'
                info['success']=False;info['reason']=self.reason
        info.update(prior_penalty=penalty,outward_violation_ticks=self.outward_violation_ticks,
            maximum_outward_mm=self.maximum_outward_mm,physical_task_complete=self.physical_task_complete)
        return observation,float(reward),done,info

    def state(self):
        result=super().state()
        for key in ['outward_squared_integral','outward_violation_ticks','maximum_outward_mm','physical_task_complete']:
            result[key]=getattr(self,key)
        return result
