"""Preserve successful time ordering and give failed episodes progress credit."""
import numpy as np
from terminal_candidate_runtime_v1 import TerminalCandidateRuntime


class JointRuntimeV5(TerminalCandidateRuntime):
    def step(self,latent):
        observation,reward,done,info=super().step(latent)
        penalty=0.
        if done and not self.success:
            remaining=max(0.,self.corridor.arcs[-1]-self.progress()[0])
            penalty=remaining/20.
            reward-=penalty
        info['failure_remaining_penalty_s']=penalty
        return observation,float(reward),done,info


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return JointRuntimeV5(reference,fixed_d=fixed_d,max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
