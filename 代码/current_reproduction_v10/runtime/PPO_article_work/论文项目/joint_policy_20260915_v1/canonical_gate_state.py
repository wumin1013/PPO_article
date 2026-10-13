"""Heading-coordinate state with continuous local gate and endpoint sensing."""
import numpy as np
import torch
from joint_runtime import SENSOR_DISTANCES
from continuous_sensor_runtime import ContinuousSensorRuntime
from continuous_gate_guidance import ContinuousGateRuntime,active_gate_transitions
from frame_feedback_policy import heading_frame
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from reflection_policy import reflect_observation


STATE_TRANSITION_WIDTH_MM=100.*.020


def canonical_observation(env,*,blend_gate=True,smooth_valid=True,gate_channel_support=True):
    """Blend observations only after expressing their vectors in one frame.

    The actual coverage tracker is restored before returning. No model output,
    speed, or ideal trajectory is prescribed by this observation encoding.
    """
    frontier=env.tracker.frontier
    def at(index):
        env.tracker.frontier=index
        raw=ContinuousSensorRuntime.observation(env)
        if smooth_valid:
            s,_=env.progress()
            raw[62:72]=np.clip(.5+(env.corridor.arcs[-1]-s-SENSOR_DISTANCES)/STATE_TRANSITION_WIDTH_MM,0.,1.)
        return heading_frame(torch.tensor(raw)).numpy()
    active=active_gate_transitions(env,STATE_TRANSITION_WIDTH_MM,
        require_channel=gate_channel_support) if blend_gate else []
    try:
        if not active:return at(frontier)
        if len(active)>1:raise ValueError('Overlapping continuous state-transition neighborhoods')
        gate,distance=active[0]
        fraction=.5+distance/STATE_TRANSITION_WIDTH_MM
        weight=fraction*fraction*(3.-2.*fraction)
        before,after=at(gate.vertex_index-1),at(gate.vertex_index)
        return ((1.-weight)*before+weight*after).astype(getattr(env,'observation_dtype',np.float32))
    finally:
        env.tracker.frontier=frontier


class CanonicalGatePolicy(KernelLocalFeedbackPolicy):
    def __init__(self,centers=256):
        super().__init__(centers)
        # Input vectors have already been rotated by the observation encoder.
        self.frame='reference'

    def hidden_pair(self,observation):
        x=observation.clone();x[...,8:11]=x.new_tensor([0.,.3,.25]);x[...,13:15]=0.
        return self.actor[:4](x),self.actor[:4](reflect_observation(x))


class CanonicalGateRuntime(ContinuousGateRuntime):
    def observation(self):
        return canonical_observation(self)


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return CanonicalGateRuntime(reference,fixed_d=fixed_d,
        max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))


def checkpoint_environment_factory(architecture):
    """Keep historical and repaired state semantics explicit in checkpoints."""
    if architecture=='canonical_gate_kernel':return environment_factory
    if architecture=='gate_band_canonical_kernel':
        from gate_band_state import environment_factory as gate_band_factory
        return gate_band_factory
    raise ValueError('Unsupported canonical checkpoint architecture: '+architecture)
