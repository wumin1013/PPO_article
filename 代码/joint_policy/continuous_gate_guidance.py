"""Candidate continuous guide target around ordered reference-segment transitions."""
import numpy as np
from joint_runtime import DT
from continuous_sensor_runtime import ContinuousSensorRuntime


GATE_TRANSITION_WIDTH_MM = 100.*DT


def active_gate_transitions(env,width,*,require_channel=True):
    """Adjacent real gate channels whose transition bands contain the position."""
    corridor=env.corridor;frontier=env.tracker.frontier
    active=[];half=.5*width
    for j in (frontier-1,frontier):
        if not 0<=j<len(corridor.gates):continue
        gate=corridor.gates[j]
        distance=float((env.position-gate.point)@gate.normal)
        if abs(distance)>=half:continue
        gate_projection=env.position-distance*gate.normal
        if require_channel and not corridor.gate_contains(gate,gate_projection):continue
        active.append((gate,distance))
    return active


def continuous_gate_target(env):
    """Blend reference queries; coverage and learned actions remain separate.

    Overlapping active neighborhoods are explicitly unsupported in this candidate.
    """
    corridor=env.corridor
    active=active_gate_transitions(env,GATE_TRANSITION_WIDTH_MM)
    if not active:
        s,_=env.progress()
        return env.sample_reference(s+env.held[2])[0]
    if len(active)>1:
        raise ValueError('Overlapping continuous guide-transition neighborhoods')
    gate,distance=active[0]
    targets=[]
    for index in (gate.vertex_index-1,gate.vertex_index):
        along=np.clip((env.position-corridor.starts[index])@corridor.tangents[index],0.,corridor.lengths[index])
        targets.append(env.sample_reference(corridor.arcs[index]+along+env.held[2])[0])
    fraction=(distance+.5*GATE_TRANSITION_WIDTH_MM)/GATE_TRANSITION_WIDTH_MM
    weight=fraction*fraction*(3.-2.*fraction)
    return (1.-weight)*targets[0]+weight*targets[1]


class ContinuousGateRuntime(ContinuousSensorRuntime):
    def guidance_target(self):
        return continuous_gate_target(self)


def environment_factory(reference,*,fixed_d=None,max_time=None):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()
    return ContinuousGateRuntime(reference,fixed_d=fixed_d,
        max_time=max(5.,float(length)/20.+1.),
        closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
