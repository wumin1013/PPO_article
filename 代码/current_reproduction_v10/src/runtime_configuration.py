"""Versioned short-segment observation/guide support correction.

v1: byte-preserved historical behavior.
v2: disjoint local transition bands; the gate tracker, corridor, commands,
KCM, action map and tolerances remain unchanged. Call before constructing envs.
"""
import numpy as np
from runtime_bootstrap import setup

_ORIGINAL = None
_ORIGINAL_TARGET = None
_VERSION = 'v1'

def _ordered_target(env):
    """A long preview must not fold behind the next unvisited gate.

    The replacement is a point on the next source segment, strictly after the
    gate. This avoids a LOS orbit around a tight turnaround while preserving
    every gate, all corridor checks, and the held learned action.
    """
    target = _ORIGINAL_TARGET(env)
    env.v3_guide_calls = getattr(env, 'v3_guide_calls', 0) + 1
    if env.tracker.frontier < len(env.corridor.gates):
        gate = env.corridor.gates[env.tracker.frontier]
        h0 = float((env.position-gate.point) @ gate.normal)
        h1 = float((target-gate.point) @ gate.normal)
        crossing_ok = h1 > 1e-8
        if h0 < -1e-8 and crossing_ok:
            crossing = env.position - h0/(h1-h0)*(target-env.position)
            crossing_ok = env.corridor.gate_contains(gate, crossing)
        queried_beyond_gate = env.progress()[0] + env.held[2] >= env.corridor.arcs[gate.vertex_index]
        if (queried_beyond_gate and not crossing_ok) or h0 > 1e-8:
            index = gate.vertex_index
            if h0 > 1e-8:
                target = gate.point - min(.25*env.corridor.lengths[index-1], .25)*env.corridor.tangents[index-1]
            else:
                target = gate.point + min(.25*env.corridor.lengths[index], .25)*env.corridor.tangents[index]
            env.v3_guide_interventions = getattr(env, 'v3_guide_interventions', 0) + 1
    return target

def _local_transitions(env, width, *, require_channel=True):
    """Return one gate and its rescaled blend coordinate, or no active gate.

    Existing consumers divide the returned coordinate by the requested width.
    Hence d_eff=d*width/local_width preserves the local smoothstep exactly.
    It is a blending coordinate and must not be reported as physical distance.
    """
    corridor=env.corridor;frontier=env.tracker.frontier
    active=[]
    for index in (frontier-1,frontier):
        if not 0<=index<len(corridor.gates): continue
        gate=corridor.gates[index]
        distance=float((env.position-gate.point)@gate.normal)
        local_width=float(width)
        for neighbor_index in (index-1,index+1):
            if not 0<=neighbor_index<len(corridor.gates): continue
            neighbor=corridor.gates[neighbor_index]
            spacing=float(np.linalg.norm(neighbor.point-gate.point))
            other_distance=abs(float((env.position-neighbor.point)@neighbor.normal))
            # The first cap gives half-width <= 0.45 adjacent vertex spacing.
            # The second makes simultaneous activation impossible: it would
            # require |d_i|<.9|d_j| and |d_j|<.9|d_i| at the same time.
            local_width=min(local_width,.9*spacing,1.8*other_distance)
        if local_width<=1e-10 or abs(distance)>=.5*local_width: continue
        gate_projection=env.position-distance*gate.normal
        if require_channel and not corridor.gate_contains(gate,gate_projection): continue
        active.append((gate,distance*width/local_width))
    if len(active)>1: raise AssertionError('Disjoint local transition invariant failed')
    return active

def configure_runtime(version='v2', tolerance=1.25):
    global _ORIGINAL,_ORIGINAL_TARGET,_VERSION
    setup()
    import continuous_gate_guidance as guide
    import canonical_gate_state as canonical
    import compact_control as compact
    if _ORIGINAL is None: _ORIGINAL=guide.active_gate_transitions
    if _ORIGINAL_TARGET is None: _ORIGINAL_TARGET=guide.continuous_gate_target
    if version not in ('v1','v2','v3'): raise ValueError('Runtime version must be v1, v2 or v3')
    from corridor_v3 import configure, restore
    if version == 'v3': configure(tolerance)
    else: restore()
    function=_ORIGINAL if version=='v1' else _local_transitions
    guide.active_gate_transitions=function
    canonical.active_gate_transitions=function
    compact.active_gate_transitions=function
    guide.continuous_gate_target = _ordered_target if version == 'v3' else _ORIGINAL_TARGET
    compact.continuous_gate_target = guide.continuous_gate_target
    _VERSION=version
    return dict(version=version,changed='local observation/guide blending supports only' if version=='v2' else 'none',
                original_geometry_and_kinematics=True,half_width_adjacent_spacing_fraction=.45 if version=='v2' else None)

def runtime_version(): return _VERSION
