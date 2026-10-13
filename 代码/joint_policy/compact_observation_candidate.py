"""Observation-only candidates derived from the frozen joint control task.

These are candidate partial observations, not a proven minimal Markov state.
No command, reward, path, gate, or kinematic constraint is changed here.
"""
import math
import numpy as np
from joint_runtime import DT, wrap


def compact_observation(env, distances=(6., 15.)):
    progress, index = env.progress()
    tangent = env.corridor.tangents[index]
    frame = np.stack([tangent, [-tangent[1], tangent[0]]], axis=1)
    angle = wrap(env.heading-math.atan2(tangent[1], tangent[0]))
    has_gate = env.tracker.frontier < len(env.corridor.gates)
    if has_gate:
        gate = env.corridor.gates[env.tracker.frontier]
        anchor = gate.point
        outgoing = env.corridor.tangents[gate.vertex_index]
        turn = math.atan2(tangent[0]*outgoing[1]-tangent[1]*outgoing[0], tangent@outgoing)
    else:
        anchor, turn = env.corridor.reference[-1], 0.
    points, _ = env.sample_reference(progress+np.asarray(distances))
    total_time = env.max_ticks*DT
    remaining = max(0., env.corridor.arcs[-1]-progress)
    return np.r_[env.velocity@frame/100., env.acceleration@frame/2000.,
        math.cos(angle), math.sin(angle),
        np.arcsinh((anchor-env.position)@frame/15.)/4., turn/math.pi,
        ((points-env.position)@frame/15.).ravel(),
        float(has_gate), float(env.tracker.pending_negative),
        math.log1p(remaining/30.)/8., math.log1p(total_time)/8.,
        (env.max_ticks-env.ticks)/env.max_ticks].astype(np.float32)


def feature_names(distances=(6., 15.)):
    return ['velocity_t', 'velocity_n', 'acceleration_t', 'acceleration_n',
        'heading_cos', 'heading_sin', 'next_anchor_t', 'next_anchor_n', 'next_turn'] + [
        f'preview_{distance:g}_{axis}' for distance in distances for axis in ('t', 'n')] + [
        'gate_exists', 'pending_gate_contact', 'remaining_length', 'episode_time_limit', 'remaining_time_fraction']
