"""Development candidate: ordered guidance in the decoded steering frame.

This changes the execution layer and is explicitly a new version. It does not
change the corridor, gate tracker, kinematic checks or completion conditions.
The original v3 configuration remains byte-preserved for matched comparisons.
"""
import math
import numpy as np

VERSION = 'v4a_action_aware_ordered_guidance'


def rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vector[0] - s * vector[1],
                     s * vector[0] + c * vector[1]])


def ordered_target(env):
    import runtime_configuration as runtime
    target = runtime._ORIGINAL_TARGET(env)
    env.v3_guide_calls = getattr(env, 'v3_guide_calls', 0) + 1
    if env.tracker.frontier >= len(env.corridor.gates):
        return target
    gate = env.corridor.gates[env.tracker.frontier]
    side = float((env.position - gate.point) @ gate.normal)
    steering = float(env.held[0])
    direction = rotate(target - env.position, steering)
    forward = float(direction @ gate.normal)
    admissible = forward > 1e-10
    if side < -1e-8 and admissible:
        intersection = env.position - side / forward * direction
        admissible = env.corridor.gate_contains(gate, intersection)
    spans_gate = env.progress()[0] + env.held[2] >= env.corridor.arcs[gate.vertex_index]
    if not ((spans_gate and not admissible) or side > 1e-8):
        return target
    vertex = gate.vertex_index
    if side > 1e-8:
        goal = gate.point - min(.25 * env.corridor.lengths[vertex-1], .25) * env.corridor.tangents[vertex-1]
    else:
        goal = gate.point + min(.25 * env.corridor.lengths[vertex], .25) * env.corridor.tangents[vertex]
    env.v3_guide_interventions = getattr(env, 'v3_guide_interventions', 0) + 1
    # The downstream heading request adds the same decoded steering angle.
    # Express the fallback in its inverse frame so it points at the gate goal.
    return env.position + rotate(goal - env.position, -steering)


def install():
    """Select this candidate only in the calling development process."""
    import runtime_configuration as runtime
    runtime._ordered_target = ordered_target
    return VERSION
