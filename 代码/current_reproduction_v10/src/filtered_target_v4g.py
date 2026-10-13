"""Evaluate a gate-direction guard using steering and the actual heading filter."""
import math
import numpy as np
VERSION = 'v4g_filtered_heading_gate_target'


def filtered_target(env):
    import joint_runtime as joint
    import runtime_configuration as runtime
    target = runtime._ORIGINAL_TARGET(env)
    env.v3_guide_calls = getattr(env, 'v3_guide_calls', 0) + 1
    if env.tracker.frontier >= len(env.corridor.gates):
        return target
    gate = env.corridor.gates[env.tracker.frontier]
    h0 = float((env.position-gate.point) @ gate.normal)
    h1 = float((target-gate.point) @ gate.normal)
    beyond = env.progress()[0] + env.held[2] >= env.corridor.arcs[gate.vertex_index]
    crossing_ok = h1 > 1e-8
    if h0 < -1e-8 and crossing_ok:
        difference = target-env.position
        beta = math.atan2(difference[1], difference[0]) if np.linalg.norm(difference)>1e-12 else env.heading
        heading = joint.wrap(env.heading+(1-math.exp(-joint.DT/joint.FILTER_TAU))*joint.wrap(beta+env.held[0]-env.heading))
        direction = np.array([math.cos(heading), math.sin(heading)])
        normal_speed = float(direction @ gate.normal)
        # A currently backward heading is allowed time to turn. It does not
        # imply that a forward target is invalid while the filter is rotating.
        if normal_speed > 1e-10:
            crossing = env.position-h0/normal_speed*direction
            crossing_ok = env.corridor.gate_contains(gate, crossing)
    if (beyond and not crossing_ok) or h0 > 1e-8:
        index = gate.vertex_index
        if h0 > 1e-8:
            target = gate.point-min(.25*env.corridor.lengths[index-1], .25)*env.corridor.tangents[index-1]
        else:
            target = gate.point+min(.25*env.corridor.lengths[index], .25)*env.corridor.tangents[index]
        env.v3_guide_interventions = getattr(env, 'v3_guide_interventions', 0)+1
    return target


def install(*, dt=.001, tolerance=1.25):
    from runtime_resolution_fixed_v4e import install as install_v4e
    result = install_v4e(dt=dt, tolerance=tolerance)
    import runtime_configuration as runtime
    runtime._ordered_target = filtered_target
    runtime.configure_runtime('v3', tolerance=tolerance)
    result.update(version=VERSION, heading_prediction='Exact next requested heading including held steering and 12ms filter; geometric heuristic only, not an executed-trajectory certificate.')
    return result
