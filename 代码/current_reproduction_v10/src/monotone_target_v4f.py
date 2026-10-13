"""Limit folded lookahead without judging a curved motion by a straight ray."""
import numpy as np
VERSION = 'v4f_forward_target_only'


def forward_target(env):
    import runtime_configuration as runtime
    target = runtime._ORIGINAL_TARGET(env)
    env.v3_guide_calls = getattr(env, 'v3_guide_calls', 0) + 1
    if env.tracker.frontier >= len(env.corridor.gates):
        return target
    gate = env.corridor.gates[env.tracker.frontier]
    h0 = float((env.position - gate.point) @ gate.normal)
    h1 = float((target - gate.point) @ gate.normal)
    beyond = env.progress()[0] + env.held[2] >= env.corridor.arcs[gate.vertex_index]
    index = gate.vertex_index
    if h0 > 1e-8:
        target = gate.point - min(.25 * env.corridor.lengths[index-1], .25) * env.corridor.tangents[index-1]
    elif beyond and h1 <= 1e-8:
        target = gate.point + min(.25 * env.corridor.lengths[index], .25) * env.corridor.tangents[index]
    else:
        return target
    env.v3_guide_interventions = getattr(env, 'v3_guide_interventions', 0) + 1
    return np.asarray(target)


def install(*, dt=.001, tolerance=1.25):
    from runtime_resolution_fixed_v4e import install as install_v4e
    result = install_v4e(dt=dt, tolerance=tolerance)
    import runtime_configuration as runtime
    runtime._ordered_target = forward_target
    runtime.configure_runtime('v3', tolerance=tolerance)
    result.update(version=VERSION,
        target_rule='A preview queried beyond the next gate must lie in its forward half-plane; a missed gate is approached again from its negative side.',
        straight_ray_gate_channel_test=False)
    return result
