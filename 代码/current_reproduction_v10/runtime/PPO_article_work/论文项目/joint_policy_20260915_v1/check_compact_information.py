"""Task-semantic counterexamples for compact observation candidates."""
import math
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, write_json
from compact_observation_candidate import compact_observation
from joint_runtime_v5 import environment_factory


def at(reference, position, frontier=0):
    env = environment_factory(np.asarray(reference, float))
    env.position = np.asarray(position, float)
    env.tracker.position = env.position.copy()
    env.tracker.frontier = frontier
    env.velocity[:] = [40., 0.]
    return env


def main():
    stage = register('compact_information_counterexamples_v1',
        [Path(__file__), BASE/'compact_observation_candidate.py', BASE/'geometry_spec.py',
         BASE/'joint_runtime.py', BASE/'joint_runtime_v5.py'],
        dict(environments=10, physical_macros=2, physical_ticks_upper=4240,
             direct_gate_tracker_updates=2, neural_calls=0, training_decisions=0),
        'Check motion, signed tracking, ordered-contact memory, endpoint/time information and an explicit remaining geometry alias before training compact policies.')
    reference = [[-20.,0.],[0.,0.],[0.,25.]]
    checks = []
    for variable in ('signed_error', 'gate_contact', 'remaining_time'):
        a, b = at(reference, [-8.,0.]), at(reference, [-8.,0.])
        if variable == 'signed_error':
            a.position[1], b.position[1] = -.4, .4
        elif variable == 'gate_contact':
            for env in (a,b):
                env.position[:] = 0.
                env.tracker.position[:] = 0.
            b.tracker.pending_negative = True
        else:
            b.ticks = b.max_ticks-100
        difference = float(np.max(abs(compact_observation(a)-compact_observation(b))))
        row = dict(variable=variable, observation_difference=difference, distinguished=difference>0)
        if variable == 'gate_contact':
            row['gates_advanced'] = [env.tracker.update(np.array([.01,.01])) for env in (a,b)]
            assert row['gates_advanced'] == [0,1]
        checks.append(row)
    # Same present motion and short geometry, but a different later corner.
    def future(angle):
        theta = math.radians(angle)
        return [[-20.,0.],[0.,0.],[20.,0.],
                [20.+20.*math.cos(theta),20.*math.sin(theta)]]
    a, b = at(future(150.),[-.5,0.]), at(future(60.),[-.5,0.])
    c18 = float(np.max(abs(compact_observation(a)-compact_observation(b))))
    c22 = float(np.max(abs(compact_observation(a,(3.,6.,12.,24.))-compact_observation(b,(3.,6.,12.,24.)))))
    c72 = float(np.max(abs(a.observation()-b.observation())))
    action = np.array([0., .2, -.7])
    _, ra, _, ia = a.step(action)
    _, rb, _, ib = b.step(action)
    next_difference = float(np.max(abs(compact_observation(a)-compact_observation(b))))
    assert c18 < 1e-7 and c22 > .01 and c72 > .01 and next_difference > .01
    alias = dict(initial_compact18_difference=c18, initial_compact22_difference=c22,
        initial_original72_difference=c72, next_compact18_difference=next_difference,
        immediate_reward_difference=abs(ra-rb),
        identical_action=action.tolist(), references=[future(150.),future(60.)],
        physical_motion_max_difference=float(np.max(abs(ia['rows']-ib['rows']))),
        conclusion='18 components do not define an exact Markov abstraction: the same observation/action has different next observations. This example does not prove a different optimal immediate action or a performance loss.',
        decision='Keep 18 as an explicitly partial candidate only; compare against the 22-component wider window if training evidence calls for it. Neither is a universal sufficiency theorem.')
    # Rotation and translation should not change the local information.
    a = at(reference,[-8.,.3])
    angle = .73
    rot = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    offset = np.array([120.,-30.])
    b = environment_factory(np.asarray(reference)@rot.T+offset)
    b.position = a.position@rot.T+offset
    b.tracker.position = b.position.copy()
    b.velocity = a.velocity@rot.T
    b.heading = a.heading+angle
    rigid_error = float(np.max(abs(compact_observation(a)-compact_observation(b))))
    assert rigid_error < 1e-6 and all(r['distinguished'] for r in checks)
    write_json(stage/'results.json',dict(checks=checks,geometry_alias=alias,
        rigid_frame_max_error=rigid_error, total_environment_constructions=10,
        training_decisions=0, goal_complete=False))
    receipt(stage,dict(status='completed_information_counterexamples_with_known_alias',
        physical_macros=2,physical_ticks=len(ia['rows'])+len(ib['rows']),
        direct_gate_tracker_updates=2,neural_calls=0,training_decisions=0))
    print(alias)


if __name__ == '__main__':
    main()
