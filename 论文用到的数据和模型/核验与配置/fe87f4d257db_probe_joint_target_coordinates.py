"""Matched physical probe of preview change with/without initial target compensation."""
import math
import time
import numpy as np
import torch
from complete_joint_function_policy import CompleteJointFunctionPolicy
from canonical_gate_state import checkpoint_environment_factory
from analyze_nonlinear_joint_response import branch
from probe_coordinated_suffix import reconstruct
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def guide_angle(env, held):
    previous = env.held.copy()
    try:
        env.held = held.copy()
        d = env.guidance_target()-env.position
        return math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-12 else env.heading
    finally:
        env.held = previous


def preserve_initial_target(env, baseline_latent, proposed_latent):
    baseline = env.decode_action(baseline_latent)
    proposed = env.decode_action(proposed_latent)
    old_beta, new_beta = guide_angle(env, baseline), guide_angle(env, proposed)
    steering = wrap(proposed[0]+old_beta-new_beta)
    if abs(steering) >= math.pi/2:
        raise ValueError('Target-coordinate action outside original steering range')
    latent = np.asarray(proposed_latent, dtype=np.float64).copy()
    latent[0] = np.arctanh(steering/(math.pi/2))
    initial_change = wrap(new_beta+env.decode_action(latent)[0]-old_beta-baseline[0])
    return latent, dict(uncompensated_initial_target_change_deg=math.degrees(wrap(new_beta-old_beta)),
                        compensated_initial_target_change_deg=math.degrees(initial_change),
                        steering_compensation_deg=math.degrees(steering-proposed[0]))


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_complete_joint_function_learning_20260919'
    result = read(source/'results.json')
    best = next(r for r in result['candidates'] if r['tag'] == result['best'])
    indexed = {r['path']: r for r in best['records']}
    md = torch.load(source/'research_candidate.pt', map_location='cpu', weights_only=False)
    policy = CompleteJointFunctionPolicy(md['joint_state'], md['feed_state'])
    with torch.no_grad():
        policy.coefficients.copy_(torch.as_tensor(md['coefficients']))
    factory = checkpoint_environment_factory(md['environment_architecture'])
    paths = [12, 13, 14, 15]
    stage = register('canonical_gate_band_joint_target_coordinate_probe_20260919',
        [__file__, source/'research_candidate.pt', source/'results.json', source/'receipt.json',
         BASE/'complete_joint_function_policy.py', BASE/'analyze_nonlinear_joint_response.py']+
        [source/indexed[p]['trace'] for p in paths],
        dict(paths=paths, maximum_states=8, maximum_complete_branches=32,
             preview_latent_offset=.1, wall_s=120, training_updates=0),
        'On newly learned complete-policy states, distinguish preview geometric '
        'advance from its instantaneous combined-target rotation. At pre-apex and '
        'largest-preview-jump states, compare +/-0.1 preview latent offsets with '
        'the same offsets accompanied by the exact steering compensation that '
        'preserves the initial combined heading target. Each lasts one original '
        '20ms decision and then resumes the SAME learned policy on real future '
        'states to complete braking/rest. This is a matched physical diagnostic '
        'of potential action coordinates, not a deployed rule or learned policy. '
        'Preserving the initial target does NOT imply preserving its subsequent '
        '20ms trajectory or better shape. Record full consequences before deciding '
        'whether learning in compensated coordinates is justified. No altered '
        'limits, time steps, guide or tests; no adoption.')
    start = time.perf_counter()
    costs = dict(actor_calls=0, physics_ticks=0, saved_prefix_geometry_ticks=0)
    output, queries = [], []
    try:
        for path in paths:
            with np.load(source/indexed[path]['trace']) as z:
                ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
            apex = int(np.argmin(np.linalg.norm(rows[:, :2]-ref[1], axis=1)))
            left, right = max(1, apex//20-5), min(len(actions)-1, apex//20+5)
            jump = left+int(abs(np.diff(actions[:, 2]))[left-1:right-1].argmax())
            frames = sorted(set([max(0, apex//20-2)*20, jump*20]))
            for frame in frames:
                env = reconstruct(ref, rows, actions, frame, costs, env_factory=factory)
                latent = policy.act(env.observation(), True)[0]
                assert np.array_equal(env.decode_action(latent), actions[frame//20])
                initial = env.state()
                query = len(queries)
                queries.append(dict(query=query, path=path, frame=frame,
                                    baseline=indexed[path]['shape']))
                for direction in (-1, 1):
                    proposed = latent.astype(float).copy()
                    proposed[2] += .1*direction
                    compensated, coupling = preserve_initial_target(env, latent, proposed)
                    assert abs(coupling['compensated_initial_target_change_deg']) < 1e-10
                    for arm, action in [('preview_only', proposed), ('target_compensated', compensated)]:
                        r, full, held, obs = branch(policy, factory, ref, rows, actions, initial,
                            action-latent, costs, start+115)
                        name = f'q{query}_{direction:+d}_{arm}.npz'
                        np.savez_compressed(stage/name, reference=ref, rows=full, actions=held,
                                            suffix_observations=obs, offset=action-latent, frame=frame)
                        r.update(query=query, path=path, direction=direction, arm=arm,
                                 coupling=coupling, trace=name)
                        output.append(r)
                    print('TARGET_COORDINATE', path, frame, direction, coupling,
                          [(r['arm'], r['success'], r['shape']['full_physical_reverse_deg'],
                            r['shape']['full_normalized_energy']) for r in output[-2:]], flush=True)
                write_json(stage/'results.json', dict(model_sha256=sha(source/'research_candidate.pt'),
                    queries=queries, records=output, costs=costs, diagnostic_only=True,
                    intervention_deployed=False, goal_complete=False))
        receipt(stage, dict(status='completed_target_coordinate_physical_probe', queries=len(queries),
            branches=len(output), successes=sum(r['success'] for r in output), costs=costs,
            wall_s=time.perf_counter()-start, intervention_deployed=False, goal_complete=False))
    except Exception as exc:
        receipt(stage, dict(status='failed_target_coordinate_physical_probe', error=str(exc),
                           costs=costs, wall_s=time.perf_counter()-start, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
