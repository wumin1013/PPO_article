"""Current saved-state joint-action counterfactuals, with complete closed feedback."""
import copy
import itertools
import sys
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from nonlinear_feed_adapter import NonlinearFeedAdapter
from probe_coordinated_suffix import reconstruct
from probe_linear_action_hold import target_motion
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


AMPLITUDE = np.array([.015, .06, .06])
METRICS = ('full_physical_reverse_deg', 'out_edge_outward_mm',
           'full_normalized_energy', 'full_spatial_peak_curvature_mm_inv',
           'apex_speed_mm_s')


def measure(ref, rows, env):
    audit = difference_audit(ref[0], rows)
    corridor = command_segment_margin(ref, rows[:, :2], tolerance=1e-10)
    success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass']
                   and corridor['max_distance_upper_mm'] <= 1.25+1e-8)
    result = dict(success=success, reason=env.reason, audit=audit, corridor=corridor,
                  geometry=env.tracker.snapshot(1.25),
                  complete_time_s=len(rows)*.001 if success else None,
                  motion=complete_motion_metrics(ref, rows) if success else None)
    if len(ref) == 3:
        result.update(shape=complete_corner_metrics(ref, rows),
                      target_motion=target_motion(ref, rows))
    return result


def branch(policy, factory, ref, rows, actions, state, offset, costs, deadline):
    env = factory(ref)
    env.restore(copy.deepcopy(state))
    frame = env.ticks
    chunks, held, observations = [], [], []
    first = True
    while not env.done:
        if time.perf_counter() >= deadline:
            raise TimeoutError('Registered complete-suffix wall budget')
        x = env.observation()
        latent = policy.act(x, True)[0]
        if first:
            latent = latent.astype(np.float64)+offset
            first = False
        _, _, _, info = env.step(latent)
        chunks.append(info['rows'])
        held.append(info['held_action'])
        observations.append(x)
        costs['actor_calls'] += 1
        costs['physics_ticks'] += len(info['rows'])
    full = np.vstack([rows[:frame], *chunks])
    all_actions = np.vstack([actions[:frame//20], held])
    return measure(ref, full, env), full, all_actions, np.asarray(observations)


def main():
    print('INTERPRETER', sys.executable, torch.__version__, flush=True)
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919'
    model = BASE/'canonical_gate_band_nonlinear_cruise_capacity_20260918/diagnostic_policy.pt'
    source_results = read(source/'results.json')
    records = {r['path']: r for r in source_results['records']}
    paths = [10, 12, 14, 15]
    stage = register('canonical_gate_band_nonlinear_joint_response_20260919',
        [__file__, model, source/'results.json', source/'receipt.json',
         BASE/'nonlinear_feed_adapter.py', BASE/'probe_coordinated_suffix.py',
         BASE/'joint_runtime.py', BASE/'gate_band_state.py',
         BASE/'complete_corner_metrics.py', BASE/'complete_motion_metrics.py'] +
        [source/records[p]['trace'] for p in paths],
        dict(paths=paths, query_states_per_path=3, axis_branches_per_state=6,
             combined_validation_branches_per_state=2, wall_s=180,
             latent_amplitude=AMPLITUDE.tolist(), training_updates=0),
        'Reuse latest learned-feed complete traces, identify approach, pre-apex and '
        'post-apex decision states from actual XY. At each state change exactly one '
        '20ms action, then use the unchanged nonlinear policy on all actual future '
        'observations through ordered coverage, braking and rest. Check reconstructed '
        'next20ms bitwise identity. Measure +/- steering/feed/preview complete suffix '
        'effects. Choose a combined three-head descent proposal from these six effects '
        'and independently measure it at full/half amplitude; do not treat sums as '
        'exact Q gradients. Offline diagnostic, not PPO or deployed switching. '
        'No constraint change, no sealed test. If composition is unreliable, use '
        'complete learned-parameter rollouts for the next joint-learning step.')
    start = time.perf_counter()
    costs = dict(actor_calls=0, physics_ticks=0, saved_prefix_geometry_ticks=0)
    md = torch.load(model, map_location='cpu', weights_only=False)
    assert sha(model) == source_results['candidate_sha256']
    policy = NonlinearFeedAdapter()
    policy.load_state_dict(md['policy'])
    factory = checkpoint_environment_factory(md['environment_architecture'])
    output, queries, qobs, effects = [], [], [], []
    try:
        for path in paths:
            r = records[path]
            with np.load(source/r['trace']) as z:
                ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
            apex = int(np.argmin(np.linalg.norm(rows[:, :2]-ref[1], axis=1)))
            u = ref[1]-ref[0]
            u /= np.linalg.norm(u)
            remaining = (ref[1]-rows[:, :2])@u
            approach = int(np.flatnonzero(remaining <= 8.)[0])//20*20
            frames = [approach, max(approach+20, (apex//20-2)*20), (apex//20+3)*20]
            for phase, frame in zip(('approach', 'pre_apex', 'post_apex'), frames):
                env = reconstruct(ref, rows, actions, frame, costs, env_factory=factory)
                x, initial = env.observation(), env.state()
                a = policy.act(x, True)[0]
                assert np.array_equal(env.decode_action(a), actions[frame//20])
                _, _, _, info = env.step(a)
                assert np.array_equal(info['rows'], rows[frame:frame+20])
                query = len(queries)
                q = dict(query=query, path=path, phase=phase, frame=frame,
                         baseline_shape=r['shape'], baseline_time_s=r['complete_time_s'],
                         identity_next20ms_bitwise=True)
                queries.append(q)
                qobs.append(x)
                local = []
                for head, direction in itertools.product(range(3), (-1, 1)):
                    offset = np.eye(3)[head]*AMPLITUDE[head]*direction
                    result, full, held, obs = branch(policy, factory, ref, rows, actions,
                        initial, offset, costs, start+175)
                    name = f'q{query}_h{head}_{direction:+d}.npz'
                    np.savez_compressed(stage/name, reference=ref, rows=full, actions=held,
                                        suffix_observations=obs, frame=frame, offset=offset)
                    result.update(query=query, path=path, head=head, direction=direction, trace=name)
                    local.append(result)
                    output.append(result)
                # Scalar only selects diagnostic direction; retain all metric responses.
                def cost(shape):
                    return shape['full_physical_reverse_deg']/5.+shape['out_edge_outward_mm']/.05+shape['full_normalized_energy']
                gradient = np.zeros(3)
                for head in range(3):
                    lo, hi = local[2*head:2*head+2]
                    if lo['success'] and hi['success']:
                        gradient[head] = (cost(hi['shape'])-cost(lo['shape']))/2
                proposal = -AMPLITUDE*gradient/max(abs(gradient).max(), 1e-12)
                for scale in (.5, 1.):
                    result, full, held, obs = branch(policy, factory, ref, rows, actions,
                        initial, scale*proposal, costs, start+175)
                    name = f'q{query}_joint_{scale}.npz'
                    np.savez_compressed(stage/name, reference=ref, rows=full, actions=held,
                                        suffix_observations=obs, frame=frame, offset=scale*proposal)
                    actual = cost(result['shape'])-cost(r['shape']) if result['success'] else None
                    predicted = float(np.sum(gradient*proposal/AMPLITUDE)*scale)
                    result.update(query=query, path=path, joint_scale=scale,
                                  offset=(scale*proposal).tolist(), predicted_cost_change=predicted,
                                  actual_cost_change=actual, trace=name)
                    output.append(result)
                effects.append(dict(query=query, signed_cost_response=gradient.tolist()))
                write_json(stage/'results.json', dict(model_sha256=sha(model), queries=queries,
                    records=output, costs=costs, diagnostic_only=True, goal_complete=False))
                print('JOINT_RESPONSE', path, phase, 'axis_success', sum(t['success'] for t in local),
                      'pred_actual', [(t['predicted_cost_change'], t['actual_cost_change']) for t in output[-2:]], flush=True)
        np.savez_compressed(stage/'queries.npz', observations=np.asarray(qobs), amplitude=AMPLITUDE)
        receipt(stage, dict(status='completed_current_joint_action_response', costs=costs,
            queries=len(queries), branches=len(output), successes=sum(r['success'] for r in output),
            wall_s=time.perf_counter()-start, learning=False, goal_complete=False))
    except Exception as exc:
        receipt(stage, dict(status='failed_current_joint_action_response', error=str(exc),
                           costs=costs, wall_s=time.perf_counter()-start, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
