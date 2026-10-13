"""Finite complete-path test of saved-state feed fitting, never automatic adoption."""
import time
import numpy as np
import torch
from nonlinear_feed_adapter import NonlinearFeedAdapter
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from probe_linear_action_hold import target_motion
from audit_function_constraint_interference import command_segment_margin
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    parent = BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    fit = BASE/'canonical_gate_band_nonlinear_cruise_capacity_20260918'
    adoption = read(parent/'adoption.json')
    original_model = BASE/adoption['model']
    candidate_model = fit/'diagnostic_policy.pt'
    assert sha(original_model) == adoption['sha256'] == read(fit/'results.json')['parent_sha256']
    pool = read(parent/'results.json')['full_candidate']
    indexed = {r['original_path']: (r, BASE/t) for r, t in zip(pool['results'], pool['traces'])}
    paths = list(range(16)) + [19, 31, 48, 57]
    stage = register('canonical_gate_band_nonlinear_feed_closed_loop_20260919',
        [__file__, BASE/'nonlinear_feed_adapter.py', BASE/'canonical_gate_state.py',
         BASE/'gate_band_state.py', BASE/'joint_runtime.py', BASE/'complete_corner_metrics.py',
         BASE/'complete_motion_metrics.py', candidate_model, original_model,
         parent/'results.json', parent/'adoption.json', fit/'results.json', fit/'receipt.json'] +
        [indexed[p][1] for p in paths],
        dict(paths=paths, new_complete_path_attempts=20, actor_calls=6000,
             physics_ticks=140000, additional_saved_state_queries=6000, wall_s=180),
        'Current86cd matched nonlinear capacity fitting reduced the saved-state maximum '
        'feed violation from19.53 to1.062mm/s but did NOT satisfy all inequalities. '
        'Test its full learned feed residual on16 existing isolated corners,2 curved '
        'training routes and2 pure straights. Reuse86cd baseline full trajectories. '
        'No online cruise labels, feed overrides, angle tables, templates, filters '
        'or physical-limit changes. Steering/preview FUNCTIONS are bitwise checked '
        'against86cd at every visited observation; their closed-loop outputs may '
        'change because feed changes future states. Each path finishes or retains '
        'its actual failure, including original coverage/braking/rest. Record raw '
        '1msXY, actual held actions, observations, shape/heading/curvature, full time '
        'and actual cruise. Failed prefixes are never efficiency evidence. This '
        'isolates the feed-function change, not the architecture from its fitting '
        'algorithm. A shape or completion regression rejects adoption and stops '
        'unchanged fitting; preserved shape plus actual curved-route cruise would '
        'justify broader joint-learning work, not final success or automatic adoption.')
    start = time.perf_counter()
    costs = dict(actor_calls=0, physics_ticks=0, baseline_queries_on_visited_states=0)
    metadata = torch.load(candidate_model, map_location='cpu', weights_only=False)
    assert metadata['parent_sha256'] == sha(original_model)
    policy = NonlinearFeedAdapter()
    policy.load_state_dict(metadata['policy'])
    original = CanonicalGatePolicy(256)
    original.load_state_dict(torch.load(original_model, map_location='cpu', weights_only=False)['policy'])
    factory = checkpoint_environment_factory(metadata['environment_architecture'])
    records = []
    try:
        for path in paths:
            old, trace = indexed[path]
            with np.load(trace) as z:
                ref, nominal = [z[k].copy() for k in ('reference', 'rows')]
            env = factory(ref)
            chunks, actions, observations, previous_outputs = [], [], [], []
            while not env.done:
                if time.perf_counter()-start >= 175 or costs['actor_calls'] >= 6000 or costs['physics_ticks'] >= 137800:
                    raise TimeoutError('Registered finite evaluation budget')
                x = env.observation()
                latent = policy.act(x, True)[0]
                baseline_latent = original.act(x, True)[0]
                costs['actor_calls'] += 1
                costs['baseline_queries_on_visited_states'] += 1
                assert np.array_equal(latent[[0, 2]], baseline_latent[[0, 2]])
                _, _, _, info = env.step(latent)
                chunks.append(info['rows'])
                actions.append(info['held_action'])
                observations.append(x)
                previous_outputs.append(env.decode_action(baseline_latent))
                costs['physics_ticks'] += len(info['rows'])
            rows = np.vstack(chunks)
            audit = difference_audit(ref[0], rows)
            corridor = command_segment_margin(ref, rows[:, :2], tolerance=1e-10)
            success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'] and
                           corridor['max_distance_upper_mm'] <= 1.25+1e-8)
            name = f'path{path}.npz'
            np.savez_compressed(stage/name, reference=ref, rows=rows, actions=np.asarray(actions),
                                observations=np.asarray(observations),
                                baseline_actions_at_visited_states=np.asarray(previous_outputs))
            r = dict(path=path, success=success, reason=env.reason, trace=name,
                     geometry=env.tracker.snapshot(1.25), audit=audit, corridor=corridor,
                     complete_time_s=len(rows)*.001 if success else None,
                     complete_time_change_s=len(rows)*.001-old['completion_time_s'] if success else None,
                     baseline_motion=complete_motion_metrics(ref, nominal),
                     motion=complete_motion_metrics(ref, rows) if success else None,
                     steering_preview_functions_unchanged=True)
            if len(ref) == 3:
                r['baseline_shape'] = complete_corner_metrics(ref, nominal)
                r['shape'] = complete_corner_metrics(ref, rows) if success else None
                r['baseline_target_motion'] = target_motion(ref, nominal)
                r['target_motion'] = target_motion(ref, rows) if success else None
            records.append(r)
            write_json(stage/'results.json', dict(parent_sha256=sha(original_model),
                candidate_sha256=sha(candidate_model), records=records, costs=costs,
                diagnostic_only=True, learned_candidate_adopted=False, goal_complete=False))
            print('NONLINEAR_CLOSED_LOOP', path, success, r['complete_time_change_s'],
                  None if not success else r['motion']['cruise_ge99_s'],
                  None if not success or len(ref) != 3 else {k: r['shape'][k]-r['baseline_shape'][k]
                  for k in ('in_edge_outward_mm', 'out_edge_outward_mm', 'full_physical_reverse_deg',
                            'full_normalized_energy')}, flush=True)
        receipt(stage, dict(status='completed_nonlinear_feed_closed_loop', costs=costs,
            successes=sum(r['success'] for r in records), wall_s=time.perf_counter()-start,
            learned_candidate_adopted=False, goal_complete=False))
    except Exception as error:
        receipt(stage, dict(status='failed_nonlinear_feed_closed_loop', error=str(error), costs=costs,
            completed_paths=len(records), wall_s=time.perf_counter()-start, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
