"""Offline two-factor attribution of a learned feed residual's shape regression."""
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
    fitted = BASE/'canonical_gate_band_nonlinear_cruise_capacity_20260918'
    full = BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919'
    model = BASE/read(parent/'adoption.json')['model']
    candidate = fitted/'diagnostic_policy.pt'
    selected = read(parent/'results.json')['full_candidate']
    indexed = {r['original_path']: (r, BASE/t) for r, t in zip(selected['results'], selected['traces'])}
    full_records = {r['path']: r for r in read(full/'results.json')['records']}
    paths = [12, 13, 14, 15]
    stage = register('canonical_gate_band_nonlinear_feed_phase_probe_20260919',
        [__file__, model, candidate, BASE/'nonlinear_feed_adapter.py', parent/'results.json',
         parent/'adoption.json', full/'results.json', full/'receipt.json',
         BASE/'prepare_physical_update_geometry.py'] + [indexed[p][1] for p in paths] +
        [full/full_records[p]['trace'] for p in paths],
        dict(paths=paths, arms=['upstream_only', 'downstream_only'], new_path_attempts=8,
             reused_paths=8, landmark_before_vertex_mm=12.4, actor_calls=3000,
             physics_ticks=80000, wall_s=180),
        'The nonlinear learned feed function produced real curved-route cruise '
        'but worsened sharp-corner shape, despite unchanged steering/preview '
        'FUNCTIONS. Isolate upstream state/decision-phase propagation from '
        'downstream feed-function changes on four complete sharp training corners. '
        'A single irreversible phase switch occurs at the first20ms decision '
        'with incoming longitudinal remaining distance<=12.4mm, the already '
        'declared curved-interior training-label clearance. Cross two binary '
        'factors: learned feed residual before this landmark and after it. '
        'Reuse00 original86cd and11 fully learned candidate; run10 and01 only. '
        'Always retain exact learned steering/preview functions and original '
        'physical/coverage/braking/rest. The switch is an OFFLINE intervention '
        'for attribution, NEVER a deployable speed schedule or adopted policy. '
        'Event times depend on each branch, so report conditional factorial '
        'effects and interaction, not additive linear decomposition. Record '
        'entry state and full raw traces, no incomplete-prefix efficiency. '
        'Stop after8 branches; no training or model adoption.')
    start = time.perf_counter()
    costs = dict(actor_calls=0, baseline_state_queries=0, physics_ticks=0)
    old = CanonicalGatePolicy(256)
    old.load_state_dict(torch.load(model, map_location='cpu', weights_only=False)['policy'])
    md = torch.load(candidate, map_location='cpu', weights_only=False)
    assert md['parent_sha256'] == sha(model)
    learned = NonlinearFeedAdapter()
    learned.load_state_dict(md['policy'])
    factory = checkpoint_environment_factory(md['environment_architecture'])
    records = []
    try:
        for path in paths:
            original_record, original_trace = indexed[path]
            with np.load(original_trace) as z:
                ref, nominal = [z[k].copy() for k in ('reference', 'rows')]
            incoming = ref[1]-ref[0]
            incoming /= np.linalg.norm(incoming)
            for arm in ('upstream_only', 'downstream_only'):
                env = factory(ref)
                chunks, actions, phases = [], [], []
                passed = False
                entry = None
                while not env.done:
                    if time.perf_counter()-start >= 175 or costs['actor_calls'] >= 3000 or costs['physics_ticks'] >= 77800:
                        raise TimeoutError('Registered phase-probe budget')
                    remaining = float((ref[1]-env.position)@incoming)
                    if not passed and remaining <= 12.4:
                        passed = True
                        entry = dict(tick=env.ticks, remaining_mm=remaining,
                            position=env.position.tolist(), velocity=env.velocity.tolist(),
                            acceleration=env.acceleration.tolist(), filtered_heading_rad=env.heading)
                    x = env.observation()
                    a = learned.act(x, True)[0]
                    b = old.act(x, True)[0]
                    costs['actor_calls'] += 1
                    costs['baseline_state_queries'] += 1
                    assert np.array_equal(a[[0, 2]], b[[0, 2]])
                    use_learned = not passed if arm == 'upstream_only' else passed
                    _, _, _, info = env.step(a if use_learned else b)
                    chunks.append(info['rows'])
                    actions.append(info['held_action'])
                    phases.append(use_learned)
                    costs['physics_ticks'] += len(info['rows'])
                rows = np.vstack(chunks)
                audit = difference_audit(ref[0], rows)
                corridor = command_segment_margin(ref, rows[:, :2], tolerance=1e-10)
                success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'] and
                               corridor['max_distance_upper_mm'] <= 1.25+1e-8)
                name = f'path{path}_{arm}.npz'
                np.savez_compressed(stage/name, reference=ref, rows=rows,
                                    actions=np.asarray(actions), learned_feed_active=np.asarray(phases))
                r = dict(path=path, arm=arm, success=success, reason=env.reason, trace=name,
                    phase_entry=entry, audit=audit, corridor=corridor, geometry=env.tracker.snapshot(1.25),
                    complete_time_s=len(rows)*.001 if success else None,
                    shape=complete_corner_metrics(ref, rows) if success else None,
                    motion=complete_motion_metrics(ref, rows) if success else None,
                    target_motion=target_motion(ref, rows) if success else None)
                records.append(r)
                write_json(stage/'results.json', dict(parent_sha256=sha(model), candidate_sha256=sha(candidate),
                    records=records, costs=costs, offline_intervention_only=True,
                    intervention_deployed=False, learned_candidate_adopted=False, goal_complete=False))
                print('FEED_PHASE', path, arm, success, None if not success else
                    {k: r['shape'][k] for k in ('full_physical_reverse_deg', 'out_edge_outward_mm',
                    'full_normalized_energy', 'apex_speed_mm_s', 'cruise_ge99_s')}, flush=True)
        receipt(stage, dict(status='completed_nonlinear_feed_phase_probe', costs=costs,
            successes=sum(r['success'] for r in records), wall_s=time.perf_counter()-start,
            intervention_deployed=False, learned_candidate_adopted=False, goal_complete=False))
    except Exception as error:
        receipt(stage, dict(status='failed_nonlinear_feed_phase_probe', error=str(error), costs=costs,
            completed_paths=len(records), wall_s=time.perf_counter()-start, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
