"""Audit saved hybrid predictions, including all errors and branch mismatches."""
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from event_adaptive_terminal_prediction import predict_terminal
from transition_waveform_model import chain_responses
from joint_runtime import terminal
from geometry_spec import Corridor
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def distances(corridor, xy):
    relative = xy[:, None, :] - corridor.starts
    fraction = np.clip(np.sum(relative * corridor.vectors, axis=2) /
                       corridor.lengths**2, 0., 1.)
    residual = relative - fraction[:, :, None] * corridor.vectors
    return np.linalg.norm(residual, axis=2).min(axis=1)


def profile_check(state, count):
    v, a, j = terminal.discrete_planar_stop_profile(state[2:4], state[4:6], .001, count)
    limits = np.array([100., 2000., 62500.*terminal.PlanarBrakingPolicy().limit_fraction])
    peaks = np.array([np.linalg.norm(x[1:], axis=1).max() for x in (v, a, j)])
    nonreverse = bool(np.linalg.norm(state[2:4]) <= 1e-12 or
                      np.all(v @ state[2:4] >= -1e-12))
    return dict(peaks=peaks.tolist(), normalized_peaks=(peaks/limits).tolist(),
                feasible=bool(np.all(peaks <= limits) and nonreverse), nonreverse=nonreverse)


def main():
    source = BASE/'canonical_gate_band_segment_cruise_credit_20260918'
    reference = BASE/'canonical_gate_band_segment_cruise_reference_20260918'
    prior = BASE/'canonical_gate_band_event_adaptive_terminal_prediction_20260918'
    data, physical, previous = [read(p/'results.json') for p in (source, reference, prior)]
    parent = BASE/data['parent_model']
    assert sha(parent) == previous['parent_sha256'] == physical['parent_sha256']
    factory = checkpoint_environment_factory(torch.load(parent, map_location='cpu',
                                            weights_only=False)['architecture'])
    stage = register('canonical_gate_band_terminal_prediction_residuals_20260918',
        [__file__, BASE/'event_adaptive_terminal_prediction.py', source/'results.json',
         reference/'results.json', reference/'basis.npz', prior/'results.json',
         prior/'receipt.json', parent],
        dict(new_actor_decisions=0, new_physics_ticks=0, wall_s=300,
             terminal_planner_queries=2500, saved_responses=116),
        'Reuse every saved response to audit ordinary-prefix XY and v/a error, '
        'corridor margins, terminal-tail containment and all two mode mismatches. '
        'For mismatches interpolate predicted to actual splice v/a and inspect '
        'the exact cubic-duration feasibility boundary. This is retrospective '
        'diagnosis, not a calibrated uncertainty bound or a learned policy. '
        'Do not remove guards or train using old-center credit.')
    start = time.perf_counter()
    costs = dict(new_actor_decisions=0, new_physics_ticks=0, terminal_planner_queries=0)
    def budget():
        assert time.perf_counter()-start < 295 and costs['terminal_planner_queries'] < 2500
    with np.load(reference/'basis.npz') as z:
        directions = z['basis']*z['amplitudes'][:, None]
    records = []
    for index, item in enumerate(data['records']):
        budget()
        with np.load(source/item['file']) as z:
            local = {k: z[k].copy() for k in z.files}
        nominal, ref = local['rows'], local['reference']
        stop = int(np.flatnonzero(nominal[:, 12])[0])
        responses = chain_responses(local, directions)
        corridor = Corridor(ref, closed=bool(np.linalg.norm(ref[0]-ref[-1]) < 1e-9))
        for candidate in physical['candidates']:
            with np.load(BASE/candidate['traces'][index]) as z:
                actual = z['rows'].copy()
            affine = nominal[:, :6] + candidate['sign']*responses[:, :, candidate['direction']]
            pred = predict_terminal(ref, affine, stop, factory, costs, budget)
            assert pred['supported']
            actual_stop = int(np.flatnonzero(actual[:, 12])[0])
            actual_count = len(actual)-actual_stop-23
            n = min(actual_stop, pred['initiation_tick'])
            error = np.linalg.norm(actual[:n, :2]-affine[:n, :2], axis=1)
            pred_margin = 1.25-distances(corridor, affine[:n, :2])
            actual_margin = 1.25-distances(corridor, actual[:n, :2])
            mode_matches = actual_stop == pred['initiation_tick'] and actual_count == pred['braking_steps']
            tail = pred['rows'][pred['initiation_tick']-1:, :2]
            tail_inside = all(corridor.segment_inside(a, b) for a, b in zip(tail[:-1], tail[1:]))
            record = dict(path=item['path'], direction=candidate['direction'], sign=candidate['sign'],
                mode_matches=mode_matches, predicted_prefix_inside=pred['predicted_prefix_inside'],
                predicted_tail_inside=tail_inside, ordinary_common_ticks=n,
                ordinary_maximum_xy_error_mm=float(error.max()),
                ordinary_rms_xy_error_mm=float(np.sqrt(np.mean(error**2))),
                ordinary_maximum_v_error=float(np.linalg.norm(actual[:n, 2:4]-affine[:n, 2:4], axis=1).max()),
                ordinary_maximum_a_error=float(np.linalg.norm(actual[:n, 4:6]-affine[:n, 4:6], axis=1).max()),
                minimum_predicted_node_margin_mm=float(pred_margin.min()),
                minimum_actual_node_margin_mm=float(actual_margin.min()),
                minimum_predicted_margin_minus_observed_error_mm=float((pred_margin-error).min()),
                maximum_terminal_xy_error_mm=float(np.linalg.norm(actual[-1, :2]-pred['rows'][-1, :2])),
                predicted_full_node_margin_mm=float((1.25-distances(corridor, pred['rows'][:, :2])).min()))
            if not mode_matches:
                assert actual_stop == pred['initiation_tick']
                initial_actual, initial_pred = actual[actual_stop-1, :6], affine[actual_stop-1, :6]
                count = pred['braking_steps']
                predicted_profile, actual_profile = profile_check(initial_pred, count), profile_check(initial_actual, count)
                assert predicted_profile['feasible'] and not actual_profile['feasible']
                lower, upper = 0., 1.
                for _ in range(45):
                    middle = (lower+upper)/2.
                    if profile_check(initial_pred+middle*(initial_actual-initial_pred), count)['feasible']:
                        lower = middle
                    else:
                        upper = middle
                record['mismatch'] = dict(initiation_tick=actual_stop,
                    predicted_braking_steps=count, actual_braking_steps=actual_count,
                    splice_xy_error_mm=float(np.linalg.norm(initial_actual[:2]-initial_pred[:2])),
                    splice_v_error=float(np.linalg.norm(initial_actual[2:4]-initial_pred[2:4])),
                    splice_a_error=float(np.linalg.norm(initial_actual[4:6]-initial_pred[4:6])),
                    shorter_duration_at_predicted_state=predicted_profile,
                    shorter_duration_at_actual_state=actual_profile,
                    actual_selected_duration=profile_check(initial_actual, actual_count),
                    boundary_fraction_toward_actual_state=upper,
                    boundary_v_displacement=float(upper*np.linalg.norm(initial_actual[2:4]-initial_pred[2:4])),
                    boundary_a_displacement=float(upper*np.linalg.norm(initial_actual[4:6]-initial_pred[4:6])),
                    cause='Same initiation, v/a prediction crosses integer cubic-duration feasibility boundary; XY does not enter duration selection.')
            records.append(record)
    errors = np.array([r['ordinary_maximum_xy_error_mm'] for r in records])
    summary = dict(saved_responses=len(records), exact_mode_matches=sum(r['mode_matches'] for r in records),
        predicted_prefix_inside=sum(r['predicted_prefix_inside'] for r in records),
        predicted_tail_inside=sum(r['predicted_tail_inside'] for r in records),
        ordinary_max_xy_error_quantiles_mm=np.quantile(errors, [0., .5, .9, .99, 1.]).tolist(),
        observed_error_exceeds_node_margin=sum(r['minimum_predicted_margin_minus_observed_error_mm'] < 0 for r in records),
        worst_ordinary_response=max(records, key=lambda r:r['ordinary_maximum_xy_error_mm']),
        mismatches=[r for r in records if not r['mode_matches']],
        predictor_safe_for_unchecked_learning=False,
        interpretation='Terminal branch prediction improves many tails but can choose an infeasible shorter duration at the actual splice. Observed errors are retrospective and are not prospective confidence bounds.')
    write_json(stage/'results.json', dict(parent_sha256=sha(parent), records=records, summary=summary,
                                        costs=costs, goal_complete=False))
    receipt(stage, dict(status='completed_saved_prediction_residual_audit', costs=costs,
                        wall_s=time.perf_counter()-start, goal_complete=False))
    state_path = BASE/'当前执行状态.json'
    state = read(state_path)
    state['current_terminal_mode_efficiency_audit'].update(prediction_status='completed_with_two_duration_mismatches',
        exec_session_id=None, residual_audit=stage.name, active_experiment_processes=[],
        next='Use explicit branch uncertainty or physical validation; do not treat 114/116 as universal predictor acceptance.')
    write_json(state_path, state)
    print('RESIDUAL_AUDIT', summary, flush=True)


if __name__ == '__main__':
    main()
