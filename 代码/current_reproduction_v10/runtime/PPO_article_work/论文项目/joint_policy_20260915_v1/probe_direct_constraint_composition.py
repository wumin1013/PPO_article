"""Small complete-feedback checks of an all-parameter credit proposal."""
import argparse
import time
from datetime import datetime, timezone
import numpy as np
import torch
from audit_current_constraint_gradients import METRICS, SHAPE_METRICS
from audit_function_constraint_interference import segment_margin
from audit_xy_curvature_resolution import curvature_metrics
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from geometry_homotopy_policy import GeometryHomotopyPolicy
from learn_single_decision_credit import review
from train_pilot import evaluate, save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def metric_rows(results, traces, summary, metric_names=METRICS):
    shapes = {s['path']: s for s in summary['shapes']}
    output = []
    for result, trace in zip(results, traces):
        path = result['original_path']
        if not result['success']:
            output.append(dict(path=path, success=False, values=None))
            continue
        shape = shapes.get(path)
        values=dict(shape) if shape is not None else dict.fromkeys(metric_names,0.)
        if shape is not None and 'negative_signed_heading_deg' in metric_names:
            values['negative_signed_heading_deg']=max(0.,-shape['heading_min_deg'])
        if shape is not None and any('full_xy_' in name for name in metric_names):
            with np.load(trace) as z:values.update(curvature_metrics(z['reference'],z['rows']))
        values.update(segment_max_distance_mm=segment_margin(trace)['max_distance_upper_mm'],
            complete_time_s=result['completion_time_s'])
        vector=[values[name] for name in metric_names]
        output.append(dict(path=path, success=True, values=vector))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=.125)
    parser.add_argument('--source', default='direct_full_subspace_constraint_solve_20260917')
    parser.add_argument('--stage', default='direct_constraint_composition_probe_20260917')
    parser.add_argument('--all-credit-paths', action='store_true')
    args = parser.parse_args()
    assert 0 < args.scale <= .25
    torch.set_num_threads(1)
    source = BASE/args.source
    parent = BASE/'recentered_reverse_shape_credit_learning_20260917'
    selection = read(parent/'training_selection.json')
    model = parent/selection['model']
    solve = read(source/'optimizer.json')
    assert (source/'receipt.json').exists() and solve['parent_sha256'] == sha(model)
    assert solve['predicted_joint_fraction'] >= 1/15
    assert solve.get('suitable_for_small_physical_probe',True)
    assert solve.get('empirical_margin_radius_pass',True),'Proposed step exceeds the radius used for its error margin'
    with np.load(source/'direction.npz') as z:
        source_ids=z['path_id'].tolist()
        geometry_delta=float(z['geometry_mix_delta']) if 'geometry_mix_delta' in z else 0.
    if abs(geometry_delta)<1e-10:geometry_delta=0.
    ids = source_ids if args.all_credit_paths else [0, 3, 5, 7, 8, 12, 13, 40]
    call_budget,tick_budget,wall_budget=(5000,125000,120) if args.all_credit_paths else (2500,60000,90)
    stage = register(args.stage,
        [__file__, source/'direction.npz', source/'optimizer.json', source/'receipt.json',
         parent/'training_selection.json', model, BASE/'train_pilot.py',
         BASE/'kernel_local_feedback_policy.py', BASE/'joint_runtime_v5.py',
         BASE/'geometry_homotopy_policy.py',BASE/'continuous_tangent_policy.py',
         BASE/'audit_current_constraint_gradients.py', BASE/'audit_function_constraint_interference.py',
         BASE/'audit_xy_curvature_resolution.py',
         BASE/'learn_single_decision_credit.py'],
        dict(paths=ids, parameter_scales=[-args.scale, args.scale], actor_calls=call_budget,
             physics_ticks=tick_budget, wall_s=wall_budget, optimizer_steps=0),
        ('The19-route full-space probe includes both formerly frozen paths16/31. ' if len(source_ids)==19 else '')+
        ('A positive geometry blend is part of this learned parameter direction. The negative diagnostic sign extrapolates that feature blend below0 only to measure local response; it is not an admissible learned candidate or deployment. Both signs still use unchanged physical acceptance. ' if geometry_delta else '')+
        ('Require each individual exit and reverse metric to be nonworse too, because this proposal explicitly added per-corner constraints. ' if solve.get('each_corner_exit_and_reverse_nonincrease',False) else '')+
        ('Extend the corrected-direction validation to all17 credit routes, including all16 isolated corners. ' if args.all_credit_paths else '')+
        'Validate one proposed shared-feedback parameter direction at small equal positive '
        'and negative scales on eight TRAINING routes covering smoothness constraints, '
        'recent single-axis failures, and previous head coupling. Reuse saved nominal '
        'traces. Measure full completion, both signed edge excursions, physical reverse, '
        'spatial curvature peak/energy, continuous corridor margin and whole-route time. '
        'Failed prefixes have no comparable complete shape/time vector. Compare actual '
        'central and positive responses against current all-time credit. These are '
        'parameter interventions, not promoted learned performance. Proceed to larger '
        'training-only candidates only if the direction improves actual aggregate shape '
        'and its predicted per-route smoothness/entry preservation survives this check; '
        'otherwise diagnose the combination error before any further update. No '
        'development or sealed paths, executor edits, teacher labels or runtime schedules.')
    started = time.perf_counter()
    costs = dict(evaluation_actor_calls=0, evaluation_physics_ticks=0)
    with np.load(source/'direction.npz') as z:
        direction = z['direction'].copy()
        predicted = dict(zip(z['path_id'].tolist(), z['predicted_changes'].tolist()))
        metric_names=z['metric_names'].tolist()
    saved = torch.load(model, map_location='cpu', weights_only=False)
    policy = GeometryHomotopyPolicy(256) if geometry_delta else KernelLocalFeedbackPolicy(256)
    old_results = [selection['results'][i] for i in ids]
    old_traces = [BASE/selection['traces'][i] for i in ids]
    baseline = review(old_results, old_traces)
    base_rows = metric_rows(old_results, old_traces, baseline, metric_names)
    geometry = read(BASE/'learning_preparation_v4/paths.json')['training']
    status_file = BASE/'当前执行状态.json'
    status = read(status_file)
    status['running'] = status['active_experiment'] = stage.name
    status['current'].update(state='direct_constraint_composition_running',
        training_running=False, physical_diagnostic_running=True, diagnostic_stage=stage.name)
    status['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(status_file, status)
    candidates = []
    for sign in (-1, 1):
        coefficient = sign*args.scale
        missing=policy.load_state_dict(saved['policy'],strict=not bool(geometry_delta))
        if geometry_delta:assert missing.missing_keys==['geometry_mix'] and not missing.unexpected_keys
        with torch.no_grad():
            policy.residual_coefficients.add_(torch.tensor(coefficient*direction, dtype=torch.float32))
            if geometry_delta:policy.geometry_mix.fill_(coefficient*geometry_delta)
        tag = f'parameter_{coefficient:+g}'
        save_torch(stage/f'{tag}.pt', dict(policy=policy.state_dict(), architecture='geometry_homotopy_kernel' if geometry_delta else 'kernel_local_feedback',
            centers=256, parent_sha256=sha(model), coefficient=coefficient,
            geometry_mix=coefficient*geometry_delta,
            diagnostic_only=True, learning_method='current_temporal_credit_direct_parameter_probe'))
        results = evaluate(policy, [np.asarray(geometry[i]) for i in ids], stage, tag,
            None, costs, started+wall_budget, environment_factory)
        assert costs['evaluation_actor_calls'] <= call_budget and costs['evaluation_physics_ticks'] <= tick_budget
        results = [dict(r, original_path=i) for r, i in zip(results, ids)]
        traces = [stage/tag/f'{j}.npz' for j in range(len(ids))]
        result = review(results, traces)
        row = dict(coefficient=coefficient, metrics=metric_rows(results, traces, result, metric_names),
            summary=result, results=results, traces=[str(t.relative_to(BASE)) for t in traces])
        candidates.append(row)
        write_json(stage/'results.json', dict(baseline=baseline, baseline_metrics=base_rows,
            predicted_per_unit=predicted, metric_names=metric_names, candidates=candidates, costs=costs,
            goal_complete=False, promoted=False))
        print('DIRECT_COMPOSITION', coefficient, {k: result[k] for k in
            ('successes', 'entry_sum_mm', 'exit_sum_mm', 'full_reverse_sum_deg', 'full_time_s')}, flush=True)
    comparisons = []
    for j, path in enumerate(ids):
        lo, hi = [r['metrics'][j] for r in candidates]
        if not (lo['success'] and hi['success']):
            comparisons.append(dict(path=path, complete_pair=False))
            continue
        old = np.asarray(base_rows[j]['values'])
        actual = (np.asarray(hi['values'])-np.asarray(lo['values']))/(2*args.scale)
        comparisons.append(dict(path=path, complete_pair=True,
            predicted_per_unit=predicted[path], actual_central_per_unit=actual.tolist(),
            actual_positive_change=(np.asarray(hi['values'])-old).tolist()))
    write_json(stage/'comparison.json', dict(metric_names=metric_names, paths=comparisons,
        direction_is_only_a_proposal=True, goal_complete=False))
    receipt(stage, dict(status='completed_small_direct_parameter_composition_check',
        costs=costs, wall_s=time.perf_counter()-started, promoted=False))
    status = read(status_file)
    status['running'] = status['active_experiment'] = None
    status['current'].update(state='direct_constraint_composition_complete',
        physical_diagnostic_running=False)
    status['completed'].append(dict(stage=stage.name, receipt_sha256=sha(stage/'receipt.json')))
    status['next_action'] = 'Analyze actual combination responses before larger candidate learning; preserve working model.'
    status['updated_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(status_file, status)


if __name__ == '__main__':
    main()
