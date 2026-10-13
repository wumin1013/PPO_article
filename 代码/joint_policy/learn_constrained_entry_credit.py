"""Matched full-return parameter directions with observed local failure exclusions."""
import copy
import itertools
import time
from datetime import datetime, timezone

import numpy as np
import torch

from learn_exit_local_credit import last_layer_features, summary
from learn_entry_interaction_feedback import policy_digest
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_entry_action_interactions import score
from search_complete_return import read
from study_io import BASE, register, receipt, sha, write_json
from train_pilot import evaluate, save_torch
from joint_runtime_v5 import environment_factory


SCALE = np.array([.025, .3, .3])
SCREEN = list(range(16)) + [16, 28, 31, 40, 41]


def solve_halfspace_qp(fisher, gradient, constraints):
    """Enumerate the <=3 active inequalities per head; verify convex KKT conditions."""
    free = np.linalg.solve(fisher, gradient)
    c = np.asarray(constraints).reshape(-1, len(gradient))
    candidates = []
    for count in range(len(c) + 1):
        for active in itertools.combinations(range(len(c)), count):
            multipliers = np.zeros(len(c))
            if active:
                a = c[list(active)]
                inverse_a = np.linalg.solve(fisher, a.T)
                lam = np.linalg.lstsq(a @ inverse_a, a @ free, rcond=1e-12)[0]
                direction = free - inverse_a @ lam
                multipliers[list(active)] = lam
            else:
                direction = free.copy()
            primal = float(max(0., np.max(c @ direction))) if len(c) else 0.
            dual = float(max(0., -multipliers.min())) if len(c) else 0.
            stationary = float(abs(fisher @ direction - gradient + c.T @ multipliers).max())
            complementary = float(abs(multipliers * (c @ direction)).max()) if len(c) else 0.
            if max(primal, dual, stationary, complementary) < 1e-8:
                objective = float(.5 * direction @ fisher @ direction - gradient @ direction)
                candidates.append((objective, direction, dict(active=list(active), primal=primal,
                    dual=dual, stationary=stationary, complementary=complementary,
                    multipliers=multipliers.tolist())))
    if not candidates:
        raise ArithmeticError('No verified KKT solution for the strictly convex local QP')
    objective, direction, audit = min(candidates, key=lambda x: x[0])
    return direction, dict(audit, objective=objective, constraints=len(c))


def complete_summary(results, traces):
    result = summary(None, results, traces)
    full_cost = 0.
    for path in traces:
        with np.load(path) as z:
            if len(z['reference']) == 3:
                full_cost += -np.expm1(-score(z['reference'], z['rows'], z['actions'])['quality_full'])
    result['full_time_plus_full_shape_s'] = (
        float(result['full_time_s'] + full_cost) if result['comparable_complete_pool'] else None)
    return result


def eligible(result, parent):
    return bool(result['comparable_complete_pool'] and result['max_entry_mm'] <= .05
        and result['median_exit_mm'] <= .8 * parent['median_exit_mm']
        and result['max_exit_mm'] <= parent['max_exit_mm']
        and result['full_reverse_sum_deg'] <= parent['full_reverse_sum_deg']
        and result['full_time_plus_full_shape_s'] < parent['full_time_plus_full_shape_s'])


def main():
    torch.set_num_threads(1)
    parent = BASE / 'exit_local_credit_parameter_learning_20260916'
    model = parent / 'candidate_1.pt'
    pool = parent / 'training_1'
    probe = BASE / 'symmetric_entry_credit_probe_20260916'
    support_file = BASE / 'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    stage = register('matched_constrained_entry_credit_20260916',
        [__file__, model, parent/'receipt.json', pool/'summary.json', support_file,
         probe/'receipt.json', probe/'results.json', probe/'sensitivities.json',
         BASE/'learn_exit_local_credit.py', BASE/'learn_entry_interaction_feedback.py',
         BASE/'probe_entry_action_interactions.py', BASE/'complete_corner_metrics.py',
         BASE/'bounded_shape_return.py', BASE/'local_geometry_feedback_policy.py',
         BASE/'frame_feedback_policy.py', BASE/'reflection_policy.py', BASE/'joint_runtime_v5.py',
         BASE/'learning_preparation_v4/paths.json'],
        dict(training_evaluation_decisions=22000, development_decisions=3000,
             physics_ticks=550000, wall_s=450, new_probe_decisions=0,
             arms=['unconstrained', 'excluded_failure_directions'], steps=[.25, .5, 1.],
             screen_paths=SCREEN, full_training_paths=64, ridge=.001,
             latent_scales=SCALE.tolist(), maximum_normalized_support_change=1.),
        'Matched local complete-return parameter learning: identical parent, full-extent return '
        'finite sensitivities, all64 saved training-state covariance, query states, ridge, '
        'normalization and three step opportunities. The only arm difference is6 observed failed '
        'axis directions excluded by homogeneous linear inequalities at their original query '
        'states. These are heuristic local exclusions, not a certified feasible region. '
        'Solve strictly convex headwise QPs and check KKT residuals before physics. '
        'No action schedules or corner templates at deployment. Screen21 paths including all '
        'isolated corners and5 previously sensitive mixed paths. For each arm, choose at most '
        'one screen-eligible step by full time+full-extent two-edge bounded cost; then evaluate '
        'only the43 missing training paths. Eligibility requires every path complete, entry '
        '<=0.05mm, median exit reduction>=20%, no increase in maximum exit or complete physical '
        'reversal sum, and improved complete time+full-extent cost. Record training selection '
        'before at most one12-path development evaluation per arm. Stop the direction if no '
        'candidate qualifies; failed-prefix aggregates remain null. The20% screen is a candidate '
        'promotion criterion, not final shape acceptance. This study is not a one-factor '
        'comparison to earlier window-return updates. No sealed tests, no independent-seed claim.')
    started = time.perf_counter()
    costs = dict(evaluation_actor_calls=0, evaluation_physics_ticks=0,
                 saved_prefix_geometry_ticks=0, new_probe_decisions=0)
    saved = torch.load(model, map_location='cpu', weights_only=False)
    policy = LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(saved['policy'])
    baseline_results = read(pool/'summary.json')['results']
    baseline_traces = [pool/f'{i}.npz' for i in range(64)]
    baseline = complete_summary(baseline_results, baseline_traces)
    screen_baseline = complete_summary([dict(baseline_results[i], original_path=i) for i in SCREEN],
                                      [baseline_traces[i] for i in SCREEN])
    n = sum(r['decisions'] for r in baseline_results)
    with np.load(support_file) as z:
        observations = z['observation'][:n].copy()
        weights = z['weight'][:n].copy()*3
    assert abs(weights.sum()-1) < 1e-12
    features = last_layer_features(policy, observations)
    base_parameters = np.column_stack([saved['policy']['actor.4.weight'].numpy(),
                                       saved['policy']['actor.4.bias'].numpy()])
    with torch.no_grad():
        actual = policy.distribution(torch.tensor(observations)).mean.numpy()
    feature_error = float(abs(np.einsum('nhf,hf->nh', features, base_parameters)-actual).max())
    assert feature_error < 1e-6
    sensitivity_rows = read(probe/'sensitivities.json')['records']
    raw = read(probe/'results.json')['records']
    qobs = []
    for row in sensitivity_rows:
        match = next(r for r in raw if (r['path'], r['anchor']) == (row['path'], row['anchor']))
        with np.load(BASE/match['trace']) as z:
            qobs.append(z['observation'][0])
    qobs = np.asarray(qobs)
    qf = last_layer_features(policy, qobs)
    sensitivities = np.asarray([r['gradient'] for r in sensitivity_rows])
    support = np.concatenate([features, qf])
    directions = {}; audits = {}; matrices = []; gradients = []; constraints = []
    failed_records = [r for r in raw if not r['success']]
    for head in range(3):
        f = features[:, head]
        fisher = f.T @ (weights[:, None]*f) + .001*np.eye(f.shape[1])
        gradient = np.mean(qf[:, head]*(sensitivities[:, head]*SCALE[head])[:, None], axis=0)
        excluded = []
        for record in failed_records:
            if record['axis'] != head:
                continue
            index = next(i for i,r in enumerate(sensitivity_rows)
                         if (r['path'],r['anchor']) == (record['path'],record['anchor']))
            excluded.append(np.sign(record['delta'][head])*qf[index, head])
        matrices.append(fisher); gradients.append(gradient)
        constraints.append(np.asarray(excluded).reshape(-1, f.shape[1]))
    for arm in ('unconstrained', 'excluded_failure_directions'):
        direction = []; head_audits = []
        for head in range(3):
            c = constraints[head] if arm != 'unconstrained' else np.empty((0, 97))
            d, a = solve_halfspace_qp(matrices[head], gradients[head], c)
            direction.append(d); head_audits.append(a)
        direction = np.asarray(direction)
        magnitude = float(abs(np.einsum('nhf,hf->nh', support, direction)).max())
        direction /= max(magnitude, 1e-12)
        delta = direction*SCALE[:, None]
        gain = float(np.mean(np.sum(np.einsum('nhf,hf->nh', qf, delta)*sensitivities, axis=1)))
        exclusions = [float((constraints[h] @ direction[h]).max()) if len(constraints[h]) else 0.
                      for h in range(3)]
        assert gain > 0
        if arm != 'unconstrained': assert max(exclusions) < 1e-8
        directions[arm] = delta
        audits[arm] = dict(kkt=head_audits, local_predicted_return_gain=gain,
            maximum_excluded_direction_by_head=exclusions, unscaled_support_max=magnitude,
            normalized_support_max=float(abs(np.einsum('nhf,hf->nh', support, direction)).max()))
    np.savez_compressed(stage/'directions.npz', query_observation=qobs,
        sensitivity=sensitivities, **directions)
    write_json(stage/'direction_audit.json', dict(arms=audits, feature_error=feature_error,
        reused_support_states=n, new_support_physics=0, failed_axis_constraints=len(failed_records),
        support_source=str(support_file), original_block_rows=n, original_weight_multiplier=3))
    print('DIRECTION', audits, flush=True)
    paths = read(BASE/'learning_preparation_v4/paths.json')
    training = [np.asarray(p) for p in paths['training']]
    cache = {}; evaluations = {}; candidates = []; selections = []

    def run(indices, tag, geometry=training):
        key = (policy_digest(policy), tuple(indices), tuple(p.tobytes() for p in geometry))
        if key in cache:
            return copy.deepcopy(cache[key])
        results = evaluate(policy, [geometry[i] for i in indices], stage, tag, None,
                           costs, started+450, environment_factory)
        results = [dict(r, original_path=i) for r,i in zip(results,indices)]
        traces = [stage/tag/f'{i}.npz' for i in range(len(indices))]
        value = (results, traces)
        cache[key] = copy.deepcopy(value)
        assert costs['evaluation_actor_calls'] <= 25000
        assert costs['evaluation_physics_ticks'] <= 550000
        return value

    for arm, direction in directions.items():
        for step in (.25, .5, 1.):
            policy.load_state_dict(saved['policy'])
            with torch.no_grad():
                policy.actor[-1].weight.add_(torch.tensor(step*direction[:, :-1], dtype=torch.float32))
                policy.actor[-1].bias.add_(torch.tensor(step*direction[:, -1], dtype=torch.float32))
            name = f'{arm}_{step:g}.pt'
            save_torch(stage/name, dict(policy=policy.state_dict(), observation_dim=72,
                architecture='local_geometry_feedback', mask_endpoint=True, frame='heading',
                history_independent=True, parent_sha256=sha(model), arm=arm, step=step,
                learning_method='finite_full_suffix_return_with_matched_local_exclusion'))
            results, traces = run(SCREEN, f'{arm}_screen_{step:g}')
            evaluations[name] = (results, traces)
            row = dict(complete_summary(results, traces), arm=arm, step=step, model=name)
            row['screen_eligible'] = eligible(row, screen_baseline)
            candidates.append(row)
            write_json(stage/'screen_results.json', dict(baseline=screen_baseline, candidates=candidates, costs=costs))
            print('SCREEN', arm, step, {k:row[k] for k in ('successes','median_exit_mm',
                'max_exit_mm','max_entry_mm','full_reverse_sum_deg','screen_eligible')}, flush=True)
    for arm in directions:
        valid = [r for r in candidates if r['arm']==arm and r['screen_eligible']]
        if not valid: continue
        chosen = min(valid, key=lambda r:r['full_time_plus_full_shape_s'])
        write_json(stage/f'{arm}_screen_selection.json', dict(selected=chosen, training_only=True))
        policy.load_state_dict(torch.load(stage/chosen['model'], map_location='cpu', weights_only=False)['policy'])
        remaining = [i for i in range(64) if i not in SCREEN]
        extra_results, extra_traces = run(remaining, f'{arm}_remaining')
        old_results, old_traces = evaluations[chosen['model']]
        combined = sorted(zip(old_results+extra_results, old_traces+extra_traces), key=lambda x:x[0]['original_path'])
        results, traces = map(list, zip(*combined))
        reviewed = dict(complete_summary(results,traces), arm=arm, model=chosen['model'])
        reviewed['eligible'] = eligible(reviewed, baseline)
        write_json(stage/f'{arm}_full_training.json', dict(summary=reviewed, baseline=baseline,
            results=results, traces=[p.relative_to(BASE).as_posix() for p in traces]))
        if reviewed['eligible']: selections.append(reviewed)
    assert costs['evaluation_actor_calls'] <= 22000
    write_json(stage/'selection.json', dict(selected=selections, training_only=True, goal_complete=False))
    before = costs['evaluation_actor_calls']; development = []
    for selected in selections:
        policy.load_state_dict(torch.load(stage/selected['model'], map_location='cpu', weights_only=False)['policy'])
        results,traces = run(list(range(12)), selected['arm']+'_development',
                            [np.asarray(p) for p in paths['validation']])
        development.append(dict(complete_summary(results,traces), arm=selected['arm'], model=selected['model']))
    assert costs['evaluation_actor_calls']-before <= 3000
    write_json(stage/'results.json', dict(selected=selections, development=development, costs=costs, goal_complete=False))
    receipt(stage, dict(status='completed_matched_failure_exclusion_parameter_learning', costs=costs,
        selected_models=[r['model'] for r in selections], wall_s=time.perf_counter()-started))
    status = read(BASE/'当前执行状态.json')
    for completed in (probe,stage):
        if completed.name not in [x['stage'] for x in status['completed']]:
            status['completed'].append(dict(stage=completed.name, receipt_sha256=sha(completed/'receipt.json')))
    status['updated_utc'] = datetime.now(timezone.utc).isoformat()
    status['current'] = dict(state='review_matched_failure_exclusion_result', training_running=False,
        retained_model=model.relative_to(BASE).as_posix(), executor='unchanged_original')
    status['next'] = 'Inspect complete trajectories and update attribution from the matched local exclusion experiment. No automatic adoption.'
    write_json(BASE/'当前执行状态.json', status)


if __name__ == '__main__':
    main()
