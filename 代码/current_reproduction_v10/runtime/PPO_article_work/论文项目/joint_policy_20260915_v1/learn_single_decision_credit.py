"""Matched bounded policy updates from same-state, single-decision consequences."""
import time
import numpy as np
import torch

from learn_exit_local_credit import last_layer_features, summary
from learn_constrained_entry_credit import solve_halfspace_qp
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_one_decision_credit import AMPLITUDE, score_complete
from search_complete_return import read
from study_io import BASE, register, receipt, sha, write_json
from train_pilot import evaluate, save_torch
from complete_motion_metrics import complete_motion_metrics
from joint_runtime_v5 import environment_factory


SCREEN = list(range(16)) + [16, 28, 31, 40, 41]
ARMS = ('original_full_return', 'independent_zero_return')


def review(results, traces):
    value = summary(None, results, traces)
    scores = [];motion=[]
    for result, trace in zip(results, traces):
        with np.load(trace) as z:
            motion.append(dict(path=result['original_path'],success=result['success'],
                **complete_motion_metrics(z['reference'],z['rows'])))
            scores.append(dict(path=result['original_path'], success=result['success'],
                **score_complete(z['reference'], z['rows'], z['actions'])))
    good = value['comparable_complete_pool']
    for arm in ARMS:
        value[arm] = sum(s[arm] for s in scores) if good else None
    value['entry_sum_mm'] = sum(s['in_edge_outward_mm'] for s in value['shapes']) if good else None
    value['exit_sum_mm'] = sum(s['out_edge_outward_mm'] for s in value['shapes']) if good else None
    value['motion']=motion
    value['complete_cruise_s']={str(threshold):sum(r[f'cruise_ge{threshold}_s'] for r in motion)
        if good else None for threshold in (95,98,99)}
    value['speed_window_definition']='Original1ms vector speed; cruise includes the full recorded path and rest. Before/after-apex speeds use5mm of actual travelled arc about closest vertex approach, not learned entry/exit points.'
    return value


def adoption_checks(value, baseline):
    if not value['comparable_complete_pool']:
        return dict(all_complete=False)
    old = {s['path']: s for s in baseline['shapes']}
    return dict(all_complete=True,
        each_entry_nonworse=all(s['in_edge_outward_mm'] <= old[s['path']]['in_edge_outward_mm']+1e-6
            for s in value['shapes']),
        entry_sum_reduced_ten_percent=value['entry_sum_mm'] <= .9*baseline['entry_sum_mm'],
        exit_sum_reduced_ten_percent=value['exit_sum_mm'] <= .9*baseline['exit_sum_mm'],
        max_exit_nonworse=value['max_exit_mm'] <= baseline['max_exit_mm']+1e-6,
        full_reverse_nonworse=value['full_reverse_sum_deg'] <= baseline['full_reverse_sum_deg']+1e-6,
        complete_time_within_two_percent=value['full_time_s'] <= 1.02*baseline['full_time_s'],
        independent_complete_return_improved=value['independent_zero_return'] > baseline['independent_zero_return'])


def main():
    torch.set_num_threads(1)
    parent = BASE/'exit_local_credit_parameter_learning_20260916'
    model = parent/'candidate_1.pt'
    pool = parent/'training_1'
    probe = BASE/'single_decision_complete_credit_probe_20260917'
    support_file = BASE/'matched_entry_interaction_feedback_20260916/window_training_data.npz'
    assert read(BASE/'single_decision_credit_signal_audit_20260917/results.json')['decision']['proceed_to_bounded_parameter_test']
    stage = register('matched_single_decision_credit_learning_20260917',
        [__file__, model, parent/'receipt.json', pool/'summary.json', support_file,
         probe/'receipt.json', probe/'results.json', probe/'sensitivities.json', probe/'queries.npz',
         BASE/'learn_exit_local_credit.py', BASE/'learn_constrained_entry_credit.py',
         BASE/'probe_one_decision_credit.py', BASE/'complete_corner_metrics.py',
         BASE/'local_geometry_feedback_policy.py', BASE/'joint_runtime_v5.py',
         BASE/'train_pilot.py', BASE/'learning_preparation_v4/paths.json'],
        dict(evaluation_actor_calls=25000, evaluation_physics_ticks=600000, wall_s=450,
             new_probe_decisions=0, steps=[.25, .5, 1.], arms=list(ARMS), screen_paths=SCREEN,
             maximum_normalized_saved_state_change=1., scales=AMPLITUDE.tolist(), ridge=.001),
        'Test whether same-state20ms complete-consequence credit can repair both entry and exit '
        'in the retained policy. Two matched arms differ only in the explicitly bundled shape '
        'objective: original full max-edge cost with allowances versus independent edge sum '
        'without position/reversal allowances. Identical160 queries,4177 original states, '
        'last-layer291 parameters, path-total gradient quadrature, ridge, one observed failed '
        'axis exclusion and three step opportunities. Complete-side gradients retain failed '
        'directions as local exclusion constraints; no claim of global safety or exact derivatives. '
        'Original trajectories are reused. Screen21 existing training paths; require every '
        'entry maximum nonworse within1e-6mm numerical slack, entry and exit sums each reduced '
        'at least10%, no increase in max exit or total full reversal, complete time<=102% of '
        'parent and improved independent complete return. These are promotion conditions, '
        'not final shape acceptance. At most one chosen step per arm receives missing43 '
        'training paths, then at most one12-path development run after recorded full-training '
        'selection. Stop this direction if no step qualifies. Do not redeploy interventions, '
        'change executor, train on development, or access sealed tests. Failed-prefix '
        'aggregate metrics are null. This differs in credit support and duration from prior studies.')
    started = time.perf_counter()
    costs = dict(evaluation_actor_calls=0, evaluation_physics_ticks=0, new_probe_decisions=0)
    saved = torch.load(model, map_location='cpu', weights_only=False)
    policy = LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(saved['policy'])
    base_results = [dict(r, original_path=i) for i, r in enumerate(read(pool/'summary.json')['results'])]
    base_traces = [pool/f'{i}.npz' for i in range(64)]
    baseline = review(base_results, base_traces)
    screen_baseline = review([base_results[i] for i in SCREEN], [base_traces[i] for i in SCREEN])
    n = sum(r['decisions'] for r in base_results)
    with np.load(support_file) as z:
        observations = z['observation'][:n].copy()
        weights = z['weight'][:n].copy()*3
    assert n == 4177 and abs(weights.sum()-1.) < 1e-12
    with np.load(probe/'queries.npz') as z:
        qobs = z['observation'].copy()
    features = last_layer_features(policy, observations)
    qf = last_layer_features(policy, qobs)
    support = np.concatenate([features, qf])
    base_parameters = np.column_stack([saved['policy']['actor.4.weight'].numpy(),
                                      saved['policy']['actor.4.bias'].numpy()])
    with torch.no_grad():
        actual = policy.distribution(torch.tensor(observations)).mean.numpy()
    error = float(abs(np.einsum('nhf,hf->nh', features, base_parameters)-actual).max())
    assert error < 1e-6
    records = read(probe/'sensitivities.json')['records']
    query_weight = np.asarray([r['path_objective_gradient_weight'] for r in records])
    failed = [r for r in read(probe/'results.json')['records'] if not r['success']]
    directions = {}; audits = {}
    for arm in ARMS:
        sensitivity = np.asarray([r['objectives'][arm]['complete_side_gradient'] for r in records])
        heads = []; kkt = []
        for head in range(3):
            f = features[:, head]
            fisher = f.T @ (weights[:, None]*f) + .001*np.eye(f.shape[1])
            gradient = np.sum(qf[:, head]*(query_weight*sensitivity[:, head]*AMPLITUDE[head])[:, None], axis=0)
            constraints = np.asarray([r['direction']*qf[r['query'], head]
                for r in failed if r['axis'] == head]).reshape(-1, f.shape[1])
            direction, audit = solve_halfspace_qp(fisher, gradient, constraints)
            heads.append(direction); kkt.append(audit)
        direction = np.asarray(heads)
        maximum = float(abs(np.einsum('nhf,hf->nh', support, direction)).max())
        direction /= max(maximum, 1e-12)
        delta = direction*AMPLITUDE[:, None]
        qchange = np.einsum('nhf,hf->nh', qf, delta)
        gain = float(np.sum(query_weight[:, None]*qchange*sensitivity))
        assert gain > 0
        directions[arm] = delta
        audits[arm] = dict(kkt=kkt, predicted_mean_path_return_gain=gain,
            qchange=qchange.tolist(), weighted_query_gain=(query_weight[:, None]*qchange*sensitivity).tolist(),
            maximum_normalized_support_change=float(abs(np.einsum('nhf,hf->nh', support, direction)).max()),
            initial_direction_scale=maximum)
    np.savez_compressed(stage/'directions.npz', **directions)
    write_json(stage/'direction_audit.json', dict(arms=audits, feature_identity_error=error,
        support_states=n, queries=len(qobs), excluded_failed_branches=failed))
    paths = read(BASE/'learning_preparation_v4/paths.json')
    training = [np.asarray(p) for p in paths['training']]

    def run(indices, tag, geometry=training):
        results = evaluate(policy, [geometry[i] for i in indices], stage, tag, None,
                           costs, started+450, environment_factory)
        assert costs['evaluation_actor_calls'] <= 25000 and costs['evaluation_physics_ticks'] <= 600000
        return [dict(r, original_path=i) for r, i in zip(results, indices)], [stage/tag/f'{i}.npz' for i in range(len(indices))]

    candidates = []; cache = {}; selections = []
    for arm, direction in directions.items():
        for step in (.25, .5, 1.):
            policy.load_state_dict(saved['policy'])
            with torch.no_grad():
                policy.actor[-1].weight.add_(torch.tensor(step*direction[:, :-1], dtype=torch.float32))
                policy.actor[-1].bias.add_(torch.tensor(step*direction[:, -1], dtype=torch.float32))
            name = f'{arm}_{step:g}.pt'
            save_torch(stage/name, dict(policy=policy.state_dict(), observation_dim=72,
                architecture='local_geometry_feedback', mask_endpoint=True, frame='heading',
                parent_sha256=sha(model), arm=arm, step=step,
                learning_method='single_decision_complete_consequence_parameter_gradient'))
            results, traces = run(SCREEN, f'{arm}_{step:g}')
            cache[name] = (results, traces)
            row = dict(review(results, traces), arm=arm, step=step, model=name)
            row['checks'] = adoption_checks(row, screen_baseline)
            row['eligible'] = all(row['checks'].values())
            candidates.append(row)
            write_json(stage/'screen_results.json', dict(baseline=screen_baseline, candidates=candidates, costs=costs))
            print('SINGLE_CREDIT', arm, step, {k:row[k] for k in ('successes','entry_sum_mm',
                'exit_sum_mm','full_reverse_sum_deg','eligible')}, row['checks'], flush=True)
    for arm in ARMS:
        eligible = [r for r in candidates if r['arm'] == arm and r['eligible']]
        if not eligible:
            continue
        chosen = max(eligible, key=lambda r:r['independent_zero_return'])
        write_json(stage/f'{arm}_screen_selection.json', dict(selected=chosen, training_only=True))
        policy.load_state_dict(torch.load(stage/chosen['model'], map_location='cpu', weights_only=False)['policy'])
        extra, traces = run([i for i in range(64) if i not in SCREEN], f'{arm}_remaining')
        old, old_traces = cache[chosen['model']]
        pairs = sorted(zip(old+extra, old_traces+traces), key=lambda p:p[0]['original_path'])
        full_results, full_traces = map(list, zip(*pairs))
        full = review(full_results, full_traces)
        full['checks'] = adoption_checks(full, baseline)
        accepted = all(full['checks'].values())
        selected = dict(arm=arm, model=chosen['model'], training=full, eligible=accepted,
                        traces=[str(p.relative_to(BASE)) for p in full_traces], results=full_results)
        write_json(stage/f'{arm}_training_selection.json', selected)
        if accepted:
            dev_results, dev_traces = run(list(range(12)), f'{arm}_development', [np.asarray(p) for p in paths['validation']])
            selected['development'] = review(dev_results, dev_traces)
        selections.append(selected)
    write_json(stage/'results.json', dict(baseline=baseline, screen=candidates, selections=selections,
        costs=costs, goal_complete=False))
    receipt(stage, dict(status='completed_matched_single_decision_credit_update', costs=costs,
        screen_eligible=sum(r['eligible'] for r in candidates),
        training_eligible=sum(r['eligible'] for r in selections), wall_s=time.perf_counter()-started))


if __name__ == '__main__':
    main()
