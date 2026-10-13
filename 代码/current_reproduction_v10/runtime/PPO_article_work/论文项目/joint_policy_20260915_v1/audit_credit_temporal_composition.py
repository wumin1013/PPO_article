"""Separate sparse temporal credit from shared-policy closed-loop composition."""
import time
import numpy as np
import torch

from learn_exit_local_credit import last_layer_features
from learn_single_decision_credit import SCREEN
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import score_complete
from search_complete_return import read
from study_io import BASE, register, receipt, write_json
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from validate_actions import difference_audit


PATHS = [13, 31]
AMOUNT = .25
METRICS = ['independent_zero_return', 'in_edge_outward_mm', 'out_edge_outward_mm', 'full_physical_reverse_deg']


def metric(row, name):
    return row[name] if name == 'independent_zero_return' else row['shape'][name]


def main():
    torch.set_num_threads(1)
    parent = BASE/'exit_local_credit_parameter_learning_20260916'
    learning = BASE/'matched_single_decision_credit_learning_20260917'
    arm = 'independent_zero_return'
    stage = register('single_decision_temporal_composition_audit_20260917',
        [__file__, parent/'candidate_1.pt', learning/'receipt.json', learning/'directions.npz',
         BASE/'probe_coordinated_suffix.py', BASE/'probe_one_decision_credit.py',
         BASE/'learn_exit_local_credit.py', BASE/'local_geometry_feedback_policy.py',
         BASE/'joint_runtime_v5.py', BASE/'validate_actions.py'],
        dict(paths=PATHS, maximum_suffixes=300, actor_calls=25000, new_physics_ticks=600000,
             wall_s=400, direction_amount=AMOUNT, policy_training_decisions=0),
        'Post-update mechanism diagnosis, not another training run: choose path13 because '
        'its entry worsens under the revised-objective quarter-step and path31 because larger '
        'steps fail despite one local failure exclusion. At EVERY original decision, apply '
        'the learned update direction for only that20ms decision, at +/-quarter-step, then '
        'resume unchanged parent feedback. Three action heads move together in their learned '
        'direction. Compare complete-pair all-time sum with original four/two quantile estimate '
        'and with full-feedback +/-quarter-step (reuse saved positive) and +/-sixteenth-step. '
        'Differences from original axial credit include vector composition and smaller amplitude; '
        'do not attribute those to temporal quadrature. The dense-vs-quantile comparison here '
        'uses identical vector probes and differs only in temporal support. No deployment '
        'or sealed/development access; any failed branches remain explicit and excluded from '
        'complete shape-derivative comparisons. Stop after this finite causal audit.')
    started = time.perf_counter()
    costs = dict(actor_calls=0, new_physics_ticks=0, saved_prefix_geometry_ticks=0,
                 evaluation_actor_calls=0, evaluation_physics_ticks=0)
    saved = torch.load(parent/'candidate_1.pt', map_location='cpu', weights_only=False)
    policy = LocalGeometryFeedbackPolicy(mask_endpoint=True)
    policy.load_state_dict(saved['policy'])
    with np.load(learning/'directions.npz') as z:
        direction = z[arm].copy()
    baselines = []; records = []; global_records = []; observations = []
    for path in PATHS:
        with np.load(parent/'training_1'/f'{path}.npz') as z:
            ref, rows, actions = [z[k].copy() for k in ('reference', 'rows', 'actions')]
        baseline = dict(path=path, decisions=len(actions), **score_complete(ref, rows, actions))
        baselines.append(baseline)
        obs = []
        policy.load_state_dict(saved['policy'])
        for anchor in range(len(actions)):
            env = reconstruct(ref, rows, actions, anchor*20, costs)
            state = env.state(); observation = env.observation(); obs.append(observation)
            delta = np.einsum('nhf,hf->nh', last_layer_features(policy, observation[None]), direction)[0]
            for sign in (-1, 1):
                env.restore(state); chunks = []; held = []; count = 0
                while not env.done:
                    assert time.perf_counter()-started < 400
                    assert costs['actor_calls']+costs['evaluation_actor_calls'] < 25000
                    assert costs['new_physics_ticks']+costs['evaluation_physics_ticks'] < 597880
                    action = policy.act(env.observation(), True)[0]; costs['actor_calls'] += 1
                    if count == 0:
                        action = action.astype(float)+sign*AMOUNT*delta
                    _, _, _, info = env.step(action)
                    chunks.append(info['rows']); held.append(info['held_action']); count += 1
                    costs['new_physics_ticks'] += len(info['rows'])
                suffix = np.vstack(chunks); suffix_actions = np.asarray(held)
                full = np.vstack([rows[:anchor*20], suffix])
                all_actions = np.vstack([actions[:anchor], suffix_actions]) if anchor else suffix_actions
                audit = difference_audit(ref[0], full)
                success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                name = f'p{path}_a{anchor}_{sign:+d}.npz'
                np.savez_compressed(stage/name, reference=ref, suffix_rows=suffix,
                    suffix_actions=suffix_actions, anchor=anchor, baseline_path=path, latent_delta=sign*AMOUNT*delta)
                records.append(dict(path=path, anchor=anchor, sign=sign, success=success,
                    reason=env.reason, trace=name, audit=audit, full_time_s=len(full)*.001 if success else None,
                    **score_complete(ref, full, all_actions, env)))
            if anchor % 10 == 0:
                write_json(stage/'partial.json', dict(baselines=baselines, records=records, costs=costs))
                print('DENSE_CREDIT', path, anchor, len(actions), 'calls', costs['actor_calls'], flush=True)
        observations.extend(obs)
        for amount in [-.25, -.0625, .0625, .25]:
            if amount == .25:
                trace = learning/f'{arm}_0.25'/f'{SCREEN.index(path)}.npz'
                result = read(trace.with_suffix('.json'))
            else:
                policy.load_state_dict(saved['policy'])
                with torch.no_grad():
                    policy.actor[-1].weight.add_(torch.tensor(amount*direction[:, :-1], dtype=torch.float32))
                    policy.actor[-1].bias.add_(torch.tensor(amount*direction[:, -1], dtype=torch.float32))
                tag = f'p{path}_full_{amount:g}'
                result = evaluate(policy, [ref], stage, tag, None, costs, started+400, environment_factory)[0]
                trace = stage/tag/'0.npz'
            with np.load(trace) as z:
                value = score_complete(z['reference'], z['rows'], z['actions'])
            # Only complete full-feedback records are used below; retain actual failures explicitly.
            if not result['success']:
                for key in ('base_return', 'original_full_return', 'independent_zero_return'):
                    value[key] = None
            global_records.append(dict(path=path, amount=amount, success=result['success'],
                trace=str(trace.relative_to(BASE)), **value))
    comparisons = []
    for baseline in baselines:
        path = baseline['path']; n = baseline['decisions']
        quantiles = [.2, .4, .6, .8] if baseline['shape'] is not None else [.33, .67]
        anchors = sorted({int(q*(n-1)) for q in quantiles})
        rows = [r for r in records if r['path'] == path]
        valid = all(r['success'] for r in rows)
        metrics = METRICS if baseline['shape'] is not None else METRICS[:1]
        values = {}
        for name in metrics:
            if not valid:
                values[name] = None
                continue
            derivative = [(metric(next(r for r in rows if r['anchor']==a and r['sign']==1), name)-
                           metric(next(r for r in rows if r['anchor']==a and r['sign']==-1), name))/(2*AMOUNT)
                          for a in range(n)]
            global_derivatives = {}
            for magnitude in [.0625, .25]:
                pair = [next(r for r in global_records if r['path']==path and r['amount']==s*magnitude) for s in (-1, 1)]
                global_derivatives[str(magnitude)] = (metric(pair[1],name)-metric(pair[0],name))/(2*magnitude) if all(r['success'] for r in pair) else None
            values[name] = dict(all_time_sum=float(sum(derivative)),
                quantile_sum=float(sum(derivative[a] for a in anchors)*n/len(anchors)),
                by_decision=derivative, full_feedback_central=global_derivatives)
        comparisons.append(dict(path=path, decisions=n, quantile_anchors=anchors,
                                all_local_complete=valid, metrics=values))
    write_json(stage/'results.json', dict(baselines=baselines, local=records, full_feedback=global_records,
        comparisons=comparisons, costs=costs, goal_complete=False))
    np.savez_compressed(stage/'observations.npz', observation=np.asarray(observations))
    receipt(stage, dict(status='completed_dense_temporal_credit_composition_audit', costs=costs,
        local_suffixes=len(records), local_successes=sum(r['success'] for r in records),
        wall_s=time.perf_counter()-started))
    print('TEMPORAL_COMPARISON', comparisons, flush=True)


if __name__ == '__main__':
    main()
