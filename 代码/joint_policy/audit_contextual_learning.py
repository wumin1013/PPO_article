"""Saved-data audit of coefficient support and complete-return learning signal."""
import time
import numpy as np
import torch
from contextual_mixture_policy import load_contextual_policy
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def quantile(x):
    x = np.asarray(x)
    return np.quantile(x, [0., .1, .5, .9, 1.], axis=0).tolist() if len(x) else []


def main():
    torch.set_num_threads(1)
    source = BASE/'contextual_learned_function_ppo_20260916'
    recovery = BASE/'contextual_checkpoint_recovery_20260916'
    stage = register('contextual_support_credit_audit_20260916',
        [__file__, source/'receipt.json', recovery/'receipt.json', source/'paired_groups.json',
         source/'episode_results.json', recovery/'policy_22509.pt', BASE/'bounded_shape_return.py'],
        dict(new_actor_decisions=0, physics_ticks=0, wall_s=120),
        'Audit saved states only: coefficient clipping, learned-function action support, '
        'and successful versus mixed-success paired return magnitudes. Report measured facts '
        'separately from causes; offline action ranges do not prove closed-loop reachability.')
    started = time.perf_counter()
    policy = load_contextual_policy(torch.load(recovery/'policy_22509.pt', map_location='cpu', weights_only=False))
    records = read(source/'episode_results.json')['episodes']
    groups = read(source/'paired_groups.json')['groups']
    observations, latent, returns, successes, isolated, straight, in_corner = [], [], [], [], [], [], []
    episode_metrics = []
    for e in records:
        with np.load(source/'episodes'/e['trace']) as z:
            obs = z['observation']
            observations.append(obs)
            latent.append(z['latent'])
            returns.extend([e['shaped_return']]*len(obs))
            successes.extend([e['success']]*len(obs))
            isolated.extend([len(z['reference']) == 3]*len(obs))
            # Raw geometric sensors: all visible tangent directions parallel to current segment.
            directions = obs[:,42:62].reshape(-1,10,2)
            parallel = np.max(abs(directions[:,:,1]), axis=1) < 1e-5
            far_end = np.expm1(obs[:,15]*8)*30 > 15
            straight.extend(parallel & far_end)
            in_corner.extend(~parallel)
            episode_metrics.append(dict(episode=e['episode'], path=e['path'], success=e['success'],
                decisions=len(obs), return_=e['shaped_return'], shape_cost=e['shape_cost']['cost']))
    x = torch.tensor(np.concatenate(observations))
    z = np.concatenate(latent)
    with torch.no_grad():
        mean = policy.distribution(x).mean.numpy()
        a = policy.shape_function.distribution(x).mean.numpy()
        b = policy.cruise_function.distribution(x).mean.numpy()
    action_scale = np.array([90., 50., 5.6])
    action_offset = np.array([0., 50., 6.4])
    action_a = np.tanh(a)*action_scale+action_offset
    action_b = np.tanh(b)*action_scale+action_offset
    masks = {'all': np.ones(len(x),bool), 'straight_far_from_end': np.asarray(straight),
             'visible_turn':np.asarray(in_corner)}
    support = {}
    for label, mask in masks.items():
        support[label] = dict(count=int(mask.sum()), raw_mean_quantiles=quantile(mean[mask]),
            coefficient_mean_quantiles=quantile(np.clip(mean[mask],0,1)),
            raw_sample_quantiles=quantile(z[mask]), clipped_zero_fraction=(z[mask]<=0).mean(0).tolist(),
            clipped_one_fraction=(z[mask]>=1).mean(0).tolist(),
            learned_function_action_difference_quantiles=quantile(action_b[mask]-action_a[mask]),
            maximum_feed_in_interval_quantiles=quantile(np.maximum(action_a[mask,1], action_b[mask,1])),
            both_feed_below95_fraction=float((np.maximum(action_a[mask,1], action_b[mask,1])<95).mean()),
            both_feed_below98_fraction=float((np.maximum(action_a[mask,1], action_b[mask,1])<98).mean()))
    group_rows = []
    for g in groups:
        subset = [records[i] for i in g['episodes']]
        total = np.array(g['returns'])
        adv = total-np.array(g['leave_one_out_baselines'])
        lengths = np.array([episode_metrics[i]['decisions'] for i in g['episodes']])
        group_rows.append(dict(path=g['path'], episodes=g['episodes'], all_success=all(e['success'] for e in subset),
            successes=sum(e['success'] for e in subset), return_range=float(np.ptp(total)),
            decision_weighted_advantage_square=float((lengths*adv**2).sum()),
            abs_return_advantage=float(np.abs(adv).sum())))
    weight = sum(g['decision_weighted_advantage_square'] for g in group_rows)
    mixed = [g for g in group_rows if not g['all_success']]
    report = dict(support=support, episodes=len(records), successes=sum(e['success'] for e in records),
        groups=len(groups), groups_with_failure=len(mixed),
        full_success_return_range_quantiles=quantile([g['return_range'] for g in group_rows if g['all_success']]),
        mixed_success_return_range_quantiles=quantile([g['return_range'] for g in mixed]),
        failed_group_advantage_square_fraction=sum(g['decision_weighted_advantage_square'] for g in mixed)/weight,
        group_rows=group_rows, final_acceptance=False,
        limits='Advantage square measures scale, not actual parameter-gradient contribution. Saved-state support is not a closed-loop impossibility proof.')
    write_json(stage/'results.json', report)
    receipt(stage, dict(status='saved_state_support_credit_audit_complete', new_decisions=0, physics_ticks=0,
                        wall_s=time.perf_counter()-started))
    print({k:v for k,v in report.items() if k != 'group_rows'}, flush=True)


if __name__ == '__main__':
    main()
