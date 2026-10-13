"""Read-only action-distribution audit; no training or trajectory evaluation."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from common import ROOT, sha, utc, write_json
from strong_policy_v4 import load_strong_policy


def physical_change_probability(mean, sigma, physical_change=1.):
    """Probability a clipped Gaussian feed differs by at least 1 mm/s."""
    center = np.clip(mean, -1., 1.)
    delta = physical_change / 50.
    lower, upper = center-delta, center+delta
    down = np.where(lower >= -1., ndtr((lower-mean)/sigma), 0.)
    up = np.where(upper <= 1., ndtr((mean-upper)/sigma), 0.)
    return down+up


def main():
    torch.set_num_threads(1)
    source = ROOT/'results/recovery_training_v6/training_data/supervision.npz'
    with np.load(source) as data:
        observations = data['observations'].copy()
        groups = data['case'].copy()
    report = dict(created_utc=utc(), scope='Analytic distribution diagnostic on retained training observations only',
        no_new_training=True, no_new_trajectory_evaluation=True, rows=len(observations),
        input=str(source.relative_to(ROOT)), input_sha256=sha(source),
        distribution='Unbounded Gaussian followed by the unchanged clipped physical decoder',
        feed_decoder='50*(1+clip(raw,-1,1)) mm/s', threshold_mm_s=1., arms={})
    for arm in ('adaptive', 'fixed'):
        checkpoint = ROOT/f'results/recovery_training_v6/seed101_{arm}/final.pt'
        policy, _ = load_strong_policy(checkpoint)
        with torch.no_grad():
            raw = np.concatenate([policy.mean(x).numpy() for x in np.array_split(observations, 32)])
        feed = raw[:, 1].astype(float)
        choices = {'existing_initial_std': np.full(len(feed), .01),
                   'candidate_overflow_aware_std': .01+np.maximum(abs(feed)-1., 0.)}
        row = dict(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
            raw_feed_quantiles=dict(zip(('min','p10','median','p90','max'), np.quantile(feed,[0,.1,.5,.9,1]).tolist())),
            fraction_feed_clipped=float((abs(feed)>=1.).mean()),
            fraction_feed_at_least_three_std_above_bound=float((feed>=1.03).mean()), distributions={})
        for name, sigma in choices.items():
            probability = physical_change_probability(feed, sigma)
            row['distributions'][name] = dict(
                average_probability_feed_change_at_least_1mm_s=float(probability.mean()),
                fraction_states_probability_below_1percent=float((probability<.01).mean()),
                by_program={str(case):dict(rows=int((groups==case).sum()),
                    average_probability=float(probability[groups==case].mean())) for case in sorted(set(groups))},
                status='Existing initial distribution' if name=='existing_initial_std' else
                       'Analytic candidate only; not an adopted policy or experiment result')
        report['arms'][arm] = row
    assert abs(physical_change_probability(np.array([0.]),np.array([.01]))[0]-2*ndtr(-2)) < 1e-15
    assert physical_change_probability(np.array([1.2]),np.array([.01]))[0] < 1e-50
    report['analytic_sanity_checks']='passed'
    write_json(ROOT/'audit/exploration_support_v7.json',report)
    lines=['# Physical feed exploration diagnostic','',
        'Read-only analytic assessment of the v6 fitted policies on their 16,527 retained training observations. '
        'No new rollout, training update, model selection, or final-test access.', '',
        '| Arm | Raw feed median | States clipped (%) | Existing probability of >=1 mm/s change (%) | Candidate probability (%) |',
        '|---|---:|---:|---:|---:|']
    for arm, row in report['arms'].items():
        values=[row['distributions'][k]['average_probability_feed_change_at_least_1mm_s']*100 for k in choices]
        lines.append(f"| {arm} | {row['raw_feed_quantiles']['median']:.6f} | {row['fraction_feed_clipped']*100:.3f} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ['', 'The existing initial raw standard deviation is 0.01. At an overflowing mean, clipping can remove '
        'almost all physical feed exploration. The analytic candidate uses sigma=0.01+max(abs(mean)-1,0); '
        'its Normal likelihood must use that same state-dependent scale. The deterministic mean is unchanged. '
        'This table measures exploration support, not training benefit or completion.', '',
        'The audit counts training rows, not independent paths or seeds. Paired arms require the same distribution '
        'rule and retained original requests; transformed executor commands cannot replace requests in PPO likelihoods.', '',
        'Regenerate: `D:\\Anaconda\\envs\\PPO\\python.exe src/audit_exploration_support_v7.py`.']
    (ROOT/'audit/exploration_support_v7.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
