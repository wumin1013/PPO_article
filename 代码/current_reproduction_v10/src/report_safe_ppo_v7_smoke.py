"""Read retained smoke evidence and document the supported integration scope."""
from collections import Counter
import numpy as np
from common import ROOT,sha,utc,write_json
from train_safe_ppo_v7 import OUT,read,verify


def main():
    reg=verify('smoke');base=OUT/'smoke';check=read(base/'completion_validation.json')
    assert check['status']=='passed';details=[]
    for arm in reg['arms']:
        records=read(base/arm/'transitions.json');episodes=read(base/arm/'episodes.json');receipt=read(base/arm/'receipt.json')
        sources=dict(Counter(row['source'] for row in records));changed=0
        for row in records:
            if row['submitted_request'] is not None:
                executed=np.asarray(row['submitted_request'])[:len(row['sample'])]
                changed+=int(not np.array_equal(executed,np.asarray(row['sample'])))
        details.append(dict(arm=arm,source_counts=sources,changed_submitted_request_rows=changed,
            transitions=len(records),elapsed_s=sum(r['elapsed_s'] for r in records),
            longest_decision_s=max(r['elapsed_s'] for r in records),
            complete_episode_results=[{k:e[k] for k in ('episode','elapsed_s','total_reward','executor_success','telescoping_reward_error')} for e in episodes if e['terminated']],
            budget_truncated_episodes=sum(e['budget_truncated'] for e in episodes),receipt=receipt,
            maximum_reward_identity_error=max(e['telescoping_reward_error'] for e in episodes)))
    result=dict(created_utc=utc(),status='completed_smoke_only',registration_sha256=sha(base/'registration.json'),
        completion_validation_sha256=sha(base/'completion_validation.json'),details=details,
        no_pilot_registration=not (OUT/'pilot').exists(),no_five_seed=True,no_holdout=True,
        prior_mean_and_gradient_review='audit/safe_ppo_interface_v7_independent_review.json',
        scope='Two96-decision integration smokes and one fixed update each, not full training or final policy evaluation')
    write_json(ROOT/'audit/safe_ppo_v7_smoke_report.json',result)
    lines=['# Unified supervised-fit / recoverable PPO interface: seed101 smoke','',
        'The existing v6 fresh students can initialize a versioned supervised-plus-PPO pipeline using the same14-input actor and rebuilt96/192-feature Wendland maps. The present work completes only the registered96-decision smoke for each arm; it does not reconstruct the old RL chain, complete a five-seed study, or replace the historical main model. Formal pilot registration is intentionally absent while original assets are being investigated.','',
        '## Existing code findings and implemented minimum path','',
        '1. `StrongPPOV4` can reuse the v6 actor without architecture changes. The old trainer hardcodes the earlier fit/hash, historical anchor corpus, v4d executor and adaptive arm; changing only the fit path would leave an inconsistent recipe.',
        '2. The existing recoverable executor obtains a deterministic teacher mean internally. The new request provider accepts exactly one externally sampled request per real decision, verifies its observation, and rejects a second read. No historical teacher is queried online.',
        '3. PPO stores and scores the original sampled Gaussian coordinates. Raw, corrected, geometric and stored-stop outputs are recorded separately as environment behavior; their executed coordinates never replace policy samples in the surrogate. Protected transitions remain in learning.',
        '4. Fixed preview uses two policy coordinates and core `fixed_d=3`. Its geometric steering is converted using the3mm guide, and every actual command records held preview. The old geometric action with a hardcoded0.8mm guide is not applied to the fixed arm.',
        '5. The old constant small feed noise can disappear under common decoder saturation. The new true Normal scale is `base_std_feed + relu(abs(mean_feed)-1)`, with base standard deviations0.008/0.08/0.02. Density replay and optimization use this same differentiable scale; deterministic actor means remain exactly those of the v6 fit. There is no straight-through approximation.',
        '6. Reward uses all actually executed physical time, including complete terminal tails and stored stops. Gamma is1 for the explicit complete-episode time objective; GAE trace decay is `0.95**(elapsed/0.02)`. Real termination clears bootstrap; a computational wall cap or rollout tail retains it. Failure cost includes elapsed time and the physical task penalty instead of a flat exception reward.',
        '7. Actor and critic gradients have separately registered caps. Both arms use the same supervision rows, four-group anchor sampling stream, optimization counts, seed allocation and fixed endpoint selection. Feature buffers remain unchanged during PPO.','',
        '## Retained smoke results','',
        '| Arm | Decisions | Completed successful episodes | Truncated episodes | Real command time (s) | Full wall (s) |','|---|---:|---:|---:|---:|---:|']
    for row in details:
        r=row['receipt'];lines.append(f"| {row['arm']} | {row['transitions']} | {r['complete_successful_episodes']} | {row['budget_truncated_episodes']} | {row['elapsed_s']:.3f} | {r['cost']['total_wall_s']:.3f} |")
    lines.extend(['','Both arms complete the same registered one-update budget: two optimization epochs over96 transitions,192 PPO sample presentations and384 anchor presentations. Original sample log probabilities replay exactly; the anchor index stream is identical. The retained successful episodes occurred during collection before the smoke optimizer update and are not performance evaluations of the resulting final checkpoint.','',
        '| Arm | Command sources | Changed submitted requests | Longest decision (ms) | Sample feed change ≥1mm/s |','|---|---|---:|---:|---:|'])
    for row in details:
        fraction=row['receipt']['sampled_physical_feed']['fraction_delta_at_least_1_mm_s']
        lines.append(f"| {row['arm']} | {row['source_counts']} | {row['changed_submitted_request_rows']} | {row['longest_decision_s']*1000:g} | {100*fraction:.2f}% |")
    lines.extend(['',
        'These feed-change fractions describe the actual smoke states, not a matched comparison against the earlier analytic whole-training-corpus diagnostic. Full episode reward sums match the elapsed-time/potential/failure-penalty identity. All fixed-arm held-preview samples equal3mm.','',
        'The smoke exercises raw and corrected requests and complete terminal tails. It does not provide a new executed geometric-resumption or intermediate stored-stop PPO case; those control mechanisms were exercised separately by the retained v5_4 development checks, and the physical-time GAE stop/truncation cases were checked independently at the interface. No new whole-program reliability claim follows from this limited smoke.','',
        '## Next bounded step, not launched','',
        'First resolve whether the original upstream training assets can be restored. If this alternative pipeline is later used, register one common training-only recipe before a single-seed pilot, retain every failed/truncated episode, and evaluate its fixed final checkpoint. Each future seed must first complete its own supervised fit; copying the seed101 checkpoint into five runs would not produce five independent full training replicates.','',
        'The shared-source implementation is `src/safe_ppo_interface_v7.py`; the registered runner is `src/train_safe_ppo_v7.py`. The former separates policy sampling and physical execution; the latter controls data, budgets and records. This avoids duplicating the frozen controller and preserves its evidence.','',
        '## Reproduction','',
        'From `D:\\PPO_article_work\\JNN_dominance_20261011_v5`, use `D:\\Anaconda\\envs\\PPO\\python.exe`. Read-only smoke verification: `src/train_safe_ppo_v7.py audit --phase smoke`. Regenerate this report: `src/report_safe_ppo_v7_smoke.py`. Creation commands deliberately reject existing outputs; reproduce the registered `register --phase smoke --wall 180`, then `train --phase smoke --arm adaptive` and `train --phase smoke --arm fixed` only in a separate workspace with no existing `results/safe_ppo_v7/smoke`.','',
        'All records, source hashes, sampled actions and probabilities, executed commands, speculative branches, stopping continuations, episode results, minibatches, anchor indices and checkpoints are retained under `results/safe_ppo_v7/smoke`.'])
    (ROOT/'audit/safe_ppo_v7_smoke_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(result['status'],[(x['arm'],x['source_counts']) for x in details])


if __name__=='__main__':main()
