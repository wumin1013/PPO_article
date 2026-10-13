"""Bounded first execution of the registered natural case, preserving old evidence."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from study_io import BASE, sha, write_json, register, receipt
from deep_joint_credit_policy import load_credit_policy
from learned_action_pulse_policy import environment_factory
from paper_state_support_policy import RemainingSupportPolicy
from paper_complete_evaluation import evaluate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    case = BASE / 'paper_evidence_review/natural_closed_case'
    checks = []
    for name in ('receipt.json', 'scope_update_receipt.json'):
        r = read(case / name)
        for group in ('sources_sha256', 'outputs_sha256'):
            for source, expected in r.get(group, {}).items():
                p = BASE / source
                if not p.exists():
                    p = case / source
                actual = sha(p) if p.is_file() else None
                checks.append(dict(receipt=name, source=source, expected=expected,
                                   actual=actual, matches=actual == expected))
    cont = read(case / 'continuation.json')
    for key in ('reference', 'geometry', 'geometry_receipt', 'geometry_reporting', 'guide'):
        p = BASE / cont[key]
        checks.append(dict(source=cont[key], actual=sha(p), expected=cont[key+'_sha256'],
                           matches=sha(p) == cont[key+'_sha256']))
    state = BASE / '当前执行状态.json'
    expected = read(case / 'state_update_receipt.json')['state_after_sha256']
    checks.append(dict(source=state.name, actual=sha(state), expected=expected,
                       matches=sha(state) == expected))
    model = BASE / 'canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt'
    assert sha(model) == '25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173'
    upper = read(BASE / 'paper_evidence_review/support_repair/results.json')['complete_policy_spec']['upper']
    stage = register('paper_evidence_review/natural_closed_case/execution_diagnostic',
        [__file__, BASE/'paper_complete_evaluation.py', BASE/'paper_state_support_policy.py',
         case/'continuation.json', case/'scope_update_receipt.json', case/'reference.csv',
         case/'geometry.json', model, state],
        dict(total_wall_s=1800, max_variants=2, training_updates=0, decision_ms=20,
             execution_ms=1, new_user_authorization=True),
        'Evaluate immutable worst_guard first on the unchanged natural case; if it fails, '
        'evaluate the already defined remaining-support transform once. Stop each rollout '
        'at physical failure, completion or wall limit. No weight fitting or sealed data. '
        'This is diagnosis, not formal adoption; complete comparison requires analytic '
        'line/arc and between-update audits in addition to inherited sample checks.')
    write_json(stage/'identity_audit.json', dict(checks=checks, all_match=all(c['matches'] for c in checks)))
    if not all(c['matches'] for c in checks):
        raise RuntimeError('Historical identity mismatch; inspect identity_audit.json before simulation')
    write_json(stage/'current_authorization.json', dict(
        time=datetime.now(timezone.utc).isoformat(), source='User instruction in current task, 2026-09-21',
        overrides_historical_two_round_cap=True, per_round_simulation_training_wall_s=1800,
        continue_only_for_core_evidence_gaps=True, no_new_agents=True,
        stop_at_evidence_sufficient_for_scoped_paper=True, weekly_remaining_percent=47, reset_credits=0))
    parent = torch.load(model, map_location='cpu', weights_only=False)
    reference = np.loadtxt(case/'reference.csv', delimiter=',', skiprows=1)
    started = time.perf_counter()
    records = []
    for label in ('original', 'remaining_support'):
        if time.perf_counter() >= started + 1800:
            break
        policy = load_credit_policy(parent)
        if label == 'remaining_support':
            policy = RemainingSupportPolicy(policy, upper)
        result = evaluate(policy, environment_factory(reference), stage/(label+'.npz'), started+1780)
        result.update(label=label, model_sha256=sha(model), reference_sha256=sha(case/'reference.csv'))
        records.append(result)
        write_json(stage/'results.json', records)
        print('NATURAL_RESULT', label, result['success'], result['reason'], result['elapsed_prefix_or_complete_s'], flush=True)
        if result['success']:
            break
    receipt(stage, dict(status='completed_execution_diagnostic', records=len(records),
        complete_successes=sum(r['success'] for r in records), wall_s=time.perf_counter()-started,
        actor_calls=sum(r['actor_calls'] for r in records), physics_ticks=sum(r['physics_ticks'] for r in records),
        ppo_updates=0, offline_fit_updates=0, parameter_search_updates=0, goal_complete=False,
        results_sha256=sha(stage/'results.json')))


if __name__ == '__main__':
    main()
