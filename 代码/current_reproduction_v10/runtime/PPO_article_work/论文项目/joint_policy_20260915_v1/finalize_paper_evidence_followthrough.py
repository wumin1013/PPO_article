"""Archive completed evidence work without loading a policy or running a simulator."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess

from study_io import sha, write_json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
REVIEW = BASE / 'paper_evidence_review'
OUT = REVIEW / 'followthrough'
STAGES = ('closed_paths', 'learning_costs', 'development', 'public_long', 'evaluation_audit')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    # Fail before any mutation if this completed archiving operation already exists.
    assert not OUT.exists(), 'Archive exists; verify it instead of rerunning.'
    checks = []

    def verify(path, expected):
        path = Path(path)
        actual = sha(path)
        assert actual == expected, str(path)
        checks.append(dict(path=str(path.relative_to(ROOT)), sha256=actual))

    previous = read(REVIEW / 'receipt.json')
    for name, digest in previous['sources_sha256'].items():
        verify(BASE / name, digest)
    for name, digest in previous['outputs_sha256'].items():
        verify(REVIEW / name, digest)
    original_checks = len(checks)

    receipts = {}
    for name in STAGES:
        folder = REVIEW / name
        receipt = read(folder / 'receipt.json')
        receipts[name] = receipt
        verify(folder / 'registration.json', receipt['registration_sha256'])
        for source, digest in read(folder / 'registration.json')['source_sha256'].items():
            verify(ROOT / source, digest)
        for field, filename in {
            'results_sha256': 'results.json', 'ledger_sha256': 'ledger.json',
            'structure_sha256': 'checkpoint_structure.json',
            'evaluation_sha256': 'evaluation.json', 'trace_sha256': 'trace.npz',
        }.items():
            if field in receipt:
                verify(folder / filename, receipt[field])

    frozen = read(REVIEW / 'frozen_evaluation_candidate.json')
    state_path = BASE / '当前执行状态.json'
    state = read(state_path)
    resume = state['latest_resume_20260919']
    boundary_before = state['paper_research_boundary'].copy()
    verify(BASE / frozen['model'], frozen['sha256'])
    verify(BASE / resume['retained_model'], resume['retained_sha256'])
    verify(BASE / boundary_before['source'], boundary_before['sha256'])

    development = read(REVIEW / 'development/results.json')
    for record in development['records']:
        verify(BASE / record['trace'], record['trace_sha256'])
    audit = read(REVIEW / 'evaluation_audit/results.json')
    for record in audit['baseline_trace_bindings']:
        verify(BASE / record['path'], record['sha256'])
    metadata = read(REVIEW / 'learning_costs/metadata_pointer_check.json')
    verify(REVIEW / 'learning_costs/ledger.json', metadata['ledger_sha256'])
    assert all(row['matches'] for row in metadata['checks'])

    # Inspect process metadata only. Do not terminate or alter any process.
    command = (
        '[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); '
        '$p=@(Get-CimInstance Win32_Process | Where-Object '
        "{$_.Name -match '^(python|pythonw|conda)\\.exe$'} | "
        'Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine,CreationDate); '
        'ConvertTo-Json -InputObject $p -Depth 4 -Compress'
    )
    process_result = subprocess.run(
        ['powershell', '-NoProfile', '-Command', command],
        capture_output=True, text=True, encoding='utf-8', check=True)
    processes = json.loads(process_result.stdout)
    others = [p for p in processes if p['ProcessId'] != os.getpid()]
    research = [p for p in others if any(term in json.dumps(p).lower()
                for term in ('ppo_article_work', 'envs\\\\ppo', 'joint_policy_20260915_v1'))]
    assert not research, 'An existing PPO process requires inspection before finalization.'
    now = datetime.now(timezone.utc).isoformat()

    OUT.mkdir()
    shutil.copy2(state_path, OUT / 'execution_state_before.json')
    write_json(OUT / 'process_snapshot.json', dict(
        checked_utc=now, audit_self_pid=os.getpid(), inspected_python_conda_processes=processes,
        other_python_conda_processes=others, other_ppo_research_processes=research,
        no_new_research_process_started=True))
    write_json(OUT / 'verification.json', dict(
        checked_utc=now, previous_review_hash_checks=original_checks,
        direct_file_hash_checks=len(checks), checks=checks,
        metadata_pointer_checks_from_saved_audit=len(metadata['checks']),
        metadata_distinct_files=len({r['input_file'] for r in metadata['checks']}),
        metadata_all_match=True, sealed_coordinates_not_parsed=True))

    cost = dict(
        actor_calls=sum(receipts[n]['costs']['actor_calls'] for n in ('development', 'public_long')),
        physics_ticks=sum(receipts[n]['costs']['physics_ticks'] for n in ('development', 'public_long')),
        measured_evaluation_wall_s=sum(receipts[n]['wall_s'] for n in ('development', 'public_long')),
        ppo_updates=0, offline_fitting_updates=0, parameter_optimization_updates=0,
        algorithm_rounds_used=0)
    assert cost['actor_calls'] == 893 and cost['physics_ticks'] == 19053
    next_action = (
        'Preserve the registered matrix and the independently confirmed public-long hard failure. '
        'No automatic optimization, retry, or checkpoint replacement. Next, register a single '
        'bounded batch for remaining public-long/fair-comparison evidence, with total run and '
        'wall limits, or explicitly narrow claims if qualified independent repetition is infeasible. '
        'Keep sealed tests closed until the full method, all arm selection, global D and seed '
        'procedure are frozen. Do not rerun completed evaluations or historical searches.')
    update = dict(
        completed_utc=now, status='development_complete_public_long_failure_independently_confirmed',
        report='paper_evidence_review/论文证据当前结论.md',
        evaluation_candidate=frozen, formal_retained_model_unchanged=True,
        development_summary=development['summary'],
        historical_comparisons={k: v['paired'] for k, v in audit['baselines'].items()},
        public_long_failure=audit['long_failure'],
        remaining_public_long_cases=['dense_corners_large', 'closed_line_arc_large'],
        remaining_public_long_status='not_evaluated_with_current_candidate',
        matched_training_budget_established=False, completed_independent_full_seeds=0,
        algorithm_rounds_used=0, algorithm_rounds_cap=2,
        sealed_test_opened=False, new_evaluation_costs=cost,
        next_action=next_action, next_batch_registered=False, goal_complete=False,
        stage_receipts={n: dict(path=f'paper_evidence_review/{n}/receipt.json',
                               sha256=sha(REVIEW / n / 'receipt.json')) for n in STAGES})
    write_json(OUT / 'continuation.json', update)
    receipt = dict(
        status='completed_evidence_followthrough_archive', completed_utc=now,
        work_type='offline_audit_plus_locked_checkpoint_evaluation_no_learning',
        previous_review_receipt_sha256=sha(REVIEW / 'receipt.json'),
        sources_sha256={str(Path(__file__).relative_to(BASE)): sha(__file__),
                        **{f'paper_evidence_review/{n}/receipt.json': sha(REVIEW/n/'receipt.json')
                           for n in STAGES},
                        'paper_evidence_review/learning_costs/metadata_pointer_check.json':
                            sha(REVIEW/'learning_costs/metadata_pointer_check.json')},
        outputs_sha256={str(p.relative_to(BASE)): sha(p) for p in (
            REVIEW/'论文证据当前结论.md', OUT/'execution_state_before.json',
            OUT/'process_snapshot.json', OUT/'verification.json', OUT/'continuation.json')},
        evaluation_costs=cost, sealed_test_opened=False, formal_adoption=False,
        prior_immutable_evidence_preserved=True, no_research_process_left_running=True,
        latest_usage=dict(weekly_remaining_percent=49, reset_credits_available=0,
                         source='live get_usage_limits during this continuation',
                         reset_used=False, purchased=False), goal_complete=False)
    write_json(OUT / 'receipt.json', receipt)

    resume['paper_evidence_review']['latest_continuation'] = dict(
        path='paper_evidence_review/followthrough/continuation.json',
        sha256=sha(OUT/'continuation.json'), receipt='paper_evidence_review/followthrough/receipt.json',
        receipt_sha256=sha(OUT/'receipt.json'), status=update['status'])
    # Keep the original no-simulation statement scoped to its immutable review stage.
    resume['paper_evidence_review']['no_new_training_or_simulation_scope'] = 'initial_review_only'
    resume['paper_evidence_review']['next_action'] = next_action
    resume['learning_status'] = 'no_learning_running_development_passed_public_long_hard_failure'
    resume['next_action'] = next_action
    resume['latest_execution_utc'] = now
    resume['no_research_session_left_running_by_this_turn'] = True
    resume['goal_complete'] = False
    state.update(updated_utc=now, goal_complete=False,
        current_action='Completed development12, closed-path audit and provenance ledger; public-long hard failure independently confirmed. No further batch running.',
        next_action=next_action, running=False, active_experiment=None,
        current_experiment=None, observed_research_processes=[], running_commands=[],
        active_parallel_stages=[])
    assert state['paper_research_boundary'] == boundary_before
    write_json(state_path, state)
    write_json(OUT / 'state_update_receipt.json', dict(
        completed_utc=now, previous_state_sha256=sha(OUT/'execution_state_before.json'),
        updated_state_sha256=sha(state_path), evidence_receipt_sha256=sha(OUT/'receipt.json'),
        research_boundary_unchanged=True, retained_model_unchanged=True))
    for path in [Path(__file__), REVIEW/'论文证据当前结论.md', state_path, *OUT.glob('*.json')]:
        raw = path.read_bytes()
        assert not raw.startswith(b'\xef\xbb\xbf'), str(path)
        raw.decode('utf-8')
    for rel, digest in receipt['outputs_sha256'].items():
        assert sha(BASE/rel) == digest, rel
    print(json.dumps(dict(status=receipt['status'], direct_hash_checks=len(checks),
                          previous_review_hash_checks=original_checks, cost=cost,
                          other_ppo_processes=len(research), goal_complete=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
