"""Record completed supplements and the currently accepted selected controller."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, BASE, write, sha
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    now = datetime.now(timezone.utc)
    records, active = [], []
    for family, roles in [('matched_direction_initialization', ['three', 'two']),
                          ('coherent_teacher_refinement', ['main_repeat', 'common_two']),
                          ('common_two_action_expansion', ['three', 'two']),
                          ('two_visited_feedback', ['']),
                          ('two_accumulated_feedback', ['']),
                          ('matched_local_recovery', ['']),
                          ('matched_local_recovery_support_complete', ['']),
                          ('matched_local_second_recovery', ['', 'attempt_2']),
                          ('strict_prefix_recovery', ['']),
                          ('single_feature_recovery', ['']),
                          ('paired_seed2_initial_recovery', ['', 'attempt_2', 'attempt_2/two', 'attempt_2/three'])]:
        for seed in SEEDS:
            for role in roles:
                stage = SUPPLEMENT/family/f'seed_{seed}'/role
                if not stage.exists():
                    continue
                item = dict(family=family, seed=seed, role=role, directory=str(stage))
                for label, directory in [('fit', stage), ('evaluation', stage/'evaluation')]:
                    receipt = directory/'replay_receipt.json'
                    if receipt.exists():
                        item[label] = read(receipt)
                    elif (directory/'registration.json').exists():
                        item[label] = dict(status='registered_without_receipt')
                        active.append(str(directory))
                log = stage/'fit_log.json'
                if log.exists():
                    item['latest_fit'] = read(log)[-1]
                gate = stage/'evaluation/joint_gate_audit/paired_resolution_extended_audit.json'
                if gate.exists():
                    item['gate_audit'] = str(gate)
                    item['extended_gates_pass'] = read(gate)['records'][0]['extended_numerical_performance_gates_pass']
                records.append(item)
    for seed in SEEDS:
        for name in [f'supplement_coherent_main_{seed}_ppo24', f'supplement_margin_main_{seed}_ppo24'] + [
                f'supplement_common_start_{seed}_{arm}_ppo{count}'
                for arm in ['two', 'three'] for count in [24,48]]:
            stage = ROOT/name
            if not (stage/'registration.json').exists():
                continue
            item = dict(family='fresh_supplement_ppo', seed=seed, directory=str(stage))
            receipt = stage/'receipt.json'
            if receipt.exists():
                item['training'] = read(receipt)
            else:
                item['training'] = dict(status='registered_without_receipt')
                active.append(str(stage))
            updates = stage/'updates.json'
            if updates.exists():
                item['completed_ppo_updates'] = len(read(updates))
            evaluation = stage/'evaluation/replay_receipt.json'
            if evaluation.exists():
                item['evaluation'] = read(evaluation)
            elif (evaluation.parent/'registration.json').exists():
                active.append(str(evaluation.parent))
            records.append(item)
    for directory in [SUPPLEMENT/'paired_seed2_original_probe/half_ms']:
        if not (directory/'registration.json').exists():
            continue
        receipt=directory/'replay_receipt.json'
        records.append(dict(family='paired_initialization_diagnostic',directory=str(directory),
                            diagnostic=read(receipt) if receipt.exists() else dict(status='registered_without_receipt')))
        if not receipt.exists():
            active.append(str(directory))
    acceptance_path=SUPPLEMENT/'three_selected_student_acceptance_v2.json'
    accepted=acceptance_path.exists() and read(acceptance_path)['all_three_selected_checkpoints_pass']
    if accepted:
        assert all(sha(ROOT/r['checkpoint'])==r['checkpoint_sha256'] for r in read(acceptance_path)['records'])
    selected_path=SUPPLEMENT/'selected_main_923202_evidence/selected_acceptance_20260924.json'
    selected=read(selected_path)
    assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
    paired_path=SUPPLEMENT/'common_two_action_expansion/completed_endpoint_selection.json'
    paired=read(paired_path)
    assert paired['all_three_seeds_complete_registered_paired_opportunities']
    nnc_receipt_path=SUPPLEMENT/'original_nnc_current_cases/receipt.json'
    nnc_receipt=read(nnc_receipt_path) if nnc_receipt_path.exists() else {}
    nnc_complete=bool(nnc_receipt.get('sources_unchanged') and len(nnc_receipt.get('records',[]))==2 and all(r['exit_code']==0 for r in nnc_receipt['records']))
    delivery_path=BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision/delivery_manifest.json'
    delivery=read(delivery_path) if delivery_path.exists() else {}
    paper_ready=bool(delivery.get('manuscript_review_ready') and sha(delivery['pdf'])==delivery['pdf_sha256'])
    current_summary=('Selected main model is seed923202 matched12000, frozen after both cases, both resolutions and all current joint gates. Three selected student histories pass with shared teacher data and different later refinement, not uniform full-pipeline independent repetitions. Matched PPO comparison is complete without a credible isolated-preview gain. Original NNC feasibility checks on both current cases are complete. English and Chinese revisions, figure instructions and source archive are ready for author review. Figure editorial status and remaining author actions are read from the current delivery manifest; no training is active.')
    report = dict(updated_utc=now.isoformat(), records=records, active_stages=active,
        scope='Fixed three-seed histories conditional on shared teacher functions/data and static features. Offline refinement is not fresh PPO. No failed seed replacement.',
        selected_frozen_model_changed=True, paired_ablation_complete=True,
        primary_selected_student_seed=923202,
        selected_main_acceptance=str(selected_path),
        paired_action_only_credible_gain_demonstrated=paired['all_pairs_show_credible_dense_gain'],
        paired_action_only_five_percent_demonstrated=paired['all_pairs_show_dense_five_percent'],
        current_case_nnc_comparison_complete=nnc_complete,
        three_seed_joint_acceptance_complete=bool(accepted),
        selected_seed_checkpoint_acceptance=str(acceptance_path) if accepted else None,
        uniform_fixed_recipe_three_repetitions_complete=False,
        final_manuscript_accepted=False)
    destination = SUPPLEMENT/'current_supplement_progress.json'
    history = SUPPLEMENT/'status_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
    history.mkdir(parents=True)
    if destination.exists():
        shutil.copy2(destination, history/destination.name)
    write(destination, report)
    for index, path in enumerate([BASE/'当前执行状态.json', ROOT/'current_progress.json',
                                  ROOT.parent/'live_research_summary.json',
                                  ROOT.parent/'all34_dominance_research/active_research_checkpoint.json']):
        state = read(path)
        shutil.copy2(path, history/f'state_{index}_{path.name}')
        state.update(updated_utc=now.isoformat(), active_stages=active,
            current_supplement_progress=str(destination),
            goal_complete=False, final_manuscript_accepted=False,
            independent_three_seed_acceptance_complete=False,
            method_frozen=True, research_acceptance_passed=True,
            core_joint_numeric_goal_achieved=True,
            selected_research_mainline=str(Path(selected['checkpoint']).parent),
            current_numeric_candidate_binding=str(selected_path),
            current_numeric_candidate_binding_sha256=sha(selected_path),
            current_numeric_checkpoint=selected['checkpoint'],
            current_numeric_checkpoint_sha256=selected['checkpoint_sha256'],
            current_numeric_performance_gates_pass=True,
            acceptance_evidence=str(selected_path),
            selected_seed_checkpoint_acceptance=str(acceptance_path),
            selected_seed_checkpoint_acceptance_sha256=sha(acceptance_path),
            selected_independent_student_checkpoint_count=3,
            three_selected_student_joint_gates_pass=bool(accepted),
            paired_ablation_complete=True,
            paired_action_only_five_percent_demonstrated=False,
            current_case_nnc_comparison_complete=nnc_complete,
            active_offline_audit=None, verified_active_exec_session=None,
            next_step=delivery.get('next_step', delivery.get('remaining_author_action_summary', 'Await new author instructions; preserve the frozen method.')),
            current=current_summary)
        if paper_ready:
            state.update(final_pdf=delivery['pdf'], review_pdf=delivery['pdf'], review_pdf_sha256=delivery['pdf_sha256'],
                manuscript_pages=delivery['pages'], manuscript_review_ready=True,
                chinese_review_pdf=delivery['chinese_pdf'], chinese_review_pdf_sha256=delivery['chinese_pdf_sha256'],
                chinese_manuscript_pages=delivery['chinese_pages'],
                revision_notes_pdf=delivery['revision_notes_pdf'], manuscript_source_archive=delivery['source_archive'],
                manuscript_delivery_manifest=str(delivery_path), manuscript_delivery_manifest_sha256=sha(delivery_path),
                remaining_author_action=delivery['remaining_author_action_summary'])
        write(path, state)
    print('SUPPLEMENT_PROGRESS', len(records), 'records;', len(active), 'without receipt')


if __name__ == '__main__':
    main()
