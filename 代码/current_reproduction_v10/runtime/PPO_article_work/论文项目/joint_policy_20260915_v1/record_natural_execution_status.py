"""Append current authorized execution state without erasing historical limits/results."""
import json,shutil
from datetime import datetime,timezone
from study_io import BASE,sha,write_json


def main():
    case=BASE/'paper_evidence_review/natural_closed_case';state=BASE/'当前执行状态.json'
    backup=case/'execution_diagnostic/execution_state_before_current_work.json'
    if not backup.exists():shutil.copy2(state,backup)
    s=json.loads(state.read_text(encoding='utf-8'));receipts=[]
    for folder in ('execution_diagnostic','support_calibration','s2_complete','fixed_historical',
                   'fixed_support_calibration','fixed_conditional_training','resolution_half_ms'):
        p=case/folder/'receipt.json'
        if p.exists():receipts.append(dict(path=str(p.relative_to(BASE)),sha256=sha(p),result=json.loads(p.read_text(encoding='utf-8'))))
    freeze_path=case/'method_freeze.json'
    freeze=json.loads(freeze_path.read_text(encoding='utf-8')) if freeze_path.exists() else {}
    delivery_path=case/'paper_delivery_continuation.json'
    delivery=json.loads(delivery_path.read_text(encoding='utf-8')) if delivery_path.exists() else None
    delivered=bool(delivery and delivery.get('scoped_delivery_complete'))
    record=dict(updated_utc=datetime.now(timezone.utc).isoformat(),user_authorization='2026-09-21 current task',
        historical_two_round_cap_superseded=True,per_round_training_simulation_wall_cap_s=1800,
        primary_case=str((case/'reference.csv').relative_to(BASE)),primary_case_sha256=sha(case/'reference.csv'),
        candidate_model='canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt',
        candidate_model_sha256='25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173',
        candidate_transform=dict(remaining_upper=.1412694752216339,other_nonlocal_inputs_changed=False),
        completed_receipts=receipts,independent_ppo_seeds=0,primary_model_count=1,full_pipeline_replication_established=False,
        current_phase='scoped natural-case manuscript delivered' if delivered else 'method frozen; manuscript QA pending',
        goal_complete=delivered,scoped_case_study_complete=delivered,method_frozen=bool(freeze.get('method_frozen')),
        independent_training_repeatability_established=False,general_superiority_established=False,
        reset_used=False,weekly_remaining_last_queried_percent=46,
        delivery=delivery,
        next_action=delivery['next_action'] if delivered else 'Complete the manuscript, nine figures and full-PDF QA; no further soft-metric optimization.')
    s['paper_natural_case_execution']=record
    if delivered:
        s['paper_delivery']=delivery
        s['latest_execution_resume']=dict(path=str(delivery_path.relative_to(BASE)),sha256=sha(delivery_path),
            legacy_resume_records_preserved=True,scope='PPO_article natural closed case; limited claim manuscript delivery')
    s['paper_research_boundary']['latest_execution_authorization']=dict(
        source='2026-09-21 latest user instruction',additional_core_training_authorized=True,
        historical_two_round_cap_superseded=True,per_round_wall_cap_s=1800,
        stop_at_scoped_paper_evidence=True,independent_seed_target=3,report_single_model_limit_if_unmet=True)
    write_json(state,s);write_json(case/'execution_continuation.json',record)
    write_json(case/'execution_state_receipt.json',dict(state_sha256=sha(state),continuation_sha256=sha(case/'execution_continuation.json'),
        prior_state_backup=str(backup.relative_to(BASE)),prior_state_sha256=sha(backup),historical_receipts_unchanged=True))
    print('CURRENT_NATURAL_STATE_RECORDED',len(receipts))


if __name__=='__main__':main()
