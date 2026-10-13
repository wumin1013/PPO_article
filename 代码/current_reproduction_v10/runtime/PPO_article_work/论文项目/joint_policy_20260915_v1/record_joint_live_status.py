"""Brief resume update; preserve original objective and retained model."""
import argparse
import shutil
from datetime import datetime, timezone
from study_io import BASE, sha, write_json
from search_complete_return import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--active', default='')
    parser.add_argument('--also-active',nargs='*',default=[])
    parser.add_argument('--kind', default='none')
    parser.add_argument('--next', required=True)
    args = parser.parse_args()
    target = BASE/'当前执行状态.json'
    backup = BASE/'before/joint_nonlinear_learning_20260919'
    backup.mkdir(parents=True, exist_ok=True)
    if not (backup/target.name).exists():
        shutil.copy2(target, backup/target.name)
    state = read(target)
    completed_names = [
        'canonical_gate_band_nonlinear_joint_response_20260919',
        'canonical_gate_band_nonlinear_joint_feedback_20260919',
        'canonical_gate_band_complete_joint_function_learning_20260919',
        'canonical_gate_band_shared_feature_joint_learning_20260919',
        'canonical_gate_band_joint_target_coordinate_probe_20260919',
        'canonical_gate_band_complete_local_credit_learning_20260919',
        'canonical_gate_band_raw_state_joint_learning_20260919',
        'canonical_gate_band_coupled_feasibility_credit_20260919',
        'canonical_gate_band_trace_resolved_joint_credit_20260919',
        'canonical_gate_band_entry_preserving_trace_credit_20260919',
        'canonical_gate_band_conditioned_joint_credit_20260919',
        'canonical_gate_band_retained_joint_growth_20260919',
        'canonical_gate_band_retained_joint_growth_identity_corrected_20260919',
        'canonical_gate_band_shape_feasible_joint_growth_20260919',
        'canonical_gate_band_growth_time_constraint_attribution_20260919',
        'canonical_gate_band_compact_joint_growth_20260919',
        'canonical_gate_band_compact_joint_growth_completion_20260919',
        'canonical_gate_band_absolute_shape_joint_credit_20260919',
        'canonical_gate_band_joint_geometry_response_consistency_20260919',
        'canonical_gate_band_calibrated_geometry_joint_credit_20260919',
        'canonical_gate_band_fast_absolute_shape_joint_credit_20260919',
        'canonical_gate_band_stratified_suffix_joint_credit_20260919',
        'canonical_gate_band_shared_network_credit_calibration_20260919',
        'canonical_gate_band_shared_credit_shape_localization_20260919',
        'canonical_gate_band_curved_local_suffix_credit_20260919',
        'canonical_gate_band_prefix_preserving_suffix_credit_20260919',
        'canonical_gate_band_numerical_prefix_suffix_credit_20260919',
        'canonical_gate_band_current_protected_joint_credit_20260919',
        'canonical_gate_band_margin_constrained_suffix_credit_20260919',
        'canonical_gate_band_linear_margin_suffix_credit_20260919',
        'canonical_gate_band_dense_boundary_suffix_credit_20260919',
        'canonical_gate_band_dense_margin_suffix_learning_20260919',
        'canonical_gate_band_complete_current_suffix_credit_20260919',
        'canonical_gate_band_complete_decision_joint_learning_20260919',
        'canonical_gate_band_per_corner_dense_credit_20260919',
        'canonical_gate_band_dense_signed_trace_joint_credit_20260919',
        'canonical_gate_band_dense_signed_trace_completion_20260919',
        'canonical_gate_band_error_buffered_trace_credit_20260919',
        'canonical_gate_band_error_buffered_trace_completion_20260919',
        'canonical_gate_band_local_amplitude_trace_credit_20260919',
        'canonical_gate_band_local_amplitude_guarded_learning_20260919',
        'canonical_gate_band_secant_calibrated_joint_learning_20260919',
        'canonical_gate_band_recentered_fine_trace_learning_20260919',
        'canonical_gate_band_recentered_training_pool_check_20260919',
        'canonical_gate_band_training_pool_failure_credit_20260919',
        'canonical_gate_band_training_pool_constraint_learning_20260919',
        'canonical_gate_band_common_horizon_constraint_learning_20260919',
        'canonical_gate_band_motion_aware_constraint_learning_20260919',
        'canonical_gate_band_iterative_motion_constraint_learning_20260919',
        'canonical_gate_band_repaired_training_pool_check_20260919',
        'canonical_gate_band_full_pool_joint_credit_learning_20260919',
        'canonical_gate_band_full_pool_compensation_learning_20260919',
        'canonical_gate_band_pathwise_full_credit_learning_20260919',
        'canonical_gate_band_active_constraint_joint_learning_20260919',
        'canonical_gate_band_full_pool_secant_guard_learning_20260919',
        'canonical_gate_band_full_pool_step_refinement_20260919',
        'canonical_gate_band_current_zero_identity_check_20260919',
        'canonical_gate_band_current_bend_capacity_learning_20260919',
        'canonical_gate_band_current_head_interaction_learning_20260919',
        'canonical_gate_band_complete_head_prism_learning_20260919',
        'canonical_gate_band_joint_interaction_subspace_learning_20260919',
        'canonical_gate_band_complete_metric_credit_learning_20260919',
        'canonical_gate_band_fine_full_policy_credit_learning_20260919',
        'canonical_gate_band_fine_secant_policy_learning_20260919',
        'canonical_gate_band_measured_joint_distribution_learning_20260919',
        'canonical_gate_band_decision_innovation_joint_learning_20260919',
        'canonical_gate_band_innovation_local_compensation_learning_20260919',
        'canonical_gate_band_innovation_motion_guard_learning_20260919',
        'canonical_gate_band_complete_joint_teacher_learning_20260919',
        'canonical_gate_band_joint_teacher_state_distillation_20260919',
        'canonical_gate_band_bounded_joint_teacher_learning_20260919',
        'canonical_gate_band_continued_joint_teacher_distillation_20260919',
        'canonical_gate_band_constrained_joint_teacher_learning_20260919',
        'canonical_gate_band_constrained_teacher_state_distillation_20260919',
        'canonical_gate_band_remaining_corner_teacher_learning_20260919',
        'canonical_gate_band_remaining_corner_state_distillation_20260919',
        'canonical_gate_band_joint_training_neighborhood_20260919',
        'canonical_gate_band_neighborhood_joint_teacher_learning_20260919',
        'canonical_gate_band_neighborhood_teacher_state_distillation_20260919',
        'canonical_gate_band_neighborhood_teacher_head_credit_20260919',
        'canonical_gate_band_full_window_joint_teacher_learning_20260919',
        'canonical_gate_band_full_window_teacher_state_distillation_20260919',
        'canonical_gate_band_target_jump_joint_teacher_learning_20260919',
        'canonical_gate_band_target_jump_teacher_state_distillation_20260919',
        'canonical_gate_band_joint_training_interpolation_20260919',
        'canonical_gate_band_shared_temporal_joint_learning_20260919',
        'canonical_gate_band_shared_temporal_full_pool_check_20260919',
        'canonical_gate_band_speed_conditioned_temporal_joint_learning_20260919',
        'canonical_gate_band_speed_conditioned_temporal_full_pool_check_20260919',
        'canonical_gate_band_target_coordinate_joint_learning_20260919',
        'canonical_gate_band_target_coordinate_full_pool_check_20260919',
        'canonical_gate_band_shared_teacher_kernel_learning_20260919',
        'canonical_gate_band_shared_teacher_kernel_direct_learning_20260920',
        'canonical_gate_band_joint_expert_residual_learning_20260920',
        'canonical_gate_band_joint_expert_full_pool_check_20260920',
        'canonical_gate_band_anticipatory_joint_expert_learning_20260920',
        'canonical_gate_band_anticipatory_joint_expert_full_pool_check_20260920',
        'canonical_gate_band_reference_envelope_joint_learning_20260920',
        'canonical_gate_band_reference_envelope_full_pool_check_20260920',
        'canonical_gate_band_adaptive_credit_joint_teacher_learning_20260920',
        'canonical_gate_band_adaptive_credit_shared_kernel_learning_20260920',
        'canonical_gate_band_visible_credit_joint_teacher_learning_20260920',
        'canonical_gate_band_visible_credit_shared_kernel_learning_20260920',
        'canonical_gate_band_event_resolved_joint_teacher_learning_20260920',
        'canonical_gate_band_event_resolved_shared_kernel_learning_20260920',
        'canonical_gate_band_expert_direction_joint_teacher_learning_20260920',
        'canonical_gate_band_expert_direction_teacher_head_credit_20260920',
        'canonical_gate_band_expert_direction_shared_kernel_learning_20260920',
        'canonical_gate_band_expert_direction_exact_state_shared_kernel_learning_20260920',
        'canonical_gate_band_lateral_aware_joint_teacher_learning_20260920',
        'canonical_gate_band_lateral_aware_shared_kernel_learning_20260920',
        'canonical_gate_band_lateral_aware_expert_weight_learning_20260920',
        'canonical_gate_band_joint_state_representation_neighborhood_20260920',
        'canonical_gate_band_spatial_shared_joint_credit_learning_20260920',
        'canonical_gate_band_spatial_shared_joint_full_pool_20260920',
        'canonical_gate_band_causal_block_joint_teacher_learning_20260920',
        'canonical_gate_band_causal_block_state_weight_learning_20260920',
        'canonical_gate_band_bounded_causal_block_joint_teacher_learning_20260920',
        'canonical_gate_band_bounded_causal_block_state_weight_learning_20260920',
        'canonical_gate_band_innovation_coordinate_response_20260920',
        'canonical_gate_band_shared_innovation_coordinate_learning_20260920',
        'canonical_gate_band_shared_innovation_coordinate_full_pool_20260920',
        'canonical_gate_band_motion_state_innovation_learning_20260920',
        'canonical_gate_band_motion_state_innovation_full_pool_20260920',
        'canonical_gate_band_fine_innovation_credit_learning_20260920',
        'canonical_gate_band_fine_innovation_credit_full_pool_20260920',
        'canonical_gate_band_fine_motion_state_credit_learning_20260920',
        'canonical_gate_band_fine_motion_state_credit_full_pool_20260920',
        'canonical_gate_band_margin_restored_innovation_learning_20260920',
        'canonical_gate_band_geometry_motion_credit_learning_20260920',
        'canonical_gate_band_geometry_motion_credit_full_pool_20260920',
        'canonical_gate_band_independent_command_credit_learning_20260920',
        'canonical_gate_band_independent_command_credit_full_pool_20260920',
        'canonical_gate_band_smooth_constrained_credit_learning_20260920',
        'canonical_gate_band_scaled_command_credit_learning_20260920',
        'canonical_gate_band_signed_slack_credit_learning_20260920',
        'canonical_gate_band_signed_slack_credit_full_pool_20260920',
        'canonical_gate_band_secant_restored_joint_credit_learning_20260920',
        'canonical_gate_band_secant_restored_joint_credit_full_pool_20260920',
        'canonical_gate_band_direct_observable_joint_learning_20260920',
        'canonical_gate_band_direct_observable_exit_learning_20260920',
        'canonical_gate_band_early_observable_joint_learning_20260920',
        'canonical_gate_band_early_fullroute_joint_learning_20260920',
        'canonical_gate_band_early_learned_pulse_joint_learning_20260920',
        'canonical_gate_band_early_joint_exploration_restoration_20260920',
        'canonical_gate_band_early_local_affine_joint_learning_20260920',
        'canonical_gate_band_early_affine_coupled_joint_learning_20260920',
        'canonical_gate_band_early_observable_memory_joint_learning_20260920',
        'canonical_gate_band_memory_calibrated_joint_credit_learning_20260920',
        'canonical_gate_band_memory_context_joint_credit_learning_20260920',
        'canonical_gate_band_compact_memory_joint_credit_learning_20260920',
        'canonical_gate_band_main_turn_memory_joint_learning_20260920',
        'canonical_gate_band_peak_memory_shape_restoration_20260920',
        'canonical_gate_band_coupled_peak_memory_learning_20260920',
        'canonical_gate_band_duration_guarded_memory_learning_20260920',
        'canonical_gate_band_postpeak_memory_joint_learning_20260920',
        'canonical_gate_band_shared_postpeak_memory_learning_20260920',
        'canonical_gate_band_context_postpeak_memory_learning_20260920',
        'canonical_gate_band_recentered_context_memory_credit_20260920',
        'canonical_gate_band_zero_margin_context_credit_20260920',
        'canonical_gate_band_anticipatory_context_memory_learning_20260920',
        'canonical_gate_band_coupled_phase_context_memory_learning_20260920',
        'canonical_gate_band_coupled_context_composition_learning_20260920',
        'canonical_gate_band_measured_joint_amplitude_restoration_20260920',
        'canonical_gate_band_bilinear_motion_context_joint_learning_20260920',
        'canonical_gate_band_quadratic_motion_context_joint_learning_20260920',
        'canonical_gate_band_approach_coupled_joint_learning_20260920',
        'canonical_gate_band_approach_conditioned_recovery_learning_20260920',
        'canonical_gate_band_conditioned_main_turn_joint_learning_20260920',
        'canonical_gate_band_geometry_conditioned_joint_learning_20260920',
        'canonical_gate_band_geometry_secant_recovery_learning_20260920',
        'canonical_gate_band_signed_heading_joint_recovery_20260920',
        'canonical_gate_band_recentered_joint_subspace_learning_20260920',
        'canonical_gate_band_bend_metric_joint_learning_20260920',
        'canonical_gate_band_metric_function_transfer_20260920',
        'canonical_gate_band_centered_bend_function_transfer_20260920',
        'canonical_gate_band_centered_metric_joint_recovery_20260920',
        'canonical_gate_band_intrinsic_curvature_joint_recovery_20260920',
        'canonical_gate_band_joint_output_precision_probe_20260920',
        'canonical_gate_band_bounded_output_precision_probe_20260920',
        'canonical_gate_band_positive_margin_joint_recovery_20260920',
        'canonical_gate_band_preemptive_joint_credit_learning_20260920',
        'canonical_gate_band_event_recentered_preemptive_credit_20260920',
        'canonical_gate_band_preapex_joint_compensation_20260920',
        'canonical_gate_band_decoded_joint_step_learning_20260920',
        'canonical_gate_band_supported_outgoing_joint_learning_20260920',
        'canonical_gate_band_midturn_supported_joint_learning_20260920',
        'canonical_gate_band_bounded_midturn_joint_learning_20260920',
        'canonical_gate_band_motion_robust_midturn_joint_learning_20260920',
        'canonical_gate_band_motion_robust_midturn_completion_20260920',
        'canonical_gate_band_stable_support_joint_credit_learning_20260920',
        'canonical_gate_band_direct_stable_joint_learning_20260920',
        'canonical_gate_band_early_geometry_joint_learning_20260920',
        'canonical_gate_band_early_geometry_joint_completion_20260920',
        'canonical_gate_band_early_geometry_credit_recovery_20260920',
        'canonical_gate_band_reversal_window_joint_credit_20260920',
        'canonical_gate_band_reversal_window_secant_recovery_20260920',
        'canonical_gate_band_reversal_window_local_recovery_20260920',
        'canonical_gate_band_active_reversal_window_joint_credit_20260920',
        'canonical_gate_band_active_window_local_joint_recovery_20260920',
        'canonical_gate_band_preturn_supported_joint_recovery_20260920',
        'canonical_gate_band_late_support_aliasing_probe_20260920',
        'canonical_gate_band_postpeak_separated_joint_recovery_20260920',
        'canonical_gate_band_postpeak_bounded_joint_recovery_20260920',
        'canonical_gate_band_reference_support_closed_loop_probe_20260920',
        'canonical_gate_band_reference_supported_joint_learning_20260920',
        'canonical_gate_band_shared_reference_transfer_joint_learning_20260920',
        'canonical_gate_band_interaction_return_credit_learning_20260920',
        'canonical_gate_band_head_counterfactual_ppo_20260920',
        'canonical_gate_band_head_counterfactual_ppo_collection_20260920',
        'canonical_gate_band_constraint_counterfactual_ppo_20260920',
        'canonical_gate_band_conditional_sample_credit_ppo_20260920',
        'canonical_gate_band_matched_context_counterfactual_ppo_20260920',
        'canonical_gate_band_scaled_context_counterfactual_ppo_20260920',
        'canonical_gate_band_roundoff_context_counterfactual_ppo_20260920',
        'canonical_gate_band_warm_context_counterfactual_ppo_20260920',
        'canonical_gate_band_stratified_head_credit_ppo_20260920',
        'canonical_gate_band_stratified_head_credit_ppo_update_20260920',
        'canonical_gate_band_stratified_constraint_credit_ppo_20260920',
        'canonical_gate_band_deterministic_complete_credit_20260920',
        'canonical_gate_band_deterministic_local_credit_20260920',
        'canonical_gate_band_minimum_norm_complete_credit_20260920',
        'canonical_gate_band_joint_secant_credit_20260920',
        'canonical_gate_band_shape_minimax_training_branch_20260920',
        'canonical_gate_band_smooth_minimax_training_branch_20260920',
        'canonical_gate_band_deterministic_full_support_credit_20260920',
        'canonical_gate_band_stratified_weighted_constraint_credit_ppo_20260920',
        'canonical_gate_band_factorial_joint_ppo_20260920',
        'canonical_gate_band_reference_control_variate_ppo_20260920',
        'canonical_gate_band_guarded_control_variate_proposal_20260920',
        'canonical_gate_band_coherent_parameter_training_20260920',
        'canonical_gate_band_dual_coherent_parameter_training_20260920',
        'canonical_gate_band_context_coherent_parameter_training_20260920',
        'canonical_gate_band_action_capacity_joint_training_20260920',
        'canonical_gate_band_causal_prefix_joint_compensation_20260920',
        'canonical_gate_band_verified_trust_joint_learning_20260920',
        'canonical_gate_band_coupled_early_late_joint_learning_20260920',
        'canonical_gate_band_coupled_early_late_joint_completion_20260920',
        'canonical_gate_band_information_refreshed_joint_learning_20260920',
        'canonical_gate_band_completion_event_joint_learning_20260920',
        'canonical_gate_band_complementary_joint_direction_learning_20260920',
        'canonical_gate_band_worst_guard_joint_learning_20260920',
        'canonical_gate_band_recentered_minimax_joint_learning_20260920',
        'canonical_gate_band_feasible_direction_current_credit_20260920']
    verified = []
    learned_outcomes = []
    training_branches = []
    for name in completed_names:
        path = BASE/name/'receipt.json'
        if path.exists():
            data = read(path)
            item = dict(stage=name, receipt_sha256=sha(path))
            if data['status'].startswith('completed') and not any(row['stage'] == name for row in state['completed']):
                state['completed'].append(item)
            verified.append(dict(item, status=data['status']))
            shadow_file = BASE/name/'shadow_training_result.json'
            if data['status'].startswith('completed') and shadow_file.exists():
                shadow = read(shadow_file)
                shadow_model = BASE/name/'shadow_training_policy.pt'
                model_hash = sha(shadow_model)
                assert model_hash == data['shadow_sha256'], ('shadow_hash_mismatch', name)
                training_branches.append(dict(stage=name, selected=shadow['tag'],
                    summary=shadow['summary'], original_all_guard_pass=shadow['original_all_guard_pass'],
                    shadow_model_sha256=model_hash, receipt_sha256=sha(path), adopted=False))
            result_file = BASE/name/'results.json'
            if data['status'].startswith('completed') and result_file.exists():
                result = read(result_file)
                if 'best' in result and not (BASE/name/'selection_hold.json').exists():
                    selected = next(r for r in result['candidates'] if r['tag'] == result['best'])
                    if 'summary' in selected:
                        learned_outcomes.append(dict(stage=name, selected=selected['tag'],
                            summary=selected['summary'], adopted=False,
                            original64_summary=selected.get('original64_summary'),
                            added_training_summary=selected.get('added_training_summary'),
                            promotion=selected.get('promotion'), receipt_sha256=sha(path),
                            research_model_sha256=sha(BASE/name/'research_candidate.pt')
                            if (BASE/name/'research_candidate.pt').exists() else None))
    now = datetime.now(timezone.utc).isoformat()
    state.update(updated_utc=now, running=args.active or None,
                 active_parallel_stages=args.also_active,
                 active_experiment=args.active or None, goal_complete=False, next=args.next)
    ppo_running=bool(args.active) and args.kind in ('ppo_interaction_training','ppo_update_reused_on_policy_data')
    parameter_running=bool(args.active) and args.kind in ('complete_closed_loop_parameter_optimization',
        'deterministic_credit_parameter_optimization','complete_closed_loop_parameter_learning')
    response_running=bool(args.active) and args.kind in ('physical_response_collection_for_parameter_learning',
        'current_joint_response_collection_for_parameter_learning',
        'deterministic_policy_gradient_counterfactual_collection','counterfactual_interaction_collection_then_ppo')
    state['current'].update(state=args.kind, training_running=ppo_running,
        parameter_optimization_running=parameter_running,
        physical_diagnostic_running=response_running,
        diagnostic_stage=args.active if response_running else None,
        candidate_final_accepted=False)
    resume = state['latest_resume_20260919']
    resume.update(latest_execution_utc=now, latest_execution_receipts=verified,
        latest_complete_parameter_results=learned_outcomes,
        latest_unadopted_training_branches=training_branches[-3:],
        active_stage=args.active or None, learning_status=args.kind,
        active_parallel_stages=args.also_active,
        next_action=args.next, no_research_session_left_running_by_this_turn=not bool(args.active),
        goal_complete=False)
    boundary_file=BASE/'论文研究边界与续接指令_20260920.md'
    if boundary_file.exists():
        boundary=dict(source=boundary_file.name,sha256=sha(boundary_file),
            objective='Evidence sufficient for a scoped research paper, not industrial deployment or universal perfection.',
            user_clarification_date='2026-09-20',hard_physical_constraints_unchanged=True,
            all_paths_all_soft_metrics_nonregression_required=False,
            representative_complete_cases_and_fair_comparisons_required=True,
            finite_budget_and_freeze_required=True,
            historical_rejected_soft_guard_candidates_may_be_reassessed_not_automatically_adopted=True)
        state['paper_research_boundary']=boundary
        resume['paper_research_boundary']=boundary
    for pool_name in ('canonical_gate_band_recentered_training_pool_check_20260919',
                      'canonical_gate_band_repaired_training_pool_check_20260919'):
        full_pool_receipt=BASE/pool_name/'receipt.json'
        if full_pool_receipt.exists():
            full_pool=read(full_pool_receipt)
            resume['latest_training_pool_check']=dict(stage=full_pool_receipt.parent.name,
                source_model_sha256=full_pool.get('source_model_sha256'),summary=full_pool.get('summary'),
                receipt_sha256=sha(full_pool_receipt),adopted=False)
    complete_pool_candidates=[]
    for row in learned_outcomes:
        if row['summary'].get('paths')==64 and row['summary'].get('successes')==64:
            complete_pool_candidates.append(row)
        elif row.get('original64_summary',{}):
            if row['original64_summary'].get('successes')==64:
                complete_pool_candidates.append(dict(row,summary=row['original64_summary'],
                    scope='Original64 subset of the actual expanded-training evaluation.'))
    if complete_pool_candidates:
        resume['latest_full64_research_candidate']=complete_pool_candidates[-1]
    expanded_candidates=[row for row in learned_outcomes if row['summary'].get('paths',0)>64
                         and row['summary'].get('successes')==row['summary']['paths']]
    if expanded_candidates:
        resume['latest_expanded_training_research_candidate']=expanded_candidates[-1]
    write_json(target, state)
    note = [
        '# 当前联合学习简记', '',
        '保留模型仍为86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86，未替换。',
        '历史阶段包含参数优化、离线拟合、物理诊断及PPO更新；当前是否运行及其类型以下方实时记录为准。',
        '', '执行器仍含引导器、向量约束投影和制动。1.25mm、20/1ms、V/A/J、完整覆盖和停稳要求未改。',
        '下表注明训练路径数；64条训练池也不能替代独立种子、长路径、公平基线或泛化验收。',
        '仅列最近六个保留的研究候选；完整旧记录留在状态JSON和原收据，不重复扩写。',
        '', '| 完整闭环学习阶段 | 路径数 | 反转总量/deg | 出边越过总量/mm | 含弯99巡航/s | 采用 |',
        '|---|---:|---:|---:|---:|---|']
    for row in learned_outcomes[-6:]:
        s = row['summary']
        note.append(f"| {row['stage']} | {s['paths']} | {s['reverse_sum_deg']:.6f} | {s['exit_sum_mm']:.6f} | {s['curved_cruise99_s']:.6f} | 否 |")
    fine_path = BASE/'canonical_gate_band_local_amplitude_trace_credit_20260919/model_validation.json'
    if fine_path.exists():
        fine = read(fine_path)
        note += ['', f"两条峰曲率问题路径的小幅度对照：三个已测候选的归一化预测误差合计由{fine['coarse_error']:.8f}降至{fine['fine_error']:.8f}；仅支持该局部响应问题。"]
    measured_path = BASE/'canonical_gate_band_local_amplitude_guarded_learning_20260919/results.json'
    if measured_path.exists():
        for measured in read(measured_path)['candidates']:
            summary=measured['summary']
            note += [f"小幅响应对照{measured['tag']}：{summary['successes']}/20完成，反转{summary['reverse_sum_deg']:.6f}deg，出边{summary['exit_sum_mm']:.6f}mm，含弯99巡航{summary['curved_cruise99_s']:.6f}s；该对照逐角保护未全过，未采用。"]
    note += ['', '当前执行类型：'+args.kind+'；活动目录：'+(args.active or '无，需核实进程与最新收据。'),
             '下一步：'+args.next, '',
             '复用原执行器和统一评价函数，避免重复逻辑；仅增加本轮必要表示与优化，按职责分开策略、响应、评价和状态记录。',
             '原历史PPO证据边界不变：支持该批次跨时间/动作信用分配问题，不能归结为唯一根因或估计器有偏。',
             '没有清理、提交、推送或新派子代理；仿真工作进程不是子代理。目标仍未完成。']
    (BASE/'当前联合学习简记.md').write_text('\n'.join(note)+'\n', encoding='utf-8')
    print('STATUS_UPDATED',now,args.active,args.kind,'receipts',len(verified),
          'latest',[r['stage'] for r in verified[-3:]],flush=True)


if __name__ == '__main__':
    main()
