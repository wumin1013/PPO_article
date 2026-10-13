"""Bind the offline evidence review and update the existing PPO_article resume."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil


BASE = Path(__file__).resolve().parent
OUT = BASE / 'paper_evidence_review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def geometry_sha(reference):
    data = json.dumps(reference, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def main():
    audit = read(OUT / 'candidate_audit.json')
    candidates = {c['label']: c for c in audit['candidates']}
    state_file = BASE / '当前执行状态.json'
    state = read(state_file)
    resume = state['latest_resume_20260919']
    now = datetime.now(timezone.utc).isoformat()
    expected = {
        'retained': resume['retained_sha256'],
        'reference_supported': 'ded2a93f85f3503a8c09bf59fd79e48c8f2cd187874f07789c53b8863c936229',
        'complementary': 'f52eaec0129ff491c7fa71d7bf09d85b458fc1c0c27eaba9514e0a67e524479b',
        'worst_guard': '25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173',
        'recentered': '54c6f68f8419c9395d5e300a54ce9ab0e8698e88188425ef58d2f32d09ce9f57',
    }
    models = []
    for name, c in candidates.items():
        actual = sha(BASE / c['model'])
        assert actual == c['model_sha256'] == expected[name]
        assert sha(BASE / c['result']) == c['result_sha256']
        models.append(dict(label=name, path=c['model'], sha256=actual, expected_match=True))
    final_stage = BASE / 'canonical_gate_band_feasible_direction_current_credit_20260920'
    assert sha(final_stage / 'shadow_training_policy.pt') == expected['recentered']
    models.append(dict(label='latest_duplicate', path=str((final_stage / 'shadow_training_policy.pt').relative_to(BASE)),
                       sha256=expected['recentered'], expected_match=True))
    process = read(OUT / 'process_snapshot.json')
    assert not process['research_processes'], process
    write(OUT / 'identity_verification.json', dict(verified_utc=now, models=models,
          latest_receipts=audit['sources_receipts'], research_boundary=state['paper_research_boundary'],
          boundary_hash_matches=sha(BASE / state['paper_research_boundary']['source']) == state['paper_research_boundary']['sha256']))

    chosen = candidates['worst_guard']
    protocol = BASE / 'learning_preparation_v4/paths.json'
    development = read(protocol)
    train = []
    for r in chosen['records']:
        if r['path'] < 64:
            assert r['reference'] == development['training'][r['path']]
        train.append(dict(path=r['path'], reference_sha256=geometry_sha(r['reference']),
                          vertices=r['vertices'], length_mm=r['length_mm']))
    assert len(train) == 88 and len({r['path'] for r in train}) == 88
    sealed_receipt = read(BASE / 'sealed_test_preparation_v1/receipt.json')
    sealed_file = BASE / 'sealed_test_preparation_v1/paths.json'
    # Read bytes only for the digest; do not parse sealed coordinates or outcomes.
    assert sha(sealed_file) == sealed_receipt['paths_sha256']
    long_cases = read(BASE / 's2_frozen_command_new_geometry_v1/results.json')['results']
    long_definitions = []
    for c, angle in zip(long_cases, [17, -23, 41]):
        ref = BASE.parent / 'revision_20260913_preview_efficiency_v2/geometry_design_v1/references' / (c['case'] + '.csv')
        assert sha(ref) == c['reference_sha256']
        long_definitions.append(dict(source=str(ref), source_sha256=sha(ref),
            scale=1.1, rotation_deg=angle, transformation='scale and rotate around the first reference point',
            coordinates_generated=False, outcomes_read=False,
            role='unselected full-length scale/rotation check, not novel topology'))
    illustrative = [dict(source='expanded_training', path=i) for i in [48, 3, 4, 11, 14, 204, 32]]
    illustrative.append(dict(source='public_long_development', path='mixed_lengths_large',
                             current_candidate_evaluation_missing=True))
    matrix = dict(created_utc=now, objective='PPO_article original research objective',
        status='evidence_scope_and_evaluation_checkpoint_locked; full_method_and_arm_selection_not_yet_frozen',
        illustrative_cases=illustrative, training88=train,
        development12=dict(source=str(protocol.relative_to(BASE)), source_sha256=sha(protocol),
            paths=[dict(path=i, reference_sha256=geometry_sha(r)) for i, r in enumerate(development['validation'])]),
        sealed30=dict(source=str(sealed_file.relative_to(BASE)), sha256=sealed_receipt['paths_sha256'],
            seed=915271, counts=sealed_receipt['counts'], coordinates_not_parsed=True, opened=False),
        new_long3=long_definitions,
        learned_arms=['joint_complete', 'fixed_global_preview', 'no_learned_steering_residual', 'no_learned_feed'],
        deterministic_baseline='S2', independent_training_seeds=[921101, 921102, 921103],
        completed_qualified_independent_seed_runs=0,
        ablations=dict(fixed_preview='also the primary fixed-D baseline; do not duplicate',
            no_learned_steering_residual='steering residual fixed to zero; independently retrain feed and preview',
            no_learned_feed='ordinary requested feed fixed to 100 mm/s; independently retrain steering and preview; same actuator/braking'),
        fixed_distance_opportunity=dict(grid_mm=[.8, 1.5, 3., 4.6, 6., 8., 12.],
            historical_new_decisions_per_distance=120000, historical_selected_distance_mm=3.,
            strongest_verified_existing_model='fixed3_gsde_refinement_v7/final.pt',
            strongest_verified_existing_model_sha256='cf77ea99f7438a4c7a7377a856e59c0396edb3779d4d112224615de7287ea828',
            development12_complete_time_s=18.464, same_setup_completed_grid_must_not_be_rerun=True),
        confirmed_primary_evaluation_max_runs=390, new_long_evaluation_max_runs=39,
        numerical_matched_training_budget=None,
        budget_gap='Requires offline provenance and total PPO/fitting/parameter-search/validation cost ledger; no training dispatch authorized by this file.',
        hard_constraints=dict(euclidean_one_sided_tolerance_mm=1.25, decision_ms=20, execution_ms=1,
            vector_limits=[100, 2000, 62500], ordered_complete=True, complete_braking=True, stopped_ms_at_least=20),
        current_turn_new_training_or_simulation_budget=0, additional_algorithm_rounds_used=0,
        additional_algorithm_rounds_max=2, per_algorithm_round_training_simulation_wall_s_max=1800,
        sealed_access_allowed=False, learned_arm_training_started=False)
    write(OUT / 'evidence_matrix.json', matrix)
    frozen = dict(created_utc=now, purpose='Single immutable checkpoint for fair-evaluation preparation',
        model=chosen['model'], sha256=chosen['model_sha256'], formal_adoption=False,
        replaces_retained_model=False, complete_training_method_frozen=False,
        sealed_test_access_allowed=False, further_optimization_default=False,
        reason='Reassessed under paper scope: smaller exit excursions and reverse lateral motion at small time/cruise cost; competing candidates remain documented.')
    write(OUT / 'frozen_evaluation_candidate.json', frozen)
    fixed = BASE / matrix['fixed_distance_opportunity']['strongest_verified_existing_model']
    assert sha(fixed) == matrix['fixed_distance_opportunity']['strongest_verified_existing_model_sha256']
    sources = [BASE / n for n in ['review_paper_evidence.py', 'finalize_paper_evidence_review.py',
        'complete_motion_metrics.py', 'complete_corner_metrics.py', 'multi_corner_raw_metrics.py',
        'audit_xy_curvature_resolution.py', 'geometry_spec.py', 'validate_actions.py', 'joint_runtime.py',
        'fixed_distance_search_results_v1/results.json', 'fixed3_gsde_refinement_v7/receipt.json',
        'fixed3_gsde_refinement_v7/selection.json', 'joint_gsde_refinement_v7/receipt.json',
        'v7_joint_long_mixed_v1/evaluation.json', 'v7_fixed_long_mixed_v1/evaluation.json',
        's2_frozen_command_new_geometry_v1/results.json', 'sealed_test_preparation_v1/receipt.json']]
    outputs = ['candidate_audit.json', 'all_path_metrics.csv', 'identity_verification.json',
               'process_snapshot.json', 'evidence_matrix.json', 'frozen_evaluation_candidate.json',
               '最小论文证据矩阵.md']
    receipt = dict(status='completed_existing_candidate_review_and_minimum_evidence_matrix',
        completed_utc=now, work_type='offline_saved_trace_diagnostic_and_scope_registration',
        candidate_records=416, unique_saved_traces=audit['unique_saved_traces'],
        raw_recomputation_wall_s=audit['wall_s'], model_files_verified=6, latest_receipts_verified=10,
        new_ppo_updates=0, new_offline_fitting_updates=0, new_parameter_optimization_updates=0,
        new_actor_calls=0, new_simulation_ticks=0, sealed_test_opened=False,
        sources_sha256={str(p.relative_to(BASE)): sha(p) for p in sources},
        outputs_sha256={p: sha(OUT / p) for p in outputs},
        latest_usage=dict(weekly_remaining_percent=49, reset_credits_available=0,
                          source='live get_usage_limits in this turn', reset_used=False, purchased=False),
        formal_retained_model_unchanged=True, goal_complete=False)
    write(OUT / 'receipt.json', receipt)
    backup = OUT / 'execution_state_before_review.json'
    if not backup.exists():
        shutil.copy2(state_file, backup)
    next_action = ('No new optimization. Complete offline closed-seam shape coverage and the reproducible learning/cost ledger, '
                   'then register bounded missing development/fair-comparison runs under the locked evidence matrix. '
                   'Keep sealed tests closed until the complete method, all arm checkpoints, global D and independent seed procedures are frozen.')
    review = dict(directory='paper_evidence_review', receipt_sha256=sha(OUT / 'receipt.json'),
        matrix_sha256=sha(OUT / 'evidence_matrix.json'), evaluation_candidate=frozen,
        current_candidate_original64=chosen['original64_summary'], current_candidate_expanded88=chosen['all_summary'],
        original64_time_faster_equal_slower=[40, 11, 13],
        algorithm_rounds_used=0, no_new_training_or_simulation=True, next_action=next_action)
    resume.update(paper_evidence_review=review, learning_status='evaluation_checkpoint_and_evidence_matrix_locked_no_training_running',
                  next_action=next_action, no_research_session_left_running_by_this_turn=True)
    state.update(updated_utc=now, next_action=next_action, running=None, active_experiment=None,
                 current_action='Completed existing-candidate raw audit and locked minimal paper evidence matrix; no algorithm experiments started.',
                 current_experiment=dict(status='no_active_experiment', type='paper_evidence_preparation',
                     prior_experiment_description_preserved_in='paper_evidence_review/execution_state_before_review.json'),
                 observed_research_processes=[], running_commands=[], active_parallel_stages=[])
    temporary = OUT / 'execution_state_updated.tmp'
    write(temporary, state)
    temporary.replace(state_file)
    write(OUT / 'state_update_receipt.json', dict(updated_utc=now,
        before_sha256=sha(backup), after_sha256=sha(state_file),
        current_resume_review_receipt=sha(OUT / 'receipt.json'), retained_model_changed=False))
    print(json.dumps(dict(receipt=str(OUT / 'receipt.json'), models_verified=6,
         primary_dataset_count=sealed_receipt['paths'], sealed_opened=False,
         candidate=chosen['label'], current_research_processes=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
