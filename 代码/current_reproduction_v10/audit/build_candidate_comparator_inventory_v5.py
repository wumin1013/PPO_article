"""Inventory retained neural comparators without training or trajectory execution."""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha, utc, write_json

PACKAGE = ROOT / 'baseline_records/supplementary_reproduction'
RESEARCH = Path('D:/PPO_article_joint_runs/natural_closed_resume/compact_dense_research')


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def asset(path):
    p = Path(path)
    return dict(path=str(p.resolve()), exists=p.is_file(),
                sha256=sha(p) if p.is_file() else None)


def model(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    keys = ('interface', 'runtime_class', 'policy_class', 'preview_rule_mm',
            'local_basis_count', 'local_feature_scope', 'observation_dimension',
            'actor_outputs', 'policy_memory_scalars', 'training_kind', 'offline_iterations',
            'steps', 'updates', 'inherited_ppo_updates', 'new_ppo_updates',
            'mixed_resolution_ppo_updates', 'ppo_updates_this_round',
            'parent_mixed_resolution_updates', 'source_checkpoint',
            'source_checkpoint_sha256', 'parent_checkpoint', 'parent_checkpoint_sha256',
            'initial_checkpoint', 'initial_checkpoint_sha256')
    return dict(**asset(path), recorded_metadata={k: saved[k] for k in keys if k in saved},
                budget_caution='Metadata fields may be inherited across fits and refinements; they do not independently establish an additive complete-pipeline training budget or equality to J-NNC.')


def summary(record):
    d = read(record)
    trace = record.with_suffix('.npz')
    return dict(result=asset(record), trace=asset(trace), success=d.get('success'),
                complete_time_s=d.get('complete_time_s'), elapsed_prefix_s=d.get('elapsed_prefix_s'),
                reason=d.get('reason'), checkpoint_sha256=d.get('checkpoint_sha256'),
                trace_matches_declared_sha256=trace.is_file() and sha(trace) == d.get('trace_sha256'),
                stored_acceptance={k: d.get(k) for k in ('audit', 'analytic', 'geometry')})


def main():
    full_audit_file = ROOT / 'audit/fixed_comparator_complete_trace_audit_v5.json'
    audited = read(full_audit_file)
    fixed = dict(id='historical_selected_fixed_3mm_40af7ea3',
        description='Independently developed two-output neural comparator with a fixed 3 mm guide distance.',
        model=model(PACKAGE / 'models/two.pt'),
        executor='Historical DirectTwoRuntime with preserved KCM, ordered guide and 0.1 mm transition blend; these traces do not use the later v4b or v5_5 executor.',
        comparison_type='Independent neural comparator on the same development paths and acceptance contract; not an equal-budget ablation or causal isolated preview intervention.',
        selection=asset(PACKAGE / 'training/two_action_distance_selection.json'),
        learning_notes=asset(PACKAGE / 'TRAINING_NOTES.md'),
        complete_current_acceptance_audit=asset(full_audit_file),
        all_four_current_acceptance_pass=audited['all_four_success'],
        runs=audited['runs'],
        recommendation='Eligible for complete-time and constraint comparison under the recorded historical comparator identity. Retain both paths and both resolutions. Do not use development results as unseen-path generalization or independent training replications.')
    faster_root = RESEARCH / 'experimental_supplement_20260923/common_two_action_expansion/seed_923201/two'
    faster = dict(id='historical_faster_dense_fixed_seed923201',
        model=model(faster_root / 'initial.pt'),
        selection_context=asset(PACKAGE / 'additional_result_bindings/dense_two_action_comparison.json'),
        runs=[summary(faster_root / 'evaluation' / f'eval_0000_{case}_{step}.json')
              for case in ('original', 'dense') for step in ('1ms', 'half_ms')],
        current_acceptance_reaudit=False,
        recommendation='Retain as a recoverable historical fixed-preview candidate; do not call the selected 15.875 s model the fastest historical neural comparator. Current arc-reserve, ordered-interval and model-interface verification remains required before adding this alternate to the new formal table.')
    nnc_restore = ROOT / 'audit/nnc_recovered_inputs_v5.json'
    restored = read(nnc_restore)
    nnc = dict(id='project_NNC_adapter_checkpoint_24d263a0',
        model=asset(restored['checkpoint']),
        source_restoration=asset(nnc_restore),
        targeted_checkpoint_search=asset(ROOT / 'audit/nnc_checkpoint_targeted_search_v5.json'),
        all_six_registered_sources_restored=restored['all_six_registered_sources_exact'],
        historical_identity_audit=asset(ROOT / 'audit/nnc_method_identity_readonly_v5.json'),
        current_reference_trace_inventory=asset(ROOT / 'audit/nnc_existing_trace_reference_inventory_v5.json'),
        old_records=[],
        architecture='Project PPOContinuous actor with 14 observations, hidden width 256 and two outputs; external paper DDPG algorithm identity is not established by this checkpoint.',
        training_budget='Configuration seed 42, 900 episodes and 7200 s; actual complete training steps and measured total cost are not established by the recovered configuration alone.',
        restored_vs_removed_protection='The archived OriginalNNCPlanarEnv resets angular-kinematic feed limits and turn-prebraking; the matching archived base CNC environment contains those guards. The old prefix therefore represents protection-removal diagnosis. Preserve the original project guards if preparing a new project-adapter comparison; do not add a new KCM and call it original NNC.',
        complete_time_comparison_eligible_now=False,
        recommendation='Inputs now permit preparation of a separately registered project-adapter replay preserving original protections. Existing one-second prefixes are diagnostic only, not complete-attempt failures or completion-time results. No full-path result is inferred from recovered model bytes.')
    for case in ('original', 'dense'):
        p = PACKAGE / 'nnc_diagnostic' / f'{case}.json'
        d = read(p)
        nnc['old_records'].append(dict(case=case, result=asset(p), trace=asset(p.with_suffix('.npz')),
            elapsed_prefix_s=d['observed_prefix_time_s'], complete_time_s=None,
            full_task_peaks=False, complete_time_comparison_eligible=False,
            reason='Registered diagnostic termination after infeasibility with up to 1 s recorded; not a complete path attempt.'))
    out = dict(schema=1, created_utc=utc(),
        scope='Read-only retained original/dense comparator inventory. No new training, rollout, model selection or manuscript edit.',
        nominal_tolerance_mm=1.25, planar_limits=[100., 2000., 62500.],
        required_preserved_baselines=['S2-Q', 'S2-R', 'SCMO', 'COPP'],
        candidates=[fixed, faster, nnc],
        update_to_previous_search='The historical NNC checkpoint-not-recovered statement is superseded within this inventory by two exact registered SHA-256 matches; old audit files remain unmodified historical records.')
    destination = ROOT / 'audit/candidate_comparator_inventory_v5.json'
    write_json(destination, out)
    print('WROTE', destination, 'candidates', len(out['candidates']))


if __name__ == '__main__':
    main()
