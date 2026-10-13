"""Keep all three retained fixed seeds eligible for the original full task."""
from pathlib import Path
import sys,json,shutil
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha,write_json

parent=OUT/'fixed_actual_context_feedback_training'
checks=json.loads((parent/'all34_selection.json').read_text(encoding='utf-8'))
inherited=json.loads((OUT/'fixed_curve_context_cruise_training/selected_policy.json').read_text(encoding='utf-8'))
stage=parent/'full_context_candidates';stage.mkdir(exist_ok=False)
retained=[dict(label=r['label'].removesuffix('_all34'),spec=r['spec'],score=r['context_key'][-1],isolated_all34_success=r['success']) for r in checks]
assert len(retained)==3
write_json(stage/'selected_policy.json',dict(spec=retained[0]['spec'],selected_from='nominated_by_actual_context_score_not_full_acceptance',
    model=inherited['model'],model_sha256=inherited['model_sha256'],seed_best=retained,goal_complete=False))
write_json(stage/'registration.json',dict(arm='fixed',kind='retained_candidates_for_full_task_validation',ppo_updates=0,
    seeds=[921221,921222,921223],parent_all34_sha256=sha(parent/'all34_selection.json'),
    reason='All three failures occurred after all local ordered gates, near isolated snippet endpoints. Preserve these failures, and assess every retained seed on the unmodified full natural task rather than exclude them from the fixed comparator.',
    selection='Keep the fastest completely successful fixed result, including previously verified results. No fitted parameters or changed task boundaries.',
    individual_round_wall_budget_s=1800,independent_full_pipeline_ppo=False))
shutil.copy2(__file__,stage/Path(__file__).name)
print(stage)
