"""Refresh a truthful live status from complete, incomplete and failed records."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from study_io import sha,write_json

def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError,json.JSONDecodeError):return None

selected=OUT/'joint_actual_context_feedback_training/exact_continuation'
main=read(selected/'full_result.json')
fixed=[]
for folder in OUT.glob('fixed_*'):
    if folder.is_dir():
        for p in folder.rglob('full_result.json'):
            r=read(p)
            if r and r.get('success'):fixed.append((r['complete_time_s'],p))
ft,fp=min(fixed)
records=[]
for family in ('joint_actual_context_feedback_training','fixed_actual_context_feedback_training'):
    for p in (OUT/family).rglob('registration.json'):
        reg=read(p);receipt=read(p.parent/'receipt.json') or read(p.parent/'process_exit_observation.json')
        if not reg:continue
        result=read(p.parent/'full_result.json') or read(p.parent/'result.json')
        records.append(dict(stage=str(p.parent),kind=reg.get('kind'),status=receipt.get('status') if receipt else 'running_or_queued',
            success=result.get('success') if result else None,complete_time_s=result.get('complete_time_s') if result else None,
            reason=(result.get('error') or result.get('reason')) if result else None,
            wall_s=(receipt.get('wall_s',receipt.get('total_wall_s')) if receipt else None)))
old=OUT/'live_research_summary.json'
if old.exists():
    backup=OUT/'live_state_history';backup.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');shutil.copy2(old,backup/(stamp+'.json'))
screen=read(selected/'numerical_acceptance.json')
status=dict(updated_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False,method_frozen=False,
    baseline18_verified=True,baseline_source_binding=str(OUT/'baseline_source_binding.json'),
    branch='codex/complex-path-adaptive-preview',historical_main_time_s=450.119,s2_time_s=445.185,
    selected_candidate=str(selected),selected_complete_success=main['success'],selected_complete_time_s=main['complete_time_s'],
    selected_result_sha256=sha(selected/'full_result.json'),strongest_fixed_time_s=ft,strongest_fixed_result=str(fp),
    provisional_gain_over_s2_s=445.185-main['complete_time_s'],provisional_gain_over_fixed_s=ft-main['complete_time_s'],
    selected_cruise99_s=main['motion']['cruise_ge99_s'],all34_first_detectable_turns_inward=True,
    last_numerical_screen=screen,current_limit='Strongest fixed full outcomes, retained conditional seeds and remaining steering intervention are still pending; rerun numerical acceptance if a stronger fixed candidate succeeds. No freeze or final-paper acceptance.',
    independent_full_pipeline_ppo_seeds=0,conditional_search_seeds=3,new_ppo_updates=0,sealed_tests_accessed_in_this_resume=False,
    no_subagents=True,no_branch_change=True,no_delete_commit_push=True,records=records,
    next='Finish immutable half-step check, all retained fixed and joint full outcomes, and at most two declared action interventions. Preserve budget prefixes and continue only after exact causal replay.')
write_json(old,status)
print(json.dumps({k:v for k,v in status.items() if k!='records'},ensure_ascii=False,indent=2))
