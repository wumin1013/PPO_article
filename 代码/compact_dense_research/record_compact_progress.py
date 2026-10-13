"""Keep the active research state separate from historical accepted controllers."""
import json
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def bind(path):return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

now=datetime.now(timezone.utc)
targets=[BASE/'当前执行状态.json',OUT/'live_research_summary.json',OUT/'all34_dominance_research/active_research_checkpoint.json']
history=ROOT/'state_history'/now.strftime('%Y%m%dT%H%M%S%fZ')
history.mkdir(parents=True,exist_ok=False)
previous=[]
for path in targets:
    copied=history/path.name
    shutil.copy2(path,copied);previous.append(bind(copied))
stages=[]
for path in sorted(ROOT.glob('ppo*/registration.json')):
    stage=path.parent;config=read(path)['config']
    result=read(stage/'result.json') if (stage/'result.json').exists() else None
    live=read(stage/'live.json') if (stage/'live.json').exists() else None
    receipt=read(stage/'receipt.json') if (stage/'receipt.json').exists() else None
    stages.append(dict(directory=str(stage),config=config,result=result,live=live,receipt=receipt))
active=[s['directory'] for s in stages if s['receipt'] is None]
state=dict(updated_utc=now.isoformat(),project='PPO_article',working_directory=str(BASE),
    branch='codex/complex-path-adaptive-preview',current_research_directory=str(ROOT),
    current='紧凑状态真实PPO训练中；新84角案例已注册，S2完整成功；新增5%及独立双动作比较尚未达成。',
    goal_complete=False,method_frozen=False,research_acceptance_passed=False,final_manuscript_accepted=False,
    final_pdf=None,active_stages=active,stage_records=stages,
    historical_states=previous,historical_state_scope='Full preserved records include the old72-dimensional selected controller, previous manuscript and all34 results; none imply compact-model acceptance.',
    dense_case=bind(ROOT/'dense_closed_case/registration.json'),dense_reference=bind(ROOT/'dense_closed_case/reference.csv'),
    dense_s2=dict(result=bind(ROOT/'dense_s2_initial/result.json'),complete_time_s=19.957,target_5percent_s=18.95915),
    compact_initial=bind(ROOT/'compact_initial_probe/results.json'),initial_ppo_updates=0,
    compact_interface=dict(verified_dimensions=[14,12,10],candidate_training_dimension=14,verified_actor_outputs=[2,3],policy_memory_scalars=0),
    compact_dimension_minimality_established=False,new_two_action_controller_trained=any(s['config']['action_dim']==2 and s['result'] and s['result']['ppo_updates_this_round']>0 for s in stages),
    allowed_vs_no_swing_training_comparison_complete=False,independent_three_seed_acceptance_complete=False,
    requested_complete_time_reduction_fraction=.05,requested_5percent_scope='Registered dense case; mean of all3 independent successful seeds against S2 and independently retrained2-output controller, both1ms and0.5ms.',
    original_case_retained=True,original_case_all34_still_required=True,original_long_case_lower_bound=bind(OUT/'all34_dominance_research/five_percent_complete_time_feasibility.json'),
    next_steps=['Inspect real PPO learning and failure causes; adapt after absent gains.',
        'Fairly train two-output and no-swing groups; compare actual paths and times.',
        'Retrain reduced sensor variants and complete three independent seeds.',
        'Verify original44.420514m/all34 and both physical resolutions; freeze only after acceptance.',
        'Revise experiments on the unique18-page source, retain first3 figures and original structure.'])
for target in targets:write(target,state)
write(ROOT/'current_progress.json',state)
print(json.dumps(dict(history=str(history),active_stages=active,stages=len(stages),goal_complete=False),ensure_ascii=False))
