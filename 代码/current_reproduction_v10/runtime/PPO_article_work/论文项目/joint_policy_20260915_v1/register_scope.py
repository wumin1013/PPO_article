"""Record confirmed scope and back up dependencies without executing experiments."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
I = ROOT / '论文项目/revision_20260913_preview_efficiency_v2'
J = I / 'ts_local_revision_v1'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

if __name__ == '__main__':
    receipt = BASE / 'scope_receipt.json'
    if receipt.exists():
        raise RuntimeError('Scope already registered; reuse its receipt')
    sources = [I/'recursive_kcm_roundoff_v1/projector.py', I/'kcm_projection_audit_v1/disks.py',
               I/'no_kcm_coverage_v2/tracking.py', ROOT/'PPO_project/src/utils/geometry.py',
               ROOT/'PPO_project/src/utils/ordered_coverage.py', J/'root_goal_observation.json']
    sources += list((I/'geometry_design_v1/references').glob('*.csv'))
    before = BASE / 'before'
    before.mkdir(exist_ok=True)
    with zipfile.ZipFile(before/'frozen_dependencies.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p,p.relative_to(ROOT).as_posix())
    stamps = {p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    value = dict(status='confirmed_user_scope_registered', created_utc=datetime.now(timezone.utc).isoformat(),
        goal_name='目标', goal_complete=False, new_training_authorized=True,
        necessary_method_description_changes_authorized=True,
        supersedes=['No new training', 'Method chapter may only add one dimension'],
        original_english_style_required=True, technical_report_manuscript_style_forbidden=True,
        backup=dict(path=(before/'frozen_dependencies.zip').relative_to(ROOT).as_posix(),sha256=digest(before/'frozen_dependencies.zip')),
        source_sha256=stamps, model_calls=0, physics_ticks=0, training_updates=0,
        current_stage='Geometry semantics and independent synthetic validation; not a performance result')
    receipt.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    prior=json.loads((J/'root_goal_observation.json').read_text(encoding='utf-8'))
    observation=json.loads((BASE/'goal_tool_observation.json').read_text(encoding='utf-8'))
    assert observation['goal']['status']=='active'
    updated=dict(source='root_report_of_actual_goal_tool', reported_tool='get_goal', status='active',
        goal_name='目标', recorded_utc=datetime.now(timezone.utc).isoformat(), is_unmodified_tool_result=False,
        raw_observation=dict(path=(BASE/'goal_tool_observation.json').relative_to(ROOT).as_posix(),sha256=digest(BASE/'goal_tool_observation.json')),
        previous_task_observation=prior, research_complete=False,
        note='User confirmed true joint learning, new training and faithful method changes; prior frozen scopes preserved as history.')
    (J/'root_goal_observation.json').write_text(json.dumps(updated,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (J/'joint_policy_scope_v1.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':value['status'],'backup':value['backup'],'goal_status':'active'},ensure_ascii=False))
