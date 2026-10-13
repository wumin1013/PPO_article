import json
from pathlib import Path
import hashlib
import re

ROOT = Path(r'D:\PPO_article_work')
BASE = ROOT / '论文项目/joint_policy_20260915_v1'
OUT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume')
R = OUT / 'all34_dominance_research'
D = BASE / 'paper_evidence_review/verified_results/all34_original_revision'
SRC = ROOT / '论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def shape(v):
    if isinstance(v, dict):
        return {k: (str(type(x).__name__) + ':' + str(len(x)) if isinstance(x, (dict, list)) else x) for k, x in v.items()}
    return str(type(v))

for p in [BASE/'当前执行状态.json', OUT/'live_research_summary.json', R/'active_research_checkpoint.json',
          D/'research_acceptance.json', OUT/'all34_original_revision_build/compile_receipt.json',
          R/'joint_transit_retained_seed922222_full/full_result.json',
          R/'joint_transit_retained_seed922222_full/full_metrics.json',
          R/'joint_transit_retained_seed922222_full/paired_acceptance_fixed_actual_failure_recovery_full/assessment.json',
          R/'joint_transit_retained_seed922222_full/actual_turn_geometry/audit.json']:
    print(p.name, json.dumps(shape(read(p)), ensure_ascii=False))
print('csv_header', (D/'all34_corner_statistics.csv').read_text(encoding='utf-8').splitlines()[:2])
print('source hash', hashlib.sha256((SRC/'manuscript.tex').read_bytes()).hexdigest())
meta = read(BASE/'manuscript_reference_metadata_v1/results.json')['results']
print('metadata key status', [{k:x.get(k) for k in ['key','status','doi','retrieved_title','retrieved_years']} for x in meta])
print('pdf info', list((OUT/'all34_original_revision_build').glob('*.bbl')))
print('original figs', list((Path(read(D/'revision_origin.json')['base_source']).parent/'figures').glob('Figure_[123].pdf')))
