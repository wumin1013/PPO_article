"""Rebind the original-format paper to the newly accepted primary student."""
from pathlib import Path

source=Path(__file__).with_name('prepare_lowdim_manuscript_evidence.py')
code=source.read_text(encoding='utf-8')
replacements={
 "selected = ROOT / 'direct14_dictionary02_resume_round73'": "selected = ROOT / 'experimental_supplement_20260923/selected_main_923202_evidence'",
 "'joint_numeric_and_mechanism_acceptance_20260923.json'": "'selected_acceptance_20260924.json'",
 "output = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'": "output = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'",
 "matched = ROOT / 'matched14_two_action_evidence_20260923.json'": "matched = ROOT / 'experimental_supplement_20260923/matched_direction_initialization/seed_923202/three/evaluation/current_two_action_comparison_v2.json'",
 "audit = read(selected / 'paired_resolution_extended_audit.json')": "audit = {'matched14_two_action_times_s': read(matched)['dense_envelope_s']}",
 "source_sha256={str(acceptance): sha(acceptance)}": "source_sha256={str(p):sha(p) for p in [acceptance,Path(__file__),source]}",
}
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'))
