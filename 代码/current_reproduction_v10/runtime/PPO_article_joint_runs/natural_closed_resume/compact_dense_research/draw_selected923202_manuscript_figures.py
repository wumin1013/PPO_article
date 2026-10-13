"""Reuse the prescribed publication layouts, replacing all primary-model data."""
from pathlib import Path

source=Path(__file__).with_name('draw_lowdim_manuscript_figures.py')
code=source.read_text(encoding='utf-8')
replacements={
 "delivery = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision'": "delivery = BASE / 'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'",
 "work = ROOT / 'lowdim_original_manuscript_work'": "work = ROOT / 'lowdim_original_manuscript_work/selected923202'",
 "ROOT / 'direct14_dictionary02_resume_round73/mechanism_0000/figure_review'": "ROOT / 'experimental_supplement_20260923/selected_main_923202_evidence/mechanism_0000/figure_review'",
 "script_sha256=sha(__file__),": "script_sha256=sha(__file__), original_drawing_script_sha256=sha(source),",
}
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'))
