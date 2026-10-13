"""Selected primary belongs to the three students; do not double-count it."""
from pathlib import Path

source=Path(__file__).with_name('draw_selected_student_efficiency.py')
code=source.read_text(encoding='utf-8')
replacements={
 "delivery = BASE/'paper_evidence_review/verified_results/lowdim_original_revision'": "delivery = BASE/'paper_evidence_review/verified_results/lowdim_original_revision/selected923202_revision'",
 "'experimental_supplement_20260923/three_selected_student_acceptance.json'": "'experimental_supplement_20260923/three_selected_student_acceptance_v2.json'",
 "students=[('Main model',[v['complete_time_s'] for v in binding['methods']['J-NNC']['dense']])]": "students=[]",
 "[binding_path, acceptance_path, args.baseline_audit, Path(__file__)]": "[binding_path, acceptance_path, args.baseline_audit, Path(__file__), source]",
 "[binding_path,acceptance_path,args.baseline_audit,Path(__file__)]": "[binding_path,acceptance_path,args.baseline_audit,Path(__file__),source]",
 "Dense gains of frozen model and three accepted students at both steps": "Dense-case gains of three accepted students including the selected primary at both steps",
 "(c) Dense-case efficiency across selected student histories": "(c) Dense-case gains of three selected policies",
 "'5% criterion'": "'5% reduction'",
 "ylim=(3.5,-.75)": "ylim=(2.55,-.75)",
 "yticks=range(4)": "yticks=range(3)",
 "for the frozen main model and three independently initialized students.": "for three independently initialized students, including the main model (seed923202).",
}
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'))
