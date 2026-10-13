"""Reuse the verified arrival diagnostic on the original complete reference."""
from pathlib import Path

wrapper_source=Path(__file__).with_name('probe_registered_dense_prefix.py')
code=wrapper_source.read_text(encoding='utf-8')
old="f'eval_*_dense_{key}.npz'"
assert code.count(old)==1
code=code.replace(old,"f'eval_*_original_{key}.npz'")
old='replacements={'
new='''replacements={
    "geometry_path=CASE/'geometry.json'": "geometry_path=Path(r'D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/natural_closed_case/geometry.json')",
    "additional_trial='Frozen73 teacher feedback from the same physical start.'": "additional_trial='Frozen73 teacher feedback; only zero-preview-offset trials may initialize the two-output control pair.'",
'''
assert code.count(old)==1
code=code.replace(old,new)
old='"[Path(__file__),source,checkpoint,trace,"'
new='"[Path(__file__),source,wrapper_source,checkpoint,trace,"'
assert code.count(old)==1
code=code.replace(old,new)
exec(compile(code,str(wrapper_source),'exec'))
