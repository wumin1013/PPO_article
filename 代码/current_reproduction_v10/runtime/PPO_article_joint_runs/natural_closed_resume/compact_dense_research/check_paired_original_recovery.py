"""Reuse the bound local checker on the original route with paired radii."""
from pathlib import Path

wrapper_source = Path(__file__).with_name('check_registered_dense_recovery.py')
code = wrapper_source.read_text(encoding='utf-8')
old = "code=generic_source.read_text(encoding='utf-8')"
new = old + '''
assert code.count("f'eval_*_dense_{key}.npz'") == 1
code=code.replace("f'eval_*_dense_{key}.npz'", "f'eval_*_original_{key}.npz'")
'''
assert code.count(old) == 1
code = code.replace(old, new)
needle = 'candidate_replacements={'
addition = '''candidate_replacements={
    "from localized_two_action_policy import policy_from": "from paired_variable_radius_policy import policy_from",
    "geometry_path=CASE/'geometry.json'": "geometry_path=Path(r'D:/PPO_article_work/论文项目/joint_policy_20260915_v1/paper_evidence_review/natural_closed_case/geometry.json')",
'''
assert code.count(needle) == 1
code = code.replace(needle, addition)
old = '"[Path(__file__),source,generic_source,checkpoint,"'
new = '"[Path(__file__),source,generic_source,wrapper_source,ROOT/\'paired_variable_radius_policy.py\',ROOT/\'variable_radius_recovery_policy.py\',checkpoint,"'
assert code.count(old) == 1
code = code.replace(old, new)
exec(compile(code, str(wrapper_source), 'exec'))
