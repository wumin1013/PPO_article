"""Existing witnessed-prefix probe with the explicit static-radius policy loader."""
from pathlib import Path

wrapper_source=Path(__file__).with_name('probe_registered_dense_prefix.py')
code=wrapper_source.read_text(encoding='utf-8')
old='replacements={'
new='''replacements={
    "from localized_two_action_policy import policy_from": "from variable_radius_recovery_policy import policy_from",
'''
assert code.count(old)==1
code=code.replace(old,new)
old='"[Path(__file__),source,checkpoint,trace,"'
new='"[Path(__file__),source,wrapper_source,ROOT/\'variable_radius_recovery_policy.py\',checkpoint,trace,"'
assert code.count(old)==1
code=code.replace(old,new)
exec(compile(code,str(wrapper_source),'exec'))
