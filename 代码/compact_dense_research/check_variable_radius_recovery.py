"""Run the registered recovery check with the explicit variable-radius loader."""
from pathlib import Path

wrapper_source=Path(__file__).with_name('check_registered_dense_recovery.py')
code=wrapper_source.read_text(encoding='utf-8')
needle='candidate_replacements={'
addition='''candidate_replacements={
    "from localized_two_action_policy import policy_from": "from variable_radius_recovery_policy import policy_from",
'''
assert code.count(needle)==1
code=code.replace(needle,addition)
old='"[Path(__file__),source,generic_source,checkpoint,"'
new='"[Path(__file__),source,generic_source,wrapper_source,ROOT/\'variable_radius_recovery_policy.py\',checkpoint,"'
assert code.count(old)==1
code=code.replace(old,new)
exec(compile(code,str(wrapper_source),'exec'))
