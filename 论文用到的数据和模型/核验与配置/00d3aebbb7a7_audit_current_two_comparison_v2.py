"""Versioned comparison adds the recovered shared start and preserves v1 audits."""
from pathlib import Path

source = Path(__file__).with_name('audit_current_two_comparison.py')
code = source.read_text(encoding='utf-8')
old = "directories=[]"
new = "directories=[sup/'paired_seed2_initial_recovery/seed_923202/attempt_2/two/evaluation']"
assert code.count(old) == 1
code = code.replace(old, new)
old = "checkpoint=Path(read(directory/'registration.json')['checkpoint'])"
new = "checkpoint=(directory/'checkpoint_0000.pt') if directory == ROOT/'direct14_dictionary02_resume_round73' else Path(read(directory/'registration.json')['checkpoint'])"
assert code.count(old) == 1
code = code.replace(old, new)
old = "bindings={str(Path(__file__)):sha(__file__)}"
new = "bindings={str(p):sha(p) for p in [Path(__file__),source]}"
assert code.count(old) == 1
code = code.replace(old, new)
old = "args.evaluation/'current_two_action_comparison.json'"
assert code.count(old) == 1
code = code.replace(old, "args.evaluation/'current_two_action_comparison_v2.json'")
exec(compile(code, str(source), 'exec'))
