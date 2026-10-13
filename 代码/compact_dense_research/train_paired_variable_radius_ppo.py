"""Paired PPO protocol unchanged; add the shared per-center-radius factory."""
from pathlib import Path

source = Path(__file__).with_name('train_paired_supplement_ppo.py')
code = source.read_text(encoding='utf-8')
old = 'from localized_two_action_policy import policy_from'
assert code.count(old) == 1
code = code.replace(old, 'from paired_variable_radius_policy import policy_from')
old = 'wrapper_sha256=sha(__file__),'
assert code.count(old) == 2
code = code.replace(old, "wrapper_sha256=sha(__file__), additional_loader_sha256={str(p):sha(p) for p in [source,ROOT/'paired_variable_radius_policy.py',ROOT/'variable_radius_recovery_policy.py',ROOT/'localized_two_action_policy.py',ROOT/'localized_joint_correction.py',ROOT/'localized_preview_head.py']},")
exec(compile(code, str(source), 'exec'))
