"""Use the unchanged physical evaluation with the additional static-radius loader."""
from pathlib import Path

source=Path(__file__).with_name('evaluate_supplement_checkpoint.py')
code=source.read_text(encoding='utf-8')
old='from train_localized_two_action_ppo import physics_worker'
assert code.count(old)==1
code=code.replace(old,'from variable_radius_recovery_policy import physics_worker')
old="sources = [Path(__file__)] + [ROOT/name for name in ("
assert code.count(old)==1
code=code.replace(old,"sources = [Path(__file__), source, ROOT/'variable_radius_recovery_policy.py'] + [ROOT/name for name in (")
exec(compile(code,str(source),'exec'))
