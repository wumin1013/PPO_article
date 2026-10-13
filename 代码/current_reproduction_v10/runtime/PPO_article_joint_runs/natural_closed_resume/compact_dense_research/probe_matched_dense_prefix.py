"""Reuse the verified prefix diagnostic for the matched12000 parent only."""
from pathlib import Path

source = Path(__file__).with_name('probe_supplement_dense_prefix.py')
code = source.read_text(encoding='utf-8')
replacements = {
    "parent=ROOT/f'supplement_margin_main_{args.seed}_ppo24'":
        "parent=ROOT/'experimental_supplement_20260923/matched_direction_initialization'/f'seed_{args.seed}'/'three'",
    "checkpoint=parent/'checkpoint_0024.pt'": "checkpoint=parent/'fit_12000.pt'",
    "trace=parent/f'evaluation/eval_0024_dense_{key}.npz'":
        "trace=parent/f'evaluation/eval_12000_dense_{key}.npz'",
    "'experimental_supplement_20260923/dense_prefix_recovery_probe'":
        "'experimental_supplement_20260923/matched_dense_prefix_recovery_probe'",
    "[Path(__file__),checkpoint,trace,": "[Path(__file__),source,checkpoint,trace,",
}
for old, new in replacements.items():
    assert code.count(old) == 1, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'))
