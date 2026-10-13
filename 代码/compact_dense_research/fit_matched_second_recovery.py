"""Bounded second local fit with unchanged actor capacity and protected long path."""
from pathlib import Path

wrapper_source=Path(__file__).with_name('fit_matched_local_recovery.py')
code=wrapper_source.read_text(encoding='utf-8')
replacements={
    "parent_stage = sup/'matched_direction_initialization'/f'seed_{args.seed}'/'three'": "parent_stage = sup/'matched_local_recovery'/f'seed_{args.seed}'",
    "parent = parent_stage/'fit_12000.pt'": "parent = parent_stage/'checkpoint_0000.pt'",
    "probe = sup/'matched_dense_prefix_recovery_probe'/f'seed_{args.seed}'/'half_ms'": "probe = sup/'matched_local_second_probe'/f'seed_{args.seed}'/'half_ms'",
    "stage = sup/('matched_local_recovery_'+args.variant)/f'seed_{args.seed}'": "stage = sup/'matched_local_second_recovery'/f'seed_{args.seed}'/'attempt_2'",
    "probe/'registration.json', Path(__file__)]": "probe/'registration.json', Path(__file__),wrapper_source]",
    "f'evaluation/eval_12000_{case}_{key}.npz'": "f'evaluation/eval_0000_{case}_{key}.npz'",
    "maximum_reallocated_features=32": "maximum_reallocated_features=16",
    "len(free)>=32": "len(free)>=16",
    "kind='stable')[:32]": "kind='stable')[:16]",
    "assert nearest.min()>1.05*radius, 'New support overlaps protected original states'":
        "eligible=nearest>1.05*radius\n    assert eligible.sum()>=20, 'Insufficient separated repair targets'\n    px,py,nearest=px[eligible],py[eligible],nearest[eligible]",
    "    assert np.all(features[:len(px),reuse].max(1)>0.), 'New coefficients do not cover every repair target'\n": "",
    "1e-5*np.eye(32)": "1e-5*np.eye(len(reuse))",
    "Matched12000 retains both complete original executions but fails half-step dense execution. A verified actual-arrival local recovery is available.":
        "The first local repair passes the original-route pair and moves dense half-step failure to12.480s. Fit16inactive original-route features from a verified recovery of this actual arrival; omit target centers whose support would touch original observations.",
}
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(wrapper_source),'exec'))
