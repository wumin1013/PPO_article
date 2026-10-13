"""Check one fitted candidate from the parent's verified arrival, not full success."""
from pathlib import Path

source=Path(__file__).with_name('probe_supplement_dense_prefix.py')
code=source.read_text(encoding='utf-8')
replacements={
    "parent=ROOT/f'supplement_margin_main_{args.seed}_ppo24'": "parent=ROOT/'experimental_supplement_20260923/matched_direction_initialization'/f'seed_{args.seed}'/'three'",
    "checkpoint=parent/'checkpoint_0024.pt'": "checkpoint=parent/'fit_12000.pt'",
    "trace=parent/f'evaluation/eval_0024_dense_{key}.npz'": "trace=parent/f'evaluation/eval_12000_dense_{key}.npz'",
    "teacher_path=ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'": "teacher_path=ROOT/'experimental_supplement_20260923/matched_local_recovery'/f'seed_{args.seed}'/'checkpoint_0000.pt'",
    "'experimental_supplement_20260923/dense_prefix_recovery_probe'": "'experimental_supplement_20260923/matched_local_recovery_execution_check'",
    "[Path(__file__),checkpoint,trace,": "[Path(__file__),source,checkpoint,trace,",
    "trials=[('offset',np.array(delta)) for delta in itertools.product([-.002,0.,.002],[-.02,0.,.02],[-.05,0.,.05])]": "trials=[]",
    "len(records)==28": "len(records)==1",
    "additional_trial='Frozen73 teacher feedback from the same physical start.'": "additional_trial='Fitted local candidate feedback from the same physical start.'",
    "offset_grid={'steer':[-.002,0.,.002],'feed':[-.02,0.,.02],'preview':[-.05,0.,.05]}": "offset_grid=None",
    "'28trials or300s; preserve failed prefixes and no-completion scope.'": "'One fitted-policy trial or300s; retain no-completion scope.'",
    "'Frozen73 teacher feedback from the same physical start.'": "'One fitted-policy check; no frozen73 execution.'",
}
# This replacement is only relevant to the human-readable old registration.
replacements.pop("'Frozen73 teacher feedback from the same physical start.'")
for old,new in replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
code=code.replace("('frozen_teacher',np.zeros(3))", "('fitted_candidate',np.zeros(3))")
code=code.replace("kind=='frozen_teacher'", "kind=='fitted_candidate'")
exec(compile(code,str(source),'exec'))
