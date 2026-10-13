"""One closed-loop candidate check at a hash-bound parent's failed arrival."""
from pathlib import Path

generic_source=Path(__file__).with_name('probe_registered_dense_prefix.py')
code=generic_source.read_text(encoding='utf-8')
code=code.replace('options=parser.parse_args()',"parser.add_argument('--candidate',type=Path,required=True)\noptions=parser.parse_args()")
needle="sys.argv=[str(Path(__file__)),'--seed',str(options.seed),'--dt',str(options.dt)]"
insertion='''
candidate_replacements={
    "teacher_path=ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'": "teacher_path=options.candidate",
    "trials=[('offset',np.array(delta)) for delta in itertools.product([-.002,0.,.002],[-.02,0.,.02],[-.05,0.,.05])]": "trials=[]",
    "len(records)==28": "len(records)==1",
    "additional_trial='Frozen73 teacher feedback from the same physical start.'": "additional_trial='One fitted-policy closed-loop recovery check.'",
    "offset_grid={'steer':[-.002,0.,.002],'feed':[-.02,0.,.02],'preview':[-.05,0.,.05]}": "offset_grid=None",
    "'28trials or300s; preserve failed prefixes and no-completion scope.'": "'One fitted-policy trial or300s; retain no-completion scope.'",
    "[Path(__file__),source,checkpoint,": "[Path(__file__),source,generic_source,checkpoint,",
}
for old,new in candidate_replacements.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
code=code.replace("('frozen_teacher',np.zeros(3))", "('fitted_candidate',np.zeros(3))")
code=code.replace("kind=='frozen_teacher'", "kind=='fitted_candidate'")
'''
assert code.count(needle)==1
code=code.replace(needle,insertion+'\n'+needle)
exec(compile(code,str(generic_source),'exec'))
