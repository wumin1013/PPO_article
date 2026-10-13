"""Validate a single selected launch setting plus a fixed terminal family."""
import sys
from pathlib import Path
import run_efficiency as study
import combined_runtime
from common import write_json,sha,utc
import evaluate

PARENT=study.ROOT
PARENT_REG=study.REG
CONTROL=PARENT/'launch_study'
study.ROOT=PARENT/'combined_study'
study.REG=study.ROOT/'registration.json'
study.__file__=__file__
sys.modules['efficiency_runtime']=combined_runtime
base_rollout=evaluate.rollout
def instrumented_rollout(*args,**kwargs):
    result=base_rollout(*args,**kwargs)
    result['terminal_candidate_diagnostics']=combined_runtime.LAST_TERMINAL
    return result
evaluate.rollout=instrumented_rollout


def register():
    assert not study.REG.exists()
    selected=study.read(CONTROL/'selection.json')
    assert selected['adoptable'] and selected['selected_omega']==60.
    reg=study.read(PARENT_REG)
    sources=[Path(study.__spec__.origin),Path(__file__),Path(combined_runtime.__file__),
        PARENT/'src/launch_tracking.py',PARENT/'terminal_research/terminal_candidate.py']
    reg.update(created_utc=utc(),version='v9_launch_and_verified_terminal',candidates=[60.],
        parent_registration_sha256=sha(PARENT_REG),own_source_sha256={},
        source_files={str(p):sha(p) for p in sources},
        control_selection_sha256=sha(CONTROL/'selection.json'),
        dependency_files={str(p):sha(p) for p in (study.BASE/'literature/ruckig_vendor').glob('*.pyd')},
        selection='Single fixed composite: retain every complete launch60 train/validation cell, then require no increase in JNN summed validation time. Otherwise do not adopt. No further bandwidth choice.',
        terminal_trigger='Same original cubic-qualified 20ms boundary; shortest fully geometry/vector accepted S-curve candidate, fallback original cubic; original final rest count retained.',
        terminal_family=dict(scalar_fraction=.98,constant_acceleration_prefix_max_s=.03,normal='cubic',endpoint_trigger_mm=1.),
        motivation='Finite launch passed validation; test a mechanically defined terminal family already full-trajectory audited on archived states. Formal and all preceding trials remain retained.')
    write_json(study.REG,reg)
    print('REGISTERED',sha(study.REG),flush=True)


base_verify=study.verify
def verify():
    reg=base_verify()
    assert sha(PARENT_REG)==reg['parent_registration_sha256']
    assert sha(CONTROL/'selection.json')==reg['control_selection_sha256']
    for key in ('source_files','dependency_files'):
        for p,value in reg[key].items():assert sha(p)==value,p
    return reg
study.verify=verify


def select():
    reg=verify();assert not (study.ROOT/'selection.json').exists()
    rows=[];lost=[];before=after=0.
    for case in reg['cases']:
        if case['historical_split'] not in ('train','validation'):continue
        for method in reg['methods']:
            new_path=study.output_path(case['id'],method,60.,.001,'tuning').with_suffix('.json')
            old_path=CONTROL/'results/tuning/omega60'/new_path.name
            new=study.read(new_path);old=study.read(old_path)
            assert new['registration_sha256']==sha(study.REG)
            assert sha(new_path.with_suffix('.npz'))==new['trace_sha256']
            if old['success'] and not new['success']:lost.append([case['id'],method])
            if case['historical_split']=='validation' and method=='JNN':
                assert old['success']
                before+=old['complete_time_s']
                after+=new['complete_time_s'] if new['success'] else 1e9
            rows.append(dict(case=case['id'],method=method,before_s=old['complete_time_s'],
                success=new['success'],after_s=new['complete_time_s'],reason=new['reason']))
    result=dict(created_utc=utc(),registration_sha256=sha(study.REG),selected_omega=60.,
        adoptable=not lost and after<=before+1e-12,lost_complete_cells=lost,
        validation_before_s=before,validation_after_s=after,records=rows)
    write_json(study.ROOT/'selection.json',result)
    print(result,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','matrix','worker','select'))
    parser.add_argument('--phase',choices=('tuning','formal','early_case_replay'),default='tuning')
    parser.add_argument('--case');parser.add_argument('--method',choices=('JNN','FIXED'))
    parser.add_argument('--omega',type=float);parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='matrix':study.matrix(args)
    elif args.command=='worker':study.worker(args)
    else:select()
