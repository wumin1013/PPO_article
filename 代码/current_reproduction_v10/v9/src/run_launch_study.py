"""Finite launch response study on the same fixed development grid."""
import sys
from pathlib import Path
import run_efficiency as study
import launch_tracking
from common import write_json,sha,utc

PARENT=study.ROOT
PARENT_REG=study.REG
study.ROOT=PARENT/'launch_study'
study.REG=study.ROOT/'registration.json'
study.__file__=__file__
sys.modules['efficiency_runtime']=launch_tracking


def register():
    assert not study.REG.exists()
    reg=study.read(PARENT_REG)
    reg.update(created_utc=utc(),version='v9_finite_launch',
        parent_registration_sha256=sha(PARENT_REG),own_source_sha256={},
        source_files={str(p):sha(p) for p in (Path(study.__spec__.origin),Path(__file__),Path(launch_tracking.__file__))},
        motivation='Global gain changed turning response; the straight-rate gate gave no validation benefit. Apply only a smooth tangential launch ramp over six fixed policy periods, preserving the subsequent original demand.',
        launch_window_s=.12,launch_blend='(1-(t/0.12)^2)^2',
        normal_and_subsequent_tracking_unchanged=True,
        all_prior_runs_preserved=True)
    write_json(study.REG,reg)
    print('REGISTERED',sha(study.REG),flush=True)


base_verify=study.verify
def verify():
    reg=base_verify()
    assert sha(PARENT_REG)==reg['parent_registration_sha256']
    for p,value in reg['source_files'].items():assert sha(p)==value,p
    return reg
study.verify=verify


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','matrix','worker','select'))
    parser.add_argument('--phase',choices=('tuning','formal'),default='tuning')
    parser.add_argument('--case');parser.add_argument('--method',choices=('JNN','FIXED'))
    parser.add_argument('--omega',type=float);parser.add_argument('--dt',type=float,default=.001)
    parser.add_argument('--jobs',type=int,default=2)
    args=parser.parse_args()
    if args.command=='register':register()
    elif args.command=='matrix':study.matrix(args)
    elif args.command=='worker':study.worker(args)
    else:study.select()
