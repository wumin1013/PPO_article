"""Reuse the registered rollout/matrix with an explicitly bound alternative demand."""
import sys
from pathlib import Path
import run_efficiency as common_study
from common import write_json, sha, utc
import straight_tracking

STUDY = Path(__file__).resolve().parents[1]/'straight_study'
PARENT = common_study.ROOT
PARENT_REG = common_study.REG
common_study.ROOT = STUDY
common_study.REG = STUDY/'registration.json'
# Matrix children must enter this driver and use this named runtime.
common_study.__file__ = __file__
sys.modules['efficiency_runtime'] = straight_tracking


def register():
    reg = common_study.read(PARENT_REG)
    assert not common_study.REG.exists()
    reg.update(created_utc=utc(),version='v9_straight_tangential',
        parent_registration=str(PARENT_REG),parent_registration_sha256=sha(PARENT_REG),
        motivating_result='Global high bandwidth caused additional corridor failures in development; preserve normal response and restrict the extra tangential demand to straight acceleration.',
        own_source_sha256={str(p.relative_to(STUDY.parent)):sha(p) for p in []},
        candidate_runtime='src/straight_tracking.py',
        source_files={str(p):sha(p) for p in (Path(common_study.__spec__.origin),Path(__file__),Path(straight_tracking.__file__))},
        straight_rate_cutoff_rad_s=.1,straight_heading_sine_cutoff=.01,
        terminal_trigger='Unchanged verified cubic at policy entry.',
        formal_scope='Four previously observed representative programs, both resolutions; one common selected setting.')
    write_json(common_study.REG,reg)
    print('REGISTERED',sha(common_study.REG),flush=True)


base_verify = common_study.verify
def verify():
    reg=base_verify()
    assert sha(PARENT_REG)==reg['parent_registration_sha256']
    for p,value in reg['source_files'].items():assert sha(p)==value,p
    return reg
common_study.verify=verify


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
    elif args.command=='matrix':common_study.matrix(args)
    elif args.command=='worker':common_study.worker(args)
    else:common_study.select()
