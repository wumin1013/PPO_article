"""Early full replays of the already selected configuration, without reselection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import run_launch_study as driver
from common import write_json,sha,utc

study=driver.study
REG=study.ROOT/'early_case_replay_registration.json'
CASES=['piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case');parser.add_argument('--method');parser.add_argument('--dt',type=float)
    args=parser.parse_args()
    config=study.verify();selection=study.read(study.ROOT/'selection.json')
    assert selection['adoptable'] and selection['selected_omega']==60.
    if args.case:
        reg=study.read(REG)
        assert sha(__file__)==reg['runner_sha256']
        assert sha(study.ROOT/'selection.json')==reg['selection_sha256']
        args.omega=60.;args.phase='early_case_replay'
        study.worker(args)
    else:
        assert not REG.exists()
        jobs=[dict(case=c,method=m,dt=d) for c in CASES for m in ('JNN','FIXED') for d in (.001,.0005)]
        write_json(REG,dict(created_utc=utc(),runner_sha256=sha(__file__),
            registration_sha256=sha(study.REG),selection_sha256=sha(study.ROOT/'selection.json'),
            purpose='Early full replay while long original runs proceed. Same selected configuration; no parameter selection or pooled replicate statistics. Formal matrix remains registered and retained.',
            jobs=jobs,omega=60.,all_attempts_retained=True))
        def run(spec):
            path=study.output_path(spec['case'],spec['method'],60.,spec['dt'],'early_case_replay')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.log').open('x',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-B',__file__,'--case',spec['case'],
                    '--method',spec['method'],'--dt',str(spec['dt'])],stdout=stream,stderr=subprocess.STDOUT,
                    timeout=config['wall_budget_s']+180.)
            print(spec,result.returncode,flush=True)
            return result.returncode
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert not any(pool.map(run,jobs))
