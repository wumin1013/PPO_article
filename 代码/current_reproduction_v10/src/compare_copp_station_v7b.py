"""Run the source-tangent repair as a new, separately registered configuration."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import compare_copp_station_v7 as driver
import compare_traditional_representative_v7 as base
from common import ROOT, sha, write_json
from g01_ordered_simplification_v7 import install as install_geometry
from copp_station_coalescing_v7b import install as install_grid

driver.OUTPUT = ROOT / 'results/new_representative_v7/traditional/copp_station_v7b'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('validation', 'formal'), required=True)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--case')
    args = parser.parse_args()
    driver.configure(args.stage)
    if args.register:
        existed = base.REG.exists()
        registration = driver.register(args.stage)
        if not existed:
            registration.update(study_id='copp_source_tangent_station_v7b_' + args.stage,
                numerical_derivative_repair='Line remnants<=1e-10mm use their untrimmed source tangent only after agreement with both adjacent curve tangents; no positions, lengths, tolerances or solver settings change.',
                previous_attempts='results/new_representative_v7/traditional/copp_station_v7')
            for name in ('compare_copp_station_v7b.py', 'copp_station_coalescing_v7b.py'):
                registration['source_sha256'][name] = sha(ROOT / 'src' / name)
            write_json(base.REG, registration)
        base.verify()
        print(json.dumps(dict(registration=str(base.REG), sha256=sha(base.REG), cells=len(registration['grid']))))
    elif args.all:
        base.verify()
        for case in base.CASES:
            if all((base.OUT / (base.run_id(case, 'COPP', dt)+'.json')).exists() for dt in base.PERIODS):
                continue
            with (base.OUT / (case+'.log')).open('x', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                    '--stage', args.stage, '--case', case], stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError('Preserved worker failure: '+case)
            receipt = driver.summarize()
            print(json.dumps(dict(attempted=receipt['attempted_cells'], successful=receipt['successful_cells'])), flush=True)
    elif args.case:
        base.verify()
        install_geometry()
        install_grid()
        base.run(args.case, 'COPP')
        driver.summarize()
    else:
        parser.error('Choose register, all or case')
