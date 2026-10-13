"""Four frozen traditional planners on the two registered tenfold programs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import compare_traditional_representative_v7 as base
from common import ROOT, sha, utc, write_json
from g01_ordered_simplification_v7 import simplify, install as install_geometry
from copp_station_coalescing_v7b import install as install_grid

base.OUT = ROOT / 'results/representative_scaled_v8/traditional'
base.REG = base.OUT / 'registration.json'
base.SOURCE = ROOT / 'evidence/scaled_representative_v8_registration.json'
base.GLOBAL = base.SOURCE
base.CASES = ('piecewise_curvature_coupon_scale10', 'asymmetric_neck_contour_scale10')
RESOURCE_LIMIT_S = 600.


def register():
    if base.REG.exists():
        return base.verify()
    checks = {}
    for relative, expected in (
        ('results/new_representative_v7/traditional/g01_v7/validation/summary.json', 16),
        ('results/new_representative_v7/traditional/copp_station_v7b/validation/summary.json', 4)):
        receipt = base.read(ROOT / relative)
        if not receipt['all_registered_cells_present'] or receipt['successful_cells'] != expected:
            raise ValueError('Required development validation has not completed: '+relative)
        checks[relative] = sha(ROOT / relative)
    r = base.register()
    r.update(study_id='traditional_scaled_representative_v8',
        scope='Explicitly registered uniform tenfold variants of the two selected synthetic programs; unchanged limits/tolerance, no scale search or new parameter tuning.',
        geometric_preprocessing='v7 ordered G01 merging with delta=0.1effective epsilon; complete original gates and command intervals remain in outer audit.',
        copp_grid='v7b stable untrimmed-source tangents only for<=1e-10mm numerical line remnants; original position queries unchanged; coalesced station retains all1/2/3-order constraints.',
        solver_and_projection='Unchanged official COPP0.2.3, three SOCP iterations, fixed third, v5d shared-resolution feasibility projection.',
        no_new_parameter_selection=True, validation_receipt_sha256=checks,
        execution_resource_limit_s=RESOURCE_LIMIT_S,
        resource_failure='A600s process limit is a planning/computation resource stop, not an executed-motion failure or evidence of speed superiority.',
        prior_attempts='results/new_representative_v7/traditional; all earlier configurations retained')
    for name in ('compare_traditional_scaled_v8.py', 'g01_ordered_simplification_v7.py',
                 'copp_station_coalescing_v7.py', 'copp_station_coalescing_v7b.py'):
        r['source_sha256'][name] = sha(ROOT / 'src' / name)
    for case in r['cases']:
        reference = np.loadtxt(ROOT / case['reference'], delimiter=',', skiprows=1)
        reduced, certificate = simplify(reference, case['evaluation_tolerance_mm'])
        file = base.OUT / (case['id']+'_geometry_certificate.json')
        write_json(file, certificate)
        np.savez_compressed(base.OUT / (case['id']+'_planning_geometry.npz'),
                            original_reference=reference, planning_reference=reduced)
        case.update(g01_certificate=str(file.relative_to(ROOT)), g01_certificate_sha256=sha(file))
    write_json(base.REG, r)
    return base.verify()


def run_all():
    registration = base.verify()
    for case in base.CASES:
        for method in base.METHODS:
            for dt in ((None,) if method=='COPP' else base.PERIODS):
                periods = base.PERIODS if dt is None else (dt,)
                if all((base.OUT / (base.run_id(case, method, period)+'.json')).exists() for period in periods):
                    continue
                label = case+'_'+method+('_both' if dt is None else f'_dt{round(dt*1e6)}us')
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--case', case, '--method', method]
                if dt is not None:
                    command += ['--dt', str(dt)]
                started = time.perf_counter()
                with (base.OUT / (label+'.log')).open('x', encoding='utf-8') as log:
                    try:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=RESOURCE_LIMIT_S, check=False)
                    except subprocess.TimeoutExpired:
                        case_row = next(x for x in registration['cases'] if x['id']==case)
                        for period in periods:
                            name = base.run_id(case, method, period)
                            if not (base.OUT / (name+'.json')).exists():
                                write_json(base.OUT / (name+'.json'), dict(run_id=name, case=case, method=method,
                                    dt_s=period, success=False, complete_time_s=None, reason='planning_resource_stop',
                                    motion_failure=False, error='Registered600s computation resource limit reached',
                                    wall_s=time.perf_counter()-started, cpu_s=None, trace=None, trace_sha256=None,
                                    registration_sha256=sha(base.REG), retained_log=label+'.log',
                                    physical_time_budget_s=case_row['physical_time_budget_s']))
                        result = None
                if result is not None and result.returncode:
                    raise RuntimeError('Preserved worker exception: '+label)
                receipt = base.summarize()
                print(json.dumps(dict(attempted=receipt['attempted_cells'], successful=receipt['successful_cells'])), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--case', choices=base.CASES)
    parser.add_argument('--method', choices=base.METHODS)
    parser.add_argument('--dt', type=float, choices=base.PERIODS)
    args=parser.parse_args()
    if args.register:
        r=register()
        print(json.dumps(dict(registration=str(base.REG), sha256=sha(base.REG), cells=len(r['grid']))))
    elif args.all:
        run_all()
    elif args.summary:
        print(json.dumps(base.summarize(), indent=2))
    elif args.case and args.method and (args.dt is not None or args.method=='COPP'):
        base.verify()
        install_geometry()
        if args.method=='COPP':
            install_grid()
        base.run(args.case, args.method, args.dt)
        base.summarize()
    else:
        parser.error('Choose register, all, summary, or case/method/dt')
