"""Validate then evaluate one frozen G01 preprocessing rule for all planners."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import compare_traditional_representative_v7 as base
from common import ROOT, sha, utc, write_json
from g01_ordered_simplification_v7 import simplify, install

BASE_OUT = ROOT / 'results/new_representative_v7/traditional/g01_v7'
VALIDATION = ('validation_trapezoid', 'validation_zigzag')


def configure(stage):
    base.OUT = BASE_OUT / stage
    base.REG = base.OUT / 'registration.json'
    base.CASES = VALIDATION if stage == 'validation' else (
        'piecewise_curvature_coupon', 'asymmetric_neck_contour')
    if stage == 'validation':
        base.SOURCE = base.OUT / 'validation_inputs.json'


def validation_inputs():
    if base.SOURCE.exists():
        return
    cases = []
    base.OUT.mkdir(parents=True, exist_ok=True)
    for name in VALIDATION:
        path = ROOT / 'data' / (name + '.csv')
        ref = np.loadtxt(path, delimiter=',', skiprows=1)
        lengths = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(lengths)]
        geometry = base.OUT / (name + '_source_geometry.json')
        write_json(geometry, dict(origin='Exact existing validation G01 program',
            primitives=[dict(kind='line', start_mm=ref[i].tolist(), end_mm=ref[i+1].tolist(),
                start_reference_index=i, end_reference_index=i+1, start_s_mm=arc[i],
                end_s_mm=arc[i+1], length_mm=lengths[i]) for i in range(len(lengths))]))
        cases.append(dict(id=name, reference=str(path.relative_to(ROOT)), reference_sha256=sha(path),
            geometry=str(geometry.relative_to(ROOT)), geometry_sha256=sha(geometry),
            origin='synthetic', nominal_tolerance_mm=1.25, arc_chord_error_bound_mm=0.,
            evaluation_tolerance_mm=1.25, physical_time_budget_s=5.+float(lengths.sum())/5.))
    write_json(base.SOURCE, dict(created_utc=utc(), scope='Existing development-validation programs only', cases=cases))


def register(stage):
    if base.REG.exists():
        return base.verify()
    if stage == 'validation':
        validation_inputs()
    else:
        validation = BASE_OUT / 'validation/summary.json'
        if not validation.exists():
            raise ValueError('Run complete existing validation comparison before formal registration')
        receipt = base.read(validation)
        if not receipt['all_registered_cells_present'] or receipt['successful_cells'] != 16:
            raise ValueError('All sixteen validation cells must complete before target execution')
    registration = base.register()
    registration.update(study_id='traditional_ordered_g01_v7_' + stage,
        scope='Existing development-validation check with no parameter selection.' if stage == 'validation' else
              'Detailed two-program comparison after previous five-program outcomes; not a new prospective test.',
        geometry_version='v7-fixed-ten-percent-ordered-G01-merging',
        geometry_design='Remove artificial very-short chord transitions with two-way ordered geometric certificates; no skipped program interval, no removed audit gate.',
        simplification_fraction=.1, fraction_selection='Fixed before either target is evaluated; no search or performance-based selection.',
        tolerance_flow='Original nominal minus only true arc chord reserve -> effective epsilon; planner merging uses0.1epsilon and original smoothing0.9epsilon; outer full-reference metrics use effective epsilon.',
        baseline_native_attempts='results/new_representative_v7/traditional',
        parameter_changes='Geometry preprocessing only; native S2 curve caps/continuous straight scheduling, SCMO overlap rule, official COPP3-iterate solver and frozen projection unchanged.',
        validation_receipt='results/new_representative_v7/traditional/g01_v7/validation/summary.json' if stage == 'formal' else None)
    for name in ('compare_traditional_g01_v7.py', 'g01_ordered_simplification_v7.py'):
        registration['source_sha256'][name] = sha(ROOT / 'src' / name)
    for case in registration['cases']:
        reference = np.loadtxt(ROOT / case['reference'], delimiter=',', skiprows=1)
        reduced, certificate = simplify(reference, case['evaluation_tolerance_mm'])
        path = base.OUT / (case['id'] + '_geometry_certificate.json')
        write_json(path, certificate)
        np.savez_compressed(base.OUT / (case['id'] + '_planning_geometry.npz'),
                            original_reference=reference, planning_reference=reduced)
        case['g01_certificate'] = str(path.relative_to(ROOT))
        case['g01_certificate_sha256'] = sha(path)
    write_json(base.REG, registration)
    return base.verify()


def run_all(stage):
    base.verify()
    for case in base.CASES:
        for method in base.METHODS:
            for dt in ((None,) if method == 'COPP' else base.PERIODS):
                periods = base.PERIODS if dt is None else (dt,)
                names = [base.run_id(case, method, x) for x in periods]
                if all((base.OUT / (name + '.json')).exists() for name in names):
                    continue
                name = case + '_' + method + ('_both' if dt is None else f'_dt{round(dt*1e6)}us')
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--stage', stage,
                           '--case', case, '--method', method]
                if dt is not None:
                    command.extend(['--dt', str(dt)])
                with (base.OUT / (name + '.log')).open('x', encoding='utf-8') as stream:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
                if completed.returncode:
                    raise RuntimeError('Preserved worker error: ' + name)
                receipt = base.summarize()
                print(json.dumps(dict(stage=stage, attempted=receipt['attempted_cells'],
                                      successful=receipt['successful_cells'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('validation', 'formal'), required=True)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--method', choices=base.METHODS)
    parser.add_argument('--dt', type=float, choices=base.PERIODS)
    args = parser.parse_args()
    configure(args.stage)
    if args.register:
        registered = register(args.stage)
        print(json.dumps(dict(registration=str(base.REG), sha256=sha(base.REG), cells=len(registered['grid']))))
    elif args.all:
        run_all(args.stage)
    elif args.summary:
        print(json.dumps(base.summarize(), indent=2))
    elif args.case and args.method and (args.dt is not None or args.method == 'COPP'):
        base.verify()
        install()
        base.run(args.case, args.method, args.dt)
        base.summarize()
    else:
        parser.error('Choose register, all, summary, or case/method/dt')
