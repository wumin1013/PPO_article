"""Registered COPP numerical-grid repair on unchanged v7 G01 geometry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import compare_traditional_g01_v7 as stage_driver
import compare_traditional_representative_v7 as base
from common import ROOT, sha, utc, write_json
from g01_ordered_simplification_v7 import install as install_geometry
from copp_station_coalescing_v7 import install as install_grid

OUTPUT = ROOT / 'results/new_representative_v7/traditional/copp_station_v7'


def configure(stage):
    stage_driver.configure(stage)
    base.OUT = OUTPUT / stage
    base.REG = base.OUT / 'registration.json'
    base.METHODS = ('COPP',)
    if stage == 'validation':
        base.SOURCE = base.OUT / 'validation_inputs.json'


def register(stage):
    if base.REG.exists():
        return base.verify()
    if stage == 'formal':
        receipt = base.read(OUTPUT / 'validation/summary.json')
        if not receipt['all_registered_cells_present'] or receipt['successful_cells'] != 4:
            raise ValueError('Require four completed validation cells before repaired formal planning')
    registration = stage_driver.register(stage)
    registration.update(study_id='copp_station_coalescing_v7_' + stage,
        numerical_grid='Coalesce only station gaps atmax(1e-12mm,16ULP); no geometric piece removed; preserve all velocity/acceleration/jerk states at merged station.',
        prior_failed_configuration='results/new_representative_v7/traditional/g01_v7/formal',
        repair_basis='Existing grid rejects floating-point straight remnants before solving; change concerns representability, not performance tuning.',
        geometric_fraction=.1, solver_or_projection_changes=False)
    if stage == 'formal':
        registration['numerical_validation_receipt_sha256'] = sha(OUTPUT / 'validation/summary.json')
    for name in ('compare_copp_station_v7.py', 'copp_station_coalescing_v7.py'):
        registration['source_sha256'][name] = sha(ROOT / 'src' / name)
    write_json(base.REG, registration)
    return base.verify()


def summarize():
    registration = base.verify()
    rows = []
    for cell in registration['grid']:
        path = base.OUT / (cell['run_id'] + '.json')
        if path.exists():
            record = base.read(path)
            if record['registration_sha256'] != sha(base.REG):
                raise ValueError('Attempt identity changed')
            rows.append({key:record.get(key) for key in ('run_id', 'case', 'method', 'dt_s', 'success',
                'complete_time_s', 'reason', 'error', 'wall_s', 'cpu_s', 'trace', 'trace_sha256',
                'vector_limit_ratios', 'max_polyline_deviation_mm', 'executed_length_mm',
                'geometry', 'terminal_rest_s')})
    summary = dict(created_utc=utc(), registration_sha256=sha(base.REG), expected_cells=len(registration['grid']),
        attempted_cells=len(rows), successful_cells=sum(x['success'] for x in rows),
        all_registered_cells_present=len(rows)==len(registration['grid']), records=rows)
    write_json(base.OUT / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('validation', 'formal'), required=True)
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--case')
    args = parser.parse_args()
    configure(args.stage)
    if args.register:
        r = register(args.stage)
        print(json.dumps(dict(registration=str(base.REG), sha256=sha(base.REG), cells=len(r['grid']))))
    elif args.all:
        base.verify()
        for case in base.CASES:
            if all((base.OUT / (base.run_id(case, 'COPP', dt)+'.json')).exists() for dt in base.PERIODS):
                continue
            with (base.OUT / (case + '.log')).open('x', encoding='utf-8') as log:
                completed = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                    '--stage', args.stage, '--case', case], stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError('Retained worker exception: ' + case)
            print(json.dumps(summarize()), flush=True)
    elif args.case:
        base.verify()
        install_geometry()
        install_grid()
        base.run(args.case, 'COPP')
        summarize()
    else:
        parser.error('Choose register, all or case')
