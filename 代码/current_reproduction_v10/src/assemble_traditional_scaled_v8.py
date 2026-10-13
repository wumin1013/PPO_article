"""Rebuild the canonical sixteen-cell comparison from immutable run records.

This command performs no planning, sampling, training or metric recomputation.
"""
import json
from pathlib import Path
from common import ROOT, sha, utc, write_json

OUTPUT = ROOT / 'results/representative_scaled_v8/traditional_final'
GLOBAL = ROOT / 'evidence/scaled_representative_v8_registration.json'
SOURCE = {
    'S2': ROOT / 'results/representative_scaled_v8/s2_local_enclosure_v8/formal',
    'S2R': ROOT / 'results/representative_scaled_v8/s2_local_enclosure_v8/formal',
    'SCMO': ROOT / 'results/representative_scaled_v8/traditional',
    'COPP': ROOT / 'results/representative_scaled_v8/traditional',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stable_save(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Canonical content changed; preserve existing release and register a new version')
        return
    write_json(path, value)


def main():
    global_registration = read(GLOBAL)
    cases = [case['id'] for case in global_registration['cases']]
    source_receipts = {}
    for folder in dict.fromkeys(SOURCE.values()):
        file = folder / 'registration.json'
        registration = read(file)
        if registration['global_registration_sha256'] != sha(GLOBAL):
            raise ValueError('Substudy global registration differs')
        for name, expected in registration['source_sha256'].items():
            if sha(ROOT / 'src' / name) != expected:
                raise ValueError('Executed source changed: '+name)
        source_receipts[str(file.relative_to(ROOT).as_posix())] = sha(file)
    records = []
    for case in cases:
        for method in SOURCE:
            folder = SOURCE[method]
            for dt in (.001, .0005):
                name = f'{case}_{method}_dt{round(dt*1e6)}us'
                file = folder / (name+'.json')
                data = read(file)
                if data['registration_sha256'] != sha(folder / 'registration.json'):
                    raise ValueError('Run identity differs: '+name)
                if data['case'] != case or data['method'] != method or data['dt_s'] != dt:
                    raise ValueError('Run grid differs: '+name)
                trace = data.get('trace')
                if trace is not None and sha(ROOT / trace) != data['trace_sha256']:
                    raise ValueError('Trace changed: '+name)
                row = {key:data.get(key) for key in ('run_id','case','method','dt_s','success',
                    'complete_time_s','reason','error','wall_s','cpu_s','trace','trace_sha256',
                    'vector_limit_ratios','max_polyline_deviation_mm','executed_length_mm','geometry',
                    'terminal_rest_s','nominal_tolerance_mm','arc_chord_error_bound_mm','evaluation_tolerance_mm')}
                row.update(record=str(file.relative_to(ROOT).as_posix()), record_sha256=sha(file),
                    source_registration=str((folder/'registration.json').relative_to(ROOT).as_posix()),
                    source_registration_sha256=sha(folder/'registration.json'),
                    method_configuration='S2-G01-v7/local-coordinate-enclosure-v8' if method in ('S2','S2R') else
                        'SCMO-G01-v7' if method=='SCMO' else 'COPP-G01-v7/source-tangent-stations-v7b',
                    cost_unit='One joint three-iterate solve shared by both command periods; do not sum the two row clocks.' if method=='COPP' else 'One independent planner/audit run for this command period.')
                if trace is not None:
                    row['trace'] = Path(trace).as_posix()
                records.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration_file = OUTPUT/'registration.json'
    old = read(registration_file) if registration_file.exists() else {}
    registration = dict(created_utc=old.get('created_utc',utc()), study_id='canonical_traditional_scaled_v8',
        global_registration=str(GLOBAL.relative_to(ROOT).as_posix()), global_registration_sha256=sha(GLOBAL),
        source_registrations=source_receipts, methods=list(SOURCE), cases=cases, periods_s=[.001,.0005],
        expected_cells=16, source_selection='Use the same final numerical implementation for each method across both complete programs and command periods; no per-case faster-run selection.',
        scope='Registered tenfold synthetic variants; unchanged nominal1.25mm and vector100/2000/62500 limits. All complete original program gates and command intervals checked.',
        failed_COPP='Both three-iteration attempts stop at the first official Clarabel NumericalError; no accepted completion time or comparative efficiency is assigned.',
        prior_attempts_retained=['results/new_representative_v7/traditional', 'results/representative_scaled_v8/traditional/superseded_process_stop.json'],
        numerical_reviews={path:sha(ROOT/path) for path in ('audit/s2_local_enclosure_v8_review.json',
            'audit/s2_local_enclosure_v8_actual_path_review.json')},
        generation_entry='src/assemble_traditional_scaled_v8.py', generation_entry_sha256=sha(__file__),
        regeneration_scope='Read existing records and verify all source/record/trace hashes; no new experiments')
    stable_save(registration_file,registration)
    summary_file=OUTPUT/'summary.json'
    previous=read(summary_file) if summary_file.exists() else {}
    summary=dict(created_utc=previous.get('created_utc',utc()),registration_sha256=sha(registration_file),
        expected_cells=16,attempted_cells=len(records),successful_cells=sum(row['success'] for row in records),
        all_registered_cells_present=len(records)==16,records=records)
    stable_save(summary_file,summary)
    print(json.dumps(dict(summary=str(summary_file),registration=str(registration_file),
        expected_cells=16,attempted_cells=len(records),successful_cells=summary['successful_cells'],
        regeneration='Summary-only; no experiments executed.'),indent=2))


if __name__=='__main__':
    main()
