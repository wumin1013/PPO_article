"""Recompute retained v8 evidence and require identical scientific tables."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def scientific_values(comparison):
    """Numerical evidence only; provenance clocks are not experimental outcomes."""
    keys = ('success', 'complete_time_s', 'elapsed_prefix_s', 'samples', 'dt_s',
        'tolerance_mm', 'evaluation_tolerance_mm', 'limits', 'vector_peaks',
        'vector_limit_ratios', 'kinematic_pass', 'rest_pass', 'terminal_rest_s',
        'geometry', 'max_polyline_deviation_mm', 'rms_polyline_deviation_mm',
        'executed_length_mm', 'reason')
    return dict(rows=[dict(case=row['case'], method=row['method'], dt_s=row['dt_s'],
        trace_sha256=row.get('trace_sha256'),
        metrics={k:row['metrics'][k] for k in keys if k in row['metrics']})
        for row in comparison['rows']], savings=comparison['savings'])


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def main():
    manifest = read(ROOT / 'DELTA_V8_MANIFEST.json')
    base = read(ROOT / 'PAYLOAD_SHA256.json')
    config = manifest['regeneration']
    for name, expected in {**base, **manifest['files']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Input file changed before regeneration: ' + name)
    expected_tables = {name:(ROOT/name).read_bytes() for name in config['byte_identical_outputs']}
    before = scientific_values(read(ROOT / config['comparison_json']))
    if canonical_digest(before) != config['scientific_values_sha256']:
        raise ValueError('Stored scientific values differ from the frozen increment')
    commands = config['commands']
    if len(commands) != 4 or not all(isinstance(command, list) and command for command in commands):
        raise ValueError('Four registered evidence regeneration commands are required')
    logs = ROOT / 'reproduction_logs_v8'
    logs.mkdir(exist_ok=False)
    completed = []
    for index, command in enumerate(commands, 1):
        path = logs / f'{index:02d}.log'
        with path.open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', *command], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=False)
        completed.append(dict(command=command, returncode=result.returncode,
                              log=path.relative_to(ROOT).as_posix()))
        if result.returncode:
            raise RuntimeError('Preserved regeneration log: ' + str(path))
    actual = scientific_values(read(ROOT / config['comparison_json']))
    if canonical_digest(actual) != config['scientific_values_sha256']:
        raise ValueError('Regenerated complete-grid scientific values differ')
    for name, expected in expected_tables.items():
        if (ROOT/name).read_bytes() != expected:
            raise ValueError('Regenerated table or CSV bytes differ: ' + name)
    for name, expected in base.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Regeneration modified a preserved base file: ' + name)
    result = dict(status='independent_v8_regeneration_verified', complete_grid_cells=len(actual['rows']),
        scientific_values_sha256=canonical_digest(actual), all_scientific_values_identical=True,
        byte_identical_outputs={name:digest(ROOT/name) for name in expected_tables},
        immutable_base_files_preserved=len(base), commands=completed,
        new_training=False, new_motion_execution=False, new_deterministic_planning=False)
    target = ROOT / 'audit/portable_regeneration_receipt_v8.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
