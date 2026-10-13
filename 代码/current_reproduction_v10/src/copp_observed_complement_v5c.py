"""Apply frozen COPP v5c to the five previously observed, registered programs.

The historical final_ prefix is retained as provenance; these are development
paths, not prospective final tests. No parameters are selected on these runs.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
import copp_path_adapter_v4 as core
import copp_adapter_v5b as parent
from copp_reserve_replay_v5c import install_for_new_planning, RESERVE
from copp_cases_v5 import load_case

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_REGISTRATION = ROOT / 'results/historical_strong_v5/registration.json'
COMPLETE_REPLAY = ROOT / 'results/copp_v5c_complete_development'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def declared_geometry(row, reference):
    """Convert registered exact circles to a complete primitive partition.

    Arc locations and angles are never inferred or fitted from the polyline.
    Between the declared arcs, the source consists of the recorded line blocks.
    """
    arcs = row.get('exact_arc_primitives')
    if not arcs:
        return None
    indexed = []
    for arc in arcs:
        endpoints = []
        for name in ('start', 'end'):
            candidates = np.flatnonzero(np.linalg.norm(reference - arc[name], axis=1) <= 2e-7)
            if len(candidates) != 1:
                raise ValueError('Declared arc endpoint absent or ambiguous')
            endpoints.append(int(candidates[0]))
        lo, hi = endpoints
        if hi - lo != arc['intervals'] or hi <= lo:
            raise ValueError('Registered exact arc does not match recorded interval count')
        indexed.append((lo, hi, arc))
    indexed.sort(key=lambda item: item[0])
    primitives, cursor, source_s = [], 0, 0.

    def append_line(lo, hi):
        nonlocal source_s
        if hi == lo:
            return
        start, end = reference[lo], reference[hi]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 0:
            raise ValueError('Zero length declared line')
        fractions = ((reference[lo:hi+1] - start) @ vector) / length**2
        residual = reference[lo:hi+1] - start - fractions[:, None] * vector
        if np.max(np.linalg.norm(residual, axis=1)) > 2e-7 or np.any(np.diff(fractions) < 0):
            raise ValueError('Non-arc interval is not the registered straight segment')
        primitives.append(dict(kind='line', start_mm=start.tolist(), end_mm=end.tolist(),
            start_reference_index=lo, end_reference_index=hi, length_mm=length,
            start_s_mm=source_s, end_s_mm=source_s + length))
        source_s += length

    for lo, hi, arc in indexed:
        if lo < cursor:
            raise ValueError('Overlapping registered exact arcs')
        append_line(cursor, lo)
        center = np.asarray(arc['center'], float)
        theta0 = math.atan2(arc['start'][1] - center[1], arc['start'][0] - center[0])
        sweep = float(arc['sweep_radian'])
        radius = float(arc['radius_mm'])
        theta = np.linspace(theta0, theta0 + sweep, hi - lo + 1)
        expected = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
        if np.max(np.linalg.norm(expected - reference[lo:hi+1], axis=1)) > 2e-7:
            raise ValueError('Registered analytic arc disagrees with complete reference')
        length = radius * abs(sweep)
        primitives.append(dict(kind='arc', start_mm=arc['start'], end_mm=arc['end'],
            center_mm=arc['center'], radius_mm=radius, start_angle_rad=theta0,
            sweep_rad=sweep, start_reference_index=lo, end_reference_index=hi,
            length_mm=length, start_s_mm=source_s, end_s_mm=source_s + length,
            chord_error_bound_mm=row['arc_chord_error_bound_mm']))
        source_s += length
        cursor = hi
    append_line(cursor, len(reference)-1)
    return dict(primitives=primitives, schema_conversion_only=True,
        source='Exact arcs explicitly recorded in the historical registration; no arc fitting',
        maximum_chord_error_bound_mm=row['arc_chord_error_bound_mm'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'results/copp_v5c_observed_additional')
    args = parser.parse_args()
    if not (COMPLETE_REPLAY / 'summary.json').exists():
        raise RuntimeError('Complete the already registered replay before additional optimization')
    if (args.output / 'registration.json').exists():
        raise FileExistsError('Do not overwrite registered additional evidence')
    original = read(ORIGINAL_REGISTRATION)
    selected = [row for row in original['cases'] if row['id'].startswith('final_')]
    assert len(selected) == 5
    assert RESERVE == .001
    args.output.mkdir(parents=True, exist_ok=True)
    overlap = []
    for row in original['cases']:
        if row['id'].startswith('final_'):
            continue
        case = load_case(row['id'])
        assert core.sha(case['path']) == row['reference_sha256']
        assert case['reference_tolerance_mm'] == row['evaluation_tolerance_mm']
        assert case['chord_error_mm'] == row['arc_chord_error_bound_mm']
        overlap.append(dict(id=row['id'], reference_sha256=row['reference_sha256'],
            evaluation_tolerance_mm=case['reference_tolerance_mm'], exactly_same_reference_and_tolerance=True))
    prepared = []
    for row in selected:
        reference_path = ROOT / row['reference']
        assert core.sha(reference_path) == row['reference_sha256']
        assert abs(row['evaluation_tolerance_mm'] + row['arc_chord_error_bound_mm'] - row['nominal_tolerance_mm']) <= 1e-14
        reference = np.loadtxt(reference_path, delimiter=',', skiprows=1)
        geometry = declared_geometry(row, reference)
        geometry_file = args.output / (row['id'] + '_source_geometry.json')
        if geometry:
            core.write(geometry_file, geometry)
        prepared.append((row, reference, geometry, geometry_file if geometry else None))
    sources = ['copp_observed_complement_v5c.py', 'copp_reserve_replay_v5c.py',
        'copp_adapter_v5b.py', 'copp_signed_geometry_v5b.py', 'copp_adapter_v5.py',
        'copp_cases_v5.py', 'copp_path_adapter_v4.py', 'copp_geometry_only_v4.py',
        'copp_time_projection_v4.py', 'metrics.py', 'metrics_exact_index_v5.py',
        'exact_distance_index_v5.py', 'runtime_configuration.py', 'runtime_bootstrap.py']
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        version='v5c-reserve0.001', settings=parent.SETTINGS, extra_time_reserve_fraction=RESERVE,
        scope='Previously observed development programs; final_ is a historical identifier only',
        method='Official COPP 0.2.3 TOPP3-SOCP with frozen v5c project geometry and common vector-command evaluation',
        policy='No tuning; same fixed third iterate, full program, all outputs/failures retained',
        limits=[100., 2000., 62500.], command_periods_s=[.001, .0005],
        original_registration=str(ORIGINAL_REGISTRATION),
        original_registration_sha256=core.sha(ORIGINAL_REGISTRATION),
        validated_reserve_selection_sha256=core.sha(ROOT / 'audit/copp_v5c_reserve_selection.json'),
        evaluator_equivalence_review_sha256=core.sha(ROOT / 'audit/distance_index_independent_review_v5.json'),
        source_sha256={name:core.sha(ROOT / 'src' / name) for name in sources},
        prior_replay_directory=str(COMPLETE_REPLAY), prior_matching_cases=overlap,
        cases=[dict(row, converted_geometry=str(path) if path else None,
            converted_geometry_sha256=core.sha(path) if path else None)
            for row, _, _, path in prepared])
    core.write(args.output / 'registration.json', registration)
    run_case = install_for_new_planning()
    records = []
    for row, reference, geometry, _ in prepared:
        record = run_case(row['id'], reference, row['evaluation_tolerance_mm'], geometry,
            args.output, settings=parent.SETTINGS)
        record.update(registered_adapter_version='v5c-reserve0.001',
            nominal_tolerance_mm=row['nominal_tolerance_mm'],
            chord_error_mm=row['arc_chord_error_bound_mm'],
            reference_sha256=row['reference_sha256'],
            registered_sources=registration['source_sha256'])
        core.write(args.output / (row['id'] + '.json'), record)
        records.append(record)
    core.write(args.output / 'summary.json', records)


if __name__ == '__main__':
    main()
