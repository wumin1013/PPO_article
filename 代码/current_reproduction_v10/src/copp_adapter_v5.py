"""Registered COPP development adapter with one grid rule for every program."""
import argparse
import math
from pathlib import Path
import numpy as np
import copp_path_adapter_v4 as core
import copp_geometry_only_v4 as frozen
from copp_signed_geometry_v5 import prepare_geometry
from copp_cases_v5 import SMALL_CASES, COMPLEMENT_CASES, load_case

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(core.SETTINGS, max_line_station_spacing_mm=5.,
                max_curve_station_spacing_mm=.5, grid_version='v5-line5-curve0.5')


def signed_derivatives(piece, local):
    local = np.asarray(local, float)
    if piece.kind != 'arc':
        return core.derivatives(piece, local)
    tangent = piece.tangent_at(local)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    curvature = piece.direction/piece.radius
    return tangent, curvature*normal, -curvature**2*tangent


def grid_geometry(curve, source_geometry, settings):
    pieces = core.flatten(curve)
    stations, first, second, third, boundaries = [], [], [], [], []
    offset = 0.
    sub_ulp = []
    counts = {}
    for index, piece in enumerate(pieces):
        if piece.length < 1e-12 and piece.length <= 4*np.spacing(max(1., offset)):
            sub_ulp.append(dict(piece_index=index, length_mm=piece.length,
                                source_start=piece.source_start, source_end=piece.source_end))
            offset += piece.length
            continue
        is_line = piece.kind == 'line'
        spacing = settings['max_line_station_spacing_mm' if is_line else 'max_curve_station_spacing_mm']
        count = max(settings['min_line_intervals' if is_line else 'min_curve_intervals'],
                    math.ceil(piece.length/spacing))
        counts[piece.kind] = counts.get(piece.kind, 0)+count
        local = np.linspace(0., piece.length, count+1)
        d1, d2, d3 = signed_derivatives(piece, local)
        if stations:
            boundaries.append(dict(index=len(stations)-1, first=d1[0], second=d2[0], third=d3[0],
                velocity_jump=float(np.linalg.norm(first[-1]-d1[0])),
                curvature_jump=float(np.linalg.norm(second[-1]-d2[0]))))
            local, d1, d2, d3 = local[1:], d1[1:], d2[1:], d3[1:]
        stations.extend((offset+local).tolist())
        first.extend(d1); second.extend(d2); third.extend(d3)
        offset += piece.length
    sigma = np.asarray(stations)
    if np.any(np.diff(sigma) <= 1e-12):
        raise ValueError('Unresolved near-duplicate station')
    if any(row['velocity_jump'] > 1e-7 or row['curvature_jump'] > 1e-7 for row in boundaries):
        raise ValueError('Signed geometric join is not C2')
    scale = settings['spatial_scale_mm']
    curve.metadata.update(sub_ulp_pieces_without_internal_stations=sub_ulp,
                          grid_intervals_by_kind=counts)
    return sigma, np.asarray(first), np.asarray(second)*scale, np.asarray(third)*scale**2, boundaries


def register(cases, version, output):
    if (output/'registration.json').exists():
        raise FileExistsError('Do not overwrite registered evidence: '+str(output))
    output.mkdir(parents=True, exist_ok=True)
    source_files = ['copp_adapter_v5.py', 'copp_signed_geometry_v5.py', 'copp_cases_v5.py',
                    'copp_path_adapter_v4.py', 'copp_geometry_only_v4.py', 'copp_time_projection_v4.py']
    record = dict(version=version, settings=core.SETTINGS if version == 'v4e' else SETTINGS,
        nominal_tolerance_mm=1.25, limits=[100., 2000., 62500.], command_periods_s=[.001, .0005],
        scope='Development only; all earlier observed tests are development. No prospective final programs.',
        selection='Fixed third SOCP iteration for every case; retain every intermediate result and failure.',
        source_sha256={name: core.sha(ROOT/'src'/name) for name in source_files},
        sources=[dict(id=c['id'], reference_sha256=core.sha(c['path']),
            geometry_sha256=core.sha(c['geometry_path']) if c['geometry_path'] else None,
            nominal_tolerance_mm=c['nominal_tolerance_mm'], reference_tolerance_mm=c['reference_tolerance_mm'],
            chord_error_mm=c['chord_error_mm']) for c in cases])
    core.write(output/'registration.json', record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v4e', 'v5'], default='v5')
    parser.add_argument('--cases', nargs='+', default=['small'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = args.cases
    if names == ['small']:
        names = SMALL_CASES
    elif names == ['complement']:
        names = COMPLEMENT_CASES
    elif names == ['all']:
        names = SMALL_CASES+['dense', 'original']
    cases = [load_case(name) for name in names]
    registration = register(cases, args.version, args.output)
    core.prepare_geometry = frozen.prepare_geometry_only if args.version == 'v4e' else prepare_geometry
    core.sample = frozen.projected_sample
    if args.version == 'v5':
        core.grid_geometry = grid_geometry
    records = []
    for case in cases:
        record = core.run_case(case['id'], case['reference'], case['reference_tolerance_mm'],
            case['source_geometry'], args.output, settings=registration['settings'])
        record.update(registered_adapter_version=args.version,
            nominal_tolerance_mm=case['nominal_tolerance_mm'], chord_error_mm=case['chord_error_mm'],
            registered_sources=registration['source_sha256'])
        if args.version == 'v5' and 'planner' in record:
            record['planner']['method'] = 'Official COPP 0.2.3 TOPP3-SOCP with signed project S2 geometry and uniform primitive-type grid adaptation'
        core.write(args.output/(case['id']+'.json'), record)
        records.append(record)
    core.write(args.output/'summary.json', records)


if __name__ == '__main__':
    main()
