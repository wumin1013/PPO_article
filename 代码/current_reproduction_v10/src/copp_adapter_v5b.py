"""COPP with dense line-end stations and coarser constant-geometry cruise spans.

Same fixed rule for all development programs; no per-case selected iterate.
"""
import argparse
import math
from pathlib import Path
import numpy as np
import copp_path_adapter_v4 as core
import copp_geometry_only_v4 as frozen
from copp_adapter_v5 import signed_derivatives
from copp_signed_geometry_v5b import prepare_geometry
from copp_cases_v5 import SMALL_CASES, load_case

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = dict(core.SETTINGS, max_line_cruise_spacing_mm=5., max_curve_station_spacing_mm=.5,
                line_endpoint_dense_length_mm=20., grid_version='v5b-dense-line-ends')


def local_stations(piece, settings):
    spacing = settings['max_curve_station_spacing_mm']
    edge = settings['line_endpoint_dense_length_mm']
    minimum = settings['min_line_intervals'] if piece.kind == 'line' else settings['min_curve_intervals']
    if piece.kind != 'line' or piece.length <= 2*edge:
        return np.linspace(0., piece.length, max(minimum, math.ceil(piece.length/spacing))+1)
    left = np.linspace(0., edge, math.ceil(edge/spacing)+1)
    middle = np.linspace(edge, piece.length-edge,
                         math.ceil((piece.length-2*edge)/settings['max_line_cruise_spacing_mm'])+1)
    right = np.linspace(piece.length-edge, piece.length, math.ceil(edge/spacing)+1)
    return np.r_[left, middle[1:], right[1:]]


def grid_geometry(curve, source_geometry, settings):
    stations, first, second, third, boundaries = [], [], [], [], []
    offset = 0.
    sub_ulp = []
    counts = {}
    for index, piece in enumerate(core.flatten(curve)):
        if piece.length < 1e-12 and piece.length <= 4*np.spacing(max(1., offset)):
            sub_ulp.append(dict(piece_index=index, length_mm=piece.length,
                                source_start=piece.source_start, source_end=piece.source_end))
            offset += piece.length
            continue
        local = local_stations(piece, settings)
        counts[piece.kind] = counts.get(piece.kind, 0)+len(local)-1
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', default=['all'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = SMALL_CASES+['dense', 'original'] if args.cases == ['all'] else args.cases
    cases = [load_case(name) for name in names]
    output = args.output
    if (output/'registration.json').exists():
        raise FileExistsError('Do not overwrite registered evidence: '+str(output))
    output.mkdir(parents=True, exist_ok=True)
    files = ['copp_adapter_v5b.py', 'copp_signed_geometry_v5b.py', 'copp_adapter_v5.py',
             'copp_cases_v5.py', 'copp_path_adapter_v4.py', 'copp_geometry_only_v4.py', 'copp_time_projection_v4.py']
    registration = dict(version='v5b', settings=SETTINGS, limits=[100., 2000., 62500.],
        command_periods_s=[.001, .0005], nominal_tolerance_mm=1.25,
        grid_choice='One rule: curve and line-end 20 mm <=0.5 mm; constant-geometry straight interior <=5 mm. Short straight remains exactly v4e mesh.',
        grid_rationale='20 mm exceeds twice V*(V/A+A/J)=16.4 mm at the common limits; retains fine stations around acceleration/deceleration and all curvature transitions. This is a discretization choice, not a continuous feasibility guarantee.',
        geometry_change='Signed arcs; remove unused S2 cruise-scheduler entry/exit-straight prerequisite only. Preserve all control points, trims, geometric checks and complete endpoints.',
        scope='Development only. No prospective final program.',
        selection='Fixed third solve; all first/second outputs and failures retained, no per-case version or iteration choice.',
        source_sha256={name: core.sha(ROOT/'src'/name) for name in files},
        cases=[dict(id=c['id'], reference_sha256=core.sha(c['path']),
            geometry_sha256=core.sha(c['geometry_path']) if c['geometry_path'] else None,
            reference_tolerance_mm=c['reference_tolerance_mm'], chord_error_mm=c['chord_error_mm']) for c in cases])
    core.write(output/'registration.json', registration)
    core.prepare_geometry = prepare_geometry
    core.grid_geometry = grid_geometry
    core.sample = frozen.projected_sample
    records = []
    for case in cases:
        record = core.run_case(case['id'], case['reference'], case['reference_tolerance_mm'],
                               case['source_geometry'], output, settings=SETTINGS)
        record.update(registered_adapter_version='v5b', nominal_tolerance_mm=1.25,
            chord_error_mm=case['chord_error_mm'], registered_sources=registration['source_sha256'])
        if 'planner' in record:
            record['planner']['method'] = 'Official COPP 0.2.3 TOPP3-SOCP, signed S2 geometry and fixed dense-line-end grid'
        core.write(output/(case['id']+'.json'), record)
        records.append(record)
    core.write(output/'summary.json', records)


if __name__ == '__main__':
    main()
