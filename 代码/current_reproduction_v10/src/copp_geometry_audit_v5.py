"""No optimization: signed derivatives, unchanged geometries and grid capacity."""
from pathlib import Path
import collections
import math
import time
import numpy as np
import copp_path_adapter_v4 as core
import copp_geometry_only_v4 as old
from copp_cases_v5 import load_case
from copp_signed_geometry_v5 import prepare_geometry, install_signed_arc
from copp_adapter_v5 import SETTINGS, grid_geometry, signed_derivatives

ROOT = Path(__file__).resolve().parents[1]


def capacity(stations, boundaries):
    n = stations-1
    acc, jerk = 32*stations, 32*(stations+boundaries)
    nonzeros = 2+11*n+2*acc+6*jerk+9*stations
    rows = 2+5*n+acc+2*jerk+8*stations
    return dict(stations=stations, decision_variables=4*stations-4,
        sparse_row_capacity=rows, nonzero_capacity=nonzeros,
        triplet_and_rhs_MiB=(24*nonzeros+8*rows)/2**20,
        csc_matrix_MiB=(16*nonzeros+8*(4*stations+1))/2**20,
        raw_constraint_coefficients_MiB=8*(3*acc+5*jerk)/2**20,
        excludes='Solver factorization, KKT workspace, Python objects, temporary copies and runtime. Not a peak-memory bound.')


def main():
    output = ROOT/'audit/copp_geometry_capacity_v5.json'
    if output.exists():
        raise FileExistsError(output)
    results = []
    for name in ['original', 'dense', 'test_line_arc', 'train_sine', 'validation_trapezoid', 'validation_zigzag']:
        case = load_case(name)
        start = time.perf_counter()
        curve, module, *_ = prepare_geometry(case['reference'], tolerance=case['reference_tolerance_mm'],
            source_geometry=case['source_geometry'])
        pieces = core.flatten(curve)
        row = dict(path=name, geometry_wall_s=time.perf_counter()-start,
            reference_vertices=len(case['reference']), length_mm=curve.length,
            primitive_counts=dict(collections.Counter(p.kind for p in pieces)),
            original_reference_sha256=core.sha(case['path']),
            original_geometry_sha256=core.sha(case['geometry_path']) if case['geometry_path'] else None)
        sigma, d1, d2, d3, boundaries = grid_geometry(curve, case['source_geometry'], SETTINGS)
        row.update(v5_grid=capacity(len(sigma), len(boundaries)),
            maximum_tangent_join_error=max((r['velocity_jump'] for r in boundaries), default=0.),
            maximum_curvature_join_error=max((r['curvature_jump'] for r in boundaries), default=0.))
        proposals = []
        for line_h, curve_h in [(.5, .5), (1., 1.), (2., 2.), (5., 5.), (5., .5), (10., .5), (5., 1.)]:
            grouped = collections.Counter()
            for p in pieces:
                grouped[p.kind] += max(2 if p.kind == 'line' else 16, math.ceil(p.length/(line_h if p.kind == 'line' else curve_h)))
            proposals.append(dict(line_spacing_mm=line_h, curve_spacing_mm=curve_h,
                intervals_by_kind=dict(grouped), **capacity(1+sum(grouped.values()), len(pieces)-1)))
        row['grid_proposals_without_optimization'] = proposals
        if name != 'test_line_arc':
            old_curve, *_ = old.prepare_geometry_only(case['reference'], tolerance=case['reference_tolerance_mm'],
                source_geometry=case['source_geometry'])
            progress = np.linspace(0., 1., 10001)
            error = float(np.max(np.linalg.norm(curve.position(progress*curve.length)-old_curve.position(progress*old_curve.length), axis=1)))
            row['maximum_position_difference_to_frozen_v4e_mm'] = error
            assert error < 2e-7, (name, error)
        results.append(row)
    tests = []
    for direction in [-1., 1.]:
        for angle in [-1.7, 0., .9]:
            arc = module.Arc([2., -3.], 9., angle, angle+direction*1.8, 0., 16.2)
            q = np.linspace(.1, arc.length-.1, 19)
            h = 1e-4
            d1, d2, d3 = signed_derivatives(arc, q)
            fd1 = (arc.position(q+h)-arc.position(q-h))/(2*h)
            fd2 = (arc.tangent_at(q+h)-arc.tangent_at(q-h))/(2*h)
            left = signed_derivatives(arc, q-h)[1]
            right = signed_derivatives(arc, q+h)[1]
            fd3 = (right-left)/(2*h)
            errors = [float(np.max(abs(a-b))) for a, b in [(d1, fd1), (d2, fd2), (d3, fd3)]]
            assert max(errors) < 1e-8, errors
            tests.append(dict(direction=direction, start_angle=angle, derivative_errors=errors))
    output.parent.mkdir(exist_ok=True, parents=True)
    core.write(output, dict(scope='Geometry/capacity only. No long or other optimization executed by this audit.',
        selected_rule='All programs: straight <=5 mm; quintic/arc <=0.5 mm; minimum intervals 2/16.',
        estimates_source='Official src/copp/copp3/opt3/clarabel_constraints.rs capacity and topp3_socp.rs capacity; 32 directions.',
        signed_arc_derivative_checks=tests, geometry_checks=results,
        source_sha256={p: core.sha(ROOT/'src'/p) for p in ['copp_geometry_audit_v5.py', 'copp_signed_geometry_v5.py', 'copp_adapter_v5.py', 'copp_cases_v5.py']}))
    print(output)
    for row in results:
        print(row['path'], row['primitive_counts'], row['v5_grid']['stations'], row.get('maximum_position_difference_to_frozen_v4e_mm'))


if __name__ == '__main__':
    main()
