"""Retire a development-related semicircle before final testing; geometry only."""
import copy
import io
import json
import math
import shutil
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json
from runtime_bootstrap import setup


def mixed_program():
    points = [np.array([0., 0.])]
    primitives = []

    def line(end):
        start = points[-1].copy()
        end = np.asarray(end, float)
        primitives.append(dict(kind='line', start_mm=start.tolist(), end_mm=end.tolist(),
                               exact_length_mm=float(np.linalg.norm(end-start))))
        points.append(end)

    def arc(center, radius, start_angle, sweep, segments):
        center = np.asarray(center, float)
        theta = np.linspace(start_angle, start_angle+sweep, segments+1)
        samples = center + radius*np.c_[np.cos(theta), np.sin(theta)]
        assert np.linalg.norm(points[-1]-samples[0]) < 1e-12
        start_index = len(points)-1
        points.extend(samples[1:])
        primitives.append(dict(kind='circular_arc', center_mm=center.tolist(), radius_mm=radius,
            start_angle_rad=start_angle, signed_sweep_rad=sweep,
            direction='counterclockwise' if sweep > 0 else 'clockwise',
            start_mm=samples[0].tolist(), end_mm=samples[-1].tolist(),
            exact_length_mm=radius*abs(sweep), chord_count=segments,
            start_vertex_index=start_index, end_vertex_index=len(points)-1,
            maximum_chord_sagitta_mm=radius*(1-math.cos(abs(sweep)/(2*segments)))))

    line([28., 0.])
    arc([28., 9.], 9., -math.pi/2, math.pi/2, 24)
    line([37., 23.])
    arc([51., 23.], 14., math.pi, -3*math.pi/4, 36)
    line(points[-1]+18*np.array([math.sqrt(.5), -math.sqrt(.5)]))
    line(points[-1]+[12., 9.])
    line(points[-1]+[18., -4.])
    return np.asarray(points), primitives


def segment_intersections(points):
    """Detect finite nonadjacent segment intersections in this open program."""
    found = []
    for i in range(len(points)-1):
        for k in range(i+2, len(points)-1):
            u, v = points[i+1]-points[i], points[k+1]-points[k]
            matrix = np.stack([u, -v], axis=1)
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            s, t = np.linalg.solve(matrix, points[k]-points[i])
            if -1e-10 <= s <= 1+1e-10 and -1e-10 <= t <= 1+1e-10:
                found.append([i, k])
    return found


def main():
    setup()
    from geometry_spec import Corridor

    manifest_file = ROOT/'data'/'manifest.json'
    old_manifest = ROOT/'data'/'manifest_before_mixed_arc_isolation.json'
    old_csv = ROOT/'data'/'development_line_semicircle.csv'
    test_csv = ROOT/'data'/'test_line_arc.csv'
    if (ROOT/'audit'/'test_freeze.json').exists():
        raise RuntimeError('Final testing is already frozen; never replace its paths')
    if list((ROOT/'results'/'unseen').glob('*.json')) or list((ROOT/'results'/'unseen').glob('*.npz')):
        raise RuntimeError('Final-test execution exists; this amendment is prohibited')
    if any('_test_' in p.name for p in (ROOT/'results'/'deterministic').glob('*')):
        raise RuntimeError('Deterministic final-test execution exists')
    if old_manifest.exists() or old_csv.exists():
        raise FileExistsError('This registered amendment must not be rerun')

    manifest = json.loads(manifest_file.read_text('utf-8'))
    protected = {r['id']:sha(ROOT/'data'/(r['id']+'.csv')) for r in manifest['paths']
                 if r['split'] in ('train', 'validation')}
    teacher_before = sha(ROOT/'data'/'teacher_records_v3.npz')
    original = next(r for r in manifest['paths'] if r['id']=='test_line_arc')
    assert sha(test_csv) == original['sha256']
    points, primitives = mixed_program()
    buffer = io.StringIO()
    np.savetxt(buffer, points, delimiter=',', header='x_mm,y_mm', comments='', fmt='%.12g')
    csv_bytes = buffer.getvalue().encode('utf-8')
    rounded = np.loadtxt(io.BytesIO(csv_bytes), delimiter=',', skiprows=1)
    coordinate_rounding_error = float(np.linalg.norm(rounded-points,axis=1).max())
    corridor = Corridor(rounded, epsilon=1.25, closed=False)
    conflicts = corridor.adjacent_gate_conflicts()
    intersections = segment_intersections(rounded)
    assert not corridor.ambiguities and not conflicts and not intersections
    assert np.all(corridor.lengths > 1e-10)
    assert all(corridor.segment_inside(a,b) for a,b in zip(rounded[:-1],rounded[1:]))

    shutil.copy2(manifest_file, old_manifest)
    shutil.copy2(test_csv, old_csv)
    previous = copy.deepcopy(original)
    previous.update(id='development_line_semicircle', split='development',
                    group='semicircular_program_development_only',
                    retired_test_id='test_line_arc',
                    retirement_reason='Related family to the semicircle already used in teacher/runtime development; no final trajectory had been executed.')
    assert sha(old_csv) == previous['sha256']
    manifest['paths'].append(previous)
    test_csv.write_bytes(csv_bytes)
    primitive_file = ROOT/'data'/'test_line_arc_primitives.json'
    max_sagitta = max(p.get('maximum_chord_sagitta_mm',0.) for p in primitives)
    write_json(primitive_file, dict(path='test_line_arc', origin='synthetic', generator='isolate_mixed_arc_test.py',
        units='mm, rad', description='Straight + CCW quarter circle R9 + straight + CW 135-degree circle R14 + straight + terminal two-line corner sequence.',
        primitives=primitives, exact_primitive_length_mm=sum(p['exact_length_mm'] for p in primitives),
        polyline_length_mm=float(corridor.lengths.sum()),
        maximum_arc_chord_sagitta_mm=max_sagitta,
        maximum_coordinate_rounding_error_mm=coordinate_rounding_error,
        analytic_primitive_to_polyline_distance_bound_mm=max_sagitta+coordinate_rounding_error,
        interpretation='The executor evaluates the retained polyline. A polyline contour bound epsilon implies at most epsilon plus this chord/rounding bound relative to the analytic primitives; exact arc following is not asserted.'))
    original.update(group='two_opposite_unequal_arcs_with_terminal_polyline', generator='isolate_mixed_arc_test.py',
        sha256=sha(test_csv), vertices=len(rounded), length_mm=float(corridor.lengths.sum()), closed=False,
        primitives_file=primitive_file.relative_to(ROOT).as_posix(), primitives_sha256=sha(primitive_file))
    manifest['primitive_note']='test_line_arc now contains two opposite-turn circular arcs (R9/90 degrees and R14/135 degrees), straight segments and a terminal polyline. Exact primitives, chord sagitta and rounding bound are in data/test_line_arc_primitives.json. The former semicircle is development-only.'
    registration_splits = {}
    for file in (ROOT/'results'/'paired').glob('*/seed_*/*/registration.json'):
        registration = json.loads(file.read_text('utf-8'))
        registration_splits[file.relative_to(ROOT).as_posix()] = registration['split_sha256']
    evidence_files = [ROOT/'results'/'teacher_selection'/'train_semicircle.json',
                      ROOT/'results'/'teacher_selection_v2'/'train_semicircle.json']
    amendment = dict(utc=utc(), prior_manifest_file=old_manifest.relative_to(ROOT).as_posix(),
        prior_manifest_sha256=sha(old_manifest), generator_sha256=sha(__file__),
        retired_csv=old_csv.relative_to(ROOT).as_posix(), retired_csv_sha256=sha(old_csv),
        reason='The old straight-semicircle-straight test was too closely related to the development semicircle, whose two teacher failures had already informed workflow diagnosis. Move both to the same development family and replace the unexecuted test with a different complete program containing unequal-radius opposite-turn arcs and terminal corners.',
        development_evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence_files},
        final_test_trajectory_executions_before_change=0,
        models_or_planners_executed_by_this_amendment=0,
        only_geometry_preflight=True,
        training_and_validation_csv_sha256=protected,
        teacher_records_v3_sha256=teacher_before,
        training_manifest_note='The manifest version changed while registered training was in progress. Only held-out/development definitions changed: all four training programs, both validation programs, group memberships and teacher-record bytes are unchanged. Registrations retain their original whole-manifest hash; this amendment and the saved manifest document the difference.',
        training_registration_manifest_hashes_observed=registration_splits)
    manifest['mixed_arc_isolation_amendment'] = amendment
    assert protected == {r['id']:sha(ROOT/'data'/(r['id']+'.csv')) for r in manifest['paths'] if r['split'] in ('train','validation')}
    assert teacher_before == sha(ROOT/'data'/'teacher_records_v3.npz')
    assert {r['id']:r for r in manifest['paths'] if r['split'] in ('train','validation')} == {r['id']:r for r in json.loads(old_manifest.read_text('utf-8'))['paths'] if r['split'] in ('train','validation')}
    write_json(manifest_file, manifest)
    report = dict(created_utc=utc(), scope='Pure input geometry; no policy, teacher, planner, rollout or final-test evaluation executed.',
        manifest_sha256=sha(manifest_file), path_sha256=sha(test_csv), old_path_sha256=sha(old_csv),
        vertices=len(rounded), polyline_length_mm=float(corridor.lengths.sum()),
        minimum_segment_length_mm=float(corridor.lengths.min()),
        ambiguities=corridor.ambiguities, adjacent_gate_conflicts=conflicts,
        nonadjacent_segment_intersections=intersections, reference_segments_inside_own_corridor=True,
        maximum_arc_chord_sagitta_mm=max_sagitta, maximum_coordinate_rounding_error_mm=coordinate_rounding_error,
        training_validation_hashes_unchanged=True, training_validation_manifest_records_unchanged=True,
        teacher_records_unchanged=True, final_test_trajectory_executions=0,
        all_geometry_checks_pass=True, amendment=amendment)
    write_json(ROOT/'audit'/'mixed_arc_isolation_preflight.json', report)
    print(json.dumps({k:report[k] for k in ('vertices','polyline_length_mm','maximum_arc_chord_sagitta_mm','all_geometry_checks_pass','final_test_trajectory_executions')}))


if __name__ == '__main__':
    main()
