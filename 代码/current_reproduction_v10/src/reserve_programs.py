"""Register five complete synthetic programs before any policy evaluation.

These programs are a prospective reserve, not evidence of machining performance.
No model, reward, runtime or experiment result is imported by this generator.
"""
import json
import math
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json


def arc(center, radius, start_deg, end_deg, spacing=.9):
    sweep = math.radians(end_deg - start_deg)
    count = max(2, math.ceil(abs(sweep) * radius / spacing))
    angle = np.linspace(math.radians(start_deg), math.radians(end_deg), count + 1)
    return np.asarray(center) + radius * np.c_[np.cos(angle), np.sin(angle)]


def join(*parts):
    points = np.vstack(parts)
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]


def programs():
    # Five lanes connected by circular U-turns: a single open raster program.
    raster = [np.array([[0., 0.], [32., 0.]])]
    for lane in range(4):
        right = lane % 2 == 0
        cx = 32. if right else 0.
        raster.append(arc((cx, 8. * lane + 4.), 4., -90., 90. if right else -270.))
        raster.append(np.array([[0. if right else 32., 8. * (lane + 1)]]))
    yield 'reserve_raster', join(*raster), 'rounded_raster_full_program', {
        'motivation': 'Successive lane reversals in planar raster motion; straight and circular blocks.',
        'construction': 'Five 32 mm lanes at 8 mm pitch, radius 4 mm semicircular links, maximum arc chord 0.9 mm.',
        'arc_chord_deviation_upper_mm': .9 ** 2 / (8 * 4) * 1.01}

    angle = math.degrees(math.atan2(6., -8.))
    keyhole = join([[-8., -6.], [0., -6.], [22., -6.]],
                  arc((30., 0.), 10., 360-angle, 360+angle),
                  [[0., 6.], [0., -6.], [-8., -6.]])
    yield 'reserve_keyhole', keyhole, 'keyhole_profile_with_entry_exit', {
        'motivation': 'A neck-to-circular-chamber profile with entry and exit strokes.',
        'construction': '10 mm chamber radius, 12 mm neck width, 8 mm entry/exit; circle sampled with maximum chord 0.9 mm.',
        'arc_chord_deviation_upper_mm': .9 ** 2 / (8 * 10) * 1.01}

    notch = np.array([[-8,0],[0,0],[48,0],[48,28],[39,28],[39,17],
                      [34,17],[34,28],[26,28],[26,11],[20,11],[20,28],
                      [11,28],[11,20],[6,20],[6,28],[0,28],[0,0],[-8,0]],float)
    yield 'reserve_notched_profile', notch, 'three_depth_castellated_profile', {
        'motivation': 'Successive concave and convex corners on a notched planar contour.',
        'construction': '48 by 28 mm envelope, three unequal inward notches, minimum edge 5 mm; full contour plus entry/exit.'}

    # A smooth asymmetric closed profile, sampled by a fixed analytic formula.
    theta = np.linspace(0., 2*math.pi, 181)
    profile = np.c_[26.*(1.-np.cos(theta)), 7.*np.sin(theta)*(1.+.35*np.cos(theta))]
    foil = join([[0., -8.]], profile, [[0., 8.]])
    yield 'reserve_asymmetric_profile', foil, 'asymmetric_oval_with_tangent_entry_exit', {
        'motivation': 'Nonuniform curvature around an elongated freeform planar profile.',
        'construction': 'x=26(1-cos theta), y=7 sin theta(1+0.35 cos theta), 180 equal parameter intervals; tangent entry/exit.'}

    theta = np.linspace(0., 2*math.pi, 241)
    radius = 20.*(1.+.18*np.cos(3*theta)+.06*np.sin(2*theta))
    lobe = np.c_[radius*np.cos(theta), radius*np.sin(theta)]
    tangent = np.array([2.4,23.6]); tangent /= np.linalg.norm(tangent)
    lobed = join([lobe[0]-8*tangent], lobe, [lobe[-1]+8*tangent])
    yield 'reserve_three_lobe', lobed, 'asymmetric_three_lobe_profile', {
        'motivation': 'Repeated curvature changes along a lobed planar contour.',
        'construction': 'Polar r=20(1+0.18 cos 3theta+0.06 sin 2theta), 240 equal parameter intervals; tangent entry/exit.'}


def main():
    destination = ROOT / 'prospective' / 'reserved_programs'
    if destination.exists():
        raise FileExistsError('Reserve already registered; do not overwrite or filter by outcome.')
    destination.mkdir(parents=True)
    rows = []
    for name, points, family, details in programs():
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert np.isfinite(points).all() and lengths.min() > 1e-8
        file = destination / (name + '.csv')
        np.savetxt(file, points, delimiter=',', header='x_mm,y_mm', comments='', fmt='%.12g')
        rows.append(dict(id=name, group=family, origin='synthetic; analytic program generated for this study',
                         status='reserved; no policy or baseline evaluation performed at registration',
                         file=file.name, sha256=sha(file), vertices=len(points), length_mm=float(lengths.sum()),
                         min_segment_mm=float(lengths.min()), max_segment_mm=float(lengths.max()), **details))
    write_json(destination / 'registration.json', dict(
        registered_utc=utc(), generator_sha256=sha(__file__), paths=rows,
        inclusion_rule='Include all five complete programs and all registered seeds; no outcome-based exclusion or replacement.',
        family_rule='All segments, rotations, scales and direct parameter variants stay in the same group.',
        geometric_interpretation='Evaluation follows the stored polylines. Analytic formulas explain construction; this is not native arc interpolation.',
        previous_test_use='All five v1 test programs are observed. Any v2 tuning on them is development use.',
        evaluation_status='NOT RUN. Not manuscript result data. Freeze models and evaluation first.',
        proposed_conditions=dict(tolerance_mm=1.25, speed_mm_s=100, acceleration_mm_s2=2000, jerk_mm_s3=62500, dt_s=.001),
        rationale='Coverage tests for contour geometry. Dimensions and tolerance have no asserted workpiece/process specification.',
        prerequisites=['Successful paired training/validation gate', 'Frozen identical evaluation rules and complete model hashes',
                       'Predeclared five-seed budget and deterministic baseline settings', 'Report all attempted outcomes']))
    print(json.dumps({'registered': len(rows), 'destination': str(destination), 'evaluations': 0}))


if __name__ == '__main__':
    main()
