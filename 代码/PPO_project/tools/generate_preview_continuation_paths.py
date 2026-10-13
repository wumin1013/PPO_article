"""Generate/register six unevaluated synthetic 2-D continuation holdouts.

This entry point performs geometry checks only. It never imports a policy,
loads a checkpoint, evaluates a planner, or changes any existing path dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from src.utils.revision_paths import generate_path, _save_path, load_manifest, load_path, path_statistics

TARGET = PROJECT / 'data/preview_continuation_v1'
PRIOR_DATA = PROJECT / 'data/preview_event_v1'
MAX_SPACING_MM = .6
SOURCE_BOUND_MM = .005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def dense_parameters(seed, count, low_length, high_length, positive_min, negative_min):
    """Explicit seeded nonrepeating alternating headings; x strictly increases."""
    rng = np.random.default_rng(seed)
    positive = rng.choice(np.arange(positive_min, 80), count//2, replace=False)
    negative = -rng.choice(np.arange(negative_min, 79), count//2, replace=False)
    headings = np.empty(count, dtype=int)
    headings[::2], headings[1::2] = positive, negative
    lengths = np.round(rng.uniform(low_length, high_length, count), 3)
    theta = np.radians(headings)
    vertices = np.vstack([[0., 0.], np.cumsum(np.c_[np.cos(theta), np.sin(theta)]*lengths[:, None], axis=0)])
    assert len(set(zip(headings.tolist(), lengths.tolist()))) == count
    return dict(seed=seed, rng='numpy.random.default_rng/PCG64',
        heading_deg=headings.tolist(), defining_segment_lengths_mm=lengths.tolist(),
        vertices_mm=vertices.tolist(), max_spacing_mm=MAX_SPACING_MM,
        construction='Alternating positive/negative headings sampled without replacement within each sign; independently sampled segment lengths. No repeated motif or transformed old path.')


def definitions():
    mixed = [
      ('a', [[0,0],[12,0],[12,.8],[12.9,1.7],[20,1.7],[20.8,1.1],[21.4,2.2],
             [21.4,10],[22.4,10.7],[31,10.7],[31,9.8],[32.3,9.8],[38.3,5.3],
             [39,6.5],[49,6.5],[49.8,5.5],[50.6,6],[57.6,10]],
       'Long straight travel interleaved with short diagonal/orthogonal connectors, changing turn signs, and an extended vertical traverse.'),
      ('b', [[0,0],[8,0],[8,-1.1],[9.5,-1.1],[15.5,3.4],[16.2,2.4],[16.2,-5.6],
             [17.3,-6.2],[18.1,-5.4],[29.1,-5.4],[30.3,-4.4],[31,-5.7],
             [37,-1.2],[37,.1],[38.2,.1],[47.2,.1],[47.8,1.5],[48.7,.7],
             [56.7,4.7],[58.1,4.7]],
       'Different-length travel blocks separated by short reentrant connectors and alternating steep/shallow transitions; distinct vertex sequence from case a.')]
    result = []
    for suffix, vertices, challenge in mixed:
        result.append(dict(id=f'mixed_lengths_continuation_v1_{suffix}_test', family='mixed_lengths',
          split='test', exposure_status='unevaluated_holdout', challenge=challenge,
          parameters=dict(vertices_mm=vertices, max_spacing_mm=MAX_SPACING_MM)))
    for suffix, parameters, challenge in [
      ('a', dense_parameters(202609101, 56, .65, 1.65, 18, 8),
       'Fifty-six unequal short blocks with alternating turn sign and nonrepeating heading/length combinations; ordered coverage must retain local protrusions.'),
      ('b', dense_parameters(202609102, 64, .55, 1.55, 9, 17),
       'Sixty-four short blocks with a different asymmetric heading range and denser geometric events; the sequence contains no repeated angle/length pair.')]:
        result.append(dict(id=f'dense_corners_continuation_v1_{suffix}_test', family='dense_corners',
          split='test', exposure_status='unevaluated_holdout', challenge=challenge, parameters=parameters))
    curves = [
      ('a', [dict(line_to=[10,0]), dict(cubic=[[13,0],[14,6],[19,6]]),
             dict(line_to=[25,6]), dict(cubic=[[27,6],[30,-5],[35,-5]]),
             dict(line_to=[39,-5]), dict(cubic=[[43,-5],[43,3],[48,3]]),
             dict(line_to=[55,3]), dict(cubic=[[58,3],[59,8],[64,8]]), dict(line_to=[70,8])],
       'Four cubic S-transition primitives separated by unequal straight runs, alternating elevation changes and varying curvature magnitudes.'),
      ('b', [dict(line_to=[7,0]), dict(cubic=[[11,0],[13,8],[17,8]]),
             dict(line_to=[20,8]), dict(cubic=[[23,8],[25,1],[28,1]]),
             dict(line_to=[32,1]), dict(cubic=[[36,1],[37,11],[43,11]]),
             dict(line_to=[46,11]), dict(cubic=[[49,11],[52,4],[55,4]]),
             dict(line_to=[59,4]), dict(cubic=[[62,4],[62,-3],[67,-3]]), dict(line_to=[70,-3])],
       'Five independently specified cubic transitions, unequal approach lengths and elevation spans, and repeated changes between straight travel and variable curvature.')]
    for suffix, primitives, challenge in curves:
        result.append(dict(id=f'line_curve_continuation_v1_{suffix}_test', family='line_curve',
          split='test', exposure_status='unevaluated_holdout', challenge=challenge,
          parameters=dict(start_mm=[0,0], primitives=primitives,
            max_spacing_mm=MAX_SPACING_MM, chord_error_bound_mm=SOURCE_BOUND_MM)))
    return result


def materialize(definition):
    if definition['family'] == 'dense_corners':
        # Reuse the existing polyline sampler; metadata retains the real family
        # and the exact defining heading/length sequence and Cartesian vertices.
        return generate_path(dict(family='mixed_lengths', parameters=definition['parameters']))
    return generate_path(definition)


def intersection_pairs(points):
    def cross(a, b):
        return float(a[0]*b[1]-a[1]*b[0])
    pairs = []
    eps = 1e-10
    for i in range(len(points)-1):
        a, b = points[i:i+2]
        for j in range(i+2, len(points)-1):
            c, d = points[j:j+2]
            if np.any(np.maximum(np.minimum(a,b),np.minimum(c,d)) >
                      np.minimum(np.maximum(a,b),np.maximum(c,d))+eps):
                continue
            o = [cross(b-a,c-a),cross(b-a,d-a),cross(d-c,a-c),cross(d-c,b-c)]
            if min(o[0],o[1]) <= eps and max(o[0],o[1]) >= -eps and min(o[2],o[3]) <= eps and max(o[2],o[3]) >= -eps:
                pairs.append([i,j])
    return pairs


def canonical(points, reverse=False):
    points = points[::-1] if reverse else points
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    sample = np.c_[np.interp(np.linspace(0,arcs[-1],513),arcs,points[:,0]),
                    np.interp(np.linspace(0,arcs[-1],513),arcs,points[:,1])]
    heading = points[1]-points[0]
    heading /= np.linalg.norm(heading)
    rotation = np.array([[heading[0],-heading[1]],[heading[1],heading[0]]])
    return (sample-points[0])@rotation/arcs[-1]


def curve_primitive_checks(parameters):
    """Analytic x-monotonicity and regular endpoints, independent of sampling."""
    start = np.array(parameters['start_mm'], dtype=float)
    rows = []
    for index, primitive in enumerate(parameters['primitives']):
        if 'line_to' in primitive:
            end = np.array(primitive['line_to'], dtype=float)
            assert end[0] > start[0]
            rows.append(dict(index=index, kind='line', length_mm=float(np.linalg.norm(end-start))))
        else:
            control = np.vstack([start, primitive['cubic']])
            derivative_controls = 3*np.diff(control, axis=0)
            assert np.all(derivative_controls[:,0] >= 0)
            assert derivative_controls[0,0] > 0 and derivative_controls[-1,0] > 0
            # A quadratic Bernstein combination with positive endpoint terms
            # and nonnegative middle term has positive x derivative everywhere.
            grid = np.linspace(0,1,1001)[:,None]
            first = ((1-grid)**2*derivative_controls[0]+2*grid*(1-grid)*derivative_controls[1]+grid**2*derivative_controls[2])
            second_controls = 2*np.diff(derivative_controls,axis=0)
            second = (1-grid)*second_controls[0]+grid*second_controls[1]
            curvature = (first[:,0]*second[:,1]-first[:,1]*second[:,0])/np.linalg.norm(first,axis=1)**3
            end = control[-1]
            rows.append(dict(index=index, kind='cubic_bezier', control_points_mm=control.tolist(),
              positive_x_derivative_analytic=True,
              start_tangent_unit=(derivative_controls[0]/np.linalg.norm(derivative_controls[0])).tolist(),
              end_tangent_unit=(derivative_controls[-1]/np.linalg.norm(derivative_controls[-1])).tolist(),
              sampled_signed_curvature_range_per_mm=[float(curvature.min()),float(curvature.max())],
              curvature_probe_count=1001, curvature_range_is_sampled_not_certified_extremum=True))
        start = end
    return rows


def geometry_checks(points, definition, priors):
    statistics = path_statistics(points)
    assert not statistics['closed']
    assert 40 <= statistics['path_length_mm'] <= 100
    assert statistics['segment_length_quantiles_mm']['max'] <= MAX_SPACING_MM+1e-9
    crossings = intersection_pairs(points)
    if crossings:
        raise ValueError(f"Unintended self intersections in {definition['id']}: {crossings}")
    delta = np.diff(points,axis=0)
    if np.any(delta[:,0] < -1e-9):
        raise ValueError('These registered constructions must be nondecreasing in x')
    signature = canonical(points)
    distances = []
    for prior_id, prior in priors:
        for reverse in (False,True):
            prior_sig = canonical(prior,reverse)
            for reflection in (1,-1):
                difference = float(np.max(np.linalg.norm(signature-prior_sig*[1,reflection],axis=1)))
                distances.append((difference,prior_id,reverse,reflection))
    best = min(distances)
    if best[0] <= 1e-5:
        raise ValueError('Normalized arclength shape duplicates a prior or another new path')
    result = dict(zero_length_segments=0, nonadjacent_intersection_pairs=crossings,
        open_path=True, x_nondecreasing=True,
        start_tangent_unit=(delta[0]/np.linalg.norm(delta[0])).tolist(),
        end_tangent_unit=(delta[-1]/np.linalg.norm(delta[-1])).tolist(),
        start_heading_deg=float(np.degrees(np.arctan2(delta[0,1],delta[0,0]))),
        end_heading_deg=float(np.degrees(np.arctan2(delta[-1,1],delta[-1,0]))),
        maximum_saved_segment_mm=statistics['segment_length_quantiles_mm']['max'],
        shape_comparison=dict(samples=513, normalization='Subtract start, align initial tangent, divide by path length; compare forward/reversed and reflected reference traces.',
            minimum_max_point_distance=best[0],closest_prior_id=best[1],prior_reversed=best[2],reflection=best[3],
            near_copy_rejection_threshold=1e-5,
            scope='Numerical polyline shape comparison plus independent explicit construction; not a symbolic curve-equivalence proof.'))
    if definition['family']=='line_curve':
        result['primitive_checks'] = curve_primitive_checks(definition['parameters'])
    else:
        result['defining_polyline_statistics'] = path_statistics(np.array(definition['parameters']['vertices_mm'],float))
    return result


def generate(target):
    if target.exists():
        raise FileExistsError('Preserve registered geometry; use verify or a new version directory')
    prior_manifest = load_manifest(PRIOR_DATA)
    prior_files = [PRIOR_DATA/'manifest.json'] + [PRIOR_DATA/p['reference_file'] for p in prior_manifest['paths']]
    prior_hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in prior_files}
    priors = [(p['id'],load_path(p['id'],PRIOR_DATA)) for p in prior_manifest['paths']]
    created = datetime.now(timezone.utc).isoformat()
    source = dict(kind='synthetic', cam_export=False,
       generator=Path(__file__).relative_to(ROOT).as_posix(), generator_sha256=sha(__file__),
       shared_path_implementation='PPO_project/src/utils/revision_paths.py',
       shared_path_implementation_sha256=sha(PROJECT/'src/utils/revision_paths.py'))
    prepared = []
    for definition in definitions():
        points = materialize(definition)
        # Validate exactly the decimal coordinates that _save_path writes.
        saved = np.array([[float(format(x,'.12g')),float(format(y,'.12g'))] for x,y in points])
        checks = geometry_checks(saved,definition,priors)
        prepared.append((definition,points,checks))
        priors.append((definition['id'],saved))
    target.mkdir(parents=True)
    entries = []
    for definition, points, checks in prepared:
        definition.update(created_utc=created,geometry_checks=checks,
          evaluation_gate='Unevaluated holdout. First policy evaluation requires a saved final method/config/model/seed freeze; no tuning on these paths.',
          model_output_inspected_during_design=False)
        source_item = dict(source, analytic_to_polyline_deviation_bound_mm=definition['parameters'].get('chord_error_bound_mm',0.),
          coordinate_serialization_significant_digits=12,
          discretization_definition='Exact defining polyline, or recursive Bezier convex-hull chord bound <=0.005 mm before decimal serialization. Source discretization is separate from motion tolerance.')
        entries.append(_save_path(target,definition,points,source_item))
    manifest = dict(schema_version='preview-continuation-paths-v1',created_utc=created,
       split_frozen_date=created[:10],source=source,paths=entries,
       units=dict(distance='mm',time='s',angle='rad'),plane='XY',
       dataset_role='Six unevaluated independent synthetic holdouts; no training/validation entries copied.',
       exposure_status='unevaluated_holdout',evaluation_count=0,
       test_usage='Freeze final design, hyperparameters, model hashes and seeds before the first policy evaluation. Any test-informed tuning converts those tests into development data and requires a replacement holdout.',
       split_rationale='New explicit vertex and primitive compositions and seeded nonrepeating dense sequences; no old path is rotated, scaled, reflected or reused.',
       design_budget=dict(families=3,test_instances_per_family=2,path_length_range_mm=[40,100],
          maximum_interpolation_segment_mm=MAX_SPACING_MM,bezier_source_chord_bound_mm=SOURCE_BOUND_MM),
       source_discretization_and_planning_tolerance='Separate definitions. No planning tolerance or motion constraints are changed by this generator.',
       prior_development_inputs_sha256=prior_hashes,
       policy_evaluation_performed=False,model_artifact_access_performed=False)
    write_json(target/'manifest.json',manifest)
    assert prior_hashes == {p.relative_to(ROOT).as_posix():sha(p) for p in prior_files}
    summary = dict(manifest=target.relative_to(ROOT).as_posix(),manifest_sha256=sha(target/'manifest.json'),
       created_utc=created,policy_evaluation_performed=False,old_geometry_unchanged=True,
       paths=[dict(id=p['id'],family=p['family'],length_mm=p['statistics']['path_length_mm'],
         segment_count=p['statistics']['segment_count'],major_corner_count=len(p['statistics']['major_corners']),
         max_turn_deg=max(p['statistics']['all_turn_angles_deg']),reference_sha256=p['reference_sha256']) for p in entries])
    write_json(target/'generation_stdout.json',summary)
    return summary


def verify(target):
    manifest = load_manifest(target)
    priors = [(p['id'],load_path(p['id'],PRIOR_DATA)) for p in load_manifest(PRIOR_DATA)['paths']]
    checked = []
    for item in manifest['paths']:
        points = load_path(item['id'],target)
        checks = geometry_checks(points,item,priors)
        if checks != item['geometry_checks']:
            raise ValueError(f"Geometry verification changed: {item['id']}")
        expected = materialize(item)
        expected = np.array([[float(format(x,'.12g')),float(format(y,'.12g'))] for x,y in expected])
        np.testing.assert_array_equal(points,expected)
        checked.append(item['id'])
        priors.append((item['id'],points))
    return dict(manifest_sha256=sha(target/'manifest.json'),verified_paths=checked,
                geometry_and_regeneration_verified=True,policy_evaluation_performed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('generate','verify'))
    parser.add_argument('--output-dir',type=Path,default=TARGET)
    args=parser.parse_args()
    output = generate(args.output_dir.resolve()) if args.command=='generate' else verify(args.output_dir.resolve())
    print(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
