"""Read immutable development programs and their declared analytic primitives."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SMALL_CASES = ['train_box', 'train_sine', 'train_bends', 'train_bezier',
               'validation_trapezoid', 'validation_zigzag',
               'test_short_segments', 'test_acute_turns', 'test_curvature',
               'test_line_arc', 'test_irregular_pocket']
COMPLEMENT_CASES = ['train_box', 'train_bends', 'train_bezier',
                    'test_short_segments', 'test_acute_turns', 'test_curvature',
                    'test_line_arc', 'test_irregular_pocket']


def normalize_declared_primitives(document, reference):
    """Convert schema only; validate exact supplied definitions, never fit arcs."""
    if all(p['kind'] in ('line', 'arc') and 'start_reference_index' in p
           for p in document['primitives']):
        return document
    output = []
    index = 0
    source_s = 0.
    for source in document['primitives']:
        start = np.asarray(source['start_mm'])
        end = np.asarray(source['end_mm'])
        if np.linalg.norm(reference[index] - start) > 2e-7:
            raise ValueError('Declared primitive start does not match ordered reference')
        candidates = np.flatnonzero(np.linalg.norm(reference[index+1:] - end, axis=1) <= 2e-7)
        if len(candidates) != 1:
            raise ValueError('Declared primitive endpoint is absent or ambiguous')
        final = index + 1 + int(candidates[0])
        length = float(source['exact_length_mm'])
        row = dict(kind='line' if source['kind'] == 'line' else 'arc',
                   start_mm=start.tolist(), end_mm=end.tolist(), length_mm=length,
                   start_reference_index=index, end_reference_index=final,
                   start_s_mm=source_s, end_s_mm=source_s+length)
        if row['kind'] == 'arc':
            if source['start_vertex_index'] != index or source['end_vertex_index'] != final:
                raise ValueError('Declared arc index mismatch')
            row.update(center_mm=source['center_mm'], radius_mm=source['radius_mm'],
                       start_angle_rad=source['start_angle_rad'],
                       sweep_rad=source['signed_sweep_rad'],
                       chord_error_bound_mm=source['maximum_chord_sagitta_mm'] +
                           document.get('maximum_coordinate_rounding_error_mm', 0.))
        output.append(row)
        index, source_s = final, source_s+length
    if index != len(reference)-1:
        raise ValueError('Declared primitives do not cover the complete reference')
    return dict(primitives=output, schema_conversion_only=True,
                source_description=document.get('description'))


def load_case(name):
    if name in ('original', 'dense'):
        directory = ROOT/'baseline_records/supplementary_reproduction/cases'/name
        path = directory/'reference.csv'
        geometry_path = directory/'geometry.json'
    else:
        path = ROOT/'data'/(name+'.csv')
        geometry_path = ROOT/'data'/(name+'_primitives.json')
    reference = np.loadtxt(path, delimiter=',', skiprows=1)
    geometry = normalize_declared_primitives(json.loads(geometry_path.read_text('utf-8')), reference) if geometry_path.exists() else None
    chord = max((p.get('chord_error_bound_mm', 0.) for p in geometry['primitives']), default=0.) if geometry else 0.
    return dict(id=name, path=path, geometry_path=geometry_path if geometry else None,
                reference=reference, source_geometry=geometry,
                reference_tolerance_mm=1.25-chord, nominal_tolerance_mm=1.25,
                chord_error_mm=chord)
