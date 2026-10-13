"""Register fresh synthetic holdouts before event-preview optimization."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.utils.revision_paths import load_manifest, generate_path, _save_path


def main():
    source, target = PROJECT / 'data/preview_v2', PROJECT / 'data/preview_event_v1'
    if target.exists():
        raise FileExistsError('Preserve registered geometry')
    target.mkdir(); (target/'paths').mkdir()
    prior = load_manifest(source)
    manifest = copy.deepcopy(prior)
    manifest.update(schema_version='preview-event-paths-v1', split_frozen_date='2026-09-09',
        generator=__file__, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), paths=[],
        test_usage='New event_v1 tests are unavailable to model evaluation until final model/config/source freeze.',
        split_rationale='Reuse development train/validation; all former tests remain exposed development. New vertex/primitive compositions, never rotations or scaled copies.')
    for old in prior['paths']:
        item = copy.deepcopy(old)
        item['previous_split'] = old['split']
        if old['split'] == 'test':
            item['split'] = 'development'
        item['exposure_status'] = 'previously_exposed_development'
        (target/item['reference_file']).write_bytes((source/item['reference_file']).read_bytes())
        (target/'paths'/f"{item['id']}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['paths'].append(item)
    definitions = [dict(id='mixed_lengths_event_v1_test', family='mixed_lengths', split='test', parameters=dict(
        vertices_mm=[[0,0],[4,0],[4,.7],[5.1,.7],[5.1,1.5],[10,3],[10,3.8],[9,3.8],[7,7],
                     [3,7],[3,6.2],[2.2,6.2],[-1,9],[-1,10.2],[.4,10.2],[5,13],[8,12]], max_spacing_mm=.6)),
        dict(id='line_curve_event_v1_test', family='line_curve', split='test', parameters=dict(start_mm=[0,0],
             primitives=[{'line_to':[4,0]},{'cubic':[[6,0],[9,2],[7,4]]},{'line_to':[5,6]},
                         {'cubic':[[3,8],[0,6],[2,4]]},{'line_to':[3,3]},
                         {'cubic':[[5,1],[9,5],[8,8]]},{'line_to':[8,11]},
                         {'cubic':[[8,14],[12,14],[14,12]]}],max_spacing_mm=.6,chord_error_bound_mm=.005))]
    headings = [[20,-55,40,-15,65],[-35,50,0,-60,25],[70,-20,35,-50,10],[-10,60,-40,15,-65], [40,-25,55,-5,30]]
    lengths = [[.7,1.1,1.6,.9,1.2],[1.4,.8,1.3,1.0,.75],[.95,1.45,.85,1.25,1.05],[1.2,.9,1.5,.8,1.1],[1.35,.75,1.,1.4,.95]]
    theta = np.radians(np.asarray(headings).ravel())
    increments = np.c_[np.cos(theta),np.sin(theta)] * np.asarray(lengths).reshape(-1,1)
    dense = np.vstack([[0,0],np.cumsum(increments,axis=0)])
    definitions.insert(1,dict(id='dense_corners_event_v1_test',family='dense_corners',split='test',parameters=dict(
        heading_blocks_deg=headings,segment_length_blocks_mm=lengths,generation='Explicit new nonrepeating heading/length composition')))
    for definition in definitions:
        points = dense if definition['family']=='dense_corners' else generate_path(definition)
        definition['exposure_status']='unevaluated_holdout'
        item = _save_path(target,definition,points,dict(kind='synthetic',cam_export=False,
            generator='generate_preview_event_paths.py',generator_sha256=manifest['generator_sha256'],
            analytic_to_polyline_deviation_bound_mm=definition['parameters'].get('chord_error_bound_mm',0.),
            discretization_definition='Exact defining polyline or Bezier convex-hull chord bound, distinct from motion tolerance'))
        manifest['paths'].append(item)
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(manifest_sha256=hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest(),
        new_test_paths=[dict(id=p['id'],statistics=p['statistics']) for p in manifest['paths'] if p['split']=='test']),ensure_ascii=False))


if __name__=='__main__':main()
