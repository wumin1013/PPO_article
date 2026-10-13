"""Fix tenfold complete-program variants before any new model or planner run."""
import json
import runpy
import numpy as np
from common import ROOT, sha, utc, write_json

FACTOR = 10.0
NAMES = ['piecewise_curvature_coupon', 'asymmetric_neck_contour']
OUT = ROOT/'data/representative_scaled_v8'
REG = ROOT/'evidence/scaled_representative_v8_registration.json'


def main():
    if REG.exists() or OUT.exists():
        raise FileExistsError('Do not replace registered scale variants')
    source_path = ROOT/'evidence/frozen_main_generalization_v6.json'
    source = json.loads(source_path.read_text('utf-8'))
    geometry_generator = ROOT/'data/final_holdout_v6/register_final_holdout_v6.py'
    api = runpy.run_path(str(geometry_generator))
    OUT.mkdir(parents=True)
    cases=[]
    for name in NAMES:
        old=next(c for c in source['cases'] if c['id']==name)
        old_geometry=json.loads((ROOT/old['geometry']).read_text('utf-8'))
        old_reference=np.loadtxt(ROOT/old['reference'],delimiter=',',skiprows=1)
        anchor=np.asarray(old_geometry['primitives'][0]['start_mm'])
        def transform(point):
            return FACTOR*(np.asarray(point)-anchor)
        program=api['Program'](np.zeros(2))
        for primitive in old_geometry['primitives']:
            if primitive['kind']=='line':
                program.line(transform(primitive['end_mm']),primitive['label'])
            else:
                program.arc(transform(primitive['center_mm']), FACTOR*primitive['radius_mm'],
                            primitive['start_angle_rad'],primitive['sweep_rad'],primitive['label'])
        identity=name+'_scale10'
        folder=OUT/identity
        folder.mkdir()
        reference_path=folder/'reference.csv'
        np.savetxt(reference_path,np.asarray(program.points),delimiter=',',header='x_mm,y_mm',comments='',fmt='%.12f')
        chord=max(p.get('chord_error_bound_mm',0.) for p in program.primitives)
        geometry=dict(case_id=identity,origin='synthetic',units='mm, rad',
            formal_reference='ordered complete line/arc program',primitives=program.primitives,
            nominal_tolerance_mm=1.25,arc_chord_error_bound_mm=chord,
            maximum_coordinate_rounding_error_mm=api['ROUNDING'],evaluation_tolerance_mm=1.25-chord,
            original_program_no_cropping=True,closed=old_geometry['closed'],lap_count=1,
            geometry_scale=FACTOR,unscaled_parent=name,scaling_anchor_mm=anchor.tolist())
        write_json(folder/'geometry.json',geometry)
        audit=api['validate'](reference_path,geometry,[])
        audit['similarity_scope']='Declared tenfold direct variant of the parent geometry; not an independent held-out family.'
        assert audit['closed']==old_geometry['closed']
        for previous,current in zip(old_geometry['primitives'],program.primitives):
            assert previous['kind']==current['kind']
            assert np.allclose(transform(previous['start_mm']),current['start_mm'],rtol=0,atol=1e-9)
            assert np.allclose(transform(previous['end_mm']),current['end_mm'],rtol=0,atol=1e-9)
            assert abs(FACTOR*previous['length_mm']-current['length_mm'])<1e-8
        if chord==0:
            reread=np.loadtxt(reference_path,delimiter=',',skiprows=1)
            assert np.allclose(reread,FACTOR*(old_reference-anchor),rtol=0,atol=1e-10)
        audit['uniform_primitive_scale_verified']=True
        write_json(folder/'static_audit.json',audit)
        case=dict(id=identity,group=old['group'],parent_case=name,
            parent_reference=old['reference'],parent_reference_sha256=old['reference_sha256'],
            parent_geometry=old['geometry'],parent_geometry_sha256=old['geometry_sha256'],
            origin='synthetic',actual_CAM=False,geometry_scale=FACTOR,
            reference=reference_path.relative_to(ROOT).as_posix(),reference_sha256=sha(reference_path),
            geometry=(folder/'geometry.json').relative_to(ROOT).as_posix(),geometry_sha256=sha(folder/'geometry.json'),
            static_audit=audit,nominal_tolerance_mm=1.25,arc_chord_error_bound_mm=chord,
            evaluation_tolerance_mm=1.25-chord,complete_program_length_mm=audit['length_mm'],
            analytic_program_length_mm=program.s,physical_time_budget_s=5+audit['length_mm']/5)
        cases.append(case)
    registration=dict(created_utc=utc(),study_id='scaled_representative_v8',
        authorization='Author requested physically larger paths; a single common factor of ten was declared before these executions.',
        geometry_scale=FACTOR,case_count=2,methods=['JNN','FIXED','S2','S2R','SCMO','COPP','NNC'],
        dt_s=[.001,.0005],nominal_tolerance_mm=1.25,planar_vector_limits=[100.,2000.,62500.],
        scale_selection='Fixed factor ten for both complete geometries; no search over scales or outcome-based replacement.',
        scope='Detailed larger-scale case studies; direct variants of the previously evaluated synthetic programs, not independent holdout families.',
        all_prior_small_scale_records_retained=True,new_training_runs=0,
        model=source['model'],model_sha256=source['model_sha256'],
        frozen_execution_source_sha256=source['source_sha256'],
        original_study_sha256=sha(source_path),generator_sha256=sha(__file__),
        arc_discretization='Re-discretize scaled analytic arcs using the unchanged 0.005 mm sagitta cap; deduct each certified chord bound from the unchanged nominal tolerance.',
        common_acceptance=source['completion'],failure_rule=source['failure_rule'],
        geometry_generator=geometry_generator.relative_to(ROOT).as_posix(),geometry_generator_sha256=sha(geometry_generator),
        cases=cases,grid=[dict(case=c['id'],method=method,dt_s=dt)
            for c in cases for method in ['JNN','FIXED','S2','S2R','SCMO','COPP','NNC'] for dt in [.001,.0005]])
    write_json(REG,registration)
    print(json.dumps(dict(registration=str(REG),sha256=sha(REG),
        cases=[{key:c[key] for key in ['id','complete_program_length_mm','evaluation_tolerance_mm']} for c in cases]),indent=2))


if __name__=='__main__':
    main()
