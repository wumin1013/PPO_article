"""Guarded fresh-entry wrapper; frozen planner and v5d rule remain unchanged."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import copp_fresh_planning_v5d_entry1 as pipeline
from copp_cases_v5 import normalize_declared_primitives

ROOT=pipeline.ROOT
core=pipeline.rule.core
VERSION='v5d-fresh-entry-v2'


def resolve_tolerance(source_geometry,tolerance,nominal_tolerance=None,chord_error=None):
    bounds=[]
    for primitive in (source_geometry or {}).get('primitives',[]):
        if primitive['kind']=='arc':
            if 'chord_error_bound_mm' not in primitive:
                raise ValueError('Every declared analytic arc requires an explicit chord-error bound')
            bound=float(primitive['chord_error_bound_mm'])
            if not np.isfinite(bound) or bound<0:
                raise ValueError('Arc chord-error bounds must be finite and nonnegative')
            bounds.append(bound)
    declared=max(bounds,default=0.)
    chord=declared if chord_error is None else float(chord_error)
    if not np.isfinite(chord) or chord<0 or chord<declared-1e-12:
        raise ValueError('Supplied chord reserve is smaller than the declared analytic-arc bound')
    nominal=float(tolerance)+chord if nominal_tolerance is None else float(nominal_tolerance)
    if not np.isfinite(tolerance) or tolerance<=0 or not np.isfinite(nominal) or nominal<=0 or abs(nominal-chord-tolerance)>1e-12:
        raise ValueError('Positive evaluation tolerance must equal nominal tolerance minus the complete arc chord reserve')
    return nominal,chord,declared


def run_case(path_id,reference,source_geometry,tolerance,output,*,input_provenance,
             nominal_tolerance=None,chord_error=None,scope='Explicitly registered caller input'):
    reference=np.asarray(reference,float)
    source_geometry=normalize_declared_primitives(source_geometry,reference) if source_geometry else None
    nominal,chord,declared=resolve_tolerance(source_geometry,tolerance,nominal_tolerance,chord_error)
    output=Path(output)
    if (output/'registration.json').exists():
        raise FileExistsError('Preserve existing registered fresh planning evidence')
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'input_reference.npz',reference=reference)
    if source_geometry:
        core.write(output/'input_geometry.json',source_geometry)
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),version=VERSION,path_id=path_id,
        scope=scope,model_identity='Unchanged deterministic COPP 0.2.3 TOPP3-SOCP and frozen v5d projection',
        underlying_entry='v5d-fresh-entry-v1',input_provenance=input_provenance,
        reference_array_file='input_reference.npz',reference_array_sha256=core.sha(output/'input_reference.npz'),
        normalized_geometry_file='input_geometry.json' if source_geometry else None,
        normalized_geometry_sha256=core.sha(output/'input_geometry.json') if source_geometry else None,
        nominal_tolerance_mm=nominal,evaluation_tolerance_mm=tolerance,
        declared_maximum_arc_chord_error_mm=declared,arc_chord_error_bound_mm=chord,
        source_sha256={name:core.sha(ROOT/'src'/name) for name in ['copp_fresh_planning_v5d_entry2.py',
            'copp_fresh_planning_v5d_entry1.py','copp_profile_projection_v5d_entry1.py',
            'copp_command_projection_v5d.py','copp_cases_v5.py']},
        policy='Only input-budget validation, standalone input persistence and exception recording are added; planner, frozen projection and canonical third SOCP iterate are unchanged')
    core.write(output/'registration.json',registration)
    started=time.perf_counter()
    inner=output/'solver_pipeline'
    try:
        result=pipeline.run_case(path_id,reference,source_geometry,tolerance,inner,
            input_provenance=input_provenance,nominal_tolerance=nominal,chord_error=chord,scope=scope)
        record=deepcopy(result)
        for item in record['iterations']:
            groups=[item['runs']]+[candidate['runs'] for candidate in item.get('projection_candidates',[])]
            for runs in groups:
                for run in runs:
                    run['trace']=str(inner/run['trace'])
        record.update(registered_adapter_version=VERSION,underlying_entry_version='v5d-fresh-entry-v1',
            underlying_record=str(inner/(path_id+'.json')),underlying_record_sha256=core.sha(inner/(path_id+'.json')),
            underlying_pipeline_wall_s=result['total_wall_s'])
    except Exception as exc:
        initial=inner/'initial_v5c'/(path_id+'.json')
        record=dict(path_id=path_id,registered_adapter_version=VERSION,status='failed',
            error_type=type(exc).__name__,error=str(exc),iterations=[],
            tolerance_mm=tolerance,nominal_tolerance_mm=nominal,chord_error_mm=chord,
            limits=list(pipeline.rule.LIMITS),initial_evidence_record=str(initial) if initial.exists() else None,
            initial_evidence_record_sha256=core.sha(initial) if initial.exists() else None,
            retained_evidence_directory=str(inner),complete_time_s=None)
    record['total_wall_s']=time.perf_counter()-started
    core.write(output/(path_id+'.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--path-id',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--geometry',type=Path)
    parser.add_argument('--tolerance',type=float,required=True)
    parser.add_argument('--nominal-tolerance',type=float)
    parser.add_argument('--chord-error',type=float)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=np.loadtxt(args.reference,delimiter=',',skiprows=1)
    geometry=json.loads(args.geometry.read_text('utf-8')) if args.geometry else None
    record=run_case(args.path_id,reference,geometry,args.tolerance,args.output,
        input_provenance=dict(reference=str(args.reference.resolve()),reference_sha256=core.sha(args.reference),
            geometry=str(args.geometry.resolve()) if args.geometry else None,
            geometry_sha256=core.sha(args.geometry) if args.geometry else None),
        nominal_tolerance=args.nominal_tolerance,chord_error=args.chord_error)
    print(json.dumps(dict(path_id=record['path_id'],status=record['status'],total_wall_s=record['total_wall_s'],
        error=record.get('error'),runs=[dict(iteration=item['iteration'],dt=run['dt_s'],
            success=run['metrics']['success'],T=run['metrics']['complete_time_s'])
            for item in record['iterations'] for run in item['runs']]),indent=2))


if __name__=='__main__':
    main()
