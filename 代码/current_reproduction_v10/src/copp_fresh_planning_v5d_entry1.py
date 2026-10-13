"""Fresh official COPP solves followed by the unchanged v5d projection rule."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import copp_command_projection_v5d as rule
from copp_cases_v5 import normalize_declared_primitives
from copp_reserve_replay_v5c import install_for_new_planning
from copp_profile_projection_v5d_entry1 import project_profile, VERSION

ROOT = Path(__file__).resolve().parents[1]


def run_case(path_id, reference, source_geometry, tolerance, output, *, input_provenance,
             nominal_tolerance=None, chord_error=0., scope='Explicitly registered caller input'):
    output = Path(output)
    if (output/'registration.json').exists():
        raise FileExistsError('Preserve every prior fresh-planning run')
    reference = np.asarray(reference,float)
    nominal_tolerance = tolerance+chord_error if nominal_tolerance is None else nominal_tolerance
    if abs(nominal_tolerance-chord_error-tolerance) > 1e-12:
        raise ValueError('Nominal tolerance must equal evaluation tolerance plus chord reserve')
    output.mkdir(parents=True,exist_ok=True)
    source_files = ['copp_fresh_planning_v5d_entry1.py','copp_profile_projection_v5d_entry1.py',
        'copp_command_projection_v5d.py','copp_reserve_replay_v5c.py','copp_adapter_v5b.py',
        'copp_signed_geometry_v5b.py','copp_adapter_v5.py','copp_cases_v5.py',
        'copp_path_adapter_v4.py','copp_geometry_only_v4.py','copp_time_projection_v4.py',
        'metrics.py','metrics_exact_index_v5.py','exact_distance_index_v5.py',
        'runtime_configuration.py','runtime_bootstrap.py']
    registration = dict(created_utc=datetime.now(timezone.utc).isoformat(),version=VERSION,
        numerical_rule='Frozen v5d-command-projection',scope=scope,path_id=path_id,
        input_provenance=input_provenance,nominal_tolerance_mm=nominal_tolerance,
        arc_chord_error_bound_mm=chord_error,evaluation_tolerance_mm=tolerance,
        model_identity='Official COPP 0.2.3 TOPP3-SOCP deterministic planner; no learned model',
        planner_settings=rule.parent.SETTINGS,periods_s=list(rule.PERIODS),limits=list(rule.LIMITS),
        frozen_projection_budget=rule.MAX_CANDIDATES,correction_reserve=rule.CORRECTION_RESERVE,
        selection='Three fresh sequential SOCP solves, fixed third iterate; every initial and projected command retained',
        source_sha256={name:rule.core.sha(ROOT/'src'/name) for name in source_files})
    rule.core.write(output/'registration.json',registration)
    started = time.perf_counter()
    initial_directory = output/'initial_v5c'
    initial = install_for_new_planning()(path_id,reference,tolerance,source_geometry,initial_directory,
        settings=rule.parent.SETTINGS,limits=rule.LIMITS)
    record = dict(path_id=path_id,registered_adapter_version=VERSION,iterations=[],
        nominal_tolerance_mm=nominal_tolerance,chord_error_mm=chord_error,tolerance_mm=tolerance,
        limits=list(rule.LIMITS),planner=initial.get('planner'),optimizer_rerun=True,
        initial_planning_record=str(initial_directory/(path_id+'.json')),
        initial_planning_record_sha256=rule.core.sha(initial_directory/(path_id+'.json')),
        initial_planning_total_wall_s=initial['total_wall_s'])
    if initial['status'] != 'completed':
        record.update(status='initial_planning_failed',error=initial.get('error'),iterations=initial['iterations'],
            total_wall_s=time.perf_counter()-started)
        rule.core.write(output/(path_id+'.json'),record)
        return record
    curve = None
    for previous in initial['iterations']:
        clock = time.perf_counter()
        runs, verification_wall = rule.verify_reused_runs(initial_directory,previous,reference,tolerance)
        raw_file = initial_directory/f"{path_id}_iteration{previous['iteration']}_profile.npz"
        with np.load(raw_file) as stored:
            raw = {name:stored[name] for name in stored.files}
        if not np.array_equal(reference,raw['reference']):
            raise ValueError('Fresh raw profile reference changed')
        if not all(run['metrics']['success'] for run in runs):
            if curve is None:
                geometry_clock = time.perf_counter()
                curve,*_ = rule.parent.prepare_geometry(reference,tolerance=tolerance,source_geometry=source_geometry)
                sigma,*_ = rule.parent.grid_geometry(curve,source_geometry,rule.parent.SETTINGS)
                record['projection_geometry_reconstruction_wall_s'] = time.perf_counter()-geometry_clock
            if not np.array_equal(sigma/rule.parent.SETTINGS['spatial_scale_mm'],raw['s']):
                raise ValueError('Fresh geometry reconstruction changed station grid')
        projected = project_profile(path_id,previous['iteration'],raw,curve,reference,tolerance,output,runs)
        projected.update(iteration=previous['iteration'],solver_wall_s=previous['solver_wall_s'],solver_rerun=True,
            parent_profile=str(raw_file),parent_profile_sha256=rule.core.sha(raw_file),
            candidate_zero_verification_wall_s=verification_wall,
            projection_phase_wall_s=time.perf_counter()-clock)
        record['iterations'].append(projected)
        rule.core.write(output/(path_id+'.json'),record)
    record.update(status='completed' if all(item['joint_acceptance_pass'] for item in record['iterations']) else 'acceptance_failed',
        total_wall_s=time.perf_counter()-started)
    rule.core.write(output/(path_id+'.json'),record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--path-id',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--geometry',type=Path)
    parser.add_argument('--tolerance',type=float,required=True)
    parser.add_argument('--nominal-tolerance',type=float)
    parser.add_argument('--chord-error',type=float,default=0.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reference=np.loadtxt(args.reference,delimiter=',',skiprows=1)
    source=json.loads(args.geometry.read_text('utf-8')) if args.geometry else None
    source=normalize_declared_primitives(source,reference) if source else None
    provenance=dict(reference=str(args.reference.resolve()),reference_sha256=rule.core.sha(args.reference),
        geometry=str(args.geometry.resolve()) if args.geometry else None,
        geometry_sha256=rule.core.sha(args.geometry) if args.geometry else None)
    record=run_case(args.path_id,reference,source,args.tolerance,args.output,input_provenance=provenance,
        nominal_tolerance=args.nominal_tolerance,chord_error=args.chord_error)
    print(json.dumps(dict(path_id=record['path_id'],status=record['status'],total_wall_s=record['total_wall_s'],
        iterations=[dict(iteration=row['iteration'],runs=[dict(dt=run['dt_s'],success=run['metrics']['success'],
            T=run['metrics']['complete_time_s']) for run in row['runs']]) for row in record['iterations']]),indent=2))


if __name__=='__main__':
    main()
