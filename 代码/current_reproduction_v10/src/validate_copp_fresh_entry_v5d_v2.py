"""Arc-budget guards plus the same two observed fresh-entry workflow checks."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from copp_fresh_planning_v5d_entry2 import run_case,resolve_tolerance,ROOT,core
from validate_copp_fresh_entry_v5d import compare_arrays,NAMES,read

OUT=ROOT/'results/copp_v5d_fresh_entry_v2_validation'


def main():
    geometry_path=ROOT/'results/copp_v5c_observed_additional/final_rounded_triangle_source_geometry.json'
    geometry=read(geometry_path)
    declared=.008564207932948903
    resolved=resolve_tolerance(geometry,1.25-declared,1.25,None)
    assert resolved==(1.25,declared,declared)
    guards=[]
    bad=deepcopy(geometry)
    del next(row for row in bad['primitives'] if row['kind']=='arc')['chord_error_bound_mm']
    for name,geom,tol,nominal,chord in [
        ('underreserved_arc',geometry,1.25,1.25,0.),
        ('missing_arc_bound',bad,1.25,1.25,None),
        ('nonfinite_nominal',None,1.25,float('nan'),None)]:
        try:
            resolve_tolerance(geom,tol,nominal,chord)
        except ValueError as exc:
            guards.append(dict(case=name,rejected=True,error=str(exc)))
        else:
            raise AssertionError('Invalid input budget accepted: '+name)
    registration_path=ROOT/'results/historical_strong_v5/registration.json'
    original=read(registration_path)
    rows=[next(row for row in original['cases'] if row['id']==name) for name in NAMES]
    if (OUT/'registration.json').exists():
        raise FileExistsError('Preserve completed v2 workflow validation')
    OUT.mkdir(parents=True,exist_ok=True)
    core.write(OUT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        version='v5d-fresh-entry-v2',scope='Two already observed programs only; arc budget checks use metadata without path execution',
        fixed_cases=NAMES,cases=rows,arc_guard_source=str(geometry_path),arc_guard_source_sha256=core.sha(geometry_path),
        arc_guard_resolved_values=list(resolved),arc_guard_checks=guards,
        original_registration_sha256=core.sha(registration_path),
        source_sha256={name:core.sha(ROOT/'src'/name) for name in ['validate_copp_fresh_entry_v5d_v2.py',
            'copp_fresh_planning_v5d_entry2.py','copp_fresh_planning_v5d_entry1.py',
            'copp_profile_projection_v5d_entry1.py','copp_command_projection_v5d.py']},
        policy='Retain prior v1 and canonical v5d; same two pure-line observed programs, three fresh SOCP iterations, two periods, no new holdout'))
    results=[]
    for row in rows:
        reference_file=ROOT/row['reference']
        assert core.sha(reference_file)==row['reference_sha256']
        reference=np.loadtxt(reference_file,delimiter=',',skiprows=1)
        folder=OUT/row['id']
        record=run_case(row['id'],reference,None,row['evaluation_tolerance_mm'],folder,
            input_provenance=dict(reference=str(reference_file),reference_sha256=row['reference_sha256']),
            nominal_tolerance=row['nominal_tolerance_mm'],chord_error=None,
            scope='Observed-development v2 guard and fresh-solver workflow check; canonical results unchanged')
        old=read(ROOT/'results/copp_v5d_complete_development'/(row['id']+'.json'))
        comparisons=[]
        for fresh,previous in zip(record['iterations'],old['iterations']):
            assert fresh['iteration']==previous['iteration']
            runs=[]
            for a,b in zip(fresh['runs'],previous['runs']):
                assert a['dt_s']==b['dt_s']
                runs.append(dict(dt_s=a['dt_s'],old_success=b['metrics']['success'],fresh_success=a['metrics']['success'],
                    old_complete_time_s=b['metrics']['complete_time_s'],fresh_complete_time_s=a['metrics']['complete_time_s'],
                    metrics_exact_equal=a['metrics']==b['metrics'],
                    array_comparison=compare_arrays(folder/a['trace'],ROOT/'results/copp_v5d_complete_development'/b['trace'])))
            comparisons.append(dict(iteration=fresh['iteration'],
                raw_profile_array_comparison=compare_arrays(fresh['parent_profile'],previous['parent_profile']),
                fresh_selected_projection_candidate=fresh['selected_projection_candidate'],runs=runs))
        complete=len(record['iterations'])==3 and all(run['metrics']['success'] for item in record['iterations'] for run in item['runs'])
        results.append(dict(path_id=row['id'],status=record['status'],fresh_record=str(folder/(row['id']+'.json')),
            fresh_record_sha256=core.sha(folder/(row['id']+'.json')),fresh_total_wall_s=record['total_wall_s'],
            underlying_pipeline_wall_s=record.get('underlying_pipeline_wall_s'),
            fresh_initial_planning_total_wall_s=record.get('initial_planning_total_wall_s'),
            fresh_solve_wall_s=sum(item['solver_wall_s'] for item in record['iterations']),
            fresh_projection_phase_wall_s=sum(item['projection_phase_wall_s'] for item in record['iterations']),
            all_selected_runs_pass=complete,comparisons=comparisons))
        print(json.dumps(dict(path_id=row['id'],status=record['status'],complete=complete,
            all_metrics_equal=all(run['metrics_exact_equal'] for item in comparisons for run in item['runs']))),flush=True)
    core.write(OUT/'summary.json',results)
    core.write(ROOT/'audit/copp_fresh_entry_v2_validation_v5d.json',dict(version='v5d-fresh-entry-v2',
        programs=2,selected_runs=12,all_selected_runs_pass=all(row['all_selected_runs_pass'] for row in results),
        arc_guard_checks=guards,arc_guard_derived_budget=resolved,no_new_holdout_programs_executed=True,records=results))


if __name__=='__main__':
    main()
