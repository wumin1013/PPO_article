"""Two prespecified, already observed programs validate the fresh v5d entry."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from copp_fresh_planning_v5d_entry1 import run_case, ROOT
from copp_command_projection_v5d import core

OUT=ROOT/'results/copp_v5d_fresh_entry_validation'
NAMES=['validation_zigzag','final_chamfer_chain']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_arrays(left,right):
    comparisons={}
    with np.load(left) as a,np.load(right) as b:
        for name in a.files:
            same_shape=name in b.files and a[name].shape==b[name].shape
            comparisons[name]=dict(same_shape=same_shape,
                exact_equal=bool(same_shape and np.array_equal(a[name],b[name])),
                max_abs_difference=float(np.max(np.abs(a[name]-b[name]),initial=0)) if same_shape else None)
    return comparisons


def main():
    registration_path=ROOT/'results/historical_strong_v5/registration.json'
    original=read(registration_path)
    rows=[next(row for row in original['cases'] if row['id']==name) for name in NAMES]
    if (OUT/'registration.json').exists():
        raise FileExistsError('Preserve completed entry validation')
    OUT.mkdir(parents=True,exist_ok=True)
    core.write(OUT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        version='v5d-fresh-entry-v1',scope='Only two previously observed development programs; no final holdout',
        fixed_cases=NAMES,original_registration=str(registration_path),
        original_registration_sha256=core.sha(registration_path),
        source_sha256={name:core.sha(ROOT/'src'/name) for name in ['validate_copp_fresh_entry_v5d.py',
            'copp_fresh_planning_v5d_entry1.py','copp_profile_projection_v5d_entry1.py','copp_command_projection_v5d.py']},
        cases=rows,policy='Fresh three sequential SOCP solves per path, two periods, fixed third iterate; all initial and projected outputs retained; compare to immutable existing v5d records without choosing faster rerun'))
    results=[]
    for row in rows:
        reference_file=ROOT/row['reference']
        assert core.sha(reference_file)==row['reference_sha256']
        assert not row.get('exact_arc_primitives') and not row.get('geometry')
        reference=np.loadtxt(reference_file,delimiter=',',skiprows=1)
        folder=OUT/row['id']
        record=run_case(row['id'],reference,None,row['evaluation_tolerance_mm'],folder,
            input_provenance=dict(reference=str(reference_file),reference_sha256=row['reference_sha256']),
            nominal_tolerance=row['nominal_tolerance_mm'],chord_error=row['arc_chord_error_bound_mm'],
            scope='Prespecified observed-development workflow validation, not replacement of prior canonical evidence')
        old=read(ROOT/'results/copp_v5d_complete_development'/(row['id']+'.json'))
        comparisons=[]
        for fresh,previous in zip(record['iterations'],old['iterations']):
            assert fresh['iteration']==previous['iteration']
            runs=[]
            for a,b in zip(fresh['runs'],previous['runs']):
                assert a['dt_s']==b['dt_s']
                old_trace=ROOT/'results/copp_v5d_complete_development'/b['trace']
                new_trace=folder/a['trace']
                runs.append(dict(dt_s=a['dt_s'],old_success=b['metrics']['success'],fresh_success=a['metrics']['success'],
                    old_complete_time_s=b['metrics']['complete_time_s'],fresh_complete_time_s=a['metrics']['complete_time_s'],
                    metrics_exact_equal=a['metrics']==b['metrics'],
                    array_comparison=compare_arrays(new_trace,old_trace)))
            comparisons.append(dict(iteration=fresh['iteration'],
                raw_profile_array_comparison=compare_arrays(fresh['parent_profile'],previous['parent_profile']),
                old_selected_projection_candidate=previous['selected_projection_candidate'],
                fresh_selected_projection_candidate=fresh['selected_projection_candidate'],runs=runs))
        results.append(dict(path_id=row['id'],status=record['status'],fresh_record=str(folder/(row['id']+'.json')),
            fresh_record_sha256=core.sha(folder/(row['id']+'.json')),fresh_total_wall_s=record['total_wall_s'],
            fresh_initial_planning_total_wall_s=record['initial_planning_total_wall_s'],
            fresh_solve_wall_s=sum(item['solver_wall_s'] for item in record['iterations']),
            fresh_projection_phase_wall_s=sum(item['projection_phase_wall_s'] for item in record['iterations']),
            all_selected_runs_pass=all(run['metrics']['success'] for item in record['iterations'] for run in item['runs']),
            comparisons=comparisons))
        print(json.dumps(dict(path_id=row['id'],status=record['status'],fresh_total_wall_s=record['total_wall_s'],
            all_metrics_equal=all(run['metrics_exact_equal'] for item in comparisons for run in item['runs']))),flush=True)
    core.write(OUT/'summary.json',results)
    core.write(ROOT/'audit/copp_fresh_entry_validation_v5d.json',dict(version='v5d-fresh-entry-v1',
        programs=2,selected_runs=12,all_selected_runs_pass=all(row['all_selected_runs_pass'] for row in results),
        no_new_holdout_programs_executed=True,records=results,
        interpretation='These are complete fresh-solver workflow checks on two prespecified observed programs; prior canonical v5d evidence remains unchanged. New solver costs are measured, not inherited.'))


if __name__=='__main__':
    main()
