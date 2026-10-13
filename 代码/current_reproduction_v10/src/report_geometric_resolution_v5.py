"""Generate an internal numerical-resolution report from retained complete runs."""
import json
from common import ROOT,sha,utc,write_json
from geometric_resolution_check_v5 import OUT,PRIOR,CASES,read,verify


def main():
    reg=verify();summary=read(OUT/'summary.json');control=read(OUT/'control_1ms.json')
    checks=[read(OUT/f'dt{value}us/unit_audit/report.json') for value in (1000,500)]
    assert control['status']=='passed' and all(x['status']=='passed' for x in checks)
    rows=summary['results'];costs=[]
    for row in rows:
        base=OUT/'dt500us'/row['case'];result=read(base/'actual.json')
        assert sha(base/'actual.npz')==result['trace_sha256']==row['trace_sha256']
        pred=base/'predictions'
        logs={name:[json.loads(line) for line in (pred/path).read_text('utf-8').splitlines()]
            for name,path in [('decision','decisions.jsonl'),('branch','branch_records.jsonl'),('stop','stops/branch_records.jsonl')]}
        costs.append(dict(case=row['case'],full_wall_s=result['cost']['full_evaluation_wall_s'],
            full_process_cpu_s=result['cost']['full_run_including_archive_and_metrics_process_cpu_s'],
            recorded_completed_decision_wall_s=sum(x.get('total_wall_s',0.) for x in logs['decision']),
            recorded_completed_planning_wall_s=sum(x.get('planning_wall_s',0.) for x in logs['decision']),
            recorded_branch_wall_s=sum(x['wall_s'] for x in logs['branch']),
            recorded_stop_wall_s=sum(x['assessment_wall_s'] for x in logs['stop']),
            recorded_stop_cpu_s=sum(x['assessment_cpu_s'] for x in logs['stop']),
            aborted_decisions=sum(x['source']=='aborted' for x in logs['decision']),
            accounting='Nested branch and stopping costs are included in full evaluation; never add them to the full total. Full process CPU includes archive and metrics and is not per-decision latency.'))
    report=dict(created_utc=utc(),registration_sha256=sha(OUT/'registration.json'),summary_sha256=sha(OUT/'summary.json'),
        registered_runs=len(rows),complete_successes=sum(x['half_ms_success'] for x in rows),results=rows,
        physical_time_configuration=reg['physical_constants'],one_ms_full_arrays_equal=all(control['all_arrays_equal'].values()),
        both_dt_stop_prechecks_pass=True,all_requested_programs_and_failures_retained=True,costs=costs,
        no_new_training=True,no_final_holdout=True,no_hardware_or_real_time_claim=True)
    write_json(ROOT/'audit/geometric_resolution_v5_4_report.json',report)
    lines=['# v5_4 physical-time resolution check','',
        'This report concerns one fixed historical teacher and the v5_4 geometric-resumption executor on three previously observed development programs. It is a numerical check, not an independent seed study or final test. The original controller, runtime files, model and main manuscript remain byte-preserved.','',
        f"The 1 ms zigzag control repeats all {len(control['all_arrays_equal'])} saved arrays exactly, including observations, requested actions, ordered action indices and command sources. At both 1 and 0.5 ms, the initial raw decision, retained full stop, stale-stop rejection and subsequent resumption passed the registered checks.",'',
        '## Physical settings','',
        '- Both resolutions retain a 20 ms decision period, 12 ms heading-filter time constant, 0.1 mm guidance transition, full length-based physical-time budget and 300 s wall cap per program.',
        '- Controller finite differences, complete-decision checks and stored-stop integration use the current dt. The frozen runner is reused with only installation, physical-time tick budget and metric-dt expressions bound by the new entry.',
        '- The original 21-sample terminal threshold is expressed as 21 ms. The cubic brake extra three hold samples are expressed as 3 ms, retaining the original 23 ms physical hold. The common final acceptance still requires at least 20 ms rest; no corridor, gate, vector bound or completion condition is relaxed.',
        '- The same raw model, correction grid, branch order, geometric resumption request and cubic stopping family are used. Existing terminal allowance remains 2.1 s.','',
        '## Complete program outcomes','',
        '| Program | 1 ms complete (s) | 0.5 ms outcome | Time change (%) |','|---|---:|---:|---:|']
    for row in rows:
        outcome=f"{row['half_ms_complete_time_s']:.4f} s" if row['half_ms_success'] else f"Incomplete; {row['half_ms_prefix_s']:.4f} s prefix"
        change='—' if row['relative_time_change_percent'] is None else f"{row['relative_time_change_percent']:+.3f}"
        lines.append(f"| {row['case']} | {row['one_ms_complete_time_s']:.3f} | {outcome} | {change} |")
    lines.extend(['',f"The finer-resolution run completed {report['complete_successes']}/{len(rows)} programs. Completion times include every intermediate stop. All registered outcomes are retained; no mean over only successful programs is presented.",'',
        '| Program | 0.5 ms max deviation (mm) | Final rest (ms) | Full evaluation wall (s) | Full process CPU (s) |','|---|---:|---:|---:|---:|'])
    for row,cost in zip(rows,costs):
        lines.append(f"| {row['case']} | {row['half_ms_deviation_mm']:.6f} | {row['half_ms_rest_s']*1000:g} | {cost['full_wall_s']:.3f} | {cost['full_process_cpu_s']:.3f} |")
    lines.extend(['',
        'Nested planning, branch and stopping costs remain available in the JSON report. These measured shared-host costs do not demonstrate hard-real-time operation. Resolution differences concern the complete hybrid controller and its discrete branch selection, not only finite-difference roundoff.','',
        '## Reproduction','',
        'From `D:\\PPO_article_work\\JNN_dominance_20261011_v5`, use `D:\\Anaconda\\envs\\PPO\\python.exe`. The registered entry is `src/geometric_resolution_check_v5.py` with modes `register`, `precheck --dt 0.001`, `run --dt 0.001 --case validation_zigzag`, `precheck --dt 0.0005`, then `run --dt 0.0005 --case <case>` once for each of the three registered programs, followed by `summary`. Creation modes reject existing output directories; use a separate workspace to reproduce the full experiment.','',
        'Regenerate this report without running a controller: `src/report_geometric_resolution_v5.py`. Full actual, speculative and stop traces are in `results/geometric_resolution_check_v5_4`. The new entry and its registration bind all frozen parent sources and original reference trajectories.'])
    (ROOT/'audit/geometric_resolution_v5_4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(completed=report['complete_successes'],runs=len(rows),one_ms_arrays_equal=True)),flush=True)


if __name__=='__main__':main()
