"""Failure-inclusive baseline tables generated from complete execution records."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common import write_json,sha,utc


def collect(folder,expected):
    rows=[]
    for p in sorted(folder.glob('*.json')):
        r=json.loads(p.read_text('utf-8'))
        if 'method' not in r or 'path_id' not in r:continue
        trace=ROOT/r['trace']
        exists=trace.is_file()
        if exists and sha(trace)!=r['trace_sha256']:
            raise ValueError('Trace hash mismatch: '+str(p))
        if r['success'] and not exists:raise ValueError('Successful run has no full trace')
        if not r['success'] and r.get('complete_time_s') is not None:
            raise ValueError('Failure has a completion time')
        planner=r.get('planner') or {}
        ratios=r.get('vector_limit_ratios',[None,None,None])
        rows.append(dict(method=r['method'],path_id=r['path_id'],dt_s=r['dt_s'],corner_scale=r['corner_scale'],
            tolerance_mm=r['tolerance_mm'],success=r['success'],reason=r.get('reason'),
            error=r.get('error'),complete_time_s=r.get('complete_time_s'),
            trace_status='full_generated_trace' if exists else 'no_trace_planning_failure',
            speed_ratio=ratios[0],acceleration_ratio=ratios[1],jerk_ratio=ratios[2],
            max_deviation_mm=r.get('max_polyline_deviation_mm'),
            ordered_complete=(r.get('geometry') or {}).get('ordered_complete'),
            rest_pass=r.get('rest_pass'),planning_wall_s=planner.get('planning_wall_s',planner.get('planning_wall_time_s')),
            planning_and_audit_wall_s=r.get('wall_s'),record=str(p.relative_to(ROOT)),record_sha256=sha(p),
            trace=r['trace'],trace_sha256=r['trace_sha256']))
    keys=[(r['method'],r['path_id'],r['dt_s'],r['corner_scale']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate method/path/resolution key')
    methods={}
    for method in ('S2','S2R','SCMO'):
        subset=[r for r in rows if r['method']==method]
        successes=[r for r in subset if r['success']]
        cost=[r['planning_wall_s'] for r in subset if r['planning_wall_s'] is not None]
        methods[method]=dict(attempted=len(subset),completed=len(successes),failed=len(subset)-len(successes),
            planning_wall_total_s=sum(cost),planning_wall_range_s=[min(cost),max(cost)] if cost else None,
            planning_and_audit_wall_total_s=sum(r['planning_and_audit_wall_s'] or 0. for r in subset),
            failed_planning_without_trace_wall_s=sum(r['planning_and_audit_wall_s'] or 0. for r in subset if r['trace_status']=='no_trace_planning_failure'),
            sum_successful_time_s=sum(r['complete_time_s'] for r in successes),
            success_only_time_sum_not_an_unconditional_efficiency_statistic=True,
            failure_records=[r['record'] for r in subset if not r['success']])
    pairs=[]
    for coarse in rows:
        if coarse['dt_s']!=.001:continue
        fine=next((r for r in rows if r['method']==coarse['method'] and r['path_id']==coarse['path_id'] and r['dt_s']==.0005 and r['corner_scale']==coarse['corner_scale']),None)
        if fine:
            both=coarse['success'] and fine['success']
            pairs.append(dict(method=coarse['method'],path_id=coarse['path_id'],both_completed=both,
                coarse_success=coarse['success'],fine_success=fine['success'],
                absolute_time_difference_s=abs(coarse['complete_time_s']-fine['complete_time_s']) if both else None,
                speed_ratio_difference=abs(coarse['speed_ratio']-fine['speed_ratio']) if both else None,
                acceleration_ratio_difference=abs(coarse['acceleration_ratio']-fine['acceleration_ratio']) if both else None,
                jerk_ratio_difference=abs(coarse['jerk_ratio']-fine['jerk_ratio']) if both else None))
    return dict(expected=expected,observed=len(rows),complete_matrix=len(rows)==expected,
        methods=methods,resolution_pairs=pairs,runs=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    frozen=ROOT/'results/baseline_v3/final_freeze_v3b.json'
    freeze=json.loads(frozen.read_text('utf-8'))
    changed=[name for name,digest in freeze['sha256'].items() if sha(ROOT/name)!=digest]
    if changed:raise ValueError('Frozen source changed: '+repr(changed))
    result=dict(created_utc=utc(),source_sha256=sha(Path(__file__)),freeze_sha256=sha(frozen),
        inference_scope='Each deterministic program is a distinct case; sampling resolutions are numerical repeats, not independent replicates. Planning wall time is local observed cost, not a real-time guarantee.',
        final_v3b=collect(ROOT/'results/baseline_v3/final_v3b',30),
        development_unified=collect(ROOT/'results/baseline_v3/development_unified',12),
        validation=collect(ROOT/'results/baseline_v3/validation_final',12))
    registry=json.loads((ROOT/'prospective/final_v3/registration.json').read_text('utf-8'))
    expected={(method,p['id'],dt) for p in registry['paths'] for method in ('S2','S2R','SCMO') for dt in (.001,.0005)}
    actual={(r['method'],r['path_id'],r['dt_s']) for r in result['final_v3b']['runs']}
    if actual-expected:raise ValueError('Unregistered final cells: '+repr(actual-expected))
    result['final_v3b']['missing_cells']=sorted(expected-actual)
    output=ROOT/'results/baseline_v3/summary_v3b.json';write_json(output,result)
    fields=list((result['final_v3b']['runs'] or result['development_unified']['runs'])[0])
    with (output.with_suffix('.csv')).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['study',*fields]);writer.writeheader()
        for study in ('final_v3b','development_unified','validation'):
            for row in result[study]['runs']:writer.writerow(dict(study=study,**row))
    print(json.dumps({name:{k:result[name][k] for k in ('expected','observed','complete_matrix','methods')} for name in ('final_v3b','development_unified','validation')},indent=2))
    if args.require_complete and not all(result[s]['complete_matrix'] for s in ('final_v3b','development_unified','validation')):
        raise SystemExit('Expected baseline matrix is not yet complete')


if __name__=='__main__':main()
