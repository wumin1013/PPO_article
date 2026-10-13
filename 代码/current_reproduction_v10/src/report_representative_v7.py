"""Combine all accepted times and unsuccessful attempts without outcome selection."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import numpy as np
from common import ROOT, sha, utc, write_json
from metrics_exact_index_v5 import metrics

REGISTRATION = ROOT / 'evidence/new_representative_v7_registration.json'
LEARNED = ROOT / 'results/new_representative_v7/learned_comparators/trace_summary.json'
METHODS = ['JNN', 'FIXED', 'S2', 'S2R', 'SCMO', 'COPP', 'NNC']
LABELS = dict(JNN='J-NNC', FIXED='Fixed preview', S2='Two-step (S2-Q)',
              S2R='Two-step (S2-R)', SCMO='SCMO', COPP='COPP', NNC='NNC')


def read(path):
    return json.loads(path.read_text('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    parser.add_argument('--learned-summary', type=Path, default=LEARNED)
    parser.add_argument('--traditional-summary', type=Path,
                        default=ROOT/'results/new_representative_v7/traditional/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT/'audit/representative_v7')
    args = parser.parse_args()
    for name in ('registration', 'learned_summary', 'traditional_summary', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    registration_path=args.registration
    learned_path=args.learned_summary
    registration = read(registration_path)
    learned = read(learned_path)
    traditional = read(args.traditional_summary)
    for summary, path in [(learned, learned_path), (traditional, args.traditional_summary)]:
        subregistration_path = path.parent/'registration.json'
        subregistration = read(subregistration_path)
        assert summary['registration_sha256'] == sha(subregistration_path)
        assert subregistration['global_registration_sha256'] == sha(registration_path)
        for source, expected_sha in subregistration.get('source_registrations', {}).items():
            assert sha(ROOT/source) == expected_sha
    rows = []
    for item in learned['rows']:
        assert sha(ROOT/item['trace']) == item['trace_sha256']
        assert sha(ROOT/item['record']) == item['record_sha256']
        metric = item['metrics']
        rows.append(dict(case=item['case'], method=item['method'], dt_s=item['dt_s'],
                         metrics=metric, trace=item['trace'], trace_sha256=item['trace_sha256'],
                         record=item['record'], record_sha256=item['record_sha256'],
                         model_sha256=item['model_sha256']))
    for item in traditional['records']:
        record_path = (ROOT/item['record'] if item.get('record') else
                       args.traditional_summary.parent / (item['run_id']+'.json'))
        if item.get('record_sha256'):
            assert sha(record_path) == item['record_sha256']
        record = read(record_path)
        if record.get('trace'):
            assert sha(ROOT/record['trace']) == record['trace_sha256']
        rows.append(dict(case=record['case'], method=record['method'], dt_s=record['dt_s'],
                         metrics=record, trace=record.get('trace'), trace_sha256=record.get('trace_sha256'),
                         record=record_path.relative_to(ROOT).as_posix(), record_sha256=sha(record_path),
                         planner_configuration=record.get('planner')))
    expected = {(x['case'],x['method'].upper(),x['dt_s']) for x in registration['grid']}
    actual = {(x['case'],x['method'],x['dt_s']) for x in rows}
    assert len(rows) == len(actual) == 28 and actual == expected
    for row in rows:
        metric = row['metrics']
        case = next(c for c in registration['cases'] if c['id'] == row['case'])
        if row['trace']:
            with np.load(ROOT/row['trace']) as archive:
                reference,positions=archive['reference'],archive['positions']
            expected_reference=np.loadtxt(ROOT/case['reference'],delimiter=',',skiprows=1)
            assert np.array_equal(reference,expected_reference)
            assert np.array_equal(positions[0],reference[0])
            independent=metrics(reference,positions,dt=row['dt_s'],
                tolerance=case['evaluation_tolerance_mm'], limits=registration['planar_vector_limits'],
                declared_success=metric['success'],reason=metric['reason'])
            assert independent['success']==metric['success']
            assert independent['complete_time_s']==metric['complete_time_s']
            assert np.allclose(independent['vector_peaks'],metric['vector_peaks'],rtol=1e-10,atol=1e-7)
            assert independent['geometry']['gates_passed']==metric['geometry']['gates_passed']
            row['metrics']=metric=independent
            row['complete_interval_audit_recomputed']=True
        else:
            assert not metric['success']
            row['complete_interval_audit_recomputed']=False
        assert metric['success'] == (metric['complete_time_s'] is not None)
        if metric['success']:
            assert metric['complete_time_s'] <= case['physical_time_budget_s'] + row['dt_s']
            assert metric['geometry']['ordered_complete']
            assert metric['geometry']['corridor_inside']
            assert metric['kinematic_pass'] and metric['rest_pass']
    rows.sort(key=lambda x:(list(c['id'] for c in registration['cases']).index(x['case']),
                            METHODS.index(x['method']),-x['dt_s']))
    savings = []
    for case in registration['cases']:
        for dt in registration['dt_s']:
            main_time = next(r['metrics']['complete_time_s'] for r in rows
                             if r['case']==case['id'] and r['method']=='JNN' and r['dt_s']==dt)
            assert main_time is not None and math.isfinite(main_time) and main_time > 0
            for row in rows:
                if row['case'] != case['id'] or row['dt_s'] != dt or row['method']=='JNN':
                    continue
                other = row['metrics']['complete_time_s']
                savings.append(dict(case=case['id'], dt_s=dt, comparator=row['method'],
                    jnn_time_s=main_time, comparator_time_s=other,
                    jnn_time_saving_percent=100*(other-main_time)/other if other is not None else None))
    output = dict(created_utc=utc(), registration_sha256=sha(registration_path),
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [learned_path,args.traditional_summary]},
        scope=registration.get('case_selection',registration.get('scope')), all_28_cells_included=True,
        scientific_unit='One complete fixed-controller or deterministic-plan execution per program and resolution.',
        rows=rows, savings=savings)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'comparison.json',output)
    stream = io.StringIO(newline='')
    writer=csv.writer(stream, lineterminator='\n')
    writer.writerow(['case','method','dt_s','complete','complete_time_s','prefix_s','reason'])
    for row in rows:
        m=row['metrics']
        writer.writerow([row['case'],row['method'],row['dt_s'],m['success'],m['complete_time_s'],
                         m.get('elapsed_prefix_s'),m['reason']])
    (args.output/'complete_comparison.csv').write_text(stream.getvalue(),encoding='utf-8')
    lines=[r'\begin{table}[tb]',r'\centering',r'\small',
        r'\caption{Complete execution times on the curvature coupon and neck contour. Each numerical entry meets the common complete-program criteria. A dash denotes no accepted complete execution.}',
        r'\label{tab:representative_comparison}',r'\begin{tabular}{lrrrr}',r'\toprule',
        r'& \multicolumn{2}{c}{Curvature coupon (s)} & \multicolumn{2}{c}{Neck contour (s)} \\',
        r'Method & 1 ms & 0.5 ms & 1 ms & 0.5 ms \\',r'\midrule']
    for method in METHODS:
        values=[]
        for case in registration['cases']:
            for dt in registration['dt_s']:
                value=next(r['metrics']['complete_time_s'] for r in rows
                           if (r['case'],r['method'],r['dt_s'])==(case['id'],method,dt))
                values.append('--' if value is None else f'{value:.4f}')
        lines.append(LABELS[method]+' & '+' & '.join(values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (args.output/'comparison_table.tex').write_text('\n'.join(lines),encoding='utf-8')
    def get(case,method,dt=.001):
        return next(r['metrics'] for r in rows if (r['case'],r['method'],r['dt_s'])==(case,method,dt))
    prose=[]
    for definition in registration['cases']:
        case=definition['id']
        title={'piecewise_curvature_coupon':'curvature coupon',
               'asymmetric_neck_contour':'neck contour'}[definition.get('parent_case',case)]
        jnn,fixed=get(case,'JNN'),get(case,'FIXED')
        prose.append(f"On the {title}, J-NNC completed the motion in {jnn['complete_time_s']:.3f} s, "
                     f"compared with {fixed['complete_time_s']:.3f} s for fixed preview. "
                     f"Their maximum command-point deviations were {jnn['max_polyline_deviation_mm']:.3f} "
                     f"and {fixed['max_polyline_deviation_mm']:.3f} mm, respectively.")
        traditional_times=[(LABELS[method],get(case,method)['complete_time_s'])
                           for method in ['S2','S2R','SCMO','COPP']]
        accepted=[(name,value) for name,value in traditional_times if value is not None]
        if accepted:
            prose.append('The corresponding complete times were '+', '.join(
                f'{value:.3f} s for {name}' for name,value in accepted)+'.')
        j_length=jnn.get('executed_length_mm')
        f_length=fixed.get('executed_length_mm')
        if j_length is not None and f_length is not None:
            prose.append(f"J-NNC and fixed preview executed {j_length:.3f} and {f_length:.3f} mm, "
                         f"with accumulated speed deficits of {100*jnn['complete_time_s']-j_length:.3f} "
                         f"and {100*fixed['complete_time_s']-f_length:.3f} mm.")
        prose.append('')
    (args.output/'comparison_numeric_prose.tex').write_text('\n'.join(prose),encoding='utf-8')
    print(stream.getvalue())


if __name__=='__main__':
    main()
