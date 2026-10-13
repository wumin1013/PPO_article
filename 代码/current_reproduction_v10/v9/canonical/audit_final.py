"""Separate integrity, complete-grid and time-order audit of canonical views."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

HERE=Path(__file__).resolve().parent
EXPECTED_CASES=('original','dense','piecewise_curvature_coupon_scale10','asymmetric_neck_contour_scale10')
EXPECTED_METHODS=('JNN','FIXED','S2','S2R','SCMO','COPP','NNC')
EXPECTED_PERIODS=(.001,.0005)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text('utf-8'))


def audit(path,cases,methods):
    data=read(path);rows=data['rows']
    expected=set(itertools.product(cases,methods,EXPECTED_PERIODS))
    actual=[(r['case'],r['method'],r['dt_s']) for r in rows]
    assert set(actual)==expected and len(actual)==len(expected)
    assert len(set(r['id'] for r in rows))==len(rows)
    verified=[]
    for path,item in data['files'].items():
        assert Path(path).is_absolute() and Path(path).stat().st_size==item['bytes']
        assert sha(path)==item['sha256'],path
    for row in rows:
        assert sha(row['record'])==row['record_sha256']
        original=read(row['record'])
        metric=original.get('metrics',original)
        if 'metrics' not in original and 'complete_time_s' not in original:
            raise ValueError('Unknown source record schema')
        assert row['metrics']['success']==metric['success']
        assert row['metrics']['complete_time_s']==metric['complete_time_s']
        if row['metrics']['success']:
            assert row['trace'] is not None
            assert row['metrics']['complete_time_s'] is not None
        else:assert row['metrics']['complete_time_s'] is None
        if row['trace'] is not None:
            assert sha(row['trace'])==row['trace_sha256']
            with np.load(row['trace'],allow_pickle=False) as archive:
                length=len(archive['positions'])
            elapsed=(length-1)*row['dt_s']
            assert abs(elapsed-row['metrics']['elapsed_prefix_s'])<1e-8
            if row['metrics']['success']:assert abs(elapsed-row['metrics']['complete_time_s'])<1e-8
        verified.append(row['id'])
    order_checks=[]
    for case,dt in itertools.product(cases,EXPECTED_PERIODS):
        same=[r for r in rows if r['case']==case and r['dt_s']==dt]
        good=[r for r in same if r['metrics']['success']]
        good.sort(key=lambda r:(r['metrics']['complete_time_s'],methods.index(r['method'])))
        declared=next(r for r in data['time_ordering'] if r['case']==case and r['dt_s']==dt)
        assert [r['id'] for r in good]==[r['id'] for r in declared['successful_order']]
        failed={r['id'] for r in same if not r['metrics']['success']}
        assert failed==set(declared['failed_ids'])
        order_checks.append(dict(case=case,dt_s=dt,
            sorted_successful_methods=[r['method'] for r in good],
            sorted_successful_times_s=[r['metrics']['complete_time_s'] for r in good],
            failed_methods=[r['method'] for r in same if not r['metrics']['success']]))
    return dict(all_passed=True,rows_verified=len(verified),files_verified=len(data['files']),
                time_order_checks=order_checks)


def main():
    final=HERE/'final_comparison.json';general=HERE/'generalization_comparison.json'
    first=audit(final,EXPECTED_CASES,EXPECTED_METHODS)
    second=audit(general,('hilbert_cell_sweep','harmonic_radial_contour','piecewise_curvature_coupon',
        'asymmetric_neck_contour','annular_sector_raster'),('JNN',))
    canonical=read(final);compatible=read(HERE/'plot_manifest.json')
    assert compatible['canonical_sha256']==sha(final)
    assert len(compatible['runs'])==28
    assert all(isinstance(x,str) for x in compatible['files'].values())
    assert set((r['case'],r['method'],r['dt_s']) for r in compatible['runs'])==set(itertools.product(('original','dense'),EXPECTED_METHODS,EXPECTED_PERIODS))
    costs=canonical['cost_summary']['scaled_successful_traditional']
    sums=[]
    for row in costs['rows']:
        old=read(row['original_cost_record'])['wall_s']
        new=read(row['refinement_cost_record'])['wall_s']
        assert old==row['original_pipeline_wall_s'] and new==row['common_startup_check_wall_s']
        assert old+new==row['combined_recorded_pipeline_wall_s']
        sums.append(old+new)
    assert min(sums)==costs['ScaledPipelineWallMin'] and max(sums)==costs['ScaledPipelineWallMax']
    failures=canonical['cost_summary']['scaled_failed_copp_shared']
    assert failures['unique_planning_attempts']==2 and sum(len(r['paired_cell_ids']) for r in failures['rows'])==4
    result=dict(final_comparison_sha256=sha(final),generalization_comparison_sha256=sha(general),
        plot_manifest_sha256=sha(HERE/'plot_manifest.json'),auditor_sha256=sha(__file__),
        main_comparison=first,generalization=second,compatible_plot_grid_passed=True,
        cost_source_arithmetic_passed=True,all_passed=True)
    target=HERE/'independent_audit.json'
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if target.exists():assert target.read_text('utf-8')==text
    else:target.write_text(text,encoding='utf-8')
    print(json.dumps(dict(all_passed=True,main_rows=first['rows_verified'],main_files=first['files_verified'],
                         generalization_rows=second['rows_verified'],cost_min=min(sums),cost_max=max(sums))))


if __name__=='__main__':main()
