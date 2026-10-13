"""Failure-aware reporting must not turn selective success into a speed gain."""
import importlib.util
from pathlib import Path

MODULE=Path(__file__).resolve().parents[1]/'tools/preview_event_report.py'
spec=importlib.util.spec_from_file_location('event_saved_report',MODULE)
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(method,seed,path,time,success=True):
    return dict(variant=method,seed=seed,path_id=path,elapsed_time_s=time,success=success,
        failure_reasons=[] if success else ['timeout'],contour_error_max_mm=4. if not success else .4,
        contour_error_rmse_mm=.2,inference_median_us=50.)


def test_failed_fast_path_excludes_entire_suite_seed():
    rows=[row('selected_fixed',s,p,10) for s in (11,22) for p in ('a','b')]
    rows += [row('selected_hold',11,'a',5),row('selected_hold',11,'b',1,False),
             row('selected_hold',22,'a',9),row('selected_hold',22,'b',11)]
    result=report.paired(rows,['selected_fixed','selected_hold'],['a','b'],[11,22])
    suite=next(r for r in result if r['path_id']=='all_paths')
    assert suite['improvement_pct']['n']==1
    assert suite['improvement_pct']['mean']==0
    assert suite['pairs'][0]['seed']==22
    assert suite['excluded']==[dict(seed=11,reason='not_jointly_successful')]
    assert suite['improvement_pct']['ci95'] is None


def test_error_and_failure_summary_preserves_failed_trajectory():
    result=report.summarize([row('m',11,'a',10),row('m',22,'a',1,False)])
    assert result['successful_time_s']['mean']==10
    assert result['all_time_s']['mean']==5.5
    assert result['all_peak_error_mm']['mean']==2.2
    assert result['failures']==1 and result['failure_reasons']=={'timeout':1}
    assert result['successful_seeds']==[11]


def test_missing_runs_are_not_zero_or_failure_rate_denominator():
    result=report.summarize([])
    assert result['n']==0 and result['successful_time_s']['mean'] is None
    pairs=report.paired([],['selected_fixed','selected_hold'],['a'],[11])
    assert all(r['improvement_pct']['n']==0 for r in pairs)
    assert pairs[-1]['excluded']==[dict(seed=11,reason='missing_result')]


def test_paired_tex_keeps_success_set_and_unestimable_interval():
    rows=[row('selected_fixed',s,'a',10) for s in (11,22)]
    rows += [row('selected_hold',11,'a',1,False),row('selected_hold',22,'a',9)]
    pairs=report.paired(rows,['selected_fixed','selected_hold'],['a'],[11,22])
    tex=report.tex_paired_table(dict(stage='formal',split='test',regime='nominal'),pairs)
    assert 'Held policy preview & 1 & 22 & 10.000 & --' in tex
    assert '[0.000, 0.000]' not in tex
