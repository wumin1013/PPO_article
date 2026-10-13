"""Physical replay equality must not be confused with variable CPU timings."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import validate_frozen_replay as replay


def test_only_measured_cpu_time_is_excluded_from_result_comparison():
    original=dict(elapsed_time_s=2.,constraints={'jerk':{'peak':3.}},inference_median_us=50.)
    fresh={**original,'inference_median_us':80.}
    assert replay.compare_results(original,fresh)['non_timing_metrics_equal']
    fresh['elapsed_time_s']=2.001
    assert replay.compare_results(original,fresh)['different_fields']==['elapsed_time_s']


def test_missing_null_physical_field_is_not_equal():
    assert replay.compare_results({'terminal_rest_satisfied':None},{})['different_fields']==['terminal_rest_satisfied']


def test_trace_comparison_checks_all_physical_columns_and_nans(tmp_path):
    columns=np.array(['timestamp_s','x_mm','control_status__direct_mapping_us'])
    original=np.array([[0.,0.,np.nan],[.001,.005,5.]])
    fresh=original.copy();fresh[1,2]=20.
    a,b=tmp_path/'original.npz',tmp_path/'replay.npz'
    np.savez(a,columns=columns,data=original);np.savez(b,columns=columns,data=fresh)
    assert replay.compare_traces(a,b)['non_timing_trace_exactly_equal']
    fresh[1,1]+=.000001;np.savez(b,columns=columns,data=fresh)
    assert list(replay.compare_traces(a,b)['different_columns'])==['x_mm']
