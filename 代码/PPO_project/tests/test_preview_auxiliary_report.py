"""Scientific summary guards without executing a policy or reading geometry."""
import importlib.util
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('preview_auxiliary_report', TOOLS / 'preview_auxiliary_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def record(seed, success, time, peak, completed=True, violation=False):
    return dict(seed=seed, metrics=dict(success=success, path_completed=completed,
        elapsed_time_s=time, contour_error_max_mm=peak, terminal_feed_mm_s=5.,
        terminal_rest_checked=False, failure_reasons=[] if success else ['constraint_jerk'],
        constraints={'jerk': {'satisfied': not violation}}))


def test_failed_fast_run_is_not_counted_as_successful_time():
    summary = report.summarize([record(11, True, 10., .5),
                                record(22, True, 12., .6),
                                record(33, False, 1., 2., violation=True)])
    assert (summary['n'], summary['success'], summary['failures']) == (3, 2, 1)
    assert summary['completed'] == 3
    assert summary['dynamic_violation_runs'] == 1
    assert summary['success_time_mean_s'] == 11.
    assert summary['success_time_sd_s'] == pytest.approx(2 ** .5)
    assert summary['all_runs_peak_error_max_mm'] == 2.
    assert summary['successful_seeds'] == [11, 22]
    assert summary['failure_reason_counts'] == {'constraint_jerk': 1}


def test_no_feasible_run_has_no_time_statistic():
    summary = report.summarize([record(None, False, 1., .2, violation=True)])
    assert summary['success_time_n'] == 0
    assert summary['success_time_mean_s'] is None
    assert summary['success_time_sd_s'] is None
    assert report.time_cell(summary) == '--'


def test_changed_source_is_rejected_before_registration(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text('0,0\n', encoding='utf-8')
    original_hash = report.gate.sha(path)
    path.write_text('1,1\n', encoding='utf-8')
    sources = {}
    with pytest.raises(ValueError, match='Artifact hash mismatch'):
        report.checked_file(path, original_hash, sources)
    assert not sources
