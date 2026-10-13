"""Offline two-step reports must not enforce constraints or invent measurements."""
import csv
import importlib.util
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("two_step_report_integrity", ROOT / "PPO_project/tools/sync_two_step_to_paper.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_trace(tmp_path, rows):
    path = tmp_path / "trace.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("method", ["J-NNC", "Traditional two-step", "NNC baseline"])
def test_measured_overlimit_values_are_preserved(tmp_path, method):
    rows = [dict(velocity=2., acceleration=-20., jerk=-150.) for _ in range(3)]
    stats = report.trace_dynamic_utilization_stats(method, "square", max_vel=1., max_acc=10., max_jerk=50., trace_path=write_trace(tmp_path, rows))
    assert stats["mean_feedrate_utilization"] == 2.
    assert stats["p95_acceleration_utilization"] == 2.
    assert stats["p95_linear_jerk_utilization"] == 3.
    assert stats["jerk_reach_rate_80_active"] == 0.
    assert stats["jerk_source"] == "stored"


def test_raw_derivatives_use_timestamps_without_limit_projection():
    rows = [dict(time=t, velocity=v) for t, v in zip([0., .1, .2, .3], [0., 0., 1., 1.])]
    motion = report._trace_kinematics(rows)
    np.testing.assert_allclose(motion["velocity"], [0., 0., 1., 1.])
    np.testing.assert_allclose(motion["acceleration"], [np.nan, 0., 10., 0.], equal_nan=True)
    np.testing.assert_allclose(motion["jerk"], [np.nan, np.nan, 100., -100.], equal_nan=True)
    assert motion["jerk_source"] == "velocity_second_difference"


def test_raw_derivative_statistics_exclude_only_unknown_boundary(tmp_path):
    rows = [dict(step=i, velocity=v) for i, v in enumerate([0., 0., 1., 1.])]
    stats = report.trace_dynamic_utilization_stats("J-NNC", "square", max_vel=1., max_acc=5., max_jerk=50., dt=.1, trace_path=write_trace(tmp_path, rows))
    assert stats["p95_linear_jerk_utilization"] == pytest.approx(2.)
    assert stats["sample_period_s"] == .1


@pytest.mark.parametrize("rows,dt", [
    ([dict(step=i, velocity=v) for i, v in enumerate([0., 0., 1.])], None),
    ([dict(step=i, velocity=v) for i, v in enumerate([0., 0., 1.])], 0.),
    ([dict(step=i, velocity=v) for i, v in zip([0, 2, 4], [0., 0., 1.])], .1),
    ([dict(time=t, velocity=v) for t, v in zip([0., 0., .1], [0., 0., 1.])], .1),
    ([dict(time=t, velocity=v) for t, v in zip([0., .1, .3], [0., 0., 1.])], .1),
    ([dict(time=t, velocity=v) for t, v in zip([0., "", .2], [0., 0., 1.])], .1),
    ([dict(step=i, velocity=v) for i, v in enumerate([0., "", 1.])], .1),
])
def test_missing_or_invalid_sampling_does_not_invent_derivatives(rows, dt):
    motion = report._trace_kinematics(rows, dt=dt)
    assert motion["jerk_source"] == "missing"
    assert np.all(np.isnan(motion["jerk"]))


@pytest.mark.parametrize("field,metric", [("velocity", "mean_feedrate_utilization"), ("acceleration", "p95_acceleration_utilization"), ("jerk", "p95_linear_jerk_utilization")])
def test_partially_missing_measurements_do_not_become_compliant_summary(tmp_path, field, metric):
    rows = [dict(velocity=0., acceleration=0., jerk=0.) for _ in range(3)]
    rows[1][field] = ""
    stats = report.trace_dynamic_utilization_stats("J-NNC", "square", max_vel=1., max_acc=1., max_jerk=1., trace_path=write_trace(tmp_path, rows))
    assert metric not in stats


def test_all_missing_motion_has_no_zero_statistics(tmp_path):
    stats = report.trace_dynamic_utilization_stats("J-NNC", "square", max_vel=1., max_acc=1., max_jerk=1., trace_path=write_trace(tmp_path, [dict(x=0., y=0.)]))
    assert not any(key in stats for key, _ in report.EFFICIENCY_METRICS)
    assert stats["jerk_source"] == "missing"


@pytest.mark.parametrize("limit", [0., -1., float("nan")])
def test_invalid_limit_has_no_utilization(tmp_path, limit):
    stats = report.trace_dynamic_utilization_stats("J-NNC", "square", max_vel=limit, max_acc=limit, max_jerk=limit, trace_path=write_trace(tmp_path, [dict(velocity=0., acceleration=0., jerk=0.)]))
    assert not any(key in stats for key, _ in report.EFFICIENCY_METRICS)


def test_missing_historical_progress_is_not_assumed_complete(monkeypatch):
    monkeypatch.setattr(report, "parse_existing_main_results", lambda: {"square": {"J-NNC": {"termination_time_s": 1.}}})
    monkeypatch.setattr(report, "trace_dynamic_utilization_stats", lambda *args, **kwargs: {})
    result = report.merge_metrics_for_tables({"square": {"original_path_length_mm": 10.}})
    assert "effective_path_speed_mm_s" not in result["square"]["J-NNC"]
    assert report.parse_latex_number(0.) == 0.


def test_plot_preserves_overlimit_values_and_expands_axes(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(report, "PAPER_FIGURES_DIR", tmp_path)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda fig, *args, **kwargs: captured.append(fig))
    reference = np.asarray([[0., 0.], [100., 0.], [100., 100.], [0., 100.], [0., 0.]])
    rows = [dict(x=i * 10., y=0., progress=i / 40., velocity=2., acceleration=-20., jerk=150.) for i in range(4)]
    report.save_two_step_comparison_figure(square_reference=reference, square_trace_rows=rows, jnnc_trace_rows=rows, half_epsilon=.1, constraints={"MAX_VEL": 1., "MAX_ACC": 10., "MAX_JERK": 50.})
    fig = captured[0]
    assert fig.axes[3].get_ylim()[0] < -2.
    assert fig.axes[4].get_ylim()[1] > 3.
    series = next(line for line in fig.axes[4].lines if line.get_label() == "J-NNC")
    np.testing.assert_allclose(series.get_ydata(), 3.)
    assert (tmp_path / "two_step_comparison_sources.json").exists()
