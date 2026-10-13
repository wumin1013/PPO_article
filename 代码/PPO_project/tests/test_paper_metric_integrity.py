"""Historical table/plot metrics must preserve evidence, including missing data."""
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "trajectory_autoresearch"))
spec = importlib.util.spec_from_file_location("paper_sync_metric_integrity", ROOT / "trajectory_autoresearch/paper_sync.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


@pytest.fixture
def rollout(tmp_path, monkeypatch):
    """Use an actual temporary CSV while avoiding any manuscript outputs."""
    paper.BEST_ROLLOUT_STATS_CACHE.clear()
    csv_path = tmp_path / "rollout.csv"
    monkeypatch.setattr(paper, "_rollout_path_csv", lambda variant, name: csv_path)
    monkeypatch.setattr(paper, "_reference_geometry", lambda *args: {
        "center": np.array([[0., 0.], [10., 0.]]), "closed": False, "half_epsilon": 1.25,
    })

    def make(velocity=(0., 0., 1.), jerk=None, *, uses_kcm=True, dt=.1, limit=50., steps=None):
        count = len(velocity)
        rows = [dict(x=float(i), y=0., velocity=v, step=i if steps is None else steps[i])
                for i, v in enumerate(velocity)]
        if jerk is not None:
            for row, value in zip(rows, jerk):
                row["jerk"] = value
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["x", "y", "velocity", "step"])
            writer.writeheader()
            writer.writerows(rows)
        monkeypatch.setattr(paper, "_variant_dt_seconds", lambda variant: dt)
        monkeypatch.setattr(paper, "_variant_linear_jerk_limit", lambda variant: limit)
        variant = dict(name="full_method_snapshot", config_path="temporary-config",
                       ablation_audit={"enable_kcm": uses_kcm},
                       rollouts_summary={"paths": {"square": {
                           "steps": max(0, count-1), "done_reason": "success", "progress": 1.,
                       }}})
        return variant

    yield make
    paper.BEST_ROLLOUT_STATS_CACHE.clear()


@pytest.mark.parametrize("uses_kcm", [False, True])
def test_velocity_reconstruction_preserves_overlimit_peak(rollout, uses_kcm):
    variant = rollout(uses_kcm=uses_kcm)
    series = paper._best_rollout_time_series(variant, "square")
    # a=[0,0,10] mm/s² and j=[0,0,100] mm/s³ at dt=0.1 s.
    np.testing.assert_allclose(series["jerk"], [0., 0., 100.])
    assert series["jerk_source"] == "velocity_second_difference"
    assert "Unclipped" in series["jerk_source_note"]
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] == pytest.approx(100.)
    assert stats["linear_jerk_overlimit"] == pytest.approx(1.)
    assert stats["uses_kcm"] is uses_kcm


@pytest.mark.parametrize("uses_kcm", [False, True])
def test_stored_signed_jerk_is_not_capped_or_replaced(rollout, uses_kcm):
    variant = rollout(jerk=[0., -150., 75.], uses_kcm=uses_kcm)
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] == 150.
    assert stats["linear_jerk_overlimit"] == 2.
    assert stats["jerk_source"] == "best_rollout_csv"
    assert "producer determines" in stats["jerk_source_note"]


@pytest.mark.parametrize("velocity", [(), (0.,), (0., 1.)])
def test_insufficient_samples_do_not_claim_kcm_compliance(rollout, velocity):
    variant = rollout(velocity=velocity)
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] is None
    assert stats["linear_jerk_overlimit"] is None


@pytest.mark.parametrize("dt", [None, 0., -1.])
def test_unknown_or_invalid_period_does_not_create_derivatives(rollout, dt):
    variant = rollout(dt=dt)
    series = paper._best_rollout_time_series(variant, "square")
    assert series["jerk_source"] == "missing"
    assert series["jerk"].size == 0
    assert paper._best_rollout_stats(variant, "square")["linear_jerk_overlimit"] is None


@pytest.mark.parametrize("steps", [[0, 2, 4], [0, 0, 2], [0, "bad", 2]])
def test_irregular_steps_do_not_assume_one_period_per_row(rollout, steps):
    variant = rollout(steps=steps)
    assert paper._best_rollout_time_series(variant, "square")["jerk_source"] == "missing"


@pytest.mark.parametrize("column", ["velocity", "jerk"])
def test_invalid_csv_motion_values_are_not_replaced_with_zero(rollout, column):
    variant = rollout(velocity=[0., "", 1.] if column == "velocity" else [0., 0., 1.],
                      jerk=[0., "NaN", 0.] if column == "jerk" else None)
    loaded = paper._load_rollout_series(variant, "square")
    assert np.isnan(loaded[column][1])
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] is None
    assert stats["linear_jerk_overlimit"] is None


def test_missing_trace_is_unknown_even_when_kcm_enabled():
    variant = {"ablation_audit": {"enable_kcm": True}}
    assert paper._display_jerk_overlimit(variant, {}, "square") is None


@pytest.mark.parametrize("limit", [0., -1., None])
def test_relative_exceedance_requires_positive_limit(rollout, limit):
    variant = rollout(jerk=[0., 150., 0.], limit=limit)
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] == 150.
    assert stats["linear_jerk_overlimit"] is None
    trace = {"candidates": [{"path_name": "square", "max_abs_linear_jerk": 150., "linear_jerk_limit": limit}]}
    assert paper._display_jerk_overlimit(variant, trace, "square") is None


def test_measured_zero_jerk_remains_a_valid_zero(rollout):
    variant = rollout(jerk=[0., 0., 0.])
    stats = paper._best_rollout_stats(variant, "square")
    assert stats["max_abs_linear_jerk"] == 0.
    assert stats["linear_jerk_overlimit"] == 0.


def test_behavior_plot_does_not_hide_real_overlimit_peak(rollout, monkeypatch, tmp_path):
    variant = rollout(jerk=[0., 150., 0.])
    captured = {}

    def inspect_figure(fig, *args, **kwargs):
        axis = fig.axes[2]
        captured["ylim"] = axis.get_ylim()
        captured["lines"] = [(line.get_label(), line.get_ydata()) for line in axis.lines]

    monkeypatch.setattr(paper, "PLOT_DPI", 40)
    monkeypatch.setattr(paper, "KCM_FIG", tmp_path / "not_written.png")
    monkeypatch.setattr(paper, "_save_figure_outputs", inspect_figure)
    paper._build_kcm_figure(variant)
    assert captured["ylim"][1] > 3.
    assert any("stored jerk" in label and np.max(values) == 3. for label, values in captured["lines"])
    assert not (tmp_path / "not_written.png").exists()


def test_jerk_comparison_refuses_missing_measurement(rollout, monkeypatch):
    variant = rollout(velocity=[0.])
    placeholders = []
    monkeypatch.setattr(paper, "_placeholder_png", lambda *args: placeholders.append(args))
    assert paper._build_jerk_constraint_figure(variant, variant) == {}
    assert placeholders and "Unavailable" in placeholders[0][1]
