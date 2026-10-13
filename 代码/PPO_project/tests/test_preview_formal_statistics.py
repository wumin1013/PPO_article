"""Synthetic-only formal statistics tests; no geometry, models or saved results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("preview_formal_statistics", TOOLS / "preview_formal_statistics.py")
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)
SEEDS = [11, 22, 33, 44, 55]
PATHS = {f"synthetic_{family}_{index}": family
         for family in ("mixed_lengths", "dense_corners", "line_curve") for index in (1, 2)}


def synthetic_rows():
    return [
        dict(seed=seed, method=method, path_id=path_id, family=family,
             success=True, elapsed_time_s=10.0 if method == "fixed3" else 8.0,
             contour_error_max_mm=0.04, contour_error_rmse_mm=0.01,
             geometry_feasible=True, path_completed=True, full_rest_satisfied=True,
             failure_reasons=[],
             constraints={"axis_X_jerk": dict(satisfied=True, peak=2.0, limit=3.0)},
             terminal_tail_s=0.1, inference_wall_s=0.2, policy_decisions=20,
             unrelated_metadata={"source": "synthetic fixture"})
        for seed in SEEDS for method in stats.METHODS for path_id, family in PATHS.items()
    ]


def mark_failed(row, time=0.01, reasons=None):
    row.update(success=False, elapsed_time_s=time, full_rest_satisfied=False,
               failure_reasons=reasons if reasons is not None else ["terminal_not_at_rest"])


def summarize(rows):
    return stats.summarize(rows, SEEDS, PATHS)


def test_complete_balanced_design_and_deterministic_cluster_ci():
    rows = synthetic_rows()
    before = deepcopy(rows)
    result = summarize(rows)
    assert rows == before
    assert result == summarize(list(reversed(rows)))
    assert result["n_rows"] == 60
    assert result["overall"]["fixed3"]["n_runs"] == 30
    assert result["overall"]["adaptive"]["success_time_s"]["mean"] == 8.0
    assert result["overall"]["adaptive"]["success_rate_ci95"] == [1.0, 1.0]
    paired = result["overall"]["paired"]
    assert paired["time_reduction_fraction"] == pytest.approx(0.2)
    assert paired["time_reduction_ci95"] == pytest.approx([0.2, 0.2])
    assert paired["n_common_success"] == 30
    assert paired["time_reduction_bootstrap"]["n_seed_clusters"] == 5
    assert paired["time_reduction_bootstrap"]["bootstrap_total_resamples"] == 10000
    assert result["statistical_protocol"]["bootstrap_seed"] == stats.BOOTSTRAP_SEED
    assert "only 5 independent" in result["statistical_protocol"]["small_sample_warning"]
    for family in stats.FAMILIES:
        assert result["by_family"][family]["fixed3"]["n_runs"] == 10
    for seed in SEEDS:
        assert result["by_seed"][str(seed)]["fixed3"]["n_runs"] == 6
        assert result["by_seed"][str(seed)]["fixed3"]["success_rate_ci95"] is None
    for path_id in PATHS:
        assert result["by_path"][path_id]["adaptive"]["n_runs"] == 5
    json.dumps(result, allow_nan=False)


def test_fast_failed_run_stays_in_denominator_but_not_successful_time():
    rows = synthetic_rows()
    failed = next(row for row in rows if row["method"] == "adaptive")
    mark_failed(failed, reasons=["axis_X_jerk", "axis_X_jerk", "terminal_not_at_rest"])
    failed["constraints"]["axis_X_jerk"]["satisfied"] = False
    result = summarize(rows)
    adaptive = result["overall"]["adaptive"]
    assert adaptive["n_success"] == 29
    assert adaptive["n_failure"] == 1
    assert adaptive["success_rate"] == pytest.approx(29 / 30)
    assert adaptive["success_time_s"]["n"] == 29
    assert adaptive["success_time_s"]["mean"] == 8.0
    assert adaptive["failures_by_reason"] == {"axis_X_jerk": 1, "terminal_not_at_rest": 1}
    assert adaptive["constraints"]["axis_X_jerk"]["n_violated"] == 1
    paired = result["overall"]["paired"]
    assert paired["n_common_success"] == 29
    assert paired["adaptive_failure_rate"] == pytest.approx(1 / 30)
    assert paired["n_fixed3_only_success"] == 1
    assert paired["time_reduction_fraction"] == pytest.approx(0.2)


def test_paired_effect_uses_identical_seed_path_intersection_and_ratio_of_totals():
    rows = synthetic_rows()
    path_ids = list(PATHS)
    for row in rows:
        if row["path_id"] == path_ids[0]:
            row["elapsed_time_s"] = 100.0 if row["method"] == "fixed3" else 50.0
        if row["method"] == "adaptive" and row["path_id"] == path_ids[1]:
            mark_failed(row)
        if row["method"] == "fixed3" and row["path_id"] == path_ids[2]:
            mark_failed(row)
        if row["path_id"] == path_ids[3]:
            mark_failed(row)
    paired = summarize(rows)["overall"]["paired"]
    assert paired["n_common_success"] == 15
    assert paired["n_both_failed"] == 5
    assert paired["n_fixed3_only_success"] == 5
    assert paired["n_adaptive_only_success"] == 5
    assert paired["time_reduction_fraction"] == pytest.approx(1 - (50 + 8 + 8) / (100 + 10 + 10))
    assert {pair["path_id"] for pair in paired["common_success_pairs"]} == {
        path_ids[0], path_ids[4], path_ids[5]}
    assert paired["both_failed_rate"] == pytest.approx(1 / 6)


def test_all_failures_report_null_time_and_ci_and_keep_unclassified_failure():
    rows = synthetic_rows()
    for row in rows:
        mark_failed(row, time=float("nan"), reasons=[])
    result = summarize(rows)
    assert result["overall"]["adaptive"]["success_time_s"]["mean"] is None
    assert result["overall"]["adaptive"]["failure_rate"] == 1.0
    assert result["overall"]["adaptive"]["failures_by_reason"] == {"unspecified_failure": 30}
    paired = result["overall"]["paired"]
    assert paired["time_reduction_fraction"] is None
    assert paired["time_reduction_ci95"] is None
    assert paired["fixed3_common_time_sum_s"] is None
    assert paired["n_both_failed"] == 30
    assert paired["time_reduction_bootstrap"]["bootstrap_valid_resamples"] == 0
    assert paired["time_reduction_bootstrap"]["bootstrap_undefined_resamples"] == 10000
    json.dumps(result, allow_nan=False)


def test_no_joint_success_is_null_even_when_each_method_succeeds():
    rows = synthetic_rows()
    for row in rows:
        if (row["seed"] in (11, 22)) == (row["method"] == "fixed3"):
            mark_failed(row)
    result = summarize(rows)["overall"]
    assert result["fixed3"]["n_success"] == 18
    assert result["adaptive"]["n_success"] == 12
    assert result["paired"]["n_common_success"] == 0
    assert result["paired"]["time_reduction_ci95"] is None


def test_undefined_bootstrap_replicates_are_counted():
    rows = synthetic_rows()
    for row in rows:
        if row["seed"] not in (11, 22):
            mark_failed(row)
    effect = summarize(rows)["overall"]["paired"]["time_reduction_bootstrap"]
    assert effect["n_seed_clusters"] == 5
    assert effect["n_contributing_seed_clusters"] == 2
    assert 0 < effect["bootstrap_valid_resamples"] < effect["bootstrap_total_resamples"]
    assert effect["bootstrap_valid_resamples"] + effect["bootstrap_undefined_resamples"] == 10000
    assert effect["ci95"] == pytest.approx([0.2, 0.2])


def test_ci_resamples_whole_seeds_and_preserves_path_dependence():
    rows = synthetic_rows()
    for row in rows:
        if row["method"] == "adaptive":
            row["elapsed_time_s"] = 10.0 * (1.0 - row["seed"] / 100.0)
    result = summarize(rows)
    overall = result["overall"]["paired"]
    assert overall["time_reduction_fraction"] == pytest.approx(0.33)
    # Each path is identical within a seed: replicating it six times must not
    # narrow the CI, as would incorrectly treating the 30 paths independently.
    for path_id in PATHS:
        assert result["by_path"][path_id]["paired"]["time_reduction_ci95"] == pytest.approx(
            overall["time_reduction_ci95"])
    assert overall["time_reduction_ci95"][0] < 0.25
    assert overall["time_reduction_ci95"][1] > 0.40


def test_success_rate_ci_uses_five_seed_rates_not_thirty_bernoulli_runs():
    rows = synthetic_rows()
    for row in rows:
        if row["method"] == "adaptive" and row["seed"] in (11, 22):
            mark_failed(row)
    result = summarize(rows)
    assert result["overall"]["adaptive"]["success_rate"] == 0.6
    for path_id in PATHS:
        assert result["by_path"][path_id]["adaptive"]["success_rate_ci95"] == (
            result["overall"]["adaptive"]["success_rate_ci95"])


def test_inference_is_weighted_by_decisions_and_preserves_arbitrary_axis_constraints():
    rows = synthetic_rows()
    for row in rows:
        row["inference_wall_s"] = 0.1 if row["seed"] == 11 else 0.9
        row["policy_decisions"] = 1 if row["seed"] == 11 else 3
        row["constraints"]["motor_Q_snap"] = dict(satisfied=True, peak=4.0, limit=8.0)
    method = summarize(rows)["overall"]["adaptive"]
    assert method["inference_per_decision_s"]["mean"] == pytest.approx((0.1 + 4 * 0.9) / 13)
    assert method["constraints"]["motor_Q_snap"]["peak"]["max"] == 4.0
    assert method["constraints"]["motor_Q_snap"]["limit"]["min"] == 8.0


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), -float("inf"), None, -1.0, 0.0])
def test_invalid_successful_elapsed_time_is_rejected(invalid_time):
    rows = synthetic_rows()
    rows[0]["elapsed_time_s"] = invalid_time
    with pytest.raises(ValueError, match="finite and positive"):
        summarize(rows)


def test_incomplete_and_duplicate_records_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="Incomplete formal design"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate formal record"):
        summarize(rows[:-1] + [deepcopy(rows[0])])
    with pytest.raises(ValueError, match="Duplicate formal record"):
        summarize(rows + [deepcopy(rows[0])])


@pytest.mark.parametrize("field,value,pattern", [
    ("seed", 66, "outside"), ("method", "other", "outside"),
    ("path_id", "unlisted_synthetic", "outside"), ("family", "other", "Family mismatch"),
    ("success", "false", "boolean"), ("geometry_feasible", 1, "boolean"),
    ("failure_reasons", "failed", "list of nonempty strings"),
    ("constraints", {"axis": {"satisfied": False}}, "Malformed constraint"),
])
def test_invalid_audited_fields_are_rejected(field, value, pattern):
    rows = synthetic_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match=pattern):
        summarize(rows)


def test_declared_design_must_be_five_seeds_six_paths_and_two_per_family():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="five formal seeds"):
        stats.summarize(rows, [11], PATHS)
    with pytest.raises(ValueError, match="five formal seeds"):
        stats.summarize(rows, [11, 22, 33, 44, 44], PATHS)
    with pytest.raises(ValueError, match="exactly six"):
        stats.summarize(rows, SEEDS, {"synthetic": "mixed_lengths"})
    with pytest.raises(ValueError, match="two paths"):
        stats.summarize(rows, SEEDS, {path_id: "mixed_lengths" for path_id in PATHS})


def test_missing_required_field_is_rejected():
    rows = synthetic_rows()
    del rows[0]["inference_wall_s"]
    with pytest.raises(ValueError, match="required audited fields"):
        summarize(rows)
