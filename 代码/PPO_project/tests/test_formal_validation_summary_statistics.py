"""Synthetic-only checks for selected development-validation statistics."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import formal_validation_summary as report


PATHS = ("synthetic_mixed", "synthetic_dense", "synthetic_curve")


def records():
    return [dict(seed=seed, method=method, path_id=path_id,
                 training_decisions=0 if method == "fixed3" else 256,
                 success=True, elapsed_time_s=10.0 if method == "fixed3" else 8.0,
                 failure_reasons=[])
            for seed in report.SEEDS for method in report.METHODS for path_id in PATHS]


def fail(row, elapsed=0.001, reason="ordered_path_incomplete"):
    row.update(success=False, elapsed_time_s=elapsed, failure_reasons=[reason])


def candidate(step, successes, total):
    return dict(step=step, successes=successes, complete_validation_time_s=total)


def test_complete_thirty_rows_are_descriptive_and_input_order_independent():
    rows = records()
    original = deepcopy(rows)
    result = report.summarize(rows, PATHS)
    assert rows == original
    assert result == report.summarize(list(reversed(rows)), PATHS)
    assert result["data_role"] == report.DATA_ROLE
    assert "not independent test" in result["data_role"]
    assert "Five independent training seeds" in result["statistical_units"]
    assert "checkpoint selection used these paths and times" in result["uncertainty"]
    assert len(result["by_seed"]) == 5
    assert len(result["paired_rows"]) == 15
    for method in report.METHODS:
        assert result["methods"][method]["runs"] == 15
        assert result["methods"][method]["successes"] == 15
        assert result["methods"][method]["failures"] == 0
    for seed in result["by_seed"]:
        assert seed["paths_per_method"] == 3
        assert seed["fixed3_selected_step"] == 0
        assert seed["adaptive_selected_step"] == 256
        assert seed["fixed3_successes"] == seed["adaptive_successes"] == 3
    assert result["common_success"]["pairs"] == 15
    assert result["common_success"]["fixed3_total_time_s"] == 150.0
    assert result["common_success"]["adaptive_total_time_s"] == 120.0
    assert result["common_success"]["time_gain_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "unknown_identity"])
def test_selected_design_requires_exactly_thirty_unique_registered_rows(fault):
    rows = records()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif fault == "extra":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["path_id"] = "unregistered_synthetic_path"
    with pytest.raises(ValueError, match="all thirty unique"):
        report.summarize(rows, PATHS)


def test_failed_fast_time_is_excluded_from_common_times_and_kept_in_denominator():
    rows = records()
    failed = next(row for row in rows if row["seed"] == 11 and row["method"] == "adaptive")
    fail(failed)
    result = report.summarize(rows, PATHS)
    adaptive = result["methods"]["adaptive"]
    assert adaptive["runs"] == 15
    assert adaptive["successes"] == 14 and adaptive["failures"] == 1
    assert adaptive["success_fraction"] == pytest.approx(14 / 15)
    assert adaptive["failure_fraction"] == pytest.approx(1 / 15)
    assert adaptive["all_three_success_seeds"] == [22, 33, 44, 55]
    assert adaptive["failure_reasons"] == {"ordered_path_incomplete": 1}
    seed = next(item for item in result["by_seed"] if item["seed"] == 11)
    assert seed["adaptive_successes"] == 2 and seed["fixed3_successes"] == 3
    assert seed["paired_count"] == 2
    assert seed["fixed3_common_time_s"] == 20.0
    assert seed["adaptive_common_time_s"] == 16.0
    assert result["common_success"]["fixed3_total_time_s"] == 140.0
    assert result["common_success"]["adaptive_total_time_s"] == 112.0
    failed_pair = next(pair for pair in result["paired_rows"]
                       if pair["seed"] == 11 and pair["path_id"] == failed["path_id"])
    assert failed_pair["common_success"] is False
    assert failed_pair["time_gain_fraction"] is None


def test_common_success_pairs_require_both_identical_seed_and_path():
    rows = records()
    # One-sided successes must not be joined to another seed or another path.
    failed_keys = {(11, "fixed3", PATHS[0]), (22, "adaptive", PATHS[0]),
                   (11, "adaptive", PATHS[1]), (11, "fixed3", PATHS[2]),
                   (11, "adaptive", PATHS[2])}
    for row in rows:
        if (row["seed"], row["method"], row["path_id"]) in failed_keys:
            fail(row)
    result = report.summarize(rows, PATHS)
    common = {(pair["seed"], pair["path_id"]) for pair in result["paired_rows"] if pair["common_success"]}
    assert common == {(seed, path_id) for seed in report.SEEDS for path_id in PATHS
                      if seed != 11 and (seed, path_id) != (22, PATHS[0])}
    assert len(common) == result["common_success"]["pairs"] == 11
    assert result["common_success"]["contributing_seeds"] == [22, 33, 44, 55]
    seed11 = next(item for item in result["by_seed"] if item["seed"] == 11)
    assert seed11["common_success_paths"] == []
    assert seed11["fixed3_common_time_s"] is None
    assert seed11["adaptive_common_time_s"] is None
    assert seed11["common_time_gain_fraction"] is None


def test_empty_intersection_is_none_even_when_both_methods_have_successes():
    rows = records()
    for row in rows:
        if (row["seed"] in (11, 22)) == (row["method"] == "fixed3"):
            fail(row)
    result = report.summarize(rows, PATHS)
    assert result["methods"]["fixed3"]["successes"] == 9
    assert result["methods"]["adaptive"]["successes"] == 6
    common = result["common_success"]
    assert common["pairs"] == 0 and common["candidate_pairs"] == 15
    assert common["contributing_seeds"] == []
    for key in ("fixed3_total_time_s", "adaptive_total_time_s", "time_gain_fraction"):
        assert common[key] is None
    assert all(pair["time_gain_fraction"] is None for pair in result["paired_rows"])


def test_gain_uses_ratio_of_common_time_totals_not_mean_of_path_percentages():
    rows = records()
    fixed, adaptive = dict(zip(PATHS, (100.0, 10.0, 2.0))), dict(zip(PATHS, (90.0, 5.0, 1.0)))
    for row in rows:
        row["elapsed_time_s"] = (fixed if row["method"] == "fixed3" else adaptive)[row["path_id"]]
    result = report.summarize(rows, PATHS)
    expected = 1 - 96.0 / 112.0
    assert result["common_success"]["time_gain_fraction"] == pytest.approx(expected)
    assert result["common_success"]["time_gain_fraction"] != pytest.approx((0.1 + 0.5 + 0.5) / 3)
    assert all(seed["common_time_gain_fraction"] == pytest.approx(expected) for seed in result["by_seed"])


def test_selection_first_maximizes_success_then_uses_time_only_for_three_successes():
    partial_early = candidate(0, 2, 1000.0)
    partial_fast = candidate(256, 2, 0.001)
    lower_success = candidate(0, 1, 0.0)
    complete_slow = candidate(512, 3, 100.0)
    complete_fast = candidate(1024, 3, 50.0)
    assert min([partial_fast, partial_early, lower_success], key=report.selection_key) is partial_early
    assert min([partial_early, complete_slow], key=report.selection_key) is complete_slow
    assert min([complete_slow, complete_fast], key=report.selection_key) is complete_fast


def test_selection_ties_choose_earliest_step_including_zero_with_unavailable_time():
    complete0, complete256 = candidate(0, 3, 50.0), candidate(256, 3, 50.0)
    assert min([complete256, complete0], key=report.selection_key) is complete0
    failed0, failed4096 = candidate(0, 0, None), candidate(4096, 0, None)
    assert min([failed4096, failed0], key=report.selection_key) is failed0
    partial0, partial256 = candidate(0, 2, None), candidate(256, 2, None)
    assert min([partial256, partial0], key=report.selection_key) is partial0
