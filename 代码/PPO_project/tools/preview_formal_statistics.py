"""Read-only statistics for the complete, already audited formal design.

This module accepts in-memory records only. It does not import an environment,
load a policy, read holdout geometry, or write a report. Failed runs remain in
every success denominator; successful-time statistics exclude their durations.
"""

from collections import Counter
from functools import lru_cache
import math
from numbers import Integral, Real
import random
import statistics


METHODS = ("fixed3", "adaptive")
FORMAL_SEEDS = (11, 22, 33, 44, 55)
FAMILIES = ("mixed_lengths", "dense_corners", "line_curve")
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _describe(values):
    finite = [float(value) for value in values if _finite(value)]
    return {
        "n": len(finite),
        "n_missing_or_nonfinite": len(values) - len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "sd": statistics.stdev(finite) if len(finite) > 1 else None,
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


@lru_cache(maxsize=8)
def _resample_indices(n_seeds):
    """A draw selects whole seeds, including all their methods and paths."""
    generator = random.Random(BOOTSTRAP_SEED)
    return tuple(
        tuple(generator.randrange(n_seeds) for _ in range(n_seeds))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )


def _percentile(ordered, probability):
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _cluster_ratio(numerators, denominators, *, complement=False):
    """Bootstrap a ratio of totals, recording every undefined resample."""
    n_seeds = len(numerators)
    n_contributing = sum(denominator > 0 for denominator in denominators)
    total_denominator = math.fsum(denominators)
    ratio = math.fsum(numerators) / total_denominator if total_denominator > 0 else None
    estimate = 1.0 - ratio if complement and ratio is not None else ratio
    draws = _resample_indices(n_seeds)
    values = []
    for indices in draws:
        denominator = math.fsum(denominators[index] for index in indices)
        if denominator > 0:
            value = math.fsum(numerators[index] for index in indices) / denominator
            values.append(1.0 - value if complement else value)
    if n_contributing == 0:
        unavailable_reason = "no_contributing_seed_clusters"
    elif n_seeds < 2 or n_contributing < 2:
        unavailable_reason = "fewer_than_two_contributing_seed_clusters"
    else:
        unavailable_reason = None
    ci95 = None
    if values and unavailable_reason is None:
        values.sort()
        alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
        ci95 = [_percentile(values, alpha), _percentile(values, 1.0 - alpha)]
    return {
        "estimate": estimate,
        "ci95": ci95,
        "n_seed_clusters": n_seeds,
        "n_contributing_seed_clusters": n_contributing,
        "bootstrap_valid_resamples": len(values),
        "bootstrap_total_resamples": len(draws),
        "bootstrap_undefined_resamples": len(draws) - len(values),
        "ci_unavailable_reason": unavailable_reason,
        "undefined_resample_policy": (
            "Zero-denominator resamples are counted and excluded from percentile endpoints; "
            "any available interval is conditional on a nonzero denominator."
        ),
    }


def _validate(rows, expected_seeds, expected_paths):
    seeds = list(expected_seeds)
    if (any(not isinstance(seed, Integral) or isinstance(seed, bool) for seed in seeds)
            or len(seeds) != len(FORMAL_SEEDS) or set(seeds) != set(FORMAL_SEEDS)):
        raise ValueError(f"Expected the five formal seeds {FORMAL_SEEDS}, without duplicates")
    if not isinstance(expected_paths, dict) or len(expected_paths) != 6:
        raise ValueError("Expected exactly six path_id -> family entries")
    if any(not isinstance(path_id, str) or not path_id for path_id in expected_paths):
        raise ValueError("Expected nonempty string path IDs")
    if Counter(expected_paths.values()) != Counter({family: 2 for family in FAMILIES}):
        raise ValueError("Expected exactly two paths in each formal family")
    expected_keys = {(seed, method, path_id) for seed in seeds
                     for method in METHODS for path_id in expected_paths}
    seen = set()
    required = {
        "seed", "method", "path_id", "family", "success", "elapsed_time_s",
        "contour_error_max_mm", "contour_error_rmse_mm", "geometry_feasible",
        "path_completed", "full_rest_satisfied", "failure_reasons", "constraints",
        "terminal_tail_s", "inference_wall_s", "policy_decisions",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Row {index} lacks required audited fields")
        seed, method, path_id = row["seed"], row["method"], row["path_id"]
        if (not isinstance(seed, Integral) or isinstance(seed, bool)
                or method not in METHODS or path_id not in expected_paths
                or seed not in seeds):
            raise ValueError(f"Row {index} is outside the expected formal design")
        key = (seed, method, path_id)
        if key in seen:
            raise ValueError(f"Duplicate formal record: {key}")
        seen.add(key)
        if row["family"] != expected_paths[path_id]:
            raise ValueError(f"Family mismatch for {key}")
        for name in ("success", "geometry_feasible", "path_completed", "full_rest_satisfied"):
            if not isinstance(row[name], bool):
                raise ValueError(f"{name} must be a boolean for {key}")
        if row["success"] and (not _finite(row["elapsed_time_s"]) or row["elapsed_time_s"] <= 0):
            raise ValueError(f"Successful elapsed_time_s must be finite and positive for {key}")
        reasons = row["failure_reasons"]
        if not isinstance(reasons, (list, tuple)) or any(
                not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError(f"failure_reasons must be a list of nonempty strings for {key}")
        if not isinstance(row["constraints"], dict):
            raise ValueError(f"constraints must be a mapping for {key}")
        for name, constraint in row["constraints"].items():
            if (not isinstance(name, str) or not name or not isinstance(constraint, dict)
                    or not {"satisfied", "peak", "limit"}.issubset(constraint)
                    or not isinstance(constraint["satisfied"], bool)):
                raise ValueError(f"Malformed constraint {name!r} for {key}")
    missing = expected_keys - seen
    if missing:
        raise ValueError(f"Incomplete formal design: expected 60 rows; missing {len(missing)} records")
    return sorted(seeds)


def _method_summary(rows, seeds):
    successful = [row for row in rows if row["success"]]
    failures = [row for row in rows if not row["success"]]
    per_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in seeds}
    success_counts = [sum(row["success"] for row in per_seed[seed]) for seed in seeds]
    rate = _cluster_ratio(success_counts, [len(per_seed[seed]) for seed in seeds])
    duration = _describe([row["elapsed_time_s"] for row in successful])
    duration_ci = _cluster_ratio(
        [math.fsum(row["elapsed_time_s"] for row in per_seed[seed] if row["success"])
         for seed in seeds], success_counts,
    )
    duration["mean_ci95"] = duration_ci["ci95"]
    duration["bootstrap"] = duration_ci
    reason_counts = Counter()
    for row in failures:
        reason_counts.update(set(row["failure_reasons"]) or {"unspecified_failure"})
    constraints = {}
    for name in sorted({name for row in rows for name in row["constraints"]}):
        entries = [row["constraints"][name] for row in rows if name in row["constraints"]]
        constraints[name] = {
            "n_reported": len(entries),
            "n_missing": len(rows) - len(entries),
            "n_satisfied": sum(entry["satisfied"] for entry in entries),
            "n_violated": sum(not entry["satisfied"] for entry in entries),
            "peak": _describe([entry["peak"] for entry in entries]),
            "limit": _describe([entry["limit"] for entry in entries]),
        }
    valid_inference = [row for row in rows if _finite(row["inference_wall_s"])
                       and row["inference_wall_s"] >= 0
                       and _finite(row["policy_decisions"]) and row["policy_decisions"] > 0]
    inference_total = math.fsum(row["inference_wall_s"] for row in valid_inference)
    decisions_total = math.fsum(row["policy_decisions"] for row in valid_inference)
    return {
        "n_runs": len(rows), "n_seeds": len(seeds),
        "n_success": len(successful), "n_failure": len(failures),
        "success_rate": rate["estimate"], "success_rate_ci95": rate["ci95"],
        "success_rate_bootstrap": rate,
        "failure_rate": len(failures) / len(rows),
        "success_time_s": duration,
        "failures_by_reason": dict(sorted(reason_counts.items())),
        "failure_reason_counting": "Each failed run counts once per distinct reason; categories may overlap.",
        "geometry_feasible_n": sum(row["geometry_feasible"] for row in rows),
        "path_completed_n": sum(row["path_completed"] for row in rows),
        "full_rest_satisfied_n": sum(row["full_rest_satisfied"] for row in rows),
        "contour_error_max_mm": _describe([row["contour_error_max_mm"] for row in rows]),
        "contour_error_rmse_mm": _describe([row["contour_error_rmse_mm"] for row in rows]),
        "constraints": constraints,
        "terminal_tail_s": _describe([row["terminal_tail_s"] for row in rows]),
        "inference_wall_s": _describe([row["inference_wall_s"] for row in rows]),
        "policy_decisions": _describe([row["policy_decisions"] for row in rows]),
        "inference_per_decision_s": {
            "mean": inference_total / decisions_total if decisions_total else None,
            "wall_s_sum": inference_total if valid_inference else None,
            "decisions_sum": decisions_total if valid_inference else None,
            "n_valid_runs": len(valid_inference),
            "n_excluded_runs": len(rows) - len(valid_inference),
            "scope": "Sum of inference wall seconds / sum of decisions over runs with valid positive decision counts; includes failures.",
        },
    }


def _paired_summary(rows, seeds):
    lookup = {(row["seed"], row["path_id"], row["method"]): row for row in rows}
    identities = sorted({(row["seed"], row["path_id"]) for row in rows})
    common, both_failed, fixed_only, adaptive_only = [], [], [], []
    fixed_time = {seed: 0.0 for seed in seeds}
    adaptive_time = {seed: 0.0 for seed in seeds}
    for seed, path_id in identities:
        fixed = lookup[seed, path_id, "fixed3"]
        adaptive = lookup[seed, path_id, "adaptive"]
        identity = {"seed": seed, "path_id": path_id}
        if fixed["success"] and adaptive["success"]:
            fixed_duration, adaptive_duration = fixed["elapsed_time_s"], adaptive["elapsed_time_s"]
            common.append({**identity, "fixed3_time_s": fixed_duration,
                           "adaptive_time_s": adaptive_duration,
                           "time_reduction_fraction": 1.0 - adaptive_duration / fixed_duration})
            fixed_time[seed] += fixed_duration
            adaptive_time[seed] += adaptive_duration
        elif fixed["success"]:
            fixed_only.append(identity)
        elif adaptive["success"]:
            adaptive_only.append(identity)
        else:
            both_failed.append(identity)
    effect = _cluster_ratio([adaptive_time[seed] for seed in seeds],
                            [fixed_time[seed] for seed in seeds], complement=True)
    return {
        "n_candidate_pairs": len(identities),
        "n_common_success": len(common),
        "n_common_success_seeds": sum(value > 0 for value in fixed_time.values()),
        "common_success_pairs": common,
        "n_both_failed": len(both_failed), "both_failed_pairs": both_failed,
        "n_fixed3_only_success": len(fixed_only), "fixed3_only_success_pairs": fixed_only,
        "n_adaptive_only_success": len(adaptive_only), "adaptive_only_success_pairs": adaptive_only,
        "fixed3_failure_rate": (len(both_failed) + len(adaptive_only)) / len(identities),
        "adaptive_failure_rate": (len(both_failed) + len(fixed_only)) / len(identities),
        "both_failed_rate": len(both_failed) / len(identities),
        "fixed3_common_time_sum_s": math.fsum(fixed_time.values()) if common else None,
        "adaptive_common_time_sum_s": math.fsum(adaptive_time.values()) if common else None,
        "time_reduction_fraction": effect["estimate"],
        "time_reduction_ci95": effect["ci95"],
        "time_reduction_bootstrap": effect,
        "effect_definition": "1 - sum(adaptive elapsed_time_s) / sum(fixed3 elapsed_time_s), over identical jointly successful seed/path pairs",
        "scope": "Conditional on joint success; excludes failed-run durations and is not an unconditional performance estimate.",
    }


def _group(rows, seeds):
    return {
        **{method: _method_summary([row for row in rows if row["method"] == method], seeds)
           for method in METHODS},
        "paired": _paired_summary(rows, seeds),
    }


def summarize(rows, expected_seeds, expected_paths):
    """Validate and summarize all 60 records without mutating the caller's data.

    ``expected_paths`` maps six opaque path IDs to the three formal families.
    Point estimates use every eligible run; all confidence intervals resample
    whole independent training seeds. Single-seed descriptive slices have no CI.
    """
    rows = list(rows)
    seeds = _validate(rows, expected_seeds, expected_paths)
    rows = sorted(rows, key=lambda row: (row["seed"], row["path_id"], row["method"]))
    return {
        "schema_version": "preview_formal_statistics_v1",
        "n_rows": len(rows), "methods": list(METHODS),
        "expected_seeds": seeds, "expected_paths": dict(sorted(expected_paths.items())),
        "statistical_protocol": {
            "sampling_unit": "independent training seed",
            "n_independent_seeds": len(seeds),
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "interval_method": "percentile bootstrap with linear quantile interpolation",
            "resampling": "Sample whole seeds with replacement; retain all paths and both methods of each selected seed, including multiplicity.",
            "success_rate_estimand": "Mean of per-seed success rates on the complete balanced design; every failure remains in the denominator.",
            "successful_time_estimand": "Pooled mean elapsed time among successful runs only, with whole-seed bootstrap uncertainty.",
            "paired_time_estimand": "Ratio of summed times on jointly successful identical seed/path pairs; recomputed in every whole-seed resample.",
            "small_sample_warning": "95% bootstrap CI with only 5 independent training seeds: uncertainty estimates are unstable and descriptive; six paths are not six extra independent training replicates.",
            "single_seed_scope": "Per-seed slices are descriptive only and have null confidence intervals.",
            "metric_scope": "Contour errors, constraints, terminal tails and inference costs include all runs; successful-time summaries use successes only.",
        },
        "overall": _group(rows, seeds),
        "by_family": {family: _group([row for row in rows if row["family"] == family], seeds)
                      for family in FAMILIES},
        "by_seed": {str(seed): _group([row for row in rows if row["seed"] == seed], [seed])
                    for seed in seeds},
        "by_path": {path_id: {"family": expected_paths[path_id],
                              **_group([row for row in rows if row["path_id"] == path_id], seeds)}
                    for path_id in sorted(expected_paths)},
    }
