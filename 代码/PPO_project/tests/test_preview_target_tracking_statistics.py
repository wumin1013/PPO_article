"""Design and paired-estimand regressions using synthetic records only."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import preview_target_tracking_statistics as stats


def design():
    paths = {f'{family}_{i}': family for family in stats.FAMILIES for i in range(4)}
    rows = [dict(seed=seed, method=method, path_id=path, family=family,
                 success=True, elapsed_time_s=10. if method == 'fixed3' else 8.,
                 contour_error_max_mm=.2, contour_error_rmse_mm=.1,
                 geometry_feasible=True, path_completed=True, full_rest_satisfied=True,
                 failure_reasons=[], constraints={name: dict(satisfied=True, peak=1., limit=2.)
                    for name in stats.CONSTRAINTS}, terminal_tail_s=.02,
                 inference_wall_s=.01, policy_decisions=100)
            for seed in stats.SEEDS for method in stats.METHODS for path, family in paths.items()]
    return rows, paths


def failed(row):
    row.update(success=False, path_completed=False, failure_reasons=['ordered_path_incomplete'], elapsed_time_s=.001)


def test_complete_design_and_seed_cluster_intervals():
    rows, paths = design()
    before = copy.deepcopy(rows)
    result = stats.summarize(rows, stats.SEEDS, paths)
    assert rows == before
    assert result['n_rows'] == 120
    assert result['overall']['adaptive']['n_success'] == 60
    paired = result['overall']['paired']
    assert paired['n_candidate_pairs'] == 60
    assert paired['time_reduction_fraction'] == pytest.approx(.2)
    assert paired['time_reduction_ci95'] == pytest.approx([.2, .2])
    assert result['overall']['paired_success_rate_difference']['estimate'] == 0
    assert all(r['paired']['time_reduction_ci95'] is None for r in result['by_seed'].values())


@pytest.mark.parametrize('mutation', ('missing', 'duplicate', 'seed', 'family', 'bool_seed', 'constraint', 'contradiction', 'nan_time'))
def test_rejects_incomplete_or_inconsistent_design(mutation):
    rows, paths = design()
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'seed': rows[0]['seed'] = 11
    elif mutation == 'family': rows[0]['family'] = 'dense_corners'
    elif mutation == 'bool_seed': rows[0]['seed'] = True
    elif mutation == 'constraint': rows[0]['constraints'].pop('feed_mm_s')
    elif mutation == 'contradiction': rows[0]['full_rest_satisfied'] = False
    else: rows[0]['elapsed_time_s'] = float('nan')
    with pytest.raises(ValueError): stats.validate(rows, stats.SEEDS, paths)


def test_fast_failures_never_enter_paired_time_and_remain_in_denominators():
    rows, paths = design()
    failed(rows[0])
    # Make a different path fail only under adaptive, leaving58 common pairs.
    failed(next(r for r in rows if r['seed'] == 111 and r['method'] == 'adaptive' and r['path_id'] == 'mixed_lengths_1'))
    result = stats.summarize(rows, stats.SEEDS, paths)['overall']
    assert result['fixed3']['n_runs'] == result['adaptive']['n_runs'] == 60
    assert result['fixed3']['n_success'] == result['adaptive']['n_success'] == 59
    assert result['paired']['n_common_success'] == 58
    assert result['paired']['fixed3_common_time_sum_s'] == 580
    assert result['paired']['adaptive_common_time_sum_s'] == 464
    assert result['paired']['time_reduction_fraction'] == pytest.approx(.2)


def test_one_contributing_seed_or_no_common_success_has_no_time_ci():
    rows, paths = design()
    for row in rows:
        if row['seed'] != 111 and row['method'] == 'adaptive': failed(row)
    report = stats.summarize(rows, stats.SEEDS, paths)['overall']['paired']
    assert report['n_common_success_seeds'] == 1
    assert report['time_reduction_ci95'] is None
    assert report['time_reduction_bootstrap']['bootstrap_undefined_resamples'] > 0
    for row in rows:
        if row['method'] == 'adaptive': failed(row)
    report = stats.summarize(rows, stats.SEEDS, paths)['overall']['paired']
    assert report['time_reduction_fraction'] is None
    assert report['time_reduction_ci95'] is None


def test_more_paths_do_not_create_more_seed_clusters():
    rows, paths = design()
    ratios = dict(zip(stats.SEEDS, (.7, .8, .9, 1., 1.1)))
    for row in rows:
        if row['method'] == 'adaptive': row['elapsed_time_s'] = 10.*ratios[row['seed']]
    report = stats.summarize(rows, stats.SEEDS, paths)['overall']['paired']
    expected = stats.base._cluster_ratio([ratios[s]*10 for s in stats.SEEDS], [10]*5, complement=True)
    assert report['time_reduction_ci95'] == pytest.approx(expected['ci95'])
    assert report['time_reduction_bootstrap']['n_seed_clusters'] == 5


def test_registered_path_and_seed_counts_are_mandatory():
    rows, paths = design()
    with pytest.raises(ValueError): stats.validate(rows, stats.SEEDS[:-1], paths)
    with pytest.raises(ValueError): stats.validate(rows, stats.SEEDS, dict(list(paths.items())[:6]))
    paths['mixed_lengths_0'] = 'dense_corners'
    with pytest.raises(ValueError): stats.validate(rows, stats.SEEDS, paths)
