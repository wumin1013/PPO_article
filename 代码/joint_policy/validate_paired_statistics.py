"""Finite synthetic checks for publication-critical paired outcome handling."""
import copy
from pathlib import Path
from paired_result_statistics import make_matrices, paired_summary
from study_io import BASE, register, receipt, write_json


def main():
    stage = register('paired_statistics_validation_v1', [Path(__file__), BASE/'paired_result_statistics.py',
        BASE/'sealed_test_protocol_v1.md'],
        dict(synthetic_bootstrap_calls=4, bootstrap_replicates_per_call=1000, physical_runs=0),
        'Validate paired failure exclusion, deterministic comparator counts, full-design checks and crossed resampling before future final evidence exists.')
    seeds, paths = [101, 102], ['p1', 'p2']
    classes = {'p1':'mixed', 'p2':'dense'}
    methods = ['joint', 'fixed', 's2']
    records = [dict(method=method, seed=seed, path=path, success=True,
                    completion_time_s=9. if method == 'joint' else 10.)
               for method in methods for seed in ([None] if method == 's2' else seeds) for path in paths]
    build = lambda data: make_matrices(data, methods, seeds, paths, classes, ['s2'])
    matrices = build(records)
    a = paired_summary(matrices, 'joint', 's2', seeds, paths, classes, replicates=1000)
    assert a['estimates']['common_success_mean_saving_percent'] == 10.
    assert a['percentile_95_intervals']['full_set_time_saving_percent']['lower'] == 10.
    assert a['physical_evaluation_records'] == {'joint':4, 's2':2}
    checks = ['constant_complete_saving_exact', 'deterministic_s2_not_counted_as_extra_runs']
    for case, bad in [('duplicate', records+[copy.deepcopy(records[0])]), ('missing', records[:-1])]:
        try:
            build(bad)
        except ValueError:
            checks.append(case+'_rejected')
        else:
            raise AssertionError(case)
    bad = copy.deepcopy(records)
    bad[0].update(success=False, completion_time_s=.001)
    try:
        build(bad)
    except ValueError:
        checks.append('failed_prefix_time_rejected')
    else:
        raise AssertionError('Failed prefix ranked')
    bad[0]['completion_time_s'] = None
    b = paired_summary(build(bad), 'joint', 's2', seeds, paths, classes, replicates=1000)
    assert b['estimates']['only_b_success'] == 1 and b['estimates']['full_set_time_saving_percent'] is None
    assert b['percentile_95_intervals']['full_set_time_saving_percent']['defined_replicates'] == 0
    assert b['estimates']['common_success_mean_saving_percent'] == 10.
    checks.append('failure_does_not_create_full_set_time_rank')
    all_failed = copy.deepcopy(records)
    for row in all_failed:
        row.update(success=False, completion_time_s=None)
    c = paired_summary(build(all_failed), 'joint', 'fixed', seeds, paths, classes, replicates=1000)
    assert c['estimates']['both_failed'] == 4
    assert c['estimates']['common_success_mean_saving_percent'] is None
    assert c['estimates']['a_wins_success_then_time'] == 0
    checks.append('no_shared_success_has_no_time_statistic')
    seed_effect = copy.deepcopy(records)
    for row in seed_effect:
        if row['method'] == 'joint':
            row['completion_time_s'] = 9. if row['seed'] == 101 else 5.
    d = paired_summary(build(seed_effect), 'joint', 's2', seeds, paths, classes, replicates=1000)
    interval = d['percentile_95_intervals']['common_success_mean_saving_percent']
    assert interval['lower'] == 10. and interval['upper'] == 50.
    checks.append('whole_seed_effect_preserved_in_interval')
    write_json(stage/'results.json', dict(checks=checks, synthetic_outputs=[a,b,c,d], real_test_opened=False))
    receipt(stage, dict(status='completed_synthetic_paired_statistics_validation', checks_passed=len(checks),
        synthetic_bootstrap_calls=4, bootstrap_replicates=4000, physical_runs=0, models=0, training_decisions=0))
    print(f'{len(checks)} synthetic checks passed; no real test data loaded.')


if __name__ == '__main__':
    main()
