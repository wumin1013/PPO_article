"""Statistics for 120 frozen outcomes; reuse the original seed-cluster kernel."""
from collections import Counter
from numbers import Integral

import preview_formal_statistics as base

METHODS = base.METHODS
SEEDS = (111, 222, 333, 444, 555)
FAMILIES = base.FAMILIES
CONSTRAINTS = frozenset(('feed_mm_s', 'feed_acceleration_mm_s2', 'feed_jerk_mm_s3',
    'angular_velocity_rad_s', 'angular_acceleration_rad_s2', 'angular_jerk_rad_s3'))


def validate(rows, expected_seeds, expected_paths):
    seeds = list(expected_seeds)
    if (len(seeds) != 5 or set(seeds) != set(SEEDS)
            or any(not isinstance(s, Integral) or isinstance(s, bool) for s in seeds)):
        raise ValueError('Require the five registered fresh seeds111/222/333/444/555')
    if (not isinstance(expected_paths, dict) or len(expected_paths) != 12
            or any(not isinstance(p, str) or not p for p in expected_paths)
            or Counter(expected_paths.values()) != Counter({family: 4 for family in FAMILIES})):
        raise ValueError('Require twelve distinct registered paths, four in each family')
    expected = {(s, m, p) for s in seeds for m in METHODS for p in expected_paths}
    required = {'seed', 'method', 'path_id', 'family', 'success', 'elapsed_time_s',
        'contour_error_max_mm', 'contour_error_rmse_mm', 'geometry_feasible',
        'path_completed', 'full_rest_satisfied', 'failure_reasons', 'constraints',
        'terminal_tail_s', 'inference_wall_s', 'policy_decisions'}
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f'Row{index} lacks audited fields')
        seed, method, path_id = row['seed'], row['method'], row['path_id']
        key = (seed, method, path_id)
        if not isinstance(seed, Integral) or isinstance(seed, bool) or key not in expected or key in seen:
            raise ValueError(f'Unexpected or duplicate frozen identity: {key}')
        seen.add(key)
        if row['family'] != expected_paths[path_id]:
            raise ValueError(f'Family mismatch: {key}')
        for name in ('success', 'geometry_feasible', 'path_completed', 'full_rest_satisfied'):
            if not isinstance(row[name], bool):
                raise ValueError(f'Non-boolean {name}: {key}')
        if not isinstance(row['failure_reasons'], (list, tuple)) or any(
                not isinstance(reason, str) or not reason for reason in row['failure_reasons']):
            raise ValueError(f'Invalid failure reasons: {key}')
        constraints = row['constraints']
        if not isinstance(constraints, dict) or set(constraints) != CONSTRAINTS:
            raise ValueError(f'All six position-difference constraints are required: {key}')
        for name, value in constraints.items():
            if (not isinstance(value, dict) or not {'satisfied', 'peak', 'limit'}.issubset(value)
                    or not isinstance(value['satisfied'], bool)):
                raise ValueError(f'Malformed {name} constraint: {key}')
        if row['success']:
            if not base._finite(row['elapsed_time_s']) or row['elapsed_time_s'] <= 0:
                raise ValueError(f'Complete duration must be finite and positive: {key}')
            if (row['failure_reasons'] or not all(row[n] for n in
                    ('geometry_feasible', 'path_completed', 'full_rest_satisfied'))
                    or not all(c['satisfied'] for c in constraints.values())):
                raise ValueError(f'Success conflicts with independent audit: {key}')
    if seen != expected:
        raise ValueError(f'Incomplete120-outcome design: missing{len(expected-seen)}')
    return sorted(seeds)


def summarize(rows, expected_seeds, expected_paths):
    rows = list(rows)
    seeds = validate(rows, expected_seeds, expected_paths)
    rows = sorted(rows, key=lambda r: (r['seed'], r['path_id'], r['method']))
    overall = base._group(rows, seeds)
    differences = [sum(r['success'] for r in rows if r['seed'] == seed and r['method'] == 'adaptive')
                   - sum(r['success'] for r in rows if r['seed'] == seed and r['method'] == 'fixed3')
                   for seed in seeds]
    overall['paired_success_rate_difference'] = base._cluster_ratio(differences, [12]*5)
    return dict(schema_version='target_tracking_statistics_v1', n_rows=len(rows), methods=list(METHODS),
        expected_seeds=seeds, expected_paths=dict(sorted(expected_paths.items())),
        statistical_protocol=dict(sampling_unit='independent complete two-phase training seed',
            n_independent_seeds=5, confidence_level=base.CONFIDENCE_LEVEL,
            bootstrap_seed=base.BOOTSTRAP_SEED, bootstrap_resamples=base.BOOTSTRAP_RESAMPLES,
            interval_method='Original percentile bootstrap with linear quantile interpolation',
            resampling='Whole seeds with all twelve paths and both methods, retaining repeated clusters',
            failure_policy='All failures remain in success denominators; their duration is never a valid completed time',
            paired_time_estimand='1-sum(adaptive time)/sum(fixed time) on identical jointly fully successful seed/path pairs; includes braking and rest',
            completion_estimand='Adaptive minus fixed complete-success proportion, paired by seed',
            undefined_resamples='Count zero-common-time denominator resamples and exclude only from conditional interval endpoints',
            limitation='Five independent seeds; twelve paths are not additional training replicates. Per-seed slices are descriptive with null CIs.'),
        overall=overall,
        by_family={family: base._group([r for r in rows if r['family'] == family], seeds) for family in FAMILIES},
        by_seed={str(seed): base._group([r for r in rows if r['seed'] == seed], [seed]) for seed in seeds},
        by_path={p: dict(family=expected_paths[p], **base._group([r for r in rows if r['path_id'] == p], seeds))
                 for p in sorted(expected_paths)})
