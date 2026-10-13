"""Paired seed-by-path summaries; failed prefixes never rank as complete times."""
from __future__ import annotations
import numpy as np


def make_matrices(records, methods, seeds, paths, classes, deterministic_methods=()):
    """Validate a complete frozen design before any performance calculation.

    Each record has method, seed, path, success and completion_time_s.
    Deterministic methods have seed=None and exactly one record per path.
    This validates table structure, not the independence or truth of source runs.
    """
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Unique nonempty methods are required')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct seed labels are required')
    if not paths or len(set(paths)) != len(paths) or set(classes) != set(paths):
        raise ValueError('Unique paths and an exact path-to-class mapping are required')
    if not set(deterministic_methods) <= set(methods):
        raise ValueError('Unknown deterministic method')
    expected = {(method, seed, path) for method in methods
                for seed in ([None] if method in deterministic_methods else seeds) for path in paths}
    lookup = {}
    for record in records:
        key = (record['method'], record['seed'], record['path'])
        if key not in expected or key in lookup:
            raise ValueError('Unexpected or duplicate method/seed/path record')
        success = record['success']
        time = record['completion_time_s']
        if not isinstance(success, bool):
            raise ValueError('Success must be an explicit boolean')
        if success and (isinstance(time, bool) or not isinstance(time, (int, float)) or
                        not np.isfinite(time) or time <= 0):
            raise ValueError('Successful records require positive finite completion time')
        if not success and time is not None:
            raise ValueError('Failed prefixes must have null completion time')
        lookup[key] = record
    if set(lookup) != expected:
        raise ValueError('The frozen method/seed/path design is incomplete')
    matrices = {}
    for method in methods:
        success = np.zeros((len(seeds), len(paths)), dtype=bool)
        times = np.full(success.shape, np.nan)
        for i, seed in enumerate(seeds):
            for j, path in enumerate(paths):
                record = lookup[method, None if method in deterministic_methods else seed, path]
                success[i, j] = record['success']
                if record['success']:
                    times[i, j] = record['completion_time_s']
        matrices[method] = dict(success=success, times=times,
            physical_evaluation_records=len(paths)*(1 if method in deterministic_methods else len(seeds)))
    return matrices


def _statistics(success_a, time_a, success_b, time_b):
    both = success_a & success_b
    only_a = success_a & ~success_b
    only_b = success_b & ~success_a
    neither = ~success_a & ~success_b
    paired_savings = 100*(time_b[both]-time_a[both])/time_b[both]
    tolerance = 1e-10
    wins_a = int(only_a.sum()+np.count_nonzero(time_a[both] < time_b[both]-tolerance))
    wins_b = int(only_b.sum()+np.count_nonzero(time_b[both] < time_a[both]-tolerance))
    all_complete = bool(both.all())
    return dict(success_rate_a=float(success_a.mean()), success_rate_b=float(success_b.mean()),
        paired_success_difference_pp=float(100*(success_a.astype(float)-success_b).mean()),
        paired_cells=int(both.size), both_success=int(both.sum()),
        only_a_success=int(only_a.sum()), only_b_success=int(only_b.sum()), both_failed=int(neither.sum()),
        a_wins_success_then_time=wins_a, b_wins_success_then_time=wins_b,
        both_success_time_ties=int(both.sum())-(wins_a-int(only_a.sum()))-(wins_b-int(only_b.sum())),
        common_success_mean_saving_percent=float(paired_savings.mean()) if len(paired_savings) else None,
        all_pairs_complete=all_complete,
        full_set_time_saving_percent=float(100*(time_b.sum()-time_a.sum())/time_b.sum()) if all_complete else None)


def paired_summary(matrices, method_a, method_b, seeds, paths, classes, *, replicates=5000, bootstrap_seed=915731):
    """Resample seeds and paths as crossed factors, with path-class stratification.

    A deterministic comparator is broadcast over the same selected seed indices;
    its physical evaluation count stays equal to the number of original paths.
    Intervals are approximate percentile intervals under factor exchangeability.
    """
    if replicates < 100:
        raise ValueError('At least 100 bootstrap replicates are required')
    a, b = matrices[method_a], matrices[method_b]
    expected_shape = (len(seeds), len(paths))
    for matrix in [a, b]:
        if matrix['success'].shape != expected_shape or matrix['times'].shape != expected_shape:
            raise ValueError('Matrix dimensions disagree with the frozen design')
    estimates = _statistics(a['success'], a['times'], b['success'], b['times'])
    keys = ['paired_success_difference_pp', 'common_success_mean_saving_percent', 'full_set_time_saving_percent']
    boot = {key: [] for key in keys}
    strata = [np.array([i for i, path in enumerate(paths) if classes[path] == label])
              for label in sorted(set(classes.values()))]
    rng = np.random.default_rng(bootstrap_seed)
    for _ in range(replicates):
        seed_indices = rng.integers(len(seeds), size=len(seeds))
        path_indices = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        selection = np.ix_(seed_indices, path_indices)
        row = _statistics(a['success'][selection], a['times'][selection],
                          b['success'][selection], b['times'][selection])
        for key in keys:
            # An absent full-set estimand is never resurrected by favourable resampling.
            value = row[key] if estimates[key] is not None else None
            if value is not None:
                boot[key].append(value)
    intervals = {key: dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)),
                           defined_replicates=len(values), requested_replicates=replicates)
                 if values else dict(lower=None, upper=None, defined_replicates=0, requested_replicates=replicates)
                 for key, values in boot.items()}
    per_seed = [dict(seed=seed, **_statistics(a['success'][i], a['times'][i], b['success'][i], b['times'][i]))
                for i, seed in enumerate(seeds)]
    per_class = []
    for label, indices in zip(sorted(set(classes.values())), strata):
        per_class.append(dict(path_class=label,
            **_statistics(a['success'][:, indices], a['times'][:, indices], b['success'][:, indices], b['times'][:, indices])))
    return dict(method_a=method_a, method_b=method_b, estimates=estimates,
        percentile_95_intervals=intervals, per_seed=per_seed, per_class=per_class,
        physical_evaluation_records={method_a:a['physical_evaluation_records'], method_b:b['physical_evaluation_records']},
        seed_labels=seeds, path_count=len(paths), bootstrap_seed=bootstrap_seed,
        method='paired crossed seed/path bootstrap, paths stratified by prespecified class',
        limitations=['independent training and source truth require separate run evidence',
                     'few seeds give uncertain interval calibration',
                     'common-success time is a selected conditional comparison',
                     'success and time intervals do not imply an automatic superiority decision'])
