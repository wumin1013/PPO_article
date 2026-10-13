"""Attribute finite-credit composition errors using saved XY, with no simulation."""
import time
import numpy as np
from geometric_turn_metrics import spatial_turn_profile
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def curvature_values(reference, rows, fixed_index=None):
    profile = spatial_turn_profile(np.vstack([reference[0], rows[:, :2]]), .1)
    values = abs(profile['turn'])/.1
    index = int(values.argmax())
    other = spatial_turn_profile(np.vstack([reference[0], rows[:, :2]]), .2)
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    moving = speed > 5.
    physical = abs(rows[moving, 2]*rows[moving, 5]-rows[moving, 3]*rows[moving, 4])/speed[moving]**3
    return np.asarray([values[index], values[index if fixed_index is None else fixed_index],
        max(abs(other['turn']))/.2, max(physical)]), index


def main():
    source = BASE/'current_dense_constraint_credit_collection_20260917'
    gradient = BASE/'current_direct_constraint_gradient_audit_20260917'
    proposal = BASE/'direct_full_subspace_constraint_solve_20260917'
    composition = BASE/'direct_constraint_composition_probe_20260917'
    paths = [3, 7, 13]
    stage = register('direct_composition_error_saved_audit_20260917',
        [__file__, source/'results.json', source/'receipt.json', gradient/'gradients.npz',
         proposal/'direction.npz', composition/'comparison.json', composition/'results.json',
         BASE/'geometric_turn_metrics.py'],
        dict(paths=paths, new_physics_ticks=0, new_actor_decisions=0, wall_s=120),
        'The small full-parameter probe completed but violates predicted nonincreasing '
        'peak curvature on paths3/7 and misses path13 exit/reversal credit. Do not scale '
        'up or promote it. Audit the saved single-action contributions, active spatial '
        'peak changes and cancellation. Compare curvature at the fixed nominal spatial '
        'peak against each branch maximum; add0.2mm spatial and velocity/acceleration '
        'physical peak diagnostics without replacing the acceptance metric. Rank '
        'high-contribution action/time pairs for a bounded lower-amplitude check if '
        'the existing finite perturbations switch active peaks. No new physical '
        'rollouts, training, reward changes or development access.')
    started = time.perf_counter()
    data = read(source/'results.json')
    whole = read(composition/'results.json')
    with np.load(gradient/'gradients.npz') as z:
        jac = z['query_jacobian'].copy()
    with np.load(proposal/'direction.npz') as z:
        direction = z['reduced_direction'].copy()
    weights = np.einsum('nhd,d->nh', jac, direction)/AMPLITUDE
    lookup = {(r['query'], r['axis'], r['direction']): r for r in data['records']}
    results = []
    for path in paths:
        base = next(r for r in data['baselines'] if r['path'] == path)
        with np.load(BASE/base['trace']) as z:
            reference, nominal = z['reference'].copy(), z['rows'].copy()
        old, index = curvature_values(reference, nominal)
        records = []
        curvature_sum = np.zeros(4)
        for q in (q for q in data['queries'] if q['path'] == path):
            for h in range(3):
                pair = [lookup[q['query'], h, s] for s in (-1, 1)]
                assert all(r['success'] for r in pair)
                values, indices = [], []
                for r in pair:
                    with np.load(BASE/r['trace']) as z:
                        rows = np.vstack([nominal[:q['anchor']*20], z['suffix_rows']])
                    value, peak = curvature_values(reference, rows, index)
                    values.append(value); indices.append(peak)
                weight = weights[q['query'], h]
                contribution = (values[1]-values[0])*.5*weight
                curvature_sum += contribution
                shape_contribution = {name: (pair[1]['shape'][name]-pair[0]['shape'][name])*.5*weight
                    for name in ('out_edge_outward_mm', 'full_physical_reverse_deg')}
                records.append(dict(query=q['query'], anchor=q['anchor'], head=h,
                    normalized_direction=float(weight), peak_indices=indices,
                    curvature_contribution=contribution.tolist(), **shape_contribution))
        actual = []
        for candidate in whole['candidates']:
            j = next(i for i, r in enumerate(candidate['results']) if r['original_path'] == path)
            with np.load(BASE/candidate['traces'][j]) as z:
                values, peak = curvature_values(reference, z['rows'], index)
            actual.append(dict(coefficient=candidate['coefficient'], values=values.tolist(), peak_index=peak))
        actual_slope = (np.asarray(actual[1]['values'])-np.asarray(actual[0]['values']))/.25
        entry = dict(path=path, nominal_values=old.tolist(), nominal_peak_index=index,
            predicted_curvature_per_unit=curvature_sum.tolist(), actual_curvature_per_unit=actual_slope.tolist(),
            actual=actual, contributions=records,
            switched_peak_pairs=sum(any(i != index for i in r['peak_indices']) for r in records),
            absolute_contribution_sum=np.abs([r['curvature_contribution'] for r in records]).sum(0).tolist(),
            top_queries_by_spatial_peak=sorted(records, key=lambda r: abs(r['curvature_contribution'][0]), reverse=True)[:10],
            top_queries_by_reverse=sorted(records, key=lambda r: abs(r['full_physical_reverse_deg']), reverse=True)[:10])
        results.append(entry)
        print('COMPOSITION_ERROR', path, {k: entry[k] for k in ('nominal_peak_index',
            'predicted_curvature_per_unit', 'actual_curvature_per_unit', 'switched_peak_pairs',
            'absolute_contribution_sum')}, flush=True)
        assert time.perf_counter()-started < 120
    write_json(stage/'results.json', dict(curvature_names=['spatial_peak_0.1mm',
        'fixed_nominal_spatial_peak', 'spatial_peak_0.2mm', 'physical_peak_speed_gt5'],
        paths=results, candidate_rejected_for_scaling=True, goal_complete=False))
    receipt(stage, dict(status='completed_saved_composition_error_audit',
        new_physics_ticks=0, new_actor_decisions=0, wall_s=time.perf_counter()-started))


if __name__ == '__main__':
    main()
