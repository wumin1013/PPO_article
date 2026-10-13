"""Duration and speed safeguards derived from complete, unsmoothed physical traces."""
from functools import lru_cache
import numpy as np
from complete_corner_metrics import complete_corner_metrics


# Existing shape/physics guards are kept separately and are never replaced here.
UPPER = (
    ('shape', 'out_edge_outward_gt02_duration_s', .02),
    ('shape', 'out_edge_outward_gt005_duration_s', .02),
    ('shape', 'heading_overshoot_deg', .5),
    ('shape', 'heading_overshoot_gt05_duration_s', .02),
    ('shape', 'reverse_lateral_significant_mm', .002),
    ('shape', 'reverse_lateral_significant_duration_s', .02),
    ('duration_audit', 'exit_time_gt02_s', .02),
    ('duration_audit', 'exit_time_gt005_s', .02),
    ('duration_audit', 'exit_area_mm_s', .002),
    ('duration_audit', 'energy_at_02mm', .02),
    ('duration_audit', 'curvature_at_02mm_inv', .2),
)
LOWER = (
    ('motion', 'before_apex_5mm_speed_mm_s', 2.),
    ('motion', 'after_apex_5mm_speed_mm_s', 2.),
    ('motion', 'cruise_ge99_s', .02),
    ('motion', 'longest_cruise_ge99_s', .02),
)


def piecewise_exceedance(values, threshold):
    """Exact duration/area for the polygonal interpolation of recorded 1ms XY."""
    a, b = values[:-1] - threshold, values[1:] - threshold
    positive_a, positive_b = a > 0., b > 0.
    both = positive_a & positive_b
    crossing = positive_a ^ positive_b
    fraction = np.zeros_like(a)
    fraction[both] = 1.
    fraction[crossing] = np.maximum(a[crossing], b[crossing]) / abs(b[crossing] - a[crossing])
    area = np.zeros_like(a)
    area[both] = .5 * (a[both] + b[both])
    area[crossing] = .5 * np.maximum(a[crossing], b[crossing]) * fraction[crossing]
    return float(fraction.sum() * .001), float(area.sum() * .001)


@lru_cache(maxsize=2048)
def raw_metrics(trace):
    with np.load(trace) as z:
        ref, rows = z['reference'].copy(), z['rows'].copy()
    u, v = np.diff(ref, axis=0)
    sign = np.sign(u[0] * v[1] - u[1] * v[0])
    v /= np.linalg.norm(v)
    normal = sign * np.array([-v[1], v[0]])
    xy = np.vstack([ref[0], rows[:, :2]])
    outward = -(xy - ref[1]) @ normal
    coarse = complete_corner_metrics(ref, rows, spacing=.2)
    return dict(exit_time_gt02_s=piecewise_exceedance(outward, .02)[0],
                exit_time_gt005_s=piecewise_exceedance(outward, .005)[0],
                exit_area_mm_s=piecewise_exceedance(outward, 0.)[1],
                energy_at_02mm=coarse['full_normalized_energy'],
                curvature_at_02mm_inv=coarse['full_spatial_peak_curvature_mm_inv'])


def augment(record):
    if record['success']:
        record['duration_audit'] = raw_metrics(str(record['trace'])).copy()
    return record


def guard_names():
    return [f'{group}.{key}' for group, key, _ in UPPER + LOWER] + ['complete_time_s']


def guards(record, baseline):
    if not record['success']:
        return np.full(len(guard_names()), -100.)
    augment(record); augment(baseline)
    values = [(baseline[g][k] - record[g][k]) / s for g, k, s in UPPER]
    values += [(record[g][k] - baseline[g][k]) / s for g, k, s in LOWER]
    values += [(baseline['complete_time_s'] - record['complete_time_s']) / .02]
    return np.asarray(values)


def penalty(record):
    if not record['success']:
        return 1e6
    augment(record)
    return (record['duration_audit']['exit_area_mm_s'] / .002) ** 2
