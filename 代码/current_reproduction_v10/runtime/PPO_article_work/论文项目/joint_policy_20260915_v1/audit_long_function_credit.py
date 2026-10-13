"""Saved-trace speed and complete-function response audit; no new simulation."""
import argparse
import time
import numpy as np
from search_complete_return import read
from study_io import BASE, register, receipt, write_json

LONG = [38, 39, 42, 43, 44, 45, 46, 47]


def trace_speed(trace):
    with np.load(trace) as z:
        rows = z['rows']; actions = z['actions']
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    requested = np.linalg.norm(rows[:, 8:10], axis=1)
    fast = (speed > 90) & (rows[:, 12] == 0)
    return dict(peak_mm_s=float(speed.max()), requested_peak_mm_s=float(requested.max()),
        ge95_s=float((speed >= 95).sum()*.001), ge98_s=float((speed >= 98).sum()*.001),
        ge99_s=float((speed >= 99).sum()*.001),
        fast_requested_median=float(np.median(requested[fast])) if fast.any() else None,
        fast_request_minus_actual_median=float(np.median((requested-speed)[fast])) if fast.any() else None,
        fast_near_acceleration_fraction=float(np.mean(np.linalg.norm(rows[fast, 4:6], axis=1) >= 1980)) if fast.any() else None,
        fast_near_jerk_fraction=float(np.mean(np.linalg.norm(rows[fast, 6:8], axis=1) >= 61875)) if fast.any() else None,
        requested_action_feed_max=float(actions[:, 1].max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    parent, source = BASE/args.parent, BASE/args.source
    selected = read(parent/'training_selection.json')
    data = read(source/'results.json')
    assert (source/'receipt.json').exists() and len(data['candidates']) == 10
    stage = register(args.stage, [__file__, parent/'training_selection.json', parent/'receipt.json',
        source/'results.json', source/'receipt.json'],
        dict(new_actor_calls=0, new_physics_ticks=0, wall_s=60),
        'Compare saved requested and executed speed on eight long TRAINING paths, and the actual '
        'whole-policy consequences of five parameter directions on those same routes. '
        'Speed>90 is a descriptive mask, not a dynamic feasibility or cruise-opportunity proof. '
        'Never rank a failed prefix as a fast full route. Retain all per-path metrics and signs; '
        'these data decide whether the existing proposal basis carries useful long-route credit.')
    started = time.perf_counter()
    baseline = [dict(path=i, success=selected['results'][i]['success'],
        time_s=selected['results'][i]['completion_time_s'],
        **trace_speed(BASE/selected['traces'][i])) for i in LONG]
    candidates = []
    for c in data['candidates']:
        rows = []
        for r, trace in zip(c['results'], c['traces']):
            if r['original_path'] in LONG:
                rows.append(dict(path=r['original_path'], success=r['success'],
                    time_s=r['completion_time_s'], **trace_speed(BASE/trace)))
        complete = all(r['success'] for r in rows) and len(rows) == len(LONG)
        candidates.append(dict(basis=c['basis'], name=c['name'], sign=c['sign'],
            successes=sum(r['success'] for r in rows),
            full_time_s=sum(r['time_s'] for r in rows) if complete else None,
            cruise_ge98_s=sum(r['ge98_s'] for r in rows) if complete else None,
            rows=rows))
    result = dict(baseline=baseline, baseline_full_time_s=sum(r['time_s'] for r in baseline),
        candidates=candidates, goal_complete=False)
    write_json(stage/'results.json', result)
    receipt(stage, dict(status='completed_saved_long_function_audit', new_actor_calls=0,
        new_physics_ticks=0, wall_s=time.perf_counter()-started))
    print('BASE_LONG', result['baseline_full_time_s'], flush=True)
    for c in candidates:
        print('LONG_RESPONSE', {k:v for k,v in c.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    main()
