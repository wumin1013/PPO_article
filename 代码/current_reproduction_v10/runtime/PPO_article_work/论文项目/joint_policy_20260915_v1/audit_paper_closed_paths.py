"""Complete saved closed-route coverage without inventing a through-seam turn."""
import json
from pathlib import Path
import time
import numpy as np

from complete_corner_metrics import complete_corner_metrics
from geometry_spec import Corridor, GateTracker
from study_io import BASE, register, receipt, sha, write_json


def phase_metrics(rows, tangent, orientation):
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    moving = speed > 5.
    normal = orientation*np.array([-tangent[1], tangent[0]])
    heading = np.rad2deg(np.arctan2(rows[:, 2:4]@normal, rows[:, 2:4]@tangent))
    omega = orientation*(rows[moving, 2]*rows[moving, 5]-rows[moving, 3]*rows[moving, 4])/speed[moving]**2
    return dict(ticks=len(rows), duration_s=len(rows)*.001,
        physical_reverse_deg=float(np.rad2deg(np.maximum(-omega, 0).sum()*.001)),
        absolute_heading_error_max_deg=float(abs(heading[moving]).max()) if moving.any() else None,
        moving_ticks=int(moving.sum()), stopped_heading_interpreted=False)


def audit_closed(trace):
    with np.load(trace) as z:
        ref, rows = z['reference'].copy(), z['rows'].copy()
    assert np.linalg.norm(ref[0]-ref[-1]) < 1e-9
    corridor = Corridor(ref, closed=True)
    tracker = GateTracker(corridor, ref[0])
    progress = []
    for xy in rows[:, :2]:
        tracker.update(xy)
        i = min(tracker.frontier, len(corridor.lengths)-1)
        along = np.clip((xy-corridor.starts[i])@corridor.tangents[i], 0., corridor.lengths[i])
        progress.append(corridor.arcs[i]+along)
    assert tracker.snapshot(1.25)['geometry_complete']
    progress = np.maximum.accumulate(progress)
    midpoints = .5*(corridor.arcs[:-1]+corridor.arcs[1:])
    boundaries = np.r_[0, np.searchsorted(progress, midpoints), len(rows)]
    assert np.all(np.diff(boundaries) > 4), boundaries.tolist()
    assert np.diff(boundaries).sum() == len(rows)
    windows = []
    signs = []
    for vertex in range(1, len(ref)-1):
        lo, hi = map(int, boundaries[vertex:vertex+2])
        local = ref[vertex-1:vertex+2]
        first = ref[0] if lo == 0 else rows[lo-1, :2]
        shape = complete_corner_metrics(local, rows[lo:hi], initial_xy=first)
        signs.append(np.sign(shape['turn_deg']))
        windows.append(dict(vertex=vertex, start_ms=lo, stop_ms=hi, shape=shape))
    # These eight registered closed development shapes are convex polygons.
    # The seam angle exists geometrically, but no continuous motion traverses it.
    assert len(set(signs)) == 1 and signs[0] != 0
    orientation = signs[0]
    start_stop = int(boundaries[1]); end_start = int(boundaries[-2])
    startup = phase_metrics(rows[:start_stop], corridor.tangents[0], orientation)
    terminal = phase_metrics(rows[end_start:], corridor.tangents[-1], orientation)
    whole = phase_metrics(rows, corridor.tangents[0], orientation)
    partition_reverse = (startup['physical_reverse_deg'] + terminal['physical_reverse_deg'] +
                         sum(w['shape']['full_physical_reverse_deg'] for w in windows))
    assert abs(partition_reverse-whole['physical_reverse_deg']) < 1e-9
    u, v = corridor.tangents[-1], corridor.tangents[0]
    return dict(trace=str(trace.relative_to(BASE)), trace_sha256=sha(trace), rows=len(rows),
        internal_corner_count=len(windows), boundaries_ms=boundaries.tolist(),
        every_tick_assigned_exactly_once=True, internal_windows=windows,
        startup=dict(start_ms=0, stop_ms=start_stop, **startup),
        terminal=dict(start_ms=end_start, stop_ms=len(rows), **terminal),
        geometric_seam_turn_deg=float(np.rad2deg(np.arctan2(u[0]*v[1]-u[1]*v[0], u@v))),
        seam_through_turn_metrics=None, seam_reason='Stopped endpoint and independent startup; no physical through-seam corner exists.',
        complete_reverse_deg=whole['physical_reverse_deg'],
        summary=dict(internal_reverse_deg=sum(w['shape']['full_physical_reverse_deg'] for w in windows),
            internal_exit_sum_mm=sum(w['shape']['out_edge_outward_mm'] for w in windows),
            internal_reverse_lateral_sum_mm=sum(w['shape']['reverse_lateral_total_mm'] for w in windows),
            internal_reverse_lateral_duration_s=sum(w['shape']['reverse_lateral_significant_duration_s'] for w in windows),
            internal_max_xy_spatial_curvature_mm_inv=max(w['shape']['full_spatial_peak_curvature_mm_inv'] for w in windows),
            startup_reverse_deg=startup['physical_reverse_deg'], terminal_reverse_deg=terminal['physical_reverse_deg']))


def main():
    prior = BASE/'paper_evidence_review'
    manifest = json.loads((prior/'candidate_audit.json').read_text(encoding='utf-8'))
    stage = register('paper_evidence_review/closed_paths',
        [__file__, prior/'candidate_audit.json', prior/'receipt.json', BASE/'geometry_spec.py',
         BASE/'complete_corner_metrics.py', BASE/'geometric_turn_metrics.py'],
        dict(new_actor_calls=0, new_simulation_ticks=0, training_updates=0, maximum_unique_saved_traces=16, wall_s=120),
        'Complete missing closed-route shape accounting from saved data only. Partition all 1ms ticks '
        'into a startup half-leg, internal corner windows bounded by adjacent reference-leg midpoints, '
        'and the final half-leg including braking/rest. Use directed-gate progress to disambiguate '
        'start versus finish at the same XY. Half-open windows do not overlap; use the real preceding '
        'XY for spatial differences. Do not splice final braking to startup or claim a through-seam tangent. '
        'All closed paths40-47 and all five already-reviewed candidates included; no optimization or selection change.')
    started=time.perf_counter(); cache={}; candidate_rows=[]
    for candidate in manifest['candidates']:
        items=[]
        for r in candidate['records']:
            if not 40 <= r['path'] <= 47:
                continue
            trace=BASE/r['trace']; digest=sha(trace)
            assert digest == r['trace_sha256']
            if digest not in cache:
                cache[digest]=audit_closed(trace)
            items.append(dict(path=r['path'], trace_sha256=digest))
        assert len(items) == 8
        values=[cache[r['trace_sha256']] for r in items]
        summary={k:sum(r['summary'][k] for r in values) for k in values[0]['summary'] if 'max_' not in k}
        summary.update(paths=8, internal_corners=sum(r['internal_corner_count'] for r in values),
                       complete_reverse_deg=sum(r['complete_reverse_deg'] for r in values),
                       complete_ticks=sum(r['rows'] for r in values),
                       all_ticks_accounted=True, seam_through_turns_claimed=0)
        candidate_rows.append(dict(label=candidate['label'], model_sha256=candidate['model_sha256'],
                                   records=items, summary=summary))
        print(candidate['label'],json.dumps(summary),flush=True)
    assert len(cache)==16
    write_json(stage/'results.json',dict(unique_trace_metrics=cache,candidates=candidate_rows,
        definition='Half-open midpoint windows; startup and braking/rest are separate real physical phases. All ticks counted once.',
        all_saved_source_hashes_match=True, new_simulation_ticks=0))
    receipt(stage,dict(status='completed_closed_path_saved_shape_accounting', paths_per_candidate=8,
        candidate_records=40, unique_saved_traces=16, new_simulation_ticks=0,new_actor_calls=0,
        results_sha256=sha(stage/'results.json'),wall_s=time.perf_counter()-started,
        formal_adoption=False, goal_complete=False))


if __name__=='__main__': main()
