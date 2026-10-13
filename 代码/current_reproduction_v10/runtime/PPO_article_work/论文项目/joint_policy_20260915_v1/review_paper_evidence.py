"""Audit saved paper candidates without policy calls, simulation, or selection updates."""
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np

from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from audit_xy_curvature_resolution import curvature_metrics
from geometry_spec import audit_commands
from multi_corner_raw_metrics import augment
from validate_actions import difference_audit


BASE = Path(__file__).resolve().parent
OUT = BASE / 'paper_evidence_review'
STAGES = {
    'retained': 'canonical_gate_band_shape_cruise_completion_feedback_20260918',
    'reference_supported': 'canonical_gate_band_reference_supported_joint_learning_20260920',
    'complementary': 'canonical_gate_band_complementary_joint_direction_learning_20260920',
    'worst_guard': 'canonical_gate_band_worst_guard_joint_learning_20260920',
    'recentered': 'canonical_gate_band_recentered_minimax_joint_learning_20260920',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def load_candidate(label):
    stage = BASE / STAGES[label]
    if label == 'retained':
        d = read(stage / 'results.json')['full_candidate']
        records = [dict(r, path=r['original_path'], trace=t,
                        complete_time_s=r['completion_time_s'])
                   for r, t in zip(d['results'], d['traces'])]
        return stage / 'candidate_0.5.pt', stage / 'results.json', records
    if label == 'reference_supported':
        d = read(stage / 'results.json')
        records = next(c for c in d['candidates'] if c['tag'] == d['best'])['records']
        return stage / 'research_candidate.pt', stage / 'results.json', records
    return (stage / 'shadow_training_policy.pt', stage / 'shadow_training_result.json',
            read(stage / 'shadow_training_result.json')['records'])


def inspect_trace(trace):
    with np.load(trace) as z:
        ref, rows = z['reference'].copy(), z['rows'].copy()
        action_count = len(z['actions']) if 'actions' in z else None
        # The terminal braking tail has a separate mode and is not a new decision.
        hold = None
        if 'executed_actions_1ms' in z:
            actions = z['executed_actions_1ms']
            changed = np.flatnonzero(np.any(actions[1:] != actions[:-1], axis=1)) + 1
            ordinary = changed[(rows[changed, 12] == 0) & (rows[changed-1, 12] == 0)]
            hold = bool(np.all(ordinary % 20 == 0))
    audit = difference_audit(ref[0], rows)
    geom, _ = audit_commands(ref, np.vstack([ref[0], rows[:, :2]]),
                             epsilon=1.25, endpoint_tolerance=1.25,
                             closed=bool(np.linalg.norm(ref[0]-ref[-1]) < 1e-9))
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    xy_velocity = np.diff(np.vstack([ref[0], rows[:, :2]]), axis=0) / .001
    result = dict(trace=str(trace.relative_to(BASE)), trace_sha256=sha(trace),
                  reference=ref.tolist(), rows=len(rows), vertices=len(ref),
                  length_mm=float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()),
                  motion=complete_motion_metrics(ref, rows), audit=audit, geometry=geom,
                  raw_velocity_max_difference=float(np.max(abs(xy_velocity-rows[:, 2:4]))),
                  saved_decisions=action_count, ordinary_actions_change_only_on_20ms=hold,
                  terminal_ticks=int(np.sum(rows[:, 12] != 0)))
    result['hard_pass'] = bool(audit['kinematic_pass'] and audit['rest_pass'] and geom['geometry_complete'])
    if len(ref) == 3:
        shape = complete_corner_metrics(ref, rows)
        result.update(shape=shape, curvature=curvature_metrics(ref, rows))
        u, v = np.diff(ref, axis=0)
        u, v = u/np.linalg.norm(u), v/np.linalg.norm(v)
        sign = np.sign(u[0]*v[1]-u[1]*v[0])
        normal = sign*np.array([-u[1], u[0]])
        heading = np.rad2deg(np.unwrap(np.arctan2(rows[:, 2:4]@normal, rows[:, 2:4]@u)))
        onsets = []
        for threshold in (.05, .5):
            indices = np.flatnonzero((speed > 5) & (abs(heading) > threshold))
            i = int(indices[0]) if len(indices) else None
            onsets.append(dict(threshold_deg=threshold,
                               direction=None if i is None else ('inside' if heading[i] > 0 else 'outside'),
                               time_s=None if i is None else (i+1)*.001,
                               distance_before_apex_projection_mm=None if i is None else float((ref[1]-rows[i, :2])@u)))
        result['first_turn'] = onsets
    elif len(ref) > 3 and np.linalg.norm(ref[0]-ref[-1]) > 1e-9 and result['hard_pass']:
        augmented = augment(dict(path=-1, success=True, trace=str(trace)))
        result['multi_corner_windows'] = augmented.get('multi_corner_windows', [])
    return result


def summary(records):
    corners = [r['shape'] for r in records if 'shape' in r]
    windows = [w['shape'] for r in records for w in r.get('multi_corner_windows', [])]
    return dict(paths=len(records), successes=sum(r['success'] for r in records),
                raw_hard_passes=sum(r['hard_pass'] for r in records),
                failures=[r['path'] for r in records if not r['success'] or not r['hard_pass']],
                complete_time_s=sum(r['motion']['recorded_time_s'] for r in records) if all(r['success'] and r['hard_pass'] for r in records) else None,
                cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records),
                curved_cruise99_s=sum(r['motion']['cruise_ge99_s'] for r in records if r['vertices'] > 2),
                longest_curved_cruise99_s=max((r['motion']['longest_cruise_ge99_s'] for r in records if r['vertices'] > 2), default=0),
                isolated_corners=len(corners),
                reverse_deg=sum(s['full_physical_reverse_deg'] for s in corners),
                exit_mm=sum(s['out_edge_outward_mm'] for s in corners),
                reverse_lateral_mm=sum(s['reverse_lateral_total_mm'] for s in corners),
                reverse_lateral_duration_s=sum(s['reverse_lateral_significant_duration_s'] for s in corners),
                normalized_energy_sum=sum(s['full_normalized_energy'] for s in corners),
                first_inside_005=sum(r['first_turn'][0]['direction'] == 'inside' for r in records if 'shape' in r),
                multi_corner_windows=len(windows),
                multi_window_reverse_deg=sum(s['full_physical_reverse_deg'] for s in windows),
                multi_window_exit_sum_mm=sum(s['out_edge_outward_mm'] for s in windows),
                max_peaks=np.max([r['audit']['peaks'] for r in records], axis=0).tolist())


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    state = read(BASE / '当前执行状态.json')
    resume = state['latest_resume_20260919']
    receipts = []
    for row in resume['latest_execution_receipts'][-10:]:
        p = BASE / row['stage'] / 'receipt.json'
        receipts.append(dict(path=str(p.relative_to(BASE)), sha256=sha(p),
                             expected=row['receipt_sha256'], matches=sha(p) == row['receipt_sha256'],
                             status=read(p)['status']))
    assert all(r['matches'] for r in receipts)
    boundaries = state['paper_research_boundary']
    assert sha(BASE / boundaries['source']) == boundaries['sha256']
    cache, references, candidates = {}, {}, []
    for label in STAGES:
        model, source, original = load_candidate(label)
        records = []
        for row in original:
            trace = (BASE / row['trace']).resolve()
            if trace not in cache:
                cache[trace] = inspect_trace(trace)
            audited = cache[trace]
            key = row['path']
            if key in references:
                assert audited['reference'] == references[key], (label, key, 'reference mismatch')
            references[key] = audited['reference']
            assert abs(audited['motion']['recorded_time_s']-row['complete_time_s']) < 1e-9
            records.append(dict(audited, path=key, success=bool(row['success'])))
        info = dict(label=label, model=str(model.relative_to(BASE)), model_sha256=sha(model),
                    result=str(source.relative_to(BASE)), result_sha256=sha(source),
                    receipt_sha256=sha(BASE / STAGES[label] / 'receipt.json'),
                    all_summary=summary(records),
                    original64_summary=summary([r for r in records if r['path'] < 64]),
                    records=records)
        candidates.append(info)
        print(label, json.dumps(info['all_summary']), flush=True)
        write(OUT / 'audit_progress.json', dict(completed_labels=[c['label'] for c in candidates],
                                              unique_saved_traces=len(cache)))
    write(OUT / 'candidate_audit.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
          work_type='saved_trace_physical_diagnostic', sources_receipts=receipts,
          candidates=candidates, unique_saved_traces=len(cache),
          new_training_decisions=0, new_actor_calls=0, new_simulation_ticks=0,
          sealed_test_opened=False, wall_s=time.perf_counter()-started))
    columns=['candidate','path','success','hard_pass','vertices','length_mm','time_s',
             'cruise99_s','longest_cruise99_s','reverse_deg','exit_mm','reverse_lateral_mm',
             'first_turn','trace_sha256']
    with (OUT / 'all_path_metrics.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for c in candidates:
            for r in c['records']:
                s=r.get('shape',{});m=r['motion']
                writer.writerow(dict(candidate=c['label'],path=r['path'],success=r['success'],hard_pass=r['hard_pass'],
                    vertices=r['vertices'],length_mm=r['length_mm'],time_s=m['recorded_time_s'],
                    cruise99_s=m['cruise_ge99_s'],longest_cruise99_s=m['longest_cruise_ge99_s'],
                    reverse_deg=s.get('full_physical_reverse_deg'),exit_mm=s.get('out_edge_outward_mm'),
                    reverse_lateral_mm=s.get('reverse_lateral_total_mm'),
                    first_turn=r.get('first_turn',[{}])[0].get('direction'),trace_sha256=r['trace_sha256']))
    print('SAVED_AUDIT_COMPLETE', len(cache), round(time.perf_counter()-started,3), flush=True)


if __name__ == '__main__':
    main()
