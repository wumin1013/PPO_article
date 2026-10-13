"""One registered geometric resumption probe at the first stopped fold state."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import math
import pickle
import time
import numpy as np
from common import ROOT, sha, utc, write_json
from runtime_physical_time_v5 import install
from predictive_action_candidate_v5 import new_environment, retained_step
from physical_teacher_v5 import guide_angle, BranchArchive
from braking_reachability_v5 import stopping_extension

PARENT = ROOT/'results/recoverable_action_candidate_v5_3/final_dense_fold'
OUT = ROOT/'results/recoverable_stall_probe_v5_1'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    decisions = [json.loads(line) for line in (PARENT/'predictions/decisions.jsonl').read_text('utf-8').splitlines()]
    states = []
    with (PARENT/'predictions/decision_states.pkl').open('rb') as stream:
        while True:
            try:
                states.append(pickle.load(stream))
            except EOFError:
                break
    selected = next(i for i in range(1, len(states))
        if decisions[i-1]['source'] == 'certified_stop' and states[i]['core']['rest_ticks'] >= 20)
    state = states[selected]
    with np.load(PARENT/'actual.npz') as data:
        reference = data['reference'].copy()
    parent = json.loads((PARENT/'actual.json').read_text('utf-8'))
    registration = dict(created_utc=utc(), scope='One development-state diagnostic only',
        state_selection='First saved real state after a certified stop with at least 20 ms rest',
        selected_decision_id=selected, selected_actual_tick=state['core']['ticks'],
        parent_trace_sha256=sha(PARENT/'actual.npz'), parent_record_sha256=sha(PARENT/'actual.json'),
        parent_states_sha256=sha(PARENT/'predictions/decision_states.pkl'),
        planned_feedback_decisions=10, control_period_s=.02, dt_s=.001,
        target='Next unvisited ordered gate point; final reference endpoint if all gates visited',
        preview_mm=.8, physical_feed_mm_s=20., heading_alignment_threshold_rad=.05,
        action='Steer toward the target after compensating the actual common guide bearing; clip to the unchanged pi/2 action bound. Request zero feed until the existing filtered heading is within 0.05 rad of that target, then 20 mm/s.',
        no_changes='No heading reset, no position reset, no new gates, no tolerance change, no skipping',
        stopping_check='Assess and retain the unchanged complete cubic stop after every actual decision; this diagnostic does not execute the stopping extension',
        outcome='Stop after 10 decisions or the first common executor termination/error; retain all intervals and outcomes',
        main_result_and_holdout_unchanged=True,
        source_sha256={name: sha(ROOT/'src'/name) for name in
            ['diagnose_recoverable_stall_v5.py', 'physical_teacher_v5.py',
             'braking_reachability_v5.py', 'runtime_physical_time_v5.py']})
    write_json(OUT/'registration.json', registration)
    (OUT/'initial_state.pkl').write_bytes(pickle.dumps(state, protocol=5))
    install(dt=.001, tolerance=parent['tolerance_mm'])
    env = new_environment(reference, state)
    archive = BranchArchive(OUT)
    stop_dir = OUT/'stops'
    stop_dir.mkdir()
    stops = BranchArchive(stop_dir)
    all_rows, all_positions, actions, details = [], [env.position.copy()], [], []
    initial_progress = float(env.progress()[0])
    started = time.perf_counter()
    try:
        for decision in range(10):
            target = (env.corridor.gates[env.tracker.frontier].point if
                env.tracker.frontier < len(env.corridor.gates) else reference[-1])
            delta = target-env.position
            bearing = math.atan2(delta[1], delta[0])
            beta = guide_angle(env.env, 0., 20., .8)
            wrap = lambda angle: (angle+math.pi) % (2*math.pi)-math.pi
            steering = float(np.clip(wrap(bearing-beta), -math.pi/2, math.pi/2))
            heading_error = abs(wrap(bearing-env.heading))
            feed = 0. if heading_error > .05 else 20.
            action = np.array([steering/(math.pi/2), feed/50.-1., -1.])
            point0 = env.position.copy()
            points = [point0]
            tracker = env.tracker
            def record(point):
                points.append(np.asarray(point).copy())
                return type(tracker).update(tracker, point)
            tracker.update = record
            rows, error = retained_step(env, action)
            points = np.asarray(points)
            detail = dict(branch_id=decision, action=action.tolist(), feed_mm_s=feed,
                target_mm=target.tolist(), heading_error_rad=heading_error,
                steering_clipped=abs(wrap(bearing-beta)) > math.pi/2,
                error=error, done=bool(env.done), success=bool(env.success), reason=env.reason,
                every_interval_inside=bool(env.tracker.inside),
                final_progress_mm=float(env.progress()[0]), final_frontier=env.tracker.frontier)
            archive.append(detail, rows, points)
            all_rows.extend(rows)
            all_positions.extend(points[1:])
            actions.append(action)
            if error is None and not env.done:
                sr, sp, stop = stopping_extension(env)
                stop.update(branch_id=decision)
                stops.append(stop, sr, sp)
                detail['stopping_extension_feasible'] = stop['feasible']
            details.append(detail)
            if error is not None or env.done:
                break
    finally:
        archive.close()
        stops.close()
    np.savez_compressed(OUT/'probe.npz', reference=reference, rows=np.asarray(all_rows),
        positions=np.asarray(all_positions), actions=np.asarray(actions))
    report = dict(created_utc=utc(), registration_sha256=sha(OUT/'registration.json'),
        decisions=details, actual_intervals=len(all_rows),
        initial_progress_mm=initial_progress, final_progress_mm=float(env.progress()[0]),
        initial_frontier=state['tracker']['frontier'], final_frontier=env.tracker.frontier,
        progress_gain_mm=float(env.progress()[0])-initial_progress,
        all_intervals_inside=all(r['every_interval_inside'] for r in details),
        all_stopping_extensions_pass=all(r.get('stopping_extension_feasible', False) for r in details),
        wall_s=time.perf_counter()-started, trace_sha256=sha(OUT/'probe.npz'),
        interpretation='A 200 ms resumption probe from one retained development state, not a full-program completion, final test, training result or main model replacement')
    write_json(OUT/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
