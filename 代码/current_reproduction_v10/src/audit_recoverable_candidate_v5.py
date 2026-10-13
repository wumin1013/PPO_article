"""Read-only independent audit of saved v5_3 development evidence.

Never calls a controller step or generates a new evaluation case. Geometry,
finite differences, decisions, saved stopping commands and costs are re-read.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import json
import pickle
import time
from collections import Counter
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL
from runtime_physical_time_v5 import install
from predictive_action_candidate_v5 import first_action, change_cost
from metrics_exact_index_v5 import metrics

OUT = ROOT/'results/recoverable_action_candidate_v5_3'
REPORT = ROOT/'audit/recoverable_action_candidate_v5_3_independent_review.json'
DT = .001
LIMITS = np.array([100., 2000., 62500.])
OFFSETS = [-.08, -.04, 0., .04, .08]
FACTORS = [.2, .4, .6, .8, 1.]


def read(path):
    return json.loads(path.read_text('utf-8'))


def jsonl(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines()]


def states_at(path):
    result = []
    with path.open('rb') as stream:
        while True:
            try:
                result.append(pickle.load(stream))
            except EOFError:
                return result


def kinematics(pp, v0, a0, previous_rest=0):
    v = np.diff(pp, axis=0)/DT
    a = np.diff(np.vstack([v0, v]), axis=0)/DT
    j = np.diff(np.vstack([a0, a]), axis=0)/DT
    peaks = [float(np.linalg.norm(x, axis=1).max(initial=0.)) for x in (v, a, j)]
    vector = bool(np.all(np.asarray(peaks)/LIMITS <= 1.00002))
    rest = (np.linalg.norm(v, axis=1) <= .001) & (np.linalg.norm(a, axis=1) <= .01)
    nonrest = np.flatnonzero(~rest)
    rest_ticks = len(rest)-(int(nonrest[-1])+1 if len(nonrest) else 0)
    if not len(nonrest):
        rest_ticks += previous_rest
    return peaks, vector, rest_ticks


class SavedArrays:
    def __init__(self, base):
        self.base, self.files, self.digests = base, {}, {}

    def get(self, record):
        path = self.base/record['trace_file']
        if path.name not in self.files:
            assert sha(path) == record['trace_sha256']
            self.digests[path.name] = record['trace_sha256']
            with np.load(path) as data:
                self.files[path.name] = {key: data[key].copy() for key in data.files}
        data = self.files[path.name]
        assert record['trace_sha256'] == self.digests[path.name]
        batch_index = np.flatnonzero(data['row_offsets'] == record['row_offset'])
        # Empty failed stop traces can share a row offset; positions are unique.
        batch_index = [i for i in batch_index if data['position_offsets'][i] == record['position_offset']]
        assert len(batch_index) == 1
        i = batch_index[0]
        assert data['branch_ids'][i] == record['branch_id']
        assert data['row_counts'][i] == record['row_count']
        assert data['position_counts'][i] == record['position_count']
        rows = data['rows'][record['row_offset']:record['row_offset']+record['row_count']]
        pp = data['positions'][record['position_offset']:record['position_offset']+record['position_count']]
        assert len(pp) == len(rows)+1
        assert np.array_equal(pp[1:], rows[:, :2])
        return rows, pp


def witness(corridor, state):
    from geometry_spec import GateTracker
    tracker = GateTracker(corridor, corridor.reference[0])
    for key, value in state.items():
        setattr(tracker, key, copy.deepcopy(value))
    return tracker


def tracker_state(tracker):
    return {key: value for key, value in vars(tracker).items()
            if key != 'corridor' and not callable(value)}


def audit_case(row, reg, teacher):
    case = row['id']
    base, pred = OUT/case, OUT/case/'predictions'
    actual = read(base/'actual.json')
    assert sha(base/'actual.npz') == actual['trace_sha256']
    assert actual['registration_sha256'] == sha(OUT/'registration.json')
    with np.load(base/'actual.npz') as data:
        arrays = {key: data[key].copy() for key in data.files}
    reference = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
    assert sha(ROOT/row['reference']) == row['reference_sha256']
    assert np.array_equal(arrays['reference'], reference)
    positions, rows = arrays['positions'], arrays['rows']
    assert len(positions) == len(rows)+1
    assert np.array_equal(positions[1:], rows[:, :2]) and np.array_equal(positions[0], reference[0])
    install(dt=DT, tolerance=row['evaluation_tolerance_mm'])
    from geometry_spec import Corridor, GateTracker
    corridor = Corridor(reference, epsilon=row['evaluation_tolerance_mm'],
                        closed=bool(np.linalg.norm(reference[0]-reference[-1]) < 1e-9))
    independent = metrics(reference, positions, DT, row['evaluation_tolerance_mm'],
                          declared_success=actual['success'], reason=actual['reason'])
    for key in ('success', 'complete_time_s', 'elapsed_prefix_s', 'vector_peaks',
                'rest_pass', 'terminal_rest_s', 'geometry', 'max_polyline_deviation_mm'):
        assert independent[key] == actual[key], (case, 'actual_metrics', key)
    assert actual['intervals_accounted']
    assert actual['success'] or actual['complete_time_s'] is None
    decisions = jsonl(pred/'decisions.jsonl')
    branches = jsonl(pred/'branch_records.jsonl')
    stops = jsonl(pred/'stops/branch_records.jsonl')
    states = states_at(pred/'decision_states.pkl')
    assert len(decisions) == actual['attempted_decisions'] == len(arrays['observations'])
    assert len(states) in (len(decisions), len(decisions)-1)
    if len(states) < len(decisions):
        assert decisions[-1]['source'] == 'aborted' and not decisions[-1].get('branch_ids')
    branch_arrays, stop_arrays = SavedArrays(pred), SavedArrays(pred/'stops')
    by_id, branch_rows, branch_positions, end_trackers, end_rests = {}, {}, {}, {}, {}
    total_branch_intervals = 0
    for index, b in enumerate(branches):
        assert b['branch_id'] == index
        state = states[b['decision_id']]
        rr, pp = branch_arrays.get(b)
        assert np.array_equal(pp[0], state['core']['position'])
        peaks, vector, rest_ticks = kinematics(pp, state['core']['velocity'],
                                             state['core']['acceleration'], state['core']['rest_ticks'])
        assert peaks == b['vector_peaks'] and vector == b['vector_pass']
        tracker = witness(corridor, state['tracker'])
        for point in pp[1:]:
            tracker.update(point)
        assert tracker.inside == b['every_interval_inside']
        assert tracker.frontier == b['final_gate_frontier']
        assert b['accounted'] and b['independent_tracker_matches']
        terminal = bool(b['executor_reason'] == 'success' and
                        tracker.snapshot(endpoint_tolerance=corridor.epsilon)['geometry_complete'] and
                        rest_ticks*DT >= .02-1e-10)
        full = bool(b['executor_reason'] is None and len(rr) == 20)
        assert terminal == b['terminal_complete'] and full == b['full_control_period']
        basic_feasible = bool(b['error'] is None and vector and tracker.inside and (terminal or full))
        if 'stopping_extension' in b:
            assert basic_feasible and not terminal
            expected = b['stopping_extension']['feasible']
        else:
            expected = basic_feasible
        assert b['feasible'] == expected
        raw = np.asarray(decisions[b['decision_id']]['raw_action'])
        order = b['candidate_order']
        expected_action = raw if order == -1 else first_action(raw, OFFSETS[order//5], FACTORS[order % 5])
        assert np.array_equal(expected_action, b['action'])
        assert change_cost(raw, expected_action) == b['action_change_cost']
        segment = min(tracker.frontier, len(corridor.lengths)-1)
        distance = float((pp[-1]-corridor.starts[segment])@corridor.tangents[segment])
        progress = float(corridor.arcs[segment]+np.clip(distance, 0., corridor.lengths[segment]))
        assert progress == b['final_ordered_progress_mm']
        assert len(rr)*DT == b['elapsed_prediction_s'] and b['wall_s'] >= 0.
        total_branch_intervals += len(rr)
        by_id[index], branch_rows[index], branch_positions[index] = b, rr, pp
        end_trackers[index], end_rests[index] = tracker_state(tracker), rest_ticks

    stop_by_key, stop_traces = {}, {}
    stop_intervals, feasible_stops = 0, 0
    for stop in stops:
        key = (stop['decision_id'], stop['branch_id'])
        assert key not in stop_by_key
        stop_by_key[key] = stop
        rr, pp = stop_arrays.get(stop)
        stop_traces[key] = (rr, pp)
        state = states[stop['decision_id']]
        if stop['kind'] == 'initial_backup':
            assert stop['branch_id'] == -1
            initial_tracker, previous_rest = state['tracker'], state['core']['rest_ticks']
            p0, v0, a0 = (state['core'][k] for k in ('position', 'velocity', 'acceleration'))
        else:
            assert stop['kind'] == 'predicted_endpoint_backup'
            b = by_id[stop['branch_id']]
            assert b['decision_id'] == stop['decision_id']
            assert b['stopping_extension']['feasible'] == stop['feasible']
            initial_tracker, previous_rest = end_trackers[stop['branch_id']], end_rests[stop['branch_id']]
            tail = branch_rows[stop['branch_id']][-1]
            p0, v0, a0 = tail[:2], tail[2:4], tail[4:6]
        assert np.array_equal(pp[0], p0)
        assert np.array_equal(stop['initial_position_mm'], p0)
        assert np.array_equal(stop['initial_velocity_mm_s'], v0)
        assert np.array_equal(stop['initial_acceleration_mm_s2'], a0)
        tracker = witness(corridor, initial_tracker)
        assert tracker.frontier == stop['initial_frontier']
        first_bad = None
        for i, point in enumerate(pp[1:], 1):
            tracker.update(point)
            if first_bad is None and not tracker.inside:
                first_bad = i
        assert first_bad == stop['first_outside_tick']
        assert tracker.inside == stop['every_interval_inside'] and tracker.frontier == stop['final_frontier']
        gate = corridor.gates[tracker.frontier] if tracker.frontier < len(corridor.gates) else None
        signed = None if gate is None else float((pp[-1]-gate.point)@gate.normal)
        forward = bool(signed is None or signed <= 1e-9)
        assert signed == stop['next_unvisited_gate_signed_mm']
        assert forward == stop['next_unvisited_gate_not_behind_stop']
        peaks, vector, rest_ticks = kinematics(pp, v0, a0, previous_rest)
        assert peaks == stop['vector_peaks'] and vector == stop['vector_pass']
        assert rest_ticks*DT == stop['terminal_rest_s']
        feasible = bool(stop['error'] is None and len(rr) > 0 and tracker.inside and forward and
                        vector and rest_ticks*DT >= .02-1e-10)
        assert feasible == stop['feasible']
        if len(rr):
            # Reconstruct every stored command from its initial physical state.
            p, old_v, old_a = p0.copy(), v0.copy(), a0.copy()
            for expected in rr:
                velocity = expected[2:4]
                acceleration = (velocity-old_v)/DT
                jerk = (acceleration-old_a)/DT
                p = p+DT*velocity
                assert np.array_equal(expected, np.r_[p, velocity, acceleration, jerk, velocity, 0., 0., 1.])
                old_v, old_a = velocity, acceleration
        else:
            assert stop['error'] is not None and not feasible
        assert stop['assessment_wall_s'] >= 0. and stop['assessment_cpu_s'] >= 0.
        stop_intervals += len(rr)
        feasible_stops += int(feasible)

    actual_tracker = GateTracker(corridor, reference[0])
    backup, cursor, submitted = None, 0, 0
    source_counts = Counter()
    source_intervals = Counter()
    aborted, completed = [], []
    for index, d in enumerate(decisions):
        assert d['decision_id'] == index and d['actual_start_tick'] == cursor
        state = states[index] if index < len(states) else None
        if state is not None:
            assert state['core']['ticks'] == cursor
            assert np.array_equal(state['core']['position'], positions[cursor])
            if cursor:
                assert np.array_equal(state['core']['velocity'], rows[cursor-1, 2:4])
                assert np.array_equal(state['core']['acceleration'], rows[cursor-1, 4:6])
            assert pickle.dumps(tracker_state(actual_tracker), protocol=5) == pickle.dumps(state['tracker'], protocol=5)
            assert teacher.mean(arrays['observations'][index]).numpy().tolist() == d['raw_action']
        group = [by_id[k] for k in d.get('branch_ids', [])]
        assert all(b['decision_id'] == index for b in group)
        if group:
            assert group[0]['candidate_order'] == -1
            orders = [-1]+[i for i in range(25) if i != 14]
            assert [b['candidate_order'] for b in group] == orders[:len(group)]
        end = d['actual_end_tick']
        assert cursor <= end <= len(rows)
        segment = rows[cursor:end]
        if d['source'] == 'aborted':
            assert d['error'] and len(segment) == d['retained_partial_rows']
            assert index == len(decisions)-1 and not actual['success']
            assert np.all(arrays['command_sources'][cursor:end] == 'aborted_actual_decision')
            aborted.append(d)
        else:
            assert d['real_state_and_recorder_unchanged'] and d['actual_rows'] == len(segment)
            if backup is None:
                backup = (index, -1)
                assert stop_by_key[backup]['feasible']
            raw = group[0]
            if d['source'] == 'raw_model':
                assert len(group) == 1 and raw['feasible']
                chosen = raw
            else:
                assert len(group) == 25 and not raw['feasible']
                valid = [b for b in group[1:] if b['feasible']]
                if d['source'] == 'corrected_model':
                    assert valid
                    chosen = min(valid, key=lambda b: (-b['final_ordered_progress_mm'],
                                 b['action_change_cost'], b['candidate_order']))
                else:
                    assert d['source'] == 'certified_stop' and not valid
                    chosen = None
            if chosen is None:
                assert d['submitted_action'] is None and d['selected_branch_id'] is None
                expected = stop_traces[backup][0]
                assert np.array_equal(segment, expected[:len(segment)])
                assert len(segment) == len(expected) or actual['reason'] == 'task_timeout'
                backup = None
            else:
                assert d['selected_branch_id'] == chosen['branch_id']
                assert d['submitted_action'] == chosen['action']
                assert arrays['action_decision_ids'][submitted] == index
                assert np.array_equal(arrays['actions'][submitted], d['submitted_action'])
                assert np.array_equal(segment, branch_rows[chosen['branch_id']])
                submitted += 1
                backup = None if chosen['terminal_complete'] else (index, chosen['branch_id'])
                if backup is not None:
                    assert stop_by_key[backup]['feasible']
            assert np.all(arrays['command_sources'][cursor:end] == d['source'])
            assert d['total_wall_s'] >= d['planning_wall_s'] >= sum(b['wall_s'] for b in group)
            completed.append(d)
            source_counts[d['source']] += 1
            source_intervals[d['source']] += len(segment)
        for point in positions[cursor+1:end+1]:
            actual_tracker.update(point)
        cursor = end
    assert cursor == len(rows) == len(arrays['command_sources'])
    assert submitted == len(arrays['actions']) == len(arrays['action_decision_ids'])
    counts = actual['controller']['counts']
    assert len(completed) == actual['completed_decisions'] == actual['controller']['decisions']
    assert len(branches) == counts['branches'] and len(stops) == counts['stopping_assessments']
    for key in ('raw_model', 'corrected_model', 'certified_stop'):
        assert source_counts[key] == counts[key]
    assert counts['actual_prediction_matches'] == submitted
    assert counts['isolation_checks'] == len(completed)
    total_decision_wall = sum(d['total_wall_s'] for d in completed)
    total_branch_wall = sum(b['wall_s'] for b in branches)
    total_stop_wall = sum(s['assessment_wall_s'] for s in stops)
    assert actual['cost']['full_evaluation_wall_s'] >= total_decision_wall
    return dict(case=case, status='passed', success=actual['success'], complete_time_s=actual['complete_time_s'],
        failed_prefix_s=None if actual['success'] else actual['elapsed_prefix_s'], reason=actual['reason'],
        actual_intervals=len(rows), decisions=len(decisions), completed_decisions=len(completed),
        branches=len(branches), branch_intervals=total_branch_intervals,
        feasible_branches=sum(b['feasible'] for b in branches), stopping_assessments=len(stops),
        stop_intervals=stop_intervals, feasible_stops=feasible_stops,
        source_decisions=dict(source_counts), source_intervals=dict(source_intervals),
        submitted_action_rows=submitted, aborted_decisions=aborted,
        all_actual_and_speculative_intervals_independently_audited=True,
        all_actual_segments_match_selected_commands=True, all_actions_and_sources_verified=True,
        stopping_cache_chain_verified=True, all_failures_retained=True,
        cost=dict(recorded_full_evaluation_wall_s=actual['cost']['full_evaluation_wall_s'],
            recorded_completed_decision_wall_s=total_decision_wall,
            recorded_completed_planning_wall_s=sum(d['planning_wall_s'] for d in completed),
            recorded_all_branch_wall_s=total_branch_wall,
            recorded_all_stop_wall_s=total_stop_wall,
            recorded_all_stop_cpu_s=sum(s['assessment_cpu_s'] for s in stops),
            full_process_cpu_recorded=False,
            accounting_note='Branch wall includes its endpoint stopping assessment. Do not add nested costs. Full evaluation includes initialization, recording and final metrics. Aborted decision total time is not separately logged; completed branch and stop costs plus full evaluation are retained.'),
        file_hashes={p.relative_to(base).as_posix(): sha(p) for p in base.rglob('*') if p.is_file()})


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    reg = read(OUT/'registration.json')
    assert all((OUT/r['id']/'actual.json').exists() for r in reg['cases']), 'Wait for existing cases; no evaluation is launched'
    assert all(sha(ROOT/'src'/key) == value for key, value in reg['source_sha256'].items())
    assert sha(MODEL) == reg['model_sha256']
    unit = read(OUT/'unit_audit/report.json')
    assert unit['status'] == 'passed' and unit['registered_sources'] == reg['source_sha256']
    assert unit['stale_backup_rejected_before_integration']
    teacher = FrozenPolicy(MODEL)
    cases = []
    for row in reg['cases']:
        print('AUDIT_CASE', row['id'], flush=True)
        cases.append(audit_case(row, reg, teacher))
    summary = read(OUT/'summary.json')
    assert summary['runs'] == len(cases) and summary['successes'] == sum(c['success'] for c in cases)
    report = dict(status='passed', created_utc=utc(), scope='Saved evidence only; no new path, rollout, training or holdout use',
        registration_sha256=sha(OUT/'registration.json'), audit_source_sha256=sha(__file__),
        source_and_model_hashes_match_registration=True, unit_audit_sha256=sha(OUT/'unit_audit/report.json'),
        static_review=dict(action_source_fix_verified=True, preintegration_full_state_fingerprint_verified=True,
            stop_array_length_precheck_verified=True, terminal_registration_20ms_verified=True,
            code_level_completion_or_geometry_relaxation_found=False),
        cases=cases, successes=sum(c['success'] for c in cases), runs=len(cases),
        actual_intervals=sum(c['actual_intervals'] for c in cases),
        branches=sum(c['branches'] for c in cases), branch_intervals=sum(c['branch_intervals'] for c in cases),
        stopping_assessments=sum(c['stopping_assessments'] for c in cases),
        stop_intervals=sum(c['stop_intervals'] for c in cases),
        audit_wall_s=time.perf_counter()-started,
        qualification='This audit verifies saved commands and reported outcomes. It does not establish recursive completeness, hard real-time feasibility, independent-test generalization or five-seed training evidence.',
        cost_gap='Full process CPU and individual aborted-decision wall time were not recorded. Existing nested wall and stopping CPU records are retained; no timing values are inferred.')
    assert not REPORT.exists(), 'Do not overwrite a prior independent review'
    write_json(REPORT, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
