"""Development controller with a retained, executable stopping continuation.

One genuine feedback decision is accepted only with a complete feasible stop.
The prior accepted stop is executed when the fixed action set has no solution.
No historical controller or evaluator is modified by this module.
"""
import copy
import hashlib
import json
import pickle
import time
import numpy as np
from common import sha
from frozen_policy import FrozenPolicy
from physical_teacher_v5 import BranchArchive
from braking_reachability_v5 import stopping_extension
from predictive_action_candidate_v5 import (snapshot, restore, state_bytes,
    recorder_lists, new_environment, retained_step, first_action, change_cost)

VERSION = 'recoverable_action_candidate_v5_3'
OFFSETS = [-.08, -.04, 0., .04, .08]
FACTORS = [.2, .4, .6, .8, 1.]
DT = .001


def one_decision(clone, state, action):
    from geometry_spec import GateTracker
    restore(clone, state)
    positions = [clone.position.copy()]
    def record(point):
        positions.append(np.asarray(point).copy())
        return type(clone.tracker).update(clone.tracker, point)
    clone.tracker.update = record
    rows, error = retained_step(clone, action)
    positions = np.asarray(positions)
    accounted = len(rows) == len(positions)-1 == clone.ticks-state['core']['ticks']
    accounted = accounted and (not len(rows) or np.array_equal(rows[:, :2], positions[1:]))
    witness = GateTracker(clone.corridor, clone.corridor.reference[0])
    for key, value in state['tracker'].items():
        setattr(witness, key, copy.deepcopy(value))
    for point in positions[1:]:
        witness.update(point)
    tracker_equal = all(pickle.dumps(getattr(witness, key), protocol=5) ==
        pickle.dumps(value, protocol=5) for key, value in vars(clone.tracker).items()
        if key not in ('corridor', 'update'))
    v = np.diff(positions, axis=0)/DT
    a = np.diff(np.vstack([state['core']['velocity'], v]), axis=0)/DT
    j = np.diff(np.vstack([state['core']['acceleration'], a]), axis=0)/DT
    peaks = [float(np.linalg.norm(x, axis=1).max(initial=0.)) for x in (v, a, j)]
    vector = bool(np.all(np.asarray(peaks)/[100., 2000., 62500.] <= 1.00002))
    terminal = bool(clone.done and clone.success and
        witness.snapshot(endpoint_tolerance=clone.corridor.epsilon)['geometry_complete'] and
        clone.rest_ticks*DT >= .02-1e-10)
    full = bool(not clone.done and len(rows) == round(.02/DT))
    feasible = bool(error is None and accounted and tracker_equal and vector and
                    witness.inside and (terminal or full))
    result = dict(feasible=feasible, error=error, accounted=accounted,
        independent_tracker_matches=tracker_equal, vector_pass=vector, vector_peaks=peaks,
        every_interval_inside=bool(witness.inside), terminal_complete=terminal,
        full_control_period=full, elapsed_prediction_s=len(rows)*DT,
        final_ordered_progress_mm=float(clone.progress()[0]),
        final_gate_frontier=witness.frontier, executor_reason=clone.reason)
    return rows, positions, result


def stop_start_fingerprint(env):
    state = snapshot(env)
    keys = ('position', 'velocity', 'acceleration', 'heading', 'held', 'ticks',
            'rest_ticks', 'brake', 'brake_index', 'fixed_d', 'max_ticks',
            'done', 'success', 'pending_pulse')
    values = {key: state['core'][key] for key in keys}
    return hashlib.sha256(pickle.dumps((values, state['tracker']), protocol=5)).hexdigest()


def execute_stop(env, positions, rows, start_fingerprint, retained_rows=None):
    """Execute every stored command, preserving all gates and ordinary rest rules."""
    if len(positions) != len(rows)+1 or len(rows) == 0:
        raise AssertionError('Stored stop has inconsistent command intervals')
    if stop_start_fingerprint(env) != start_fingerprint or not np.array_equal(env.position, positions[0]):
        raise AssertionError('Stored stop starts at a different real physical or ordered state')
    core = env.env
    core.decisions += 1
    core.pending_pulse = np.zeros(3)
    core._executed = None
    executed = [] if retained_rows is None else retained_rows
    for expected_position, expected in zip(positions[1:], rows):
        old_v, old_a = core.velocity.copy(), core.acceleration.copy()
        velocity = expected[2:4].copy()
        acceleration = (velocity-old_v)/DT
        jerk = (acceleration-old_a)/DT
        core.position = core.position+DT*velocity
        core.velocity, core.acceleration = velocity, acceleration
        core.tracker.update(core.position)
        core.ticks += 1
        actual = np.r_[core.position, velocity, acceleration, jerk, velocity, 0., 0., 1.]
        executed.append(actual)
        if not np.array_equal(core.position, expected_position) or not np.array_equal(actual, expected):
            raise AssertionError('Executed stop differs from the stored command certificate')
        rest = np.linalg.norm(velocity) <= .001 and np.linalg.norm(acceleration) <= .01
        core.rest_ticks = core.rest_ticks+1 if rest else 0
        if not core.tracker.inside:
            core.done, core.reason = True, 'corridor_violation'
        elif max(np.linalg.norm(velocity)/100., np.linalg.norm(acceleration)/2000.,
                 np.linalg.norm(jerk)/62500.) > 1.00001:
            core.done, core.reason = True, 'kinematic_violation'
        elif core.ticks >= core.max_ticks:
            core.done, core.reason = True, 'task_timeout'
        if core.done:
            break
    if not core.done and core.rest_ticks*DT >= .02-1e-10 and \
            core.tracker.snapshot(endpoint_tolerance=1.)['geometry_complete']:
        core.done, core.success, core.reason = True, True, 'success'
    return np.asarray(executed)


class RecoverableActionCandidate:
    def __init__(self, model, output, deadline=None):
        self.teacher = FrozenPolicy(model)
        self.model_sha256 = sha(model)
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.archive = BranchArchive(output)
        stop_dir = output/'stops'
        stop_dir.mkdir()
        self.stops = BranchArchive(stop_dir)
        self.log = (output/'decisions.jsonl').open('x', encoding='utf-8')
        self.states = (output/'decision_states.pkl').open('xb')
        self.deadline = deadline
        self.clone = None
        self.backup = None
        self.branch_id = 0
        self.decisions = 0
        self.actual_partial_rows = []
        self.active_record = None
        self.counts = dict(raw_model=0, corrected_model=0, certified_stop=0,
                           branches=0, stopping_assessments=0, isolation_checks=0,
                           actual_prediction_matches=0)

    def step(self, env):
        self.actual_partial_rows = []
        self.active_record = dict(decision_id=self.decisions, actual_start_tick=env.ticks)
        try:
            return self._step(env)
        except Exception as exc:
            self.active_record.update(error=type(exc).__name__+': '+str(exc),
                source='aborted', actual_end_tick=env.ticks,
                retained_partial_rows=len(self.actual_partial_rows))
            self.log.write(json.dumps(self.active_record, allow_nan=False)+'\n')
            self.log.flush()
            self.states.flush()
            raise

    def _step(self, env):
        started = time.perf_counter()
        if self.deadline is not None and started >= self.deadline:
            raise TimeoutError('Registered recovery-controller wall budget')
        state = snapshot(env)
        before = state_bytes(env)
        recorder = env.tracker.update
        list_sizes = recorder_lists(recorder)
        pickle.dump(state, self.states, protocol=5)
        raw = self.teacher.mean(env.observation()).numpy()
        if self.clone is None:
            self.clone = new_environment(env.corridor.reference, state)
        if self.backup is None:
            sr, sp, stop = stopping_extension(env)
            stop.update(branch_id=-1, decision_id=self.decisions, kind='initial_backup')
            self.stops.append(stop, sr, sp)
            self.counts['stopping_assessments'] += 1
            if not stop['feasible']:
                raise RuntimeError('Initial state has no valid stopping continuation')
            self.backup = (sr, sp, stop_start_fingerprint(env))
        record = dict(decision_id=self.decisions, actual_start_tick=env.ticks,
            raw_action=raw.tolist(), source=None, selected_branch_id=None, branch_ids=[])
        self.active_record = record

        def evaluate(action, order):
            if self.deadline is not None and time.perf_counter() >= self.deadline:
                raise TimeoutError('Registered recovery-controller wall budget')
            branch_start = time.perf_counter()
            rows, positions, result = one_decision(self.clone, state, action)
            stop_trace = None
            if result['feasible'] and not result['terminal_complete']:
                sr, sp, stop = stopping_extension(self.clone)
                stop.update(branch_id=self.branch_id, decision_id=self.decisions,
                            kind='predicted_endpoint_backup')
                self.stops.append(stop, sr, sp)
                self.counts['stopping_assessments'] += 1
                result['feasible'] = bool(stop['feasible'])
                result['stopping_extension'] = stop.copy()
                if stop['feasible']:
                    stop_trace = (sr, sp, stop_start_fingerprint(self.clone))
            result.update(branch_id=self.branch_id, decision_id=self.decisions,
                action=action.tolist(), candidate_order=order,
                action_change_cost=change_cost(raw, action),
                wall_s=time.perf_counter()-branch_start)
            self.archive.append(result, rows, positions)
            record['branch_ids'].append(self.branch_id)
            self.branch_id += 1
            self.counts['branches'] += 1
            return action, rows, result, stop_trace

        selected = evaluate(raw, -1)
        source = 'raw_model'
        if not selected[2]['feasible']:
            valid = []
            for oi, offset in enumerate(OFFSETS):
                for fi, factor in enumerate(FACTORS):
                    if offset == 0. and factor == 1.:
                        continue
                    candidate = evaluate(first_action(raw, offset, factor), oi*len(FACTORS)+fi)
                    if candidate[2]['feasible']:
                        valid.append(candidate)
            if valid:
                selected = min(valid, key=lambda item:(-item[2]['final_ordered_progress_mm'],
                    item[2]['action_change_cost'], item[2]['candidate_order']))
                source = 'corrected_model'
            else:
                source = 'certified_stop'
        unchanged = before == state_bytes(env) and recorder is env.tracker.update and \
                    list_sizes == recorder_lists(recorder)
        if not unchanged:
            raise AssertionError('Speculative calculation changed the real environment')
        self.counts['isolation_checks'] += 1
        record['real_state_and_recorder_unchanged'] = unchanged
        planning_wall = time.perf_counter()-started
        if source == 'certified_stop':
            rows = execute_stop(env, self.backup[1], self.backup[0], self.backup[2], self.actual_partial_rows)
            action = None
            self.backup = None
        else:
            action, expected, result, backup = selected
            rows, error = retained_step(env, action)
            self.actual_partial_rows = rows
            if error is not None:
                raise RuntimeError(error)
            if not np.array_equal(rows, expected):
                raise AssertionError('Accepted one-step prediction differs from actual commands')
            self.counts['actual_prediction_matches'] += 1
            record['selected_branch_id'] = result['branch_id']
            self.backup = backup
        record.update(source=source, submitted_action=None if action is None else action.tolist(), actual_rows=len(rows),
            actual_end_tick=env.ticks, planning_wall_s=planning_wall,
            total_wall_s=time.perf_counter()-started)
        self.counts[source] += 1
        self.log.write(json.dumps(record, allow_nan=False)+'\n')
        self.log.flush()
        self.states.flush()
        self.decisions += 1
        return action, rows, record

    def close(self):
        self.archive.close()
        self.stops.close()
        self.log.close()
        self.states.close()
        return dict(version=VERSION, model_sha256=self.model_sha256,
                    decisions=self.decisions, counts=self.counts)
