"""Independent six-decision predictive candidate; never the historical main.

The policy preserves a feasible raw model action exactly. A failed prediction
triggers the registered 15 first-action alternatives; all later actions come
from the unchanged model. Every actual and speculative outcome is retained.
"""
import copy
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
from common import sha
from frozen_policy import FrozenPolicy
from physical_teacher_v5 import BranchArchive

VERSION = 'predictive_action_candidate_v5_1'
OFFSETS = [-.08, -.04, 0., .04, .08]
FACTORS = [.6, .8, 1.]
DECISIONS = 6
DT = .001
LIMITS = np.array([100., 2000., 62500.])
EXCLUDED_CORE = {'corridor', 'tracker', 'experimental_projector'}


def snapshot(env):
    """Copy values, never bound methods, construction closures, or recorders."""
    core = {k: copy.deepcopy(v) for k, v in vars(env.env).items()
            if k not in EXCLUDED_CORE and not callable(v)}
    tracker = {k: copy.deepcopy(v) for k, v in vars(env.tracker).items()
               if k != 'corridor' and not callable(v)}
    wrapper = {k: copy.deepcopy(v) for k, v in vars(env).items()
               if k != 'env' and not callable(v)}
    assert 'max_ticks' in core and 'fixed_d' in core
    return dict(core=core, tracker=tracker, wrapper=wrapper,
                projector_counts=copy.deepcopy(env.experimental_projector.counts))


def restore(env, state):
    for obj, values, excluded in ((env.env, state['core'], EXCLUDED_CORE),
            (env.tracker, state['tracker'], {'corridor'}),
            (env, state['wrapper'], {'env'})):
        for key in list(vars(obj)):
            if key not in excluded and not callable(vars(obj)[key]) and key not in values:
                delattr(obj, key)
        for key, value in values.items():
            setattr(obj, key, copy.deepcopy(value))
    # Recorder closures are deliberately absent from the copied tracker.
    if 'update' in vars(env.tracker):
        delattr(env.tracker, 'update')
    env.experimental_projector.counts = copy.deepcopy(state['projector_counts'])


def state_bytes(env):
    return pickle.dumps(snapshot(env), protocol=5)


def recorder_lists(function, seen=None):
    """Read-only sizes of list objects captured by an evaluation recorder."""
    seen = set() if seen is None else seen
    if id(function) in seen:
        return []
    seen.add(id(function))
    found = []
    for cell in getattr(function, '__closure__', ()) or ():
        value = cell.cell_contents
        if isinstance(value, list):
            found.append((id(value), len(value)))
        elif callable(value):
            found.extend(recorder_lists(value, seen))
    return sorted(set(found))


def new_environment(reference, state=None):
    from compact_farvector_runtime import FarVectorRuntime
    env = FarVectorRuntime(reference, potential_speed=100.)
    if state is not None:
        restore(env, state)
    return env


def physical(action):
    bounded = np.clip(np.asarray(action, float), -1.+1e-10, 1.-1e-10)
    return np.array([bounded[0]*math.pi/2, 50.*(1.+bounded[1]),
                     6.4+5.6*bounded[2]])


def first_action(raw, offset, factor):
    value = physical(raw)
    candidate = np.asarray(raw, float).copy()
    candidate[0] = np.clip((value[0]+offset)/(math.pi/2), -1.+1e-10, 1.-1e-10)
    candidate[1] = value[1]*factor/50.-1.
    # Preserve the original raw preview, even if outside the decoder interval.
    candidate[2] = raw[2]
    if offset == 0. and factor == 1.:
        return raw.copy()
    return candidate


def change_cost(raw, action):
    delta = (physical(action)-physical(raw))/np.array([math.pi/2, 100., 11.2])
    return float(delta@delta)


def retained_step(env, action):
    error = None
    rows = np.empty((0, 13))
    try:
        _, _, _, info = env.step(action)
        rows = info['rows']
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        frame = exc.__traceback__
        while frame is not None:
            if frame.tb_frame.f_code.co_name == 'step' and frame.tb_frame.f_code.co_filename.endswith('joint_runtime.py'):
                found = frame.tb_frame.f_locals.get('rows', [])
                if len(found):
                    rows = np.asarray(found)
            frame = frame.tb_next
    return rows, error


def prediction(clone, state, action, teacher, deadline=None):
    """Use six genuine feedback decisions and an independent interval witness."""
    started, cpu = time.perf_counter(), time.process_time()
    restore(clone, state)
    positions = [clone.position.copy()]
    tracker = clone.tracker
    def record(point):
        positions.append(np.asarray(point).copy())
        return type(tracker).update(tracker, point)
    tracker.update = record
    actions, observations, chunks = [], [], []
    error = None
    model_calls = 0
    for index in range(DECISIONS):
        if deadline is not None and time.perf_counter() >= deadline:
            error = 'TimeoutError: registered complete-run wall budget during prediction'
            break
        obs = clone.observation().copy()
        chosen = action.copy() if index == 0 else teacher.mean(obs).numpy()
        model_calls += int(index > 0)
        observations.append(obs)
        actions.append(chosen.copy())
        rows, error = retained_step(clone, chosen)
        chunks.append(rows)
        if error is not None or clone.done:
            break
    rows = np.vstack(chunks) if chunks else np.empty((0, 13))
    positions = np.asarray(positions)
    accounted = (len(positions)-1 == len(rows) == clone.ticks-state['core']['ticks'])
    if accounted and len(rows):
        accounted = bool(np.array_equal(rows[:, :2], positions[1:]))
    velocity = np.diff(positions, axis=0)/DT
    acceleration = np.diff(np.vstack([state['core']['velocity'], velocity]), axis=0)/DT
    jerk = np.diff(np.vstack([state['core']['acceleration'], acceleration]), axis=0)/DT
    peaks = [float(np.linalg.norm(x, axis=1).max(initial=0.))
             for x in (velocity, acceleration, jerk)]
    vector_pass = bool(np.all(np.asarray(peaks)/LIMITS <= 1.+2e-5))
    from geometry_spec import GateTracker
    witness = GateTracker(clone.corridor, clone.corridor.reference[0])
    for key, value in state['tracker'].items():
        setattr(witness, key, copy.deepcopy(value))
    for point in positions[1:]:
        type(witness).update(witness, point)
    same_tracker = (pickle.dumps({k:v for k,v in vars(witness).items() if k!='corridor'},protocol=5) ==
                    pickle.dumps({k:v for k,v in vars(tracker).items() if k not in ('corridor','update')},protocol=5))
    geometry = witness.snapshot(endpoint_tolerance=clone.corridor.epsilon)
    terminal_valid = bool(clone.done and clone.success and geometry['geometry_complete'] and
        clone.rest_ticks*DT >= .02-1e-10 and np.linalg.norm(clone.velocity) <= .001 and
        np.linalg.norm(clone.acceleration) <= .01)
    full_horizon = bool(len(actions) == DECISIONS and not clone.done and
                        clone.ticks-state['core']['ticks'] == round(.12/DT))
    feasible = bool(error is None and accounted and same_tracker and vector_pass and
                    witness.inside and (terminal_valid or full_horizon))
    if not accounted:
        error = error or 'TraceAccountingError: speculative endpoints/rows/ticks differ'
    if not same_tracker:
        error = error or 'GeometryWitnessError: independent tracker differs'
    outcome = dict(feasible=feasible, error=error, decisions_executed=len(actions),
        model_calls=model_calls, elapsed_prediction_s=(clone.ticks-state['core']['ticks'])*DT,
        nominal_prediction_s=.12, terminal_tail=bool(clone.brake is not None),
        terminal_absorbing_rest=terminal_valid, full_six_decision_horizon=full_horizon,
        done=bool(clone.done), success=bool(clone.success), executor_reason=clone.reason,
        accounted=accounted, independent_tracker_matches=same_tracker,
        every_interval_inside=bool(witness.inside), vector_pass=vector_pass, vector_peaks=peaks,
        initial_gate_frontier=state['tracker']['frontier'], final_gate_frontier=witness.frontier,
        final_ordered_progress_mm=float(clone.progress()[0]),
        initial_position_mm=state['core']['position'].tolist(),
        initial_velocity_mm_s=state['core']['velocity'].tolist(),
        initial_acceleration_mm_s2=state['core']['acceleration'].tolist(),
        predicted_actions=[a.tolist() for a in actions],
        predicted_observations=[o.tolist() for o in observations],
        branch_wall_s=time.perf_counter()-started, branch_cpu_s=time.process_time()-cpu)
    return rows, positions, outcome


class PredictiveActionCandidate:
    def __init__(self, model, output, *, deadline=None):
        self.teacher = FrozenPolicy(model)
        self.model_sha256 = sha(model)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.archive = BranchArchive(self.output)
        self.decision_log = (self.output/'decisions.jsonl').open('x', encoding='utf-8')
        self.state_log = (self.output/'decision_states.pkl').open('xb')
        self.deadline = deadline
        self.clone = None
        self.decisions = 0
        self.branches = 0
        self.counts = dict(raw_model=0, predictive_correction=0, no_feasible_candidate=0,
                           aborted=0, isolation_checks=0, branch_model_calls=0,
                           branch_ticks=0, failed_branches=0)
        self.cost = dict(branch_wall_s=0., branch_cpu_s=0., total_action_wall_s=0.,
                         total_action_cpu_s=0., clone_initialization_wall_s=0.)

    def action_for(self, env):
        started, cpu = time.perf_counter(), time.process_time()
        state = snapshot(env)
        before = pickle.dumps(state, protocol=5)
        recorder = env.tracker.update
        list_sizes = recorder_lists(recorder)
        record = dict(decision_id=self.decisions, actual_start_tick=env.ticks,
                      actual_start_progress_mm=float(env.progress()[0]),
                      max_ticks=env.max_ticks, fixed_d=env.fixed_d,
                      raw_model_action=None, submitted_action=None, source='aborted',
                      selected_branch_id=None, branch_ids=[], error=None)
        pickle.dump(state, self.state_log, protocol=5)
        try:
            raw = self.teacher.mean(env.observation()).numpy()
            record['raw_model_action'] = raw.tolist()
            if self.clone is None:
                init = time.perf_counter()
                self.clone = new_environment(env.corridor.reference, state)
                self.cost['clone_initialization_wall_s'] += time.perf_counter()-init
                assert self.clone.env is not env.env and self.clone.tracker is not env.tracker
                assert self.clone.env.step is not env.env.step
            def evaluate(action, kind, order, offset=None, factor=None):
                rows, positions, outcome = prediction(self.clone, state, action, self.teacher, self.deadline)
                outcome.update(branch_id=self.branches, decision_id=self.decisions,
                    kind=kind, candidate_order=order, first_action=action.tolist(),
                    steering_offset_rad=offset, feed_factor=factor,
                    action_change_cost=change_cost(raw, action))
                self.archive.append(outcome, rows, positions)
                record['branch_ids'].append(self.branches)
                self.branches += 1
                self.counts['branch_model_calls'] += outcome['model_calls']
                self.counts['branch_ticks'] += len(rows)
                self.counts['failed_branches'] += int(not outcome['feasible'])
                self.cost['branch_wall_s'] += outcome['branch_wall_s']
                self.cost['branch_cpu_s'] += outcome['branch_cpu_s']
                if outcome['error'] and outcome['error'].startswith('TimeoutError:'):
                    raise TimeoutError('Registered complete-run wall budget during prediction')
                return action, outcome
            _, base = evaluate(raw, 'original_model_prediction', -1)
            if base['feasible']:
                submitted, selected, source = raw.copy(), base, 'raw_model'
            else:
                valid = []
                for offset in OFFSETS:
                    for factor in FACTORS:
                        order = OFFSETS.index(offset)*len(FACTORS)+FACTORS.index(factor)
                        chosen, outcome = evaluate(first_action(raw, offset, factor),
                                                   'candidate_first_action', order, offset, factor)
                        if outcome['feasible']:
                            valid.append((chosen, outcome))
                if valid:
                    submitted, selected = min(valid, key=lambda pair:(
                        -pair[1]['final_ordered_progress_mm'], pair[1]['action_change_cost'],
                        pair[1]['candidate_order']))
                    source = 'predictive_correction'
                else:
                    submitted, selected, source = raw.copy(), base, 'no_feasible_candidate'
            record.update(submitted_action=submitted.tolist(), source=source,
                          selected_branch_id=selected['branch_id'])
            self.counts[source] += 1
            return submitted
        except Exception as exc:
            record['error'] = type(exc).__name__+': '+str(exc)
            self.counts['aborted'] += 1
            raise
        finally:
            unchanged = (before == state_bytes(env) and recorder is env.tracker.update and
                         list_sizes == recorder_lists(env.tracker.update))
            record.update(real_state_and_recorder_unchanged=unchanged,
                          recorder_list_sizes=list_sizes,
                          wall_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu)
            self.cost['total_action_wall_s'] += record['wall_s']
            self.cost['total_action_cpu_s'] += record['cpu_s']
            self.counts['isolation_checks'] += int(unchanged)
            self.decision_log.write(json.dumps(record, allow_nan=False)+'\n')
            self.decision_log.flush()
            self.state_log.flush()
            self.decisions += 1
            if not unchanged:
                raise AssertionError('Predictive clone mutated actual state or recorder')

    def close(self):
        self.archive.close()
        self.decision_log.close()
        self.state_log.close()
        return dict(version=VERSION, decisions=self.decisions, branches=self.branches,
            counts=self.counts, cost=self.cost, model_sha256=self.model_sha256,
            interpretation='New development-only predictive execution candidate; not historical main, training evidence, final test or hard-real-time validation')
