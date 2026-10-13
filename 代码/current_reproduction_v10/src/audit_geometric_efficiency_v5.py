"""Audit every saved interval for the registered bounded efficiency comparison."""
import argparse
import math
import time
import types
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from predictive_action_candidate_v5 import new_environment
from replay_strong_checkpoint_v4 import MODEL
import audit_recoverable_candidate_v5 as base
import audit_geometric_resolution_v5 as resolution_audit
import geometric_resolution_check_v5 as resolution
import run_geometric_efficiency_v5 as experiment
from geometric_efficiency_candidates_v5 import CANDIDATES


def expected_geometric(spec, reference, state):
    env = new_environment(reference, state)
    frontier = state['tracker']['frontier']
    target = env.corridor.gates[frontier].point if frontier < len(env.corridor.gates) else reference[-1]
    direction = target-state['core']['position']
    bearing = math.atan2(direction[1], direction[0])
    old = env.env.held.copy()
    try:
        env.env.held = np.array([0., spec['feed_mm_s'], .8])
        ray = env.guidance_target()-env.position
        beta = math.atan2(ray[1], ray[0]) if np.linalg.norm(ray) > 1e-12 else env.heading
    finally:
        env.env.held = old
    error = (bearing-state['core']['heading']+math.pi) % (2*math.pi)-math.pi
    correction = (bearing-beta+math.pi) % (2*math.pi)-math.pi
    steering = float(np.clip(correction, -math.pi/2, math.pi/2))
    feed = 0. if abs(error) > spec['heading_tolerance_rad'] else spec['feed_mm_s']
    return np.array([steering/(math.pi/2), feed/50.-1., -1.])


def case_auditor(spec, dt, folder):
    resolution_audit.installed_constants(dt)
    original = resolution_audit.resolution_audit_case(dt, folder)
    namespace = dict(original.__globals__)
    namespace['independent_geometric_action'] = lambda ref, state: expected_geometric(spec, ref, state)
    return types.FunctionType(original.__code__, namespace, original.__name__,
                              original.__defaults__, original.__closure__)


def audit_candidate(spec, dt, teacher):
    folder = experiment.OUT/spec['id']/f'dt{round(dt*1e6)}us'
    reg = base.read(folder/'registration.json')
    assert reg['candidate'] == spec and reg['dt_s'] == dt
    assert all(sha(ROOT/'src'/name) == digest for name, digest in reg['source_sha256'].items())
    unit = base.read(folder/'unit_audit/report.json')
    assert unit['status'] == 'passed' and unit['registered_sources'] == reg['source_sha256']
    audit = case_auditor(spec, dt, folder)
    results = []
    for row in reg['cases']:
        print('AUDIT_EFFICIENCY', spec['id'], dt, row['id'], flush=True)
        result = audit(row, reg, teacher)
        result.update(candidate=spec, dt_s=dt)
        results.append(result)
    return dict(candidate=spec, dt_s=dt, status='passed', cases=results,
        successes=sum(row['success'] for row in results), unit_report_sha256=sha(folder/'unit_audit/report.json'))


def main(mode):
    torch.set_num_threads(1)
    started = time.perf_counter()
    reg = experiment.verify()
    teacher = FrozenPolicy(MODEL)
    if mode == 'nominal':
        runs = [audit_candidate(spec, .001, teacher) for spec in CANDIDATES]
        path = experiment.OUT/'nominal_independent_review.json'
        assert not path.exists()
        write_json(path, dict(status='passed', created_utc=utc(), registration_sha256=sha(experiment.OUT/'registration.json'),
            audit_source_sha256=sha(__file__), candidate_runs=runs, new_full_program_runs=9,
            scope='All saved nominal candidates, including failures; no new rollout',
            audit_wall_s=time.perf_counter()-started))
    else:
        nominal = base.read(experiment.OUT/'nominal_independent_review.json')
        assert nominal['status'] == 'passed'
        selection = base.read(experiment.OUT/'selection.json')
        assert selection['registration_sha256'] == sha(experiment.OUT/'registration.json')
        scores = [('frozen_v5_4', 1., 0)]
        for index, run in enumerate(nominal['candidate_runs'], 1):
            if run['successes'] == 3:
                score = sum(row['complete_time_s']/reg['baseline']['results'][row['case']]['complete_time_s']
                            for row in run['cases'])/3.
                scores.append((run['candidate']['id'], score, index))
        minimum = min(row[1] for row in scores)
        selected = min((row for row in scores if row[1] <= minimum+1e-12), key=lambda row: row[2])[0]
        assert selected == selection['selected']
        if selected == 'frozen_v5_4':
            finer = dict(reused_frozen_evidence=True,
                independent_review_sha256=sha(ROOT/'audit/geometric_resolution_v5_4_independent_review.json'))
        else:
            spec = next(spec for spec in CANDIDATES if spec['id'] == selected)
            finer = audit_candidate(spec, .0005, teacher)
        report = dict(status='passed', created_utc=utc(), selected=selected,
            registration_sha256=sha(experiment.OUT/'registration.json'), audit_source_sha256=sha(__file__),
            nominal_review_sha256=sha(experiment.OUT/'nominal_independent_review.json'),
            selection_sha256=sha(experiment.OUT/'selection.json'), selection_recomputed=True,
            nominal_results=[dict(candidate=run['candidate'], successes=run['successes'],
                cases=[{key: row[key] for key in ('case', 'success', 'complete_time_s', 'failed_prefix_s',
                        'reason', 'source_decisions', 'source_intervals', 'cost')} for row in run['cases']])
                for run in nominal['candidate_runs']],
            selected_resolution=finer, audit_wall_s=time.perf_counter()-started,
            qualification='Bounded development comparison; all registered candidates retained. Selection uses only 1 ms development results. No final-test, five-seed or hardware claim.')
        path = ROOT/'audit/geometric_efficiency_v5_5_independent_review.json'
        assert not path.exists()
        write_json(path, report)
        print('EFFICIENCY_AUDIT_PASSED', selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['nominal', 'selected-resolution'])
    main(parser.parse_args().mode)
