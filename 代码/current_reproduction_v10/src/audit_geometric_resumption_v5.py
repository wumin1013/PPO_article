"""Saved-evidence audit of the one-action extension; never launches a rollout."""
import copy
import inspect
import json
import math
import textwrap
import time
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from frozen_policy import FrozenPolicy
from replay_strong_checkpoint_v4 import MODEL
from predictive_action_candidate_v5 import new_environment
import audit_recoverable_candidate_v5 as base

OUT = ROOT/'results/geometric_resumption_candidate_v5_4'
REPORT = ROOT/'audit/geometric_resumption_candidate_v5_4_independent_review.json'


def independent_geometric_action(reference, state):
    """Rebuild the request from the saved state, without executing any command."""
    env = new_environment(reference, state)
    frontier = state['tracker']['frontier']
    target = env.corridor.gates[frontier].point if frontier < len(env.corridor.gates) else reference[-1]
    difference = target-state['core']['position']
    desired_heading = math.atan2(difference[1], difference[0])
    old = env.env.held.copy()
    try:
        env.env.held = np.array([0., 20., .8])
        ray = env.guidance_target()-env.position
        beta = math.atan2(ray[1], ray[0]) if np.linalg.norm(ray) > 1e-12 else env.heading
    finally:
        env.env.held = old
    heading_error = (desired_heading-state['core']['heading']+math.pi) % (2*math.pi)-math.pi
    correction = (desired_heading-beta+math.pi) % (2*math.pi)-math.pi
    steering = float(np.clip(correction, -math.pi/2, math.pi/2))
    feed = 0. if abs(heading_error) > .05 else 20.
    return np.array([steering/(math.pi/2), feed/50.-1., -1.])


def extended_audit_case():
    """Preserve the audited v5_3 code and extend only its action/source cases."""
    prior = base.read(ROOT/'audit/recoverable_action_candidate_v5_3_independent_review.json')
    assert sha(base.__file__) == prior['audit_source_sha256']
    source = textwrap.dedent(inspect.getsource(base.audit_case))
    replacements = [
        ("expected_action = raw if order == -1 else first_action(raw, OFFSETS[order//5], FACTORS[order % 5])",
         "expected_action = (independent_geometric_action(reference, state) if order == 25 else\n            raw if order == -1 else first_action(raw, OFFSETS[order//5], FACTORS[order % 5]))"),
        ("orders = [-1]+[i for i in range(25) if i != 14]",
         "orders = [-1]+[i for i in range(25) if i != 14]+[25]"),
        ("assert len(group) == 25 and not raw['feasible']\n                valid = [b for b in group[1:] if b['feasible']]",
         "assert len(group) in (25, 26) and not raw['feasible']\n                valid = [b for b in group[1:25] if b['feasible']]"),
        ("if d['source'] == 'corrected_model':\n                    assert valid",
         "if d['source'] == 'corrected_model':\n                    assert valid and len(group) == 25"),
        ("else:\n                    assert d['source'] == 'certified_stop' and not valid\n                    chosen = None",
         "elif d['source'] == 'geometric_resumption':\n                    assert not valid and len(group) == 26 and group[-1]['feasible']\n                    chosen = group[-1]\n                else:\n                    assert d['source'] == 'certified_stop' and not valid\n                    assert len(group) == 26 and not group[-1]['feasible']\n                    chosen = None"),
        ("for key in ('raw_model', 'corrected_model', 'certified_stop'):",
         "for key in ('raw_model', 'corrected_model', 'certified_stop', 'geometric_resumption'):"),
        ("full_process_cpu_recorded=False,", "full_process_cpu_recorded=True,\n            recorded_full_run_including_archive_and_metrics_process_cpu_s=actual['cost']['full_run_including_archive_and_metrics_process_cpu_s'],")]
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(base.__dict__)
    namespace.update(OUT=OUT, independent_geometric_action=independent_geometric_action)
    output = {}
    exec(compile(source, __file__, 'exec'), namespace, output)
    return output['audit_case']


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    reg = base.read(OUT/'registration.json')
    assert all((OUT/r['id']/'actual.json').exists() for r in reg['cases'])
    assert all(sha(ROOT/'src'/key) == value for key, value in reg['source_sha256'].items())
    assert sha(MODEL) == reg['model_sha256']
    assert reg['per_program_wall_limit_s'] == 300. and reg['run_order'] == [r['id'] for r in reg['cases']]
    unit = base.read(OUT/'unit_audit/report.json')
    assert unit['status'] == 'passed' and unit['registered_sources'] == reg['source_sha256']
    audit_case = extended_audit_case()
    teacher, cases = FrozenPolicy(MODEL), []
    for row in reg['cases']:
        print('AUDIT_GEOMETRIC_CASE', row['id'], flush=True)
        actual = base.read(OUT/row['id']/'actual.json')
        assert actual['cost']['full_run_including_archive_and_metrics_process_cpu_s'] >= 0.
        assert actual['action_source'].startswith('Four logged sources:')
        assert 'submitted action requests' in actual['action_array_definition']
        cases.append(audit_case(row, reg, teacher))
    summary = base.read(OUT/'summary.json')
    assert summary['runs'] == len(cases) and summary['successes'] == sum(c['success'] for c in cases)
    report = dict(status='passed', created_utc=utc(),
        scope='Saved existing development evidence only; no rollout, new case, training or holdout use',
        registration_sha256=sha(OUT/'registration.json'), audit_source_sha256=sha(__file__),
        parent_audit_source_sha256=sha(base.__file__), source_and_model_hashes_match_registration=True,
        static_review=dict(clone_restored_before_geometric_request=True, clone_restored_again_before_evaluation=True,
            parent_source_exact_match_checked=True, original_raw_preserved_when_admissible=True,
            geometric_evaluated_only_after_original_25_fail=True, no_state_reset_or_geometry_relaxation=True),
        cases=cases, successes=sum(c['success'] for c in cases), runs=len(cases),
        actual_intervals=sum(c['actual_intervals'] for c in cases), branches=sum(c['branches'] for c in cases),
        branch_intervals=sum(c['branch_intervals'] for c in cases),
        stopping_assessments=sum(c['stopping_assessments'] for c in cases),
        stop_intervals=sum(c['stop_intervals'] for c in cases), audit_wall_s=time.perf_counter()-started,
        qualification='Saved commands, sources and outcomes verified; no recursive completion, independent final-test, hard real-time or five-seed claim.',
        cost_gap='Full process CPU includes archival and final metrics; it is not per-step online CPU. Individual aborted-decision wall time was not separately recorded. Nested costs must not be added.')
    assert not REPORT.exists(), 'Never overwrite a prior audit'
    write_json(REPORT, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
