"""Bounded replay of a hash-verified historical controller on the fixed dense case."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent
BASE = Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0] = [str(BASE), str(OUT), str(ROOT)]
HIST = OUT / 'all34_dominance_research/joint_transit_retained_seed922222_full'
CASE = ROOT / 'dense_closed_case'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify_bound_inputs():
    manifest = json.loads((HIST / 'runtime_provenance/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['imported_source_files']:
        for key in ('original', 'snapshot'):
            assert sha(record[key]) == record['sha256'], record[key]
    assert sha(manifest['model']) == manifest['model_sha256']
    assert sha(HIST / 'selected_policy.json') == manifest['selection_sha256']
    assert sha(CASE / 'reference.csv') == '08a5c2ee631e6fc78004d172dcdebcd51656376b1230b85091073e5e1b33fb59'
    return manifest


def worker(stage, arc_support=False, preview_floor=0., deadline=None, feed_reserve=0., feedback=None):
    import numpy as np
    import torch
    from candidate_policy_factory import candidate_policy
    from turn_aware_projector import environment_factory
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from analyze_dense_trace import analyze

    verify_bound_inputs()
    torch.set_num_threads(1)
    selection = json.loads((stage / 'selected_policy.json').read_text(encoding='utf-8'))
    reference = np.loadtxt(CASE / 'reference.csv', delimiter=',', skiprows=1)
    geometry = json.loads((CASE / 'geometry.json').read_text(encoding='utf-8'))
    metadata = torch.load(selection['model'], map_location='cpu', weights_only=False)
    policy = candidate_policy(metadata, selection)
    support = None
    if arc_support:
        from local_arc_support import install_local_arc_support
        policy, support = install_local_arc_support(policy)
    preview_support = None
    if preview_floor:
        from local_arc_support import ClusteredPreviewFloorPolicy
        policy = preview_support = ClusteredPreviewFloorPolicy(policy, preview_floor)
    feed_support = None
    if feed_reserve:
        from local_arc_support import ClusteredCornerFeedPolicy
        policy = feed_support = ClusteredCornerFeedPolicy(policy, feed_reserve)
    feedback_support = None
    if feedback is not None:
        from local_arc_support import ClusteredJointFeedbackPolicy
        policy = feedback_support = ClusteredJointFeedbackPolicy(policy, feedback)
    env = environment_factory(reference)
    if selection['fixed_d'] is not None:
        env.fixed_d = selection['fixed_d']
    start = time.perf_counter()
    result = evaluate(policy, env, stage / 'trace.npz', min(start + 500, deadline) if deadline else start + 500)
    rows = np.load(stage / 'trace.npz')['rows']
    if len(rows):
        result['analytic'] = analytic_command_audit(reference[0], rows, geometry)
        if not result['analytic']['analytic_corridor_pass']:
            result['success'] = False
            result['complete_time_s'] = None
            result['motion_scope'] = 'failure_prefix_only'
    result.update(goal_complete=False, ppo_updates=0, parameter_updates=0,
                  selected_historical_candidate=str(HIST),
                  historical_model_sha256=selection['model_sha256'],
                  selection_sha256=sha(stage / 'selected_policy.json'),
                  case_sha256=sha(CASE / 'reference.csv'),
                  experimental_projector_counts=env.experimental_projector.counts)
    result['local_arc_support'] = dict(enabled=arc_support,
        suppressed_calls=support.suppressed_calls if support else 0,
        total_calls=support.total_calls if support else 0)
    result['clustered_preview_floor'] = dict(floor_mm=preview_floor,
        changed_calls=preview_support.changed_calls if preview_support else 0)
    result['clustered_corner_feed'] = dict(reserve=feed_reserve,
        changed_calls=feed_support.changed_calls if feed_support else 0)
    result['clustered_joint_feedback'] = dict(coefficients=feedback,
        changed_calls=feedback_support.changed_calls if feedback_support else 0)
    write(stage / 'gate_events.json', env.tracker.events)
    write(stage / 'result.json', result)
    if result['success']:
        metrics = analyze(stage / 'trace.npz', stage / 'result.json')
        write(stage / 'all84_metrics.json', metrics)
    verify_bound_inputs()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='historical_best_dense_initial')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--arc-support', action='store_true')
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    stage = ROOT / args.name
    if args.worker:
        worker(stage, args.arc_support)
        return
    manifest = verify_bound_inputs()
    stage.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), stage / 'source.py')
    shutil.copy2(HIST / 'selected_policy.json', stage / 'selected_policy.json')
    if args.arc_support:
        shutil.copy2(ROOT / 'local_arc_support.py', stage / 'local_arc_support.py')
    write(stage / 'registration.json', dict(
        kind='Historical best with explicitly recorded local arc-support diagnostic' if args.arc_support else 'Unchanged historical best controller on the already fixed dense development case',
        wall_budget_s=600, child_timeout_s=550, simulation_deadline_s=500,
        ppo_updates=0, parameter_updates=0, physics_dt_s=.001,
        historical_candidate=str(HIST), source_sha256=sha(Path(__file__)),
        case_registration_sha256=sha(CASE / 'registration.json'),
        case_reference_sha256=sha(CASE / 'reference.csv'),
        verified_historical_source_count=len(manifest['imported_source_files']),
        historical_manifest_sha256=sha(HIST / 'runtime_provenance/manifest.json'),
        historical_manifest=manifest,
        selection_sha256=sha(stage / 'selected_policy.json'),
        local_arc_support=dict(enabled=args.arc_support,
            source_sha256=sha(ROOT / 'local_arc_support.py') if args.arc_support else None,
            definition='Suppress only supplementary arc feed if >=2 discrete heading changes >10deg occur within +-30mm reference progress.'),
        evidence_gap='The historical all34 result does not establish complete feasibility or efficiency on the fixed dense case.',
        adoption='Only complete ordered coverage, original analytic corridor, vector V/A/J, braking and >=20ms rest allow full-time comparison. Full success alone is not research acceptance.',
        stop='Stop at physical failure or the deadline; preserve all prefixes; no source, geometry or S2 edits.'))
    start = time.perf_counter()
    with (stage / 'stdout.txt').open('w', encoding='utf-8') as out, (stage / 'stderr.txt').open('w', encoding='utf-8') as err:
        try:
            command = [sys.executable, '-B', str(Path(__file__)), '--name', args.name, '--worker']
            if args.arc_support:
                command.append('--arc-support')
            run = subprocess.run(command,
                                 stdout=out, stderr=err, timeout=550)
            status, code = 'finished', run.returncode
        except subprocess.TimeoutExpired:
            status, code = 'timeout', None
    write(stage / 'receipt.json', dict(status=status, exit_code=code,
          wall_s=time.perf_counter()-start, goal_complete=False,
          new_ppo_updates=0, new_parameter_updates=0))
    print((stage / 'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code != 0:
        print((stage / 'stderr.txt').read_text(encoding='utf-8')[-6000:])


if __name__ == '__main__':
    main()
