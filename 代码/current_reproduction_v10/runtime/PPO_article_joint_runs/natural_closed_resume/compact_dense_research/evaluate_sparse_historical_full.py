"""Bounded complete original-case validation of structural memory pruning."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
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
from evaluate_historical_dense import HIST, sha, write, verify_bound_inputs

CASE = BASE / 'paper_evidence_review/natural_closed_case'
STAGE = ROOT / 'historical_sparse_memory_full_closed_loop'


def worker():
    import numpy as np
    import torch
    from candidate_policy_factory import candidate_policy
    from turn_aware_projector import environment_factory
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    from sparse_historical_memory import sparsify_memory
    verify_bound_inputs()
    torch.set_num_threads(1)
    selection = json.loads((STAGE / 'selected_policy.json').read_text(encoding='utf-8'))
    metadata = torch.load(selection['model'], map_location='cpu', weights_only=False)
    policy = sparsify_memory(candidate_policy(metadata, selection))
    reference = np.loadtxt(CASE / 'reference.csv', delimiter=',', skiprows=1)
    geometry = json.loads((CASE / 'geometry.json').read_text(encoding='utf-8'))
    env = environment_factory(reference)
    assert selection['fixed_d'] is None
    start = time.perf_counter()
    result = evaluate(policy, env, STAGE / 'full_trace.npz', start + 1620)
    data = np.load(STAGE / 'full_trace.npz')
    rows = data['rows']
    result['analytic'] = analytic_command_audit(reference[0], rows, geometry)
    if not result['analytic']['analytic_corridor_pass']:
        result.update(success=False, complete_time_s=None, motion_scope='failure_prefix_only')
    result.update(goal_complete=False, method_frozen=False, ppo_updates=0,
                  observation_dimension=72, memory_scalars=154,
                  historical_manifest_sha256=sha(HIST / 'runtime_provenance/manifest.json'),
                  experimental_projector_counts=env.experimental_projector.counts)
    write(STAGE / 'full_result.json', result)
    write(STAGE / 'gate_events.json', env.tracker.events)
    old = np.load(HIST / 'full_trace.npz')
    comparisons = {}
    for key in ['reference', 'rows', 'observations', 'actions', 'latents', 'executed_actions_1ms']:
        a, b = data[key], old[key]
        equal_shape = a.shape == b.shape
        comparisons[key] = dict(shape=list(a.shape), historical_shape=list(b.shape),
            equal_shape=equal_shape, bitwise_equal=bool(equal_shape and np.array_equal(a, b)),
            max_abs_error=float(np.max(np.abs(a-b))) if equal_shape and a.size else None)
    accepted = False
    if result['success']:
        m = metrics(reference, rows, geometry)
        write(STAGE / 'full_metrics.json', m)
        previous = json.loads((HIST / 'full_metrics.json').read_text(encoding='utf-8'))
        deltas = [a['speed_deficit_mm']-b['speed_deficit_mm'] for a,b in zip(m['corners'], previous['corners'])]
        comparisons['all34'] = dict(complete=m['all34_windows_complete'],
            largest_speed_deficit_change_mm=max(deltas),
            largest_abs_speed_deficit_change_mm=max(abs(v) for v in deltas),
            cruise99_time_change_s=m['cruise']['99']['time_s']-previous['cruise']['99']['time_s'])
        accepted = bool(m['all34_windows_complete'] and result['complete_time_s'] == previous['complete_recorded_time_s']
            and comparisons['rows']['max_abs_error'] <= 1e-8
            and comparisons['actions']['max_abs_error'] <= 1e-8
            and max(abs(v) for v in deltas) <= 1e-8)
    write(STAGE / 'comparison.json', dict(adopt_sparse_implementation=accepted,
        research_goal_complete=False, method_frozen=False, comparison=comparisons,
        scope='Structural memory reduction only; 72-dimensional observations and the historical controller remain.'))
    verify_bound_inputs()
    print(json.dumps(dict(result=result, adopt_sparse_implementation=accepted), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    manifest = verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    shutil.copy2(HIST / 'selected_policy.json', STAGE / 'selected_policy.json')
    sources = []
    for path in [Path(__file__), ROOT/'sparse_historical_memory.py', ROOT/'evaluate_historical_dense.py']:
        target = STAGE/'source'/path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        sources.append(dict(original=str(path), snapshot=str(target), sha256=sha(path)))
    write(STAGE / 'registration.json', dict(kind='Complete closed-loop structural memory reduction validation',
        wall_budget_s=1800, child_timeout_s=1710, simulation_deadline_s=1620,
        ppo_updates=0, parameter_updates=0, physics_dt_s=.001,
        original_case=str(CASE), reference_sha256=sha(CASE/'reference.csv'),
        geometry_sha256=sha(CASE/'geometry.json'), historical_candidate=str(HIST),
        historical_manifest=manifest, selection_sha256=sha(STAGE/'selected_policy.json'), sources=sources,
        evidence_gap='All recorded actions agree; a fresh complete closed loop remains unverified.',
        adoption='Full success, unchanged time, all34 windows and <=1e-8 action/row/deficit differences.',
        stop='Physical failure, evaluation deadline or child timeout; retain all results; no baseline changes.'))
    start = time.perf_counter()
    with (STAGE/'stdout.txt').open('w',encoding='utf-8') as out, (STAGE/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable,'-B',str(Path(__file__)),'--worker'], stdout=out,stderr=err,timeout=1710)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:
            status,code='timeout',None
    write(STAGE/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        goal_complete=False,method_frozen=False,new_ppo_updates=0,
        sources_unchanged=all(sha(x['original'])==x['sha256'] for x in sources)))
    print((STAGE/'stdout.txt').read_text(encoding='utf-8')[-6000:])
    if code != 0:
        print((STAGE/'stderr.txt').read_text(encoding='utf-8')[-6000:])


if __name__ == '__main__':
    main()
