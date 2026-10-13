"""Finite joint error-feedback parameter selection on a fixed development case."""
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluate_historical_dense import ROOT, HIST, CASE, sha, write, verify_bound_inputs

STAGE = ROOT / 'historical_dense_cluster_joint_feedback_probe'
COEFFICIENTS = [list(c) for c in itertools.product([-.2, -.1, 0.], [-.5, -.2, 0.], [0., .2, .4]) if any(c)]


def worker():
    from evaluate_historical_dense import worker as evaluate
    start = time.perf_counter()
    results = []
    for index, coefficients in enumerate(COEFFICIENTS):
        if time.perf_counter()-start > 1000:
            break
        folder = STAGE / f'candidate_{index:02d}'
        folder.mkdir(exist_ok=False)
        shutil.copy2(HIST / 'selected_policy.json', folder / 'selected_policy.json')
        write(folder / 'candidate.json', dict(coefficients=coefficients, clustered_preview_floor_mm=3., local_arc_support=True))
        evaluate(folder, arc_support=True, preview_floor=3., feedback=coefficients, deadline=start+1050)
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        results.append(dict(index=index, coefficients=coefficients, directory=str(folder), success=result['success'],
            complete_time_s=result['complete_time_s'], elapsed_prefix_s=result['elapsed_prefix_or_complete_s'],
            gates_passed=result['geometry']['gates_passed'], reason=result['reason'],
            trace_sha256=result['trace_sha256']))
        write(STAGE / 'results.json', results)
        print('CANDIDATE', results[-1], flush=True)
        if result['success'] and result['complete_time_s'] <= 18.95915:
            break


def main():
    if '--worker' in sys.argv:
        worker()
        return
    verify_bound_inputs()
    STAGE.mkdir(exist_ok=False)
    source = STAGE / 'source'
    source.mkdir()
    files = [Path(__file__), ROOT / 'evaluate_historical_dense.py', ROOT / 'local_arc_support.py']
    for path in files:
        shutil.copy2(path, source / path.name)
    write(STAGE / 'registration.json', dict(kind='Registered joint-feedback coefficient parameter search',
        wall_budget_s=1200, child_timeout_s=1100, execution_deadline_s=1050,
        candidates=COEFFICIENTS, clustered_preview_floor_mm=3., ppo_updates=0,
        source_sha256={str(path): sha(path) for path in files},
        candidate=str(HIST), model_selection_sha256=sha(HIST / 'selected_policy.json'),
        dense_reference_sha256=sha(CASE / 'reference.csv'),
        evidence_gap='Changing preview or geometric feed alone has no complete success; test coordinated feedback on the same observed error.',
        baseline_evidence=str(ROOT / 'historical_dense_cluster_preview_probe/floor_3/result.json'),
        adoption='Full success is mandatory; rank successful candidates by full time. Fits are not PPO training or independent training seeds.',
        stop='On first full success <=18.95915s stop this diagnostic for validation, or finish26 candidates/1100s. S2 threshold alone does not establish superiority to a strong retrained two-output ablation.'))
    start = time.perf_counter()
    with (STAGE / 'stdout.txt').open('w', encoding='utf-8') as out, (STAGE / 'stderr.txt').open('w', encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker'], stdout=out, stderr=err, timeout=1100)
            status, code = 'finished', run.returncode
        except subprocess.TimeoutExpired:
            status, code = 'timeout', None
    write(STAGE / 'receipt.json', dict(status=status, exit_code=code, wall_s=time.perf_counter()-start,
          goal_complete=False, ppo_updates=0))
    results = json.loads((STAGE / 'results.json').read_text(encoding='utf-8')) if (STAGE / 'results.json').exists() else []
    print(json.dumps(dict(evaluated=len(results), successful=[r for r in results if r['success']],
        maximum_gates=max((r['gates_passed'] for r in results), default=0)), indent=2))
    if code != 0:
        print((STAGE / 'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__ == '__main__':
    main()
