"""Finite local-preview parameter diagnostic; no PPO or inherited source edits."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_historical_dense import ROOT, HIST, CASE, sha, write, verify_bound_inputs

STAGE = ROOT / 'historical_dense_cluster_preview_probe'
FLOORS = [2., 2.5, 3., 3.5, 4.]


def worker():
    from evaluate_historical_dense import worker as evaluate
    start = time.perf_counter()
    results = []
    for floor in FLOORS:
        if time.perf_counter()-start > 720:
            break
        folder = STAGE / f'floor_{floor:g}'
        folder.mkdir(exist_ok=False)
        shutil.copy2(HIST / 'selected_policy.json', folder / 'selected_policy.json')
        evaluate(folder, arc_support=True, preview_floor=floor, deadline=start+750)
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        results.append(dict(floor_mm=floor, directory=str(folder), success=result['success'],
            complete_time_s=result['complete_time_s'], elapsed_prefix_s=result['elapsed_prefix_or_complete_s'],
            gates_passed=result['geometry']['gates_passed'], reason=result['reason'],
            trace_sha256=result['trace_sha256']))
        write(STAGE / 'results.json', results)
        print('CANDIDATE', results[-1], flush=True)


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
    write(STAGE / 'registration.json', dict(kind='Five-value conditional preview-floor diagnostic',
        wall_budget_s=900, child_timeout_s=800, execution_deadline_s=750,
        preview_floor_values_mm=FLOORS, ppo_updates=0,
        source_sha256={str(path): sha(path) for path in files},
        candidate=str(HIST), model_selection_sha256=sha(HIST / 'selected_policy.json'),
        dense_reference_sha256=sha(CASE / 'reference.csv'),
        evidence_gap='After false arc lift removal, a112.5deg corner exit still exceeds the corridor; preceding preview falls to1.26mm at high speed.',
        adoption='Complete success and valid motion only; accept a floor only if performance improves. Retain all failures; test all registered values within deadline.',
        stop='Stop each candidate on physical failure; stop the round at800s. No global feed cap, geometry changes or S2 changes.'))
    start = time.perf_counter()
    with (STAGE / 'stdout.txt').open('w', encoding='utf-8') as out, (STAGE / 'stderr.txt').open('w', encoding='utf-8') as err:
        try:
            run = subprocess.run([sys.executable, '-B', str(Path(__file__)), '--worker'], stdout=out, stderr=err, timeout=800)
            status, code = 'finished', run.returncode
        except subprocess.TimeoutExpired:
            status, code = 'timeout', None
    write(STAGE / 'receipt.json', dict(status=status, exit_code=code,
          wall_s=time.perf_counter()-start, goal_complete=False, ppo_updates=0))
    print((STAGE / 'results.json').read_text(encoding='utf-8') if (STAGE / 'results.json').exists() else 'No candidate result')
    if code != 0:
        print((STAGE / 'stderr.txt').read_text(encoding='utf-8')[-5000:])


if __name__ == '__main__':
    main()
