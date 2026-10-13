"""Run the predeclared local experiment matrix with bounded CPU concurrency."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name('revision_experiment.py')
ROOT = SCRIPT.parents[2]
OUT = ROOT / '论文项目/revision_20260908/experiments'
VARIANTS = ('coupled', 'decoupled', 'fixed_short', 'fixed_mid', 'fixed_long')


def call(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND: python ' + str(SCRIPT.relative_to(ROOT)) + ' ' + ' '.join(map(str, args)) + '\n')
        handle.flush()
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}; preserved log: {log}')


def job(variant, seed, stage, steps, evaluate=True):
    run = OUT / stage / f'{variant}_seed{seed}'
    call(['train', '--variant', variant, '--seed', seed, '--stage', stage, '--steps', steps], run / 'console.log')
    if evaluate:
        call(['evaluate', '--run', run, '--split', 'validation'], run / 'console.log')
    return f'{stage}/{run.name}'


def fixed_selection():
    scores = []
    for variant in ('fixed_short', 'fixed_mid', 'fixed_long'):
        path = OUT / 'screen' / f'{variant}_seed101/evaluation_validation_nominal/summary.json'
        rows = json.loads(path.read_text(encoding='utf-8'))['rows']
        times = [r['elapsed_time_s'] for r in rows if r['success']]
        score = (sum(r['success'] for r in rows), sum(r['path_completed'] for r in rows),
                 sum(r['ordered_progress'] for r in rows)/len(rows), -sum(times)/len(times) if times else -1e9)
        scores.append((score, variant))
    # Preserve declared distance order on exact ties.
    return max(scores, key=lambda pair: pair[0])[1]


def eval_job(run, split, regime):
    call(['evaluate', '--run', run, '--split', split, '--regime', regime], run / ('console_eval_' + split + '.log'))
    return f'{split}/{run.name}/{regime}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'main', 'test', 'sensitivity', 'regression'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == 'screen':
            futures = [pool.submit(job, v, 101, 'screen', 4096) for v in VARIANTS]
        elif args.stage == 'main':
            fixed = fixed_selection()
            (OUT / 'main_selection.json').write_text(json.dumps(dict(selected_fixed=fixed,
                candidate='decoupled', selected_using='screen validation only', main_steps=8192,
                seeds=[11,22,33,44,55]), indent=2), encoding='utf-8')
            futures = [pool.submit(job, v, seed, 'main', 8192, False)
                       for v in ('coupled', 'decoupled', fixed) for seed in (11,22,33,44,55)]
        else:
            freeze = json.loads((OUT / 'freeze.json').read_text(encoding='utf-8'))
            runs = [OUT / run for run in freeze['models']]
            split = 'regression' if args.stage == 'regression' else 'test'
            regimes = ('tight', 'low_jerk') if args.stage == 'sensitivity' else ('nominal',)
            # Sensitivity is an explicitly limited seed-11 diagnostic.
            if args.stage == 'sensitivity': runs = [r for r in runs if r.name.endswith('seed11')]
            futures = [pool.submit(eval_job, run, split, regime) for run in runs for regime in regimes]
        errors = []
        for future in as_completed(futures):
            try: print('COMPLETED', future.result(), flush=True)
            except Exception as exc:
                errors.append(str(exc)); print('FAILED', exc, flush=True)
        if errors: raise SystemExit('\n'.join(errors))


if __name__ == '__main__': main()
