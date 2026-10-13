"""Run the declared second-round budget; every subprocess keeps its own log."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name('preview_research_v2.py')
ROOT = SCRIPT.parents[2]
OUT = ROOT / '论文项目/revision_20260908/preview_v2'
SEEDS = (11, 22, 33, 44, 55)


def call(args, log, script=SCRIPT):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND: python ' + str(script.relative_to(ROOT)) + ' ' + ' '.join(map(str, args)) + '\n')
        handle.flush()
        result = subprocess.run([sys.executable, str(script), *map(str, args)], cwd=ROOT,
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Exit {result.returncode}; preserved log: {log}')


def train(variant, seed, stage, steps):
    run = OUT / stage / f'{variant}_seed{seed}'
    call(['train', '--variant', variant, '--seed', seed, '--stage', stage, '--steps', steps], run/'console.log')
    call(['evaluate', '--run', run, '--split', 'validation'], run/'console.log')
    return str(run.relative_to(OUT))


def evaluate(variant, seed, split, regime):
    run = OUT / 'finetune' / f'{variant}_seed{seed}'
    call(['evaluate', '--run', run, '--split', split, '--regime', regime], run/f'console_{split}_{regime}.log')
    return f'{run.name}/{split}/{regime}'


def nnc(seed, train_first):
    run = OUT/'nnc_auxiliary'/'runs'/f'nnc_seed{seed}'
    script = SCRIPT.with_name('preview_v2_nnc.py')
    if train_first:
        call(['train', '--seed', seed], run/'console.log', script)
    split = 'validation' if train_first else 'test'
    call(['evaluate', '--seed', seed, '--split', split], run/f'console_{split}.log', script)
    return f'nnc/{seed}/{split}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['screen', 'pretrain', 'finetune', 'test', 'regression', 'sensitivity', 'nnc', 'nnc_test'])
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == 'screen':
            futures = [pool.submit(train, v, 101, 'screen', 32768)
                       for v in ('pretrain_short', 'pretrain_mid', 'pretrain_long')]
        elif args.stage == 'pretrain':
            futures = [pool.submit(train, 'pretrain_selected', seed, 'pretrain', 65536) for seed in SEEDS]
        elif args.stage in ('nnc', 'nnc_test'):
            futures = [pool.submit(nnc, seed, args.stage == 'nnc') for seed in SEEDS]
        elif args.stage == 'finetune':
            futures = [pool.submit(train, v, seed, 'finetune', 32768)
                       for v in ('fixed', 'residual', 'absolute') for seed in SEEDS]
        else:
            split = 'regression' if args.stage == 'regression' else 'test'
            regimes = ('tight', 'low_jerk') if args.stage == 'sensitivity' else ('nominal',)
            seeds = (11,) if args.stage == 'sensitivity' else SEEDS
            futures = [pool.submit(evaluate, v, seed, split, regime)
                       for v in ('fixed', 'residual', 'absolute') for seed in seeds for regime in regimes]
        errors = []
        for future in as_completed(futures):
            try: print('COMPLETED', future.result(), flush=True)
            except Exception as exc:
                errors.append(str(exc)); print('FAILED', exc, flush=True)
        if errors: raise SystemExit('\n'.join(errors))


if __name__ == '__main__': main()
