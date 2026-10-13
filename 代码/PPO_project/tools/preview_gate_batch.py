"""Bounded parallel orchestration for the registered geometry-gate study."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import preview_gate_research as study
from preview_v2_batch import call

SCRIPT = study.Path(study.__file__)


def train(variant, seed, stage, steps):
    run = study.OUT/stage/f'{variant}_seed{seed}'
    call(['train','--variant',variant,'--seed',seed,'--stage',stage,'--steps',steps],run/'console.log',SCRIPT)
    call(['evaluate','--run',run,'--split','validation'],run/'console.log',SCRIPT)
    return f'{stage}/{run.name}'


def evaluate(variant, seed, split, regime):
    run=study.OUT/'formal'/f'{variant}_seed{seed}'
    call(['evaluate','--run',run,'--split',split,'--regime',regime],run/f'console_{split}_{regime}.log',SCRIPT)
    return f'{run.name}/{split}/{regime}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['pilot','formal','test','regression','sensitivity'])
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.stage=='pilot':
            futures=[pool.submit(train,v,101,'pilot',study.PILOT_STEPS) for v in study.PILOT_VARIANTS]
        elif args.stage=='formal':
            futures=[pool.submit(train,v,s,'formal',study.FORMAL_STEPS) for v in study.FORMAL_VARIANTS for s in study.SEEDS]
        else:
            split='regression' if args.stage=='regression' else 'test'
            regimes=('tight','low_jerk') if args.stage=='sensitivity' else ('nominal',)
            seeds=(11,) if args.stage=='sensitivity' else study.SEEDS
            futures=[pool.submit(evaluate,v,s,split,r) for v in study.FORMAL_VARIANTS for s in seeds for r in regimes]
        errors=[]
        for future in as_completed(futures):
            try: print('COMPLETED',future.result(),flush=True)
            except Exception as exc:
                errors.append(str(exc)); print('FAILED',exc,flush=True)
        if errors: raise SystemExit('\n'.join(errors))


if __name__=='__main__': main()
