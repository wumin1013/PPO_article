"""Run registered event-preview jobs, retaining every console and failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import preview_event_study as study


def call(arguments,log):
    command=[sys.executable,str(Path(study.__file__).resolve()),*map(str,arguments)]
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as stream:
        stream.write('\n'+json.dumps(dict(command=command,timestamp_utc=datetime.now(timezone.utc).isoformat()))+'\n');stream.flush()
        result=subprocess.run(command,cwd=study.ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if result.returncode:raise RuntimeError(f'{result.returncode}: {command}; log={log}')


def train(variant,seed,stage):
    run=study.OUT/stage/f'{variant}_seed{seed}'
    call(['train','--variant',variant,'--seed',seed,'--stage',stage,'--steps',study.PILOT_STEPS if stage=='pilot' else study.FORMAL_STEPS],run/'console.log')
    call(['evaluate','--run',run,'--split','validation'],run/'console.log')
    return run.name


def evaluate(variant,seed,split,regime):
    run=study.OUT/'formal'/f'{variant}_seed{seed}'
    call(['evaluate','--run',run,'--split',split,'--regime',regime],run/f'console_{split}_{regime}.log')
    return f'{run.name}/{split}/{regime}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pilot','formal','test','regression','sensitivity'])
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1 or args.workers>12:raise ValueError('workers must be 1..12')
    if args.stage in ('pilot','formal'):study.check_protocol()
    began=time.perf_counter();results=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        if args.stage=='pilot':jobs=[pool.submit(train,v,101,'pilot') for v in study.PILOTS]
        elif args.stage=='formal':jobs=[pool.submit(train,v,s,'formal') for v in study.FORMAL for s in study.SEEDS]
        else:
            split='regression' if args.stage=='regression' else 'test'
            regimes=('tight','low_jerk') if args.stage=='sensitivity' else ('nominal',)
            seeds=(11,) if args.stage=='sensitivity' else study.SEEDS
            jobs=[pool.submit(evaluate,v,s,split,r) for v in study.FORMAL for s in seeds for r in regimes]
        for future in as_completed(jobs):
            try:
                result=future.result();results.append(result);print('COMPLETED',result,flush=True)
            except Exception as error:
                errors.append(str(error));print('FAILED',error,flush=True)
    study.write(study.OUT/f'batch_{args.stage}_{time.time_ns()}.json',dict(stage=args.stage,workers=args.workers,completed=results,errors=errors,
        wall_s=time.perf_counter()-began,command='python '+' '.join(sys.argv),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
