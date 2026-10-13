"""Bounded independent replay of frozen historical training stages, not seed relabeling."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ast
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs

SUPPLEMENT=ROOT/'experimental_supplement_20260923'
SEEDS=(923201,923202,923203)


class SeedConstants(ast.NodeTransformer):
    def __init__(self, seed): self.seed=seed
    def visit_Constant(self,node):
        if isinstance(node.value,int) and not isinstance(node.value,bool) and node.value==922302:
            return ast.copy_location(ast.Constant(self.seed),node)
        return node


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['compression','ppo_initial'],required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--worker',action='store_true')
    return p.parse_args()


def paths(args):
    stage=SUPPLEMENT/'independent_training'/f'seed_{args.seed}'/args.stage
    source=ROOT/('fit_direct_compact_policy.py' if args.stage=='compression' else 'train_direct_compact_balanced_ppo.py')
    return stage,source


def worker(args):
    stage,source=paths(args)
    source=stage/'source'/source.name
    tree=SeedConstants(args.seed).visit(ast.parse(source.read_text(encoding='utf-8')))
    ast.fix_missing_locations(tree)
    namespace={'__name__':'independent_stage_replay','__file__':str(source)}
    exec(compile(tree,str(source),'exec'),namespace)
    namespace['STAGE']=stage
    namespace['SEED']=args.seed
    if args.stage=='ppo_initial':
        namespace['INITIAL']=stage.parent/'compression/fit_0600.pt'
    started=time.perf_counter()
    try:
        namespace['worker']()
    except BaseException:
        write(stage/'replay_error.json',dict(error=traceback.format_exc(),wall_s=time.perf_counter()-started))
        raise


def main(args):
    if args.worker:worker(args);return
    stage,source=paths(args)
    stage.mkdir(parents=True,exist_ok=False);(stage/'source').mkdir()
    shutil.copy2(source,stage/'source'/source.name)
    shutil.copy2(__file__,stage/'source'/Path(__file__).name)
    verify_bound_inputs()
    initial=stage.parent/'compression/fit_0600.pt'
    if args.stage=='ppo_initial':assert initial.exists()
    common=ROOT/'direct14_saved_motion_fit_seed922302_bound/supervision.npz'
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Independent random initialization' if args.stage=='compression' else 'Independent-seed replay of first historical PPO stage',
        seed=args.seed,all_preregistered_seeds=list(SEEDS),stage=args.stage,
        original_source=str(source),original_source_sha256=sha(source),runner_sha256=sha(__file__),
        changes='Replace numeric seed922302 with preregistered seed and bind only stage output and previous own-seed initial checkpoint. Teacher paths, losses, physics and evaluation checks unchanged.',
        initialization='Fresh neural weights; shared fixed recorded training data; no fitted checkpoint loaded' if args.stage=='compression' else str(initial),
        initial_sha256=sha(initial) if args.stage=='ppo_initial' else None,
        reference_training_data=str(common),reference_training_data_sha256=sha(common),
        budget_s=1200 if args.stage=='compression' else 1800,
        child_timeout_s=1150 if args.stage=='compression' else 1740,
        procedure='Original1600offline epochs,checks200/600/1600;fixed600checkpoint continues regardless of which seed is faster' if args.stage=='compression' else 'Original24PPO updates with fixed original/dense sampling and original scheduled checks;fixed16checkpoint follows the adopted historical lineage if reached',
        adoption='Supplementary repetition evidence only. Retain all seeds and all failures. Do not replace the frozen73candidate0 or report this partial stage as a full-pipeline repeat.',
        stop='Fixed iteration/update cap or wall deadline; stop on implementation/physical exception and preserve cause. Do not substitute successful seeds.',
        independence_scope='Independent optimizer/network randomness conditional on the shared recorded training corpus. Historical teacher generation and method discovery are not independently repeated.',
        full_pipeline_complete=False,new_model_selection_allowed=False))
    started=time.perf_counter();code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--stage',args.stage,'--seed',str(args.seed),'--worker'],
                stdout=out,stderr=err,timeout=1150 if args.stage=='compression' else 1740)
            code=result.returncode;status='finished' if code==0 else 'error'
        except subprocess.TimeoutExpired:status='timeout'
    write(stage/'replay_receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        original_source_unchanged=sha(source)==sha(stage/'source'/source.name),full_pipeline_complete=False))
    print('REPLAY_STAGE',args.seed,args.stage,status,round(time.perf_counter()-started,2),flush=True)
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-1800:],flush=True)
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-1800:],flush=True)


if __name__=='__main__':main(arguments())
