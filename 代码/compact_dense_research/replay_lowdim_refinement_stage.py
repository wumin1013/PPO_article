"""Continue each independent initialization through fixed historical fit/PPO stages."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ast
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from replay_lowdim_training_stage import SeedConstants, SUPPLEMENT, SEEDS

# Selection checkpoints are fixed from the original adopted lineage before these runs.
STAGES={
    'geometry_fit':('fit_direct_geometry_conditioned.py','ppo_initial/checkpoint_0016.pt','fit_1800.pt'),
    'saturation_fit':('fit_direct_saturation_aware.py','geometry_fit/fit_1800.pt','fit_6000.pt'),
    'ppo_deficit':('train_direct_deficit_ppo.py','saturation_fit/fit_6000.pt','checkpoint_0024.pt'),
    'preview_fit':('fit_direct_previewcue.py','ppo_deficit/checkpoint_0024.pt','fit_12000.pt'),
    'ppo_preview':('train_direct_previewcue_ppo.py','preview_fit/fit_12000.pt','checkpoint_0024.pt'),
}


def args():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=list(STAGES),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--worker',action='store_true')
    return p.parse_args()


def worker(a,stage,source,initial):
    tree=SeedConstants(a.seed).visit(ast.parse(source.read_text(encoding='utf-8')))
    ast.fix_missing_locations(tree)
    namespace={'__name__':'independent_refinement_replay','__file__':str(source)}
    exec(compile(tree,str(source),'exec'),namespace)
    namespace.update(STAGE=stage,INITIAL=initial,SEED=a.seed)
    try:namespace['worker']()
    except BaseException:
        write(stage/'replay_error.json',dict(error=traceback.format_exc()))
        raise


def main(a):
    source_name,previous,selected=STAGES[a.stage]
    seedroot=SUPPLEMENT/'independent_training'/f'seed_{a.seed}'
    stage=seedroot/a.stage;source=ROOT/source_name;initial=seedroot/previous
    if a.worker:
        worker(a,stage,stage/'source'/source_name,initial);return
    assert initial.exists(),f'Missing own-seed parent {initial}'
    stage.mkdir(parents=True,exist_ok=False);(stage/'source').mkdir()
    for p in (source,Path(__file__),ROOT/'replay_lowdim_training_stage.py'):
        shutil.copy2(p,stage/'source'/p.name)
    verify_bound_inputs()
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Independent-seed replay of a fixed historical training stage',stage=a.stage,seed=a.seed,
        all_preregistered_seeds=list(SEEDS),source_sha256={str(p):sha(p) for p in (source,Path(__file__),ROOT/'replay_lowdim_training_stage.py')},
        initial_checkpoint=str(initial),initial_checkpoint_sha256=sha(initial),fixed_continuation_checkpoint=selected,
        wall_budget_s=1800,child_timeout_s=1740,
        evidence_gap='Independent repeat of the adopted training lineage has not reached its final actor/refinement/evaluation stages.',
        procedure='Replay unchanged original worker, replacing numeric random seed and output/own-seed initializer globals only. Shared recorded demonstration data and hyperparameters are fixed. Preserve original scheduled evaluation opportunities; continuation checkpoint is fixed in advance.',
        adoption='Complete both-case paired-resolution evaluation at final pipeline output; partial-stage results and failed runs remain in record. No automatic change to frozen paper model.',
        stop='Original iteration cap or registered wall deadline. Stop on exception; never replace an unsuccessful seed with another.',
        full_pipeline_complete=False,independence_scope='Independent initialization and subsequent optimizer randomness conditional on the shared fixed training corpus; historical corpus construction is not repeated.'))
    started=time.perf_counter();code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--stage',a.stage,'--seed',str(a.seed),'--worker'],stdout=out,stderr=err,timeout=1740)
            code=result.returncode;status='finished' if code==0 else 'error'
        except subprocess.TimeoutExpired:status='timeout'
    write(stage/'replay_receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        source_unchanged=sha(source)==sha(stage/'source'/source_name),selected_checkpoint_exists=(stage/selected).exists(),full_pipeline_complete=False))
    print('REPLAY_REFINEMENT',a.seed,a.stage,status,round(time.perf_counter()-started,2),flush=True)
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-1400:],flush=True)
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-1800:],flush=True)


if __name__=='__main__':main(args())
