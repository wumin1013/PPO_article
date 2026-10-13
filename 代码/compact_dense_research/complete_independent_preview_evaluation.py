"""Finish only deadline-truncated checks of a fixed own-seed PPO checkpoint."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from evaluate_historical_dense import ROOT, BASE, CASE, write, sha, verify_bound_inputs
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def worker(stage, checkpoint):
    import numpy as np
    import torch
    from compact_fine_policy import FineCompactPolicy
    from previewcue_learning_runtime import PreviewCueLearningRuntime
    from paper_complete_evaluation import evaluate
    from natural_case_audit import analytic_command_audit
    from natural_case_metrics import metrics
    from analyze_dense_trace import analyze
    started = time.perf_counter()
    torch.set_num_threads(1)
    policy = FineCompactPolicy(feed_gain=1.)
    policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['policy'])
    policy.eval()
    records = {}
    registration = read(stage/'registration.json')
    folders = {'dense':CASE, 'original':BASE/'paper_evidence_review/natural_closed_case'}
    for case in registration['missing_or_timed_out_cases']:
        reference = np.loadtxt(folders[case]/'reference.csv',delimiter=',',skiprows=1)
        geometry = read(folders[case]/'geometry.json')
        trace = stage/f'eval_0024_{case}.npz'
        env = PreviewCueLearningRuntime(reference,potential_speed=100.)
        result = evaluate(policy,env,trace,started+1640.)
        with np.load(trace) as data:
            rows = data['rows']
        result['analytic'] = analytic_command_audit(reference[0],rows,geometry)
        if not result['analytic']['analytic_corridor_pass']:
            result.update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        original = checkpoint.parent/trace.name
        prefix_exact = None
        if original.exists():
            with np.load(original) as data:
                old = data['rows']
            assert len(rows) >= len(old), 'Completion check stopped before previously executed prefix'
            prefix_exact = bool(np.array_equal(rows[:len(old)],old))
            assert prefix_exact, 'Re-evaluation changed the previously executed physical prefix'
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            source_completion_check=str(stage),previous_prefix_bitwise_identical=prefix_exact,
            observation_dimension=14,policy_memory_scalars=0,actor_outputs=3,
            new_ppo_updates=0,ppo_updates_at_checkpoint=24)
        write(trace.with_suffix('.json'),result)
        write(stage/f'eval_0024_{case}_gates.json',env.tracker.events)
        if result['success']:
            value = metrics(reference,rows,geometry) if case=='original' else analyze(trace,trace.with_suffix('.json'))
            write(stage/f'eval_0024_{case}_metrics.json',value)
        records[case] = result
        print('COMPLETION_CHECK',case,result['success'],result['complete_time_s'],result['error'],flush=True)
    write(stage/'result.json',dict(results=records,new_ppo_updates=0,wall_s=time.perf_counter()-started))


def main(args):
    root = SUPPLEMENT/'independent_training'/f'seed_{args.seed}'
    original = root/'ppo_preview'
    stage = root/'ppo_preview_evaluation_completion'
    checkpoint = original/'checkpoint_0024.pt'
    if args.worker:
        worker(stage,checkpoint)
        return
    receipt = read(original/'replay_receipt.json')
    assert checkpoint.exists(), 'Need an actually completed fixed update24 checkpoint'
    missing = []
    prior = {}
    for case in ('dense','original'):
        result_path = original/f'eval_0024_{case}.json'
        if result_path.exists():
            result = read(result_path)
            prior[case] = dict(path=str(result_path),sha256=sha(result_path),result=result)
            if str(result.get('error','')).startswith('TimeoutError:'):
                missing.append(case)
        else:
            missing.append(case)
    if not missing:
        print('NO_INCOMPLETE_CHECKS',args.seed)
        return
    stage.mkdir(exist_ok=False)
    shutil.copy2(__file__,stage/Path(__file__).name)
    verify_bound_inputs()
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=args.seed,
        stage='ppo_preview_evaluation_completion',initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint),
        previous_training_receipt=receipt,previous_checks=prior,missing_or_timed_out_cases=missing,
        source_sha256=sha(__file__),wall_budget_s=1800,child_timeout_s=1740,
        evidence_gap='Training reached the fixed checkpoint but a scheduled complete evaluation hit the wall deadline.',
        procedure='Re-evaluate only absent/deadline-truncated cases. Compare every previously recorded physical-prefix row bitwise. No training or changed model selection.',
        adoption='Complete physical checks at this fixed intermediate checkpoint; not final pipeline acceptance.',
        stop='Complete registered cases,physical failure,exception,or wall deadline. Preserve prior incomplete records.',new_ppo_updates=0))
    started=time.perf_counter()
    code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),'--seed',str(args.seed),'--worker'],stdout=out,stderr=err,timeout=1740)
            code=run.returncode
            status='finished' if code==0 else 'error'
        except subprocess.TimeoutExpired:
            status='timeout'
    published={}
    if code==0:
        previous=stage/'previous_incomplete_records'
        previous.mkdir()
        results=read(stage/'result.json')['results']
        for case,result in results.items():
            if result.get('error') is not None:
                continue
            for path in stage.glob(f'eval_0024_{case}*'):
                target=original/path.name
                if target.exists():
                    shutil.copy2(target,previous/path.name)
                shutil.copy2(path,target)
                published[str(target)]=dict(source=str(path),sha256=sha(path))
        write(stage/'canonical_result_binding.json',dict(published=published,
            preserved_previous_directory=str(previous),checkpoint_sha256=sha(checkpoint)))
    write(stage/'replay_receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,
        new_ppo_updates=0,source_unchanged=sha(__file__)==sha(stage/Path(__file__).name),published_count=len(published),full_pipeline_complete=False))
    print('PREVIEW_EVALUATION_COMPLETION',args.seed,status,round(time.perf_counter()-started,2),flush=True)
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-1000:],flush=True)
    if code!=0:
        print((stage/'stderr.txt').read_text(encoding='utf-8')[-1200:],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--worker',action='store_true')
    main(parser.parse_args())
