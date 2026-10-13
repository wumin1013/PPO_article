"""Bounded shared PPO driver for matched2/3-output conditional residual arms."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from evaluate_historical_dense import ROOT,BASE,CASE,sha,write,verify_bound_inputs


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--actions',type=int,choices=[2,3],required=True)
    p.add_argument('--preview',type=float,default=3.)
    p.add_argument('--seed',type=int,default=922302)
    p.add_argument('--initial',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--initial-evaluation',type=Path)
    p.add_argument('--steps',type=int,default=16384)
    p.add_argument('--budget',type=float,default=1200.)
    p.add_argument('--guarded-update',action='store_true')
    p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    assert (a.initial is None)!=(a.resume is None)
    assert Path(a.name).name==a.name and 300<=a.budget<=1800
    return a


def worker(args,stage):
    import numpy as np
    import torch
    import train_compact_ppo_mc as training
    if args.guarded_update:
        from learning_update_guarded_mc import update
        training.update=update
    from historical_residual_ppo import HistoricalResidualRuntime,ResidualPolicy
    from historical_two_action_residual import HistoricalTwoActionRuntime,TwoActionResidualPolicy
    from natural_case_audit import analytic_command_audit
    verify_bound_inputs();torch.set_num_threads(1)
    runtime_class,policy_class=(HistoricalTwoActionRuntime,TwoActionResidualPolicy) if args.actions==2 else (HistoricalResidualRuntime,ResidualPolicy)
    if args.initial:
        saved=torch.load(args.initial,map_location='cpu',weights_only=False)
        class Initialized(policy_class):
            def __init__(self,dimension=17,action_dim=args.actions,seed=args.seed):
                super().__init__(dimension,action_dim,seed)
                self.load_state_dict(saved['policy'])
        training.CompactPolicy=Initialized
    else:training.CompactPolicy=policy_class
    training.TrainingRuntime=runtime_class
    config=SimpleNamespace(seed=args.seed,dimension=17,action_dim=args.actions,no_swing=False,preview=args.preview,
        potential_speed=30.,remaining_encoding='log',resume=args.resume,steps=args.steps,batch=1024,
        evaluate_every=4,budget=args.budget-150.)
    training.worker(config,stage)
    result=json.loads((stage/'result.json').read_text(encoding='utf-8'))
    result.update(training_kind=f'Conditional{args.actions}-output residual PPO around the common historical backbone',
        residual_observation_dimension=17,inherited_observation_dimension=72,inherited_causal_memory_scalars=474,
        independent_full_method_training=False,goal_complete=False,method_frozen=False,
        initialization_checkpoint=str(args.initial or args.resume),initialization_checkpoint_sha256=sha(args.initial or args.resume),
        success_scope='Conditional training of a fresh2-output arm or continued3-output arm;shared historical model and teacher data explicitly retained.')
    result['post_step_kl_guard']=bool(args.guarded_update)
    if args.guarded_update:
        updates=json.loads((stage/'updates.json').read_text(encoding='utf-8'))
        result['maximum_post_update_exact_kl']=max((r['post_update_exact_kl'] for r in updates),default=None)
        result['maximum_post_update_sampled_kl']=max((r['post_update_sampled_kl'] for r in updates),default=None)
    geometry=json.loads((CASE/'geometry.json').read_text(encoding='utf-8'))
    path=stage/'evaluations.json';evaluations=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    best=None
    for record in evaluations:
        trace=stage/f"eval_{record['update']:04d}.npz"
        data=np.load(trace)
        assert data['latents'].shape[1]==args.actions
        if args.actions==2:assert np.all(data['actions'][:,2]==args.preview)
        if not record['result']['success']:continue
        record['result']['analytic']=analytic_command_audit(data['reference'][0],data['rows'],geometry)
        if not record['result']['analytic']['analytic_corridor_pass']:
            record['result'].update(success=False,complete_time_s=None,motion_scope='failure_prefix_only')
        elif best is None or record['result']['complete_time_s']<best['complete_time_s']:
            best=dict(checkpoint=record['checkpoint'],complete_time_s=record['result']['complete_time_s'],evaluation=str(trace.with_suffix('.json')),selection_scope='post-PPO scheduled checkpoint')
        write(trace.with_suffix('.json'),record['result'])
    result['best_after_this_ppo_round']=best
    alternatives=[] if best is None else [best]
    if args.initial_evaluation:
        prior=json.loads(args.initial_evaluation.read_text(encoding='utf-8'))
        if prior['success']:
            alternatives.append(dict(checkpoint=str(args.initial or args.resume),complete_time_s=prior['complete_time_s'],
                evaluation=str(args.initial_evaluation),selection_scope='preserved qualifying input checkpoint'))
    result['best_with_qualifying_input']=min(alternatives,key=lambda x:x['complete_time_s']) if alternatives else None
    result['best']=best
    write(stage/'evaluations.json',evaluations);write(stage/'result.json',result)


def main():
    args=arguments();stage=ROOT/args.name
    if args.worker:worker(args,stage);return
    manifest=verify_bound_inputs();stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    files=[Path(__file__),ROOT/'historical_two_action_residual.py',ROOT/'historical_residual_ppo.py',
        ROOT/'train_compact_ppo_mc.py',ROOT/'learning_update_compact_mc.py',ROOT/'compact_control.py',ROOT/'local_arc_support.py']
    if args.guarded_update:files.append(ROOT/'learning_update_guarded_mc.py')
    hashes={str(p):sha(p) for p in files}
    for p in files:shutil.copy2(p,stage/'source'/p.name)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write(stage/'registration.json',dict(kind='Bounded shared2/3-output conditional PPO protocol',configuration=configuration,
        wall_budget_s=args.budget,child_timeout_s=args.budget-50.,training_budget_s=args.budget-150.,
        decision_budget=args.steps,batch_size=1024,evaluate_every_updates=4,source_sha256=hashes,
        initialization_sha256=sha(args.initial or args.resume),inherited_manifest=manifest,
        post_step_kl_guard=bool(args.guarded_update),
        guarded_update_protocol='Full fresh-batch exact Gaussian and sampled KL <=0.03 after each step; restore actor parameters and Adam moments on rejection; at most8step halvings; retain accepted reduced learning rate' if args.guarded_update else None,
        information_scope='Same17local/nominal inputs and shared historical backbone. Two-output actor and Gaussian have no trained or sampled preview action; executed preview is a single fixed distance.',
        adoption='Only full corridor/vector dynamics/ordered coverage/braking/rest success qualifies. Preserve qualifying starting checkpoint if fine-tuning regresses; distinguish its training history.',
        stop='At the decision/wall budget; no distance or checkpoint-schedule expansion. Total historical,teacher,fit and PPO budgets are reported separately.'))
    start=time.perf_counter()
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            run=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],stdout=out,stderr=err,timeout=args.budget-50.)
            status,code='finished',run.returncode
        except subprocess.TimeoutExpired:status,code='timeout',None
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-start,
        sources_unchanged=all(sha(p)==h for p,h in hashes.items()),goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-5000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-3000:])


if __name__=='__main__':main()
