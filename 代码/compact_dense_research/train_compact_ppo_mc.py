"""Registered, bounded PPO training of the compact controller from fresh weights."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
from compact_control import CompactPolicy, CompactRuntime, feature_names, BASE
from compact_training_environment import TrainingRuntime
from learning_update_compact_mc import optimizers, update
from paper_complete_evaluation import evaluate

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--seed',type=int,default=922301)
    p.add_argument('--dimension',type=int,choices=[14,12,10],default=14)
    p.add_argument('--action-dim',type=int,choices=[2,3],default=3)
    p.add_argument('--no-swing',action='store_true')
    p.add_argument('--preview',type=float,default=1.5)
    p.add_argument('--potential-speed',type=float,default=100.)
    p.add_argument('--remaining-encoding',choices=['terminal','log'],default='terminal')
    p.add_argument('--steps',type=int,default=32768)
    p.add_argument('--batch',type=int,default=2048)
    p.add_argument('--evaluate-every',type=int,default=2)
    p.add_argument('--budget',type=float,default=1200.)
    p.add_argument('--resume',type=Path)
    p.add_argument('--worker',action='store_true')
    return p.parse_args()

def worker(args,stage):
    started=time.perf_counter();deadline=started+args.budget-45.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    reference=np.loadtxt(ROOT/'dense_closed_case/reference.csv',delimiter=',',skiprows=1)
    def environment():
        return TrainingRuntime(reference,dimension=args.dimension,action_dim=args.action_dim,
            preview_rule_mm=args.preview,allow_swing=not args.no_swing,
            potential_speed=args.potential_speed,remaining_encoding=args.remaining_encoding)
    policy=CompactPolicy(args.dimension,args.action_dim,args.seed)
    optimizer=optimizers(policy);env=environment()
    total_steps=0;total_updates=0;episode_reward=0.;episodes=0
    evaluations=[];updates=[];episode_records=[];best=None;prior_steps=0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        required=(args.dimension,args.action_dim,args.seed,args.no_swing,args.preview,args.potential_speed,args.remaining_encoding)
        assert tuple(saved['interface'])==required
        policy.load_state_dict(saved['policy'])
        for key in optimizer:optimizer[key].load_state_dict(saved['optimizers'][key])
        env.restore(saved['environment'])
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        total_steps=prior_steps=saved['steps'];total_updates=saved['updates']
        episode_reward=saved['episode_reward'];episodes=saved['episodes'];best=saved['best']

    def checkpoint(label):
        path=stage/(label+'.pt')
        torch.save(dict(policy=policy.state_dict(),optimizers={k:v.state_dict() for k,v in optimizer.items()},
            environment=env.state(),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),steps=total_steps,updates=total_updates,episode_reward=episode_reward,
            episodes=episodes,best=best,interface=[args.dimension,args.action_dim,args.seed,args.no_swing,args.preview,args.potential_speed,args.remaining_encoding],
            observation_features=env.feature_names(),policy_memory_scalars=0),path)
        return path

    checkpoint('initial')
    target=prior_steps+args.steps
    while total_steps<target and time.perf_counter()<deadline-35.:
        batch={key:[] for key in ['observation','latent','log_probability','value','reward','next_value','terminal']}
        held=[];progress=[];prefixes=[];collect_started=time.perf_counter()
        while len(batch['reward'])<min(args.batch,target-total_steps) and time.perf_counter()<deadline-35.:
            x=env.observation();latent,logp,value=policy.act(x)
            x2,reward,done,info=env.step(latent)
            next_value=0. if done else float(policy.value(torch.as_tensor(x2)).detach())
            values=[x,latent,logp,value,reward,next_value,done]
            for key,value in zip(batch,values):batch[key].append(value)
            held.append(info['held_action']);progress.append(env.progress()[0]);prefixes.append(env.ticks*.001)
            episode_reward+=reward
            if done:
                record=dict(episode=episodes,step=total_steps+len(batch['reward']),success=env.success,reason=env.reason,
                    complete_time_s=env.ticks*.001 if env.success else None,prefix_time_s=env.ticks*.001,
                    gates=env.tracker.frontier,progress_mm=env.progress()[0],reward=episode_reward)
                episode_records.append(record)
                with (stage/'episodes.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(record)+'\n')
                episodes+=1;episode_reward=0.;env=environment()
        if len(batch['reward'])<2:break
        collect_wall=time.perf_counter()-collect_started
        np.savez_compressed(stage/f'batch_{total_updates+1:04d}.npz',**{k:np.asarray(v) for k,v in batch.items()},
            held_action=np.asarray(held),progress_mm=np.asarray(progress),episode_time_s=np.asarray(prefixes))
        update_started=time.perf_counter();statistics=update(policy,optimizer,batch)
        total_updates+=1;total_steps+=len(batch['reward'])
        stats=dict(update=total_updates,steps=total_steps,collect_wall_s=collect_wall,update_wall_s=time.perf_counter()-update_started,
            completed_episodes=episodes,episode_reasons=dict(Counter(r['reason'] for r in episode_records)),
            requested_action_mean=np.mean(held,axis=0).tolist(),latent_std=policy.log_std.exp().detach().tolist(),**statistics)
        updates.append(stats);write(stage/'updates.json',updates)
        print('UPDATE',json.dumps(stats),flush=True)
        path=checkpoint(f'checkpoint_{total_updates:04d}')
        if total_updates%args.evaluate_every==0 and time.perf_counter()<deadline-30.:
            result=evaluate(policy,environment(),stage/f'eval_{total_updates:04d}.npz',deadline)
            record=dict(update=total_updates,steps=total_steps,checkpoint=str(path),checkpoint_sha256=sha(path),result=result)
            evaluations.append(record)
            if result['success'] and (best is None or result['complete_time_s']<best['complete_time_s']):
                best=dict(checkpoint=str(path),complete_time_s=result['complete_time_s'],evaluation=str(stage/f'eval_{total_updates:04d}.json'))
            write(stage/'evaluations.json',evaluations)
            print('EVALUATION',total_updates,result['success'],result['reason'],result['complete_time_s'],result['elapsed_prefix_or_complete_s'],flush=True)
        write(stage/'live.json',dict(status='training',steps=total_steps,updates=total_updates,best=best,wall_s=time.perf_counter()-started,goal_complete=False))
    final=checkpoint('final')
    write(stage/'result.json',dict(status='step_budget_completed' if total_steps>=target else 'wall_budget_completed',
        independent_seed=args.seed,observation_dimension=args.dimension,actor_outputs=args.action_dim,allow_swing=not args.no_swing,
        ppo_updates_this_round=len(updates),ppo_updates_total=total_updates,new_decisions=total_steps-prior_steps,
        total_decisions=total_steps,episodes_this_round=len(episode_records),episode_reasons=dict(Counter(r['reason'] for r in episode_records)),
        best=best,final_checkpoint=str(final),wall_s=time.perf_counter()-started,goal_complete=False,
        success_scope='Exploratory training and scheduled development evaluation; not final three-seed acceptance.'))

def main():
    args=arguments()
    assert 60<=args.budget<=1800 and args.steps>0 and args.batch>1
    stage=ROOT/args.name
    assert stage.parent==ROOT and stage.name==args.name
    if args.worker:worker(args,stage);return
    stage.mkdir(exist_ok=False);(stage/'source').mkdir()
    sources=[Path(__file__),ROOT/'compact_control.py',ROOT/'compact_training_environment.py',ROOT/'learning_update_compact_mc.py',ROOT.parent/'turn_aware_projector.py',
        BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',
        BASE/'terminal_candidate_runtime_v1.py',BASE/'learning_update_v4.py',BASE/'joint_policy.py',
        BASE/'continuous_gate_guidance.py',BASE/'geometry_spec.py',BASE/'paper_complete_evaluation.py',
        ROOT/'dense_closed_case/registration.json']
    manifest={}
    for index,path in enumerate(sources):
        if not path.exists():
            alternate=ROOT.parent/path.name
            if alternate.exists():path=alternate
        target=stage/'source'/f'{index:02d}_{path.name}'
        shutil.copy2(path,target);manifest[str(path)]={'sha256':sha(path),'snapshot':str(target)}
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write(stage/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),kind='Fresh PPO training' if not args.resume else 'PPO continuation',
        config=config,wall_budget_s=args.budget,child_timeout_s=args.budget-10.,sources=manifest,
        reference_sha256=sha(ROOT/'dense_closed_case/reference.csv'),policy_memory_scalars=0,
        sampler='Raw independent Gaussian latent and its unmodified joint likelihood; clipping only in common action decoder.',
        optimizer='Separate Adam; actor3e-4,critic1e-3; critic32epochs of MonteCarlo targets with unfinished-segment bootstrap, then refreshedGAE; actor6epochs,minibatch256;clip0.2,KLstop0.03,GAElambda0.98,gamma1.',
        adoption='Scheduled complete success first, then shorter full physical time. No failure prefix counted as complete.',
        stop='At registered decision or wall budget; preserve failed evaluations, batches, checkpoints, optimizer and RNG. Diagnose absent gains before continuation.',
        baseline='S2 complete19.957s at1ms; target18.95915s. Independent two-action comparison still required.',goal_complete=False))
    started=time.perf_counter();code=None
    with (stage/'stdout.txt').open('w',encoding='utf-8') as out,(stage/'stderr.txt').open('w',encoding='utf-8') as err:
        try:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:],'--worker'],stdout=out,stderr=err,timeout=args.budget-10.)
            code=result.returncode;status='finished'
        except subprocess.TimeoutExpired:status='timeout'
    unchanged=all(sha(path)==info['sha256'] for path,info in manifest.items())
    write(stage/'receipt.json',dict(status=status,exit_code=code,wall_s=time.perf_counter()-started,sources_unchanged=unchanged,goal_complete=False))
    print((stage/'stdout.txt').read_text(encoding='utf-8')[-12000:])
    if code!=0:print((stage/'stderr.txt').read_text(encoding='utf-8')[-5000:])

if __name__=='__main__':main()
