"""Registered DEVELOPMENT pilot; never evaluate or select on final-test paths.

Frozen baseline modules are imported unchanged. New records, teachers, models,
all failed prefixes and budgets live under results/development_v2/learning_001.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import random
import time
import numpy as np
import torch
from torch import nn
from common import ROOT, load_path, manifest, sha, utc, write_json
from evaluate import rollout
from paired_learning import Policy, fit

OUT=ROOT/'results'/'development_v2'/'learning_001'
CANDIDATES=[
    dict(id='reference_f40_k2_h12_d3',feed=40.,turn_weight=2.,horizon=12.,preview=3.),
    dict(id='f60_k1_h12_d3',feed=60.,turn_weight=1.,horizon=12.,preview=3.),
    dict(id='f80_k1_h12_d3',feed=80.,turn_weight=1.,horizon=12.,preview=3.),
    dict(id='f100_k1_h12_d3',feed=100.,turn_weight=1.,horizon=12.,preview=3.),
    dict(id='f100_k2_h8_d3',feed=100.,turn_weight=2.,horizon=8.,preview=3.),
    dict(id='f80_k1_h8_d1p5',feed=80.,turn_weight=1.,horizon=8.,preview=1.5),
]
PROTECTED=['paired_learning.py','common.py','evaluate.py','metrics.py','runtime_configuration.py','runtime_bootstrap.py','execution_settings.py']


def read(path):return json.loads(path.read_text('utf-8'))


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    file=OUT/'registration.json'
    if file.exists():return read(file)
    rows=[r for r in manifest()['paths'] if r['split'] in ('train','validation')]
    registration=dict(created_utc=utc(),scope='DEVELOPMENT PILOT ONLY; does not replace the v1 main experiment',
        source_sha256=sha(__file__),protected_source_sha256={n:sha(ROOT/'src'/n) for n in PROTECTED},
        candidates=CANDIDATES,paths=rows,final_test_executions_permitted=0,
        teacher_rule='feed = candidate.feed / (1 + candidate.turn_weight * maximum tangent angle over candidate.horizon); zero residual steering; selected fixed preview. Same teacher records supplied to both arms.',
        selection='Require all four training and both validation programs to complete the unchanged independent audit. Among qualifying candidates minimize summed validation completion time, then candidate order. Failed candidates and traces remain.',
        paired_training=dict(seed=101,fit_steps=2000,fit_batch=256,ppo_decisions=8192,checkpoints=[0,4096,8192],
            ppo_clip=.2,ppo_epochs=4,minibatch=64,rollout=256,ppo_lr=3e-5,feature_count=192,
            features='Training-only teacher observations; same seed/centers/scale/radii and initial tensors for both arms.',
            selection='Inspect predeclared initial, 4096 and 8192 checkpoints on validation; no automatic best-checkpoint selection. No other seeds until both 8192 validation paths pass and pilot improvement is reviewed.',
            training_error_rule='Record every episode, partial failed command prefix, exception and unfinished final episode. Count partial physics ticks.',
            fairness='Identical allocated network, fitting targets, fitting and PPO budgets, development paths, KCM/reward/executor. Fixed masks preview output; adaptive initially outputs the same selected preview. No historical learned checkpoint.',
            primary_budget_change='Pilot doubles PPO decisions from 4096 to 8192 equally for both arms; all development search costs separately retained.'))
    write_json(file,registration);return registration


class Teacher:
    def __init__(self,candidate):self.c=candidate
    def action_for(self,env):
        s,index=env.progress();c=self.c;tangents=env.corridor.tangents
        end=min(len(tangents)-1,int(np.searchsorted(env.corridor.arcs,s+c['horizon'],side='right')-1))
        turn=float(np.arccos(np.clip(tangents[index:end+1]@tangents[index],-1.,1.)).max(initial=0.))
        return np.array([0.,c['feed']/(1+c['turn_weight']*turn)/50.-1.])


def prepare():
    registration=register();candidates=[];grid=[]
    if sha(__file__)!=registration['source_sha256']:raise ValueError('Pilot source differs from registered source')
    for c in registration['candidates']:
        records=[]
        for row in registration['paths']:
            file=OUT/'teachers'/c['id']/(row['id']+'.npz')
            if file.with_suffix('.json').exists():result=read(file.with_suffix('.json'))
            else:result=rollout(Teacher(c),load_path(row['id']),file,fixed_d=c['preview'],collect=False,wall_s=90.)
            rec=dict(candidate=c['id'],path=row['id'],split=row['split'],result_file=str(file.relative_to(ROOT)),**result)
            records.append(rec);grid.append(rec)
            write_json(OUT/'teacher_grid.json',dict(scope='development',expected_runs=36,records=grid))
        candidates.append(dict(candidate=c,successes=sum(r['success'] for r in records),runs=len(records),
            all_development_paths_pass=all(r['success'] for r in records),
            validation_sum_s=sum(r['complete_time_s'] for r in records if r['split']=='validation' and r['success']),records=records))
    valid=[r for r in candidates if r['all_development_paths_pass']]
    report=dict(scope='DEVELOPMENT',candidates=candidates,all_failures_retained=True,gate_pass=bool(valid))
    if not valid:
        write_json(OUT/'teacher_selection.json',report);return
    selected=min(valid,key=lambda r:r['validation_sum_s']);report['selected']=selected['candidate']
    report['selected_validation_sum_s']=selected['validation_sum_s']
    xs=[];ys=[];groups=[]
    for row in registration['paths']:
        if row['split']!='train':continue
        file=OUT/'teachers'/selected['candidate']['id']/(row['id']+'.npz')
        with np.load(file) as data:
            x=data['observations'];y=np.c_[data['actions'],np.full(len(x),(selected['candidate']['preview']-6.4)/5.6)]
            xs.append(x);ys.append(y);groups.extend([row['id']]*len(x))
    dataset=OUT/'shared_teacher_records.npz'
    np.savez_compressed(dataset,observations=np.vstack(xs),targets=np.vstack(ys),groups=np.asarray(groups))
    report.update(dataset_sha256=sha(dataset),training_records=len(groups),normalization_and_features_use_validation=False,
        recorded_teacher_actor_calls=sum(r['actor_calls'] for r in grid),teacher_wall_s=sum(r['wall_s'] for r in grid))
    write_json(OUT/'teacher_selection.json',report)
    print('PILOT_TEACHER_SELECTED',selected['candidate'],selected['validation_sum_s'],len(groups),flush=True)


def initialize(seed,arm,selection):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed);sample_rng=np.random.default_rng(seed+10000)
    data=np.load(OUT/'shared_teacher_records.npz');x=torch.tensor(data['observations'],dtype=torch.float32);y=torch.tensor(data['targets'],dtype=torch.float32)
    scale=np.maximum(x.numpy().std(0),.05);centers=x.numpy()[rng.choice(len(x),size=192,replace=len(x)<192)]
    policy=Policy(seed,centers,scale,np.full(192,2.),selection['selected']['preview'],arm)
    return policy,x,y,rng,sample_rng


def evaluate_checkpoint(policy,stage,step,d,registration):
    results=[];started=time.perf_counter()
    for row in registration['paths']:
        if row['split']!='validation':continue
        output=stage/'validation'/f'step_{step:05d}_{row["id"]}.npz'
        result=rollout(policy,load_path(row['id']),output,fixed_d=d if policy.arm=='fixed' else None,wall_s=90.)
        results.append(dict(path=row['id'],**result))
    write_json(stage/f'validation_{step:05d}.json',dict(scope='DEVELOPMENT',step=step,records=results,wall_s=time.perf_counter()-started))


def train(arm,seed=101):
    registration=register();selection=read(OUT/'teacher_selection.json')
    if not selection['gate_pass']:raise RuntimeError('Teacher gate failed')
    if sha(__file__)!=registration['source_sha256']:raise ValueError('Pilot source differs from registered source')
    if sha(OUT/'shared_teacher_records.npz')!=selection['dataset_sha256']:raise ValueError('Teacher dataset hash mismatch')
    from execution_settings import configure_dt
    from runtime_configuration import configure_runtime
    configure_dt(.001);configure_runtime('v2')
    from compact_farvector_runtime import FarVectorRuntime
    from direct_two_action_control import DirectTwoRuntime
    stage=OUT/'paired'/f'seed_{seed}'/arm;stage.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();cost=dict(training_interactions=0,physics_ticks=0,ppo_updates=0,ppo_optimizer_steps=0,ppo_samples=0,
        supervised_optimizer_steps=0,supervised_samples=0,supervised_wall_s=0.,validation_wall_s=0.)
    error=None;curves=[];episodes=[];transitions=[];current=None;policy=None
    write_json(stage/'registration.json',dict(scope='DEVELOPMENT PILOT',seed=seed,arm=arm,source_sha256=sha(__file__),
        pilot_registration_sha256=sha(OUT/'registration.json'),dataset_sha256=sha(OUT/'shared_teacher_records.npz'),budget=registration['paired_training'],
        action_capacity='Unbounded raw Gaussian mean; common physical clipping; selected shared fixed preview at initialization.',
        runtime_version='v2',random_streams=dict(initialization=seed,minibatches=seed,training_path_sampling=seed+10000,action_sampling=seed)))
    try:
        policy,x,y,rng,sample_rng=initialize(seed,arm,selection);d=selection['selected']['preview']
        cost['feature_and_initialization_wall_s']=time.perf_counter()-started
        optimizer=torch.optim.Adam(policy.actor.parameters(),lr=2e-4)
        fit_log=fit(policy,x,y,rng,2000,optimizer,cost);write_json(stage/'fit_curve.json',fit_log)
        def save(step):
            file=stage/f'checkpoint_{step:05d}.pt'
            torch.save(dict(policy=policy.state_dict(),seed=seed,arm=arm,d=d,steps_completed=step,
                torch_rng_state=torch.get_rng_state(),batch_rng_state=rng.bit_generator.state,
                path_rng_state=sample_rng.bit_generator.state,scope='DEVELOPMENT',cost=cost),file)
            before=time.perf_counter();evaluate_checkpoint(policy,stage,step,d,registration);cost['validation_wall_s']+=time.perf_counter()-before
        save(0)
        train_rows=[r for r in registration['paths'] if r['split']=='train']
        def new_env():
            row=train_rows[sample_rng.integers(len(train_rows))];ref=load_path(row['id'])
            env=FarVectorRuntime(ref,potential_speed=100.) if arm=='adaptive' else DirectTwoRuntime(ref,preview_rule_mm=d,potential_speed=100.)
            env.env.max_ticks=int(round((5.+np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()/5.)/.001))
            positions=[ref[0].copy()];original_update=env.tracker.update
            def observe(p):positions.append(np.asarray(p).copy());return original_update(p)
            env.tracker.update=observe
            return dict(env=env,path=row['id'],reference=ref,positions=positions,observations=[],actions=[],rewards=[])
        def retain_episode(done,info):
            index=len(episodes);file=stage/'episodes'/f'episode_{index:04d}.npz';file.parent.mkdir(exist_ok=True)
            np.savez_compressed(file,reference=current['reference'],positions=np.array(current['positions']),
                observations=np.asarray(current['observations']),actions=np.asarray(current['actions']),rewards=np.asarray(current['rewards']))
            episodes.append(dict(path=current['path'],complete=done,executor_success=bool(info.get('success')) if done else None,
                reason=info.get('reason') if done else 'budget_truncated',return_value=float(sum(current['rewards'])),
                decisions=len(current['actions']),ticks=current['env'].ticks,trace_file=str(file.relative_to(ROOT)),trace_sha256=sha(file)))
            write_json(stage/'episodes.json',episodes)
        current=new_env();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-5);buffer=[];ppo_started=time.perf_counter()
        for decision in range(8192):
            if time.perf_counter()-started>900:raise TimeoutError('Registered pilot training interval exceeds 15 minutes')
            env=current['env'];obs=env.observation();action,lp,value=policy.act(obs)
            current['observations'].append(obs);current['actions'].append(action);tick_before=env.ticks
            try:
                next_obs,reward,done,info=env.step(action);next_value=0. if done else float(policy.value(torch.tensor(next_obs)).detach())
            except Exception as exc:
                done=True;reward=-10.;next_value=0.;info=dict(success=False,reason=type(exc).__name__+': '+str(exc))
            finally:cost['physics_ticks']+=env.ticks-tick_before
            current['rewards'].append(reward);buffer.append((obs,action,lp,value,reward,next_value,done))
            transitions.append((decision+1,current['path'],tick_before,env.ticks,reward,done,bool(info.get('success'))))
            cost['training_interactions']+=1
            if done:retain_episode(True,info);current=new_env()
            if len(buffer)==256:
                ob=torch.tensor(np.array([b[0] for b in buffer]),dtype=torch.float32);ac=torch.tensor(np.array([b[1] for b in buffer]),dtype=torch.float32)
                old=torch.tensor([b[2] for b in buffer]);adv=np.zeros(len(buffer));gae=0.
                for k in range(len(buffer)-1,-1,-1):
                    _,_,_,v,r,nv,terminal=buffer[k];delta=r+.99*nv-v;gae=delta+.99*.95*(not terminal)*gae;adv[k]=gae
                targets=torch.tensor(adv+np.array([b[3] for b in buffer]),dtype=torch.float32)
                advantage=torch.tensor(adv,dtype=torch.float32);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                losses=[];kls=[];fractions=[]
                for epoch in range(4):
                    ids=rng.permutation(len(buffer))
                    for start in range(0,len(ids),64):
                        ix=ids[start:start+64];dist=policy.dist(ob[ix]);logp=dist.log_prob(ac[ix]).sum(-1);ratio=torch.exp(logp-old[ix])
                        loss=-torch.minimum(ratio*advantage[ix],ratio.clamp(.8,1.2)*advantage[ix]).mean()+.5*(policy.value(ob[ix])-targets[ix]).square().mean()
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                        losses.append(float(loss.detach()));kls.append(float((old[ix]-logp).mean().detach()));fractions.append(float(((ratio-1).abs()>.2).float().mean()))
                        cost['ppo_optimizer_steps']+=1;cost['ppo_samples']+=len(ix)
                cost['ppo_updates']+=1
                curves.append(dict(step=decision+1,loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),clip_fraction=float(np.mean(fractions)),
                    completed_episodes=len(episodes),executor_successes=sum(bool(e['executor_success']) for e in episodes),mean_batch_reward=float(np.mean([b[4] for b in buffer]))))
                buffer=[];write_json(stage/'training_curve.json',curves)
                if (decision+1)%2048==0:print('PILOT_PPO',seed,arm,decision+1,len(episodes),flush=True)
                if decision+1 in (4096,8192):save(decision+1)
        cost['ppo_interval_with_checkpoint_validation_wall_s']=time.perf_counter()-ppo_started
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if current is not None and current['actions']:
            retain_episode(False,dict(reason=error or 'budget_truncated'))
        cost['total_wall_s']=time.perf_counter()-started
        if policy is not None:
            torch.save(dict(policy=policy.state_dict(),seed=seed,arm=arm,d=selection['selected']['preview'],scope='DEVELOPMENT',cost=cost,error=error),stage/'final.pt')
        write_json(stage/'transitions.json',dict(columns=['decision','program','tick_before','tick_after','reward','done','executor_success'],rows=transitions))
        write_json(stage/'receipt.json',dict(scope='DEVELOPMENT',created_utc=utc(),complete=error is None and cost['training_interactions']==8192,
            error=error,cost=cost,model_sha256=sha(stage/'final.pt') if policy is not None else None,
            source_unchanged=sha(__file__)==read(stage/'registration.json')['source_sha256'],
            protected_sources_unchanged=all(sha(ROOT/'src'/name)==digest for name,digest in registration['protected_source_sha256'].items())))
        print('PILOT_TRAIN_COMPLETE',seed,arm,error,cost,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['register','prepare','train']);parser.add_argument('--arm',choices=['fixed','adaptive']);parser.add_argument('--seed',type=int,default=101)
    args=parser.parse_args()
    if args.mode=='register':register()
    elif args.mode=='prepare':prepare()
    else:train(args.arm,args.seed)
