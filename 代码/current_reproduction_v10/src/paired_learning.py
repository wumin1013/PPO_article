"""Fresh paired PPO with deterministic training-only records and complete ledgers.

This is a newly registered lower-budget comparison, not a reproduction of the
historical four-stage optimization. Neither arm inherits a trained network.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import ROOT, load_path, manifest, sha, utc, write_json
from evaluate import rollout

SEEDS=[101,211,307,401,503]
FIXED_CANDIDATES=[.8,1.5,3.,4.5,6.,12.]


class Teacher:
    """Geometric LOS teacher with conservative feed; no learned checkpoint."""
    def __init__(self,d):self.d=d
    def action_for(self,env):
        s,index=env.progress()
        tangents=env.corridor.tangents
        end=min(len(tangents)-1,int(np.searchsorted(env.corridor.arcs,s+12,side='right')-1))
        dots=np.clip(tangents[index:end+1]@tangents[index],-1,1)
        turn=float(np.arccos(dots).max(initial=0))
        feed=40./(1+2*turn)
        return np.array([0.,feed/50.-1.])


class Actor(nn.Module):
    def __init__(self,seed,centers,scale,radii,d,arm):
        super().__init__();self.arm=arm;self.d=d
        torch.manual_seed(seed)
        self.base=nn.Sequential(nn.Linear(14,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,3))
        self.correction=nn.Linear(len(centers),3,bias=False)
        self.register_buffer('centers',torch.tensor(centers,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.register_buffer('radii',torch.tensor(radii,dtype=torch.float32))
        self.register_buffer('offset',torch.tensor([0.,-.6,(d-6.4)/5.6],dtype=torch.float32))
        self.register_buffer('gain',torch.tensor([.03,.6,.12],dtype=torch.float32))
        nn.init.zeros_(self.base[-1].weight);nn.init.zeros_(self.base[-1].bias);nn.init.zeros_(self.correction.weight)
    def forward(self,x):
        z=(x[...,None,:]-self.centers)/self.scale
        f=torch.exp(-.5*z.square().sum(-1)/self.radii.square())
        # Unbounded Gaussian means retain the complete physical action range
        # after the common executor's clip. A final tanh here would cap feed
        # at 50 mm/s and confound comparisons to 100 mm/s planners.
        return self.offset+self.gain*(self.base(x)+self.correction(f))


class Policy(nn.Module):
    def __init__(self,seed,centers,scale,radii,d,arm):
        super().__init__();self.arm=arm;self.d=d
        self.actor=Actor(seed,centers,scale,radii,d,arm)
        torch.manual_seed(seed+100000)
        self.critic=nn.Sequential(nn.Linear(14,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        self.log_std=nn.Parameter(torch.log(torch.tensor([.008,.03,.02])))
    def dist(self,x):
        mu=self.actor(x)
        n=2 if self.arm=='fixed' else 3
        return torch.distributions.Normal(mu[...,:n],self.log_std[:n].clamp(-8,-2).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    @torch.no_grad()
    def act(self,x,deterministic=False):
        x=torch.as_tensor(x,dtype=torch.float32);d=self.dist(x)
        a=d.mean if deterministic else d.sample()
        return a.numpy(),float(d.log_prob(a).sum()),float(self.value(x))


def select_and_collect():
    folder=ROOT/'results'/'teacher_selection_v3'
    folder.mkdir(parents=True,exist_ok=True)
    val=[r for r in manifest()['paths'] if r['split']=='validation']
    results=[]
    for d in FIXED_CANDIDATES:
        record=[]
        for path in val:
            output=folder/(path['id']+'_d'+str(d)+'.npz')
            if output.with_suffix('.json').exists():r=json.loads(output.with_suffix('.json').read_text('utf-8'))
            else:r=rollout(Teacher(d),load_path(path['id']),output,fixed_d=d)
            record.append(r)
        results.append(dict(d=d,successes=sum(r['success'] for r in record),
                            total_time=sum(r['complete_time_s'] or 1e6 for r in record),results=record))
    selected=min(results,key=lambda r:(-r['successes'],r['total_time'],r['d']))
    if selected['successes']!=len(val):
        write_json(folder/'failed_gate.json',dict(candidates=results,reason='No teacher completed all validation programs'))
        raise RuntimeError('Teacher validation gate failed; do not launch five-seed training')
    d=selected['d']
    write_json(folder/'selection.json',dict(created_utc=utc(),candidates=results,selected_preview_mm=d,
        rule='Maximize validation complete successes, then minimize summed complete time; shared initial D for both arms.',
        test_paths_used=0,teacher='Deterministic geometric LOS/feed rule, no historical policy or optimized teacher',
        steering_conversion='Both arm demonstrations use identical preview and guide, hence delta_fixed = delta_adaptive + beta_adaptive - beta_fixed = delta_adaptive = 0.'))
    xs=[];ys=[];groups=[];receipts=[]
    for row in manifest()['paths']:
        if row['split']!='train':continue
        output=folder/(row['id']+'.npz')
        if output.with_suffix('.json').exists():
            r=json.loads(output.with_suffix('.json').read_text('utf-8')); data=dict(np.load(output))
        else:r,data=rollout(Teacher(d),load_path(row['id']),output,fixed_d=d,collect=True)
        receipts.append(dict(path=row['id'],result=r))
        # Failed trajectories remain in ledger; never provide infeasible motion as successful supervision.
        if r['success']:
            x=data['observations'];target=np.c_[data['actions'],np.full(len(x),(d-6.4)/5.6)]
            xs.append(x);ys.append(target);groups.extend([row['id']]*len(x))
    if not xs:raise RuntimeError('No complete feasible teacher programs; cannot pass the training pipeline gate')
    dataset=ROOT/'data'/'teacher_records_v3.npz'
    np.savez_compressed(dataset,observations=np.vstack(xs),targets=np.vstack(ys),groups=np.array(groups))
    write_json(ROOT/'data'/'teacher_records_v3.json',dict(created_utc=utc(),sha256=sha(dataset),records=sum(map(len,xs)),
        selected_preview_mm=d,programs=receipts,failed_demonstrations_excluded_from_fitting_but_retained=True,
        data_scope='training programs only; no validation/test observations in normalization or features'))


def fit(policy,x,y,rng,steps,optimizer,cost):
    log=[];started=time.perf_counter()
    for step in range(steps):
        ids=torch.as_tensor(rng.integers(len(x),size=256))
        predicted=policy.actor(x[ids])
        loss=((predicted-y[ids])/torch.tensor([.03,.3,.12])).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.actor.parameters(),1.);optimizer.step()
        cost['supervised_optimizer_steps']+=1;cost['supervised_samples']+=256
        if step%100==0 or step==steps-1:log.append(dict(step=step+1,loss=float(loss.detach())))
    cost['supervised_wall_s']+=time.perf_counter()-started
    return log


def train(seed,arm,steps=4096,clip=.2,tag='fullrange'):
    from runtime_bootstrap import setup
    setup()
    from runtime_configuration import configure_runtime
    configure_runtime('v2')
    from compact_farvector_runtime import FarVectorRuntime
    from direct_two_action_control import DirectTwoRuntime
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed);sample_rng=np.random.default_rng(seed+10000)
    stage=ROOT/'results'/'paired'/tag/f'seed_{seed}'/arm
    stage.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=started+1680
    data=np.load(ROOT/'data'/'teacher_records_v3.npz');x=torch.tensor(data['observations'],dtype=torch.float32);y=torch.tensor(data['targets'],dtype=torch.float32)
    d=json.loads((ROOT/'data'/'teacher_records_v3.json').read_text('utf-8'))['selected_preview_mm']
    scale=np.maximum(x.numpy().std(0),.05)
    centers=x.numpy()[rng.choice(len(x),size=192,replace=len(x)<192)]
    radii=np.full(192,2.)
    policy=Policy(seed,centers,scale,radii,d,arm)
    source_hash=sha(__file__)
    write_json(stage/'registration.json',dict(created_utc=utc(),seed=seed,arm=arm,steps=steps,clip=clip,
        source_sha256=source_hash,dataset_sha256=sha(ROOT/'data'/'teacher_records_v3.npz'),split_sha256=sha(ROOT/'data'/'manifest.json'),
        runtime_version='v2',runtime_configuration_sha256=sha(ROOT/'src'/'runtime_configuration.py'),
        action_capacity='Unbounded raw Gaussian mean; common physical clipping gives steering +/-pi/2, feed 0--100 mm/s, preview 0.8--12 mm. Fixed preview remains selected 3 mm.',
        feature_scope='192 centers selected only from successful training teacher observations, per-seed; feature scales training-only; radius fixed before fitting',
        initialization='No learned checkpoint. Paired seeds have identical actor/critic initial tensors, records, centers, scales and feature counts. Fixed arm masks the third output.',
        supervised_iterations=2000,final_fit_iterations=0,ppo_epochs=4,batch_size=64,rollout_size=256,
        wall_budget_s=1680,selection='Final checkpoint, no success-conditioned or test-driven selection',
        random_streams=dict(initialization=seed,training_path_sampling=seed+10000,minibatches=seed,action_sampling=seed),
        network_trainable_parameters=sum(p.numel() for p in policy.parameters()),
        capacity_difference='Same allocated network; third mean and log_std inactive in fixed PPO. Both fit identical 3-channel labels before PPO.'))
    cost=dict(training_interactions=0,physics_ticks=0,ppo_updates=0,ppo_optimizer_steps=0,ppo_samples=0,
              supervised_optimizer_steps=0,supervised_samples=0,supervised_wall_s=0.,ppo_wall_s=0.,
              inherited_learned_training=0,hyperparameter_trials=0,feature_construction_wall_s=time.perf_counter()-started)
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=2e-4)
    fit_log=fit(policy,x,y,rng,2000,optimizer,cost)
    torch.save(dict(policy=policy.state_dict(),seed=seed,arm=arm,d=d,stage='fitted'),stage/'initial.pt')
    write_json(stage/'fit_curve.json',fit_log)
    train_paths=[load_path(r['id']) for r in manifest()['paths'] if r['split']=='train']
    def new_env():
        ref=train_paths[sample_rng.integers(len(train_paths))]
        result=(FarVectorRuntime(ref,potential_speed=100.) if arm=='adaptive' else DirectTwoRuntime(ref,preview_rule_mm=d,potential_speed=100.))
        result.env.max_ticks=int(round((5.+np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()/5.)/.001))
        return result
    env=new_env();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-5)
    buffer=[];curves=[];episodes=[];episode_return=0.;episode_steps=0;error=None
    train_started=time.perf_counter()
    try:
        for decision in range(steps):
            if time.perf_counter()>deadline:raise TimeoutError('Registered training deadline')
            obs=env.observation();action,lp,value=policy.act(obs)
            tick_before=env.ticks
            try:
                next_obs,reward,done,info=env.step(action)
                next_value=0. if done else float(policy.value(torch.tensor(next_obs)).detach())
            except Exception as exc:
                done=True;reward=-10.;next_value=0.;info={'reason':type(exc).__name__+':'+str(exc),'success':False}
            finally:
                cost['physics_ticks']+=env.ticks-tick_before
            buffer.append((obs,action,lp,value,reward,next_value,done))
            cost['training_interactions']+=1;episode_return+=reward;episode_steps+=1
            if done:
                episodes.append(dict(end_decision=decision+1,return_value=episode_return,decisions=episode_steps,
                    success=bool(info.get('success')),reason=info.get('reason')))
                env=new_env();episode_return=0.;episode_steps=0
            if len(buffer)==256 or decision+1==steps:
                ob=torch.tensor(np.array([b[0] for b in buffer]),dtype=torch.float32)
                ac=torch.tensor(np.array([b[1] for b in buffer]),dtype=torch.float32)
                old=torch.tensor([b[2] for b in buffer]);adv=np.zeros(len(buffer));gae=0.
                for k in range(len(buffer)-1,-1,-1):
                    _,_,_,v,r,nv,terminal=buffer[k]
                    delta=r+.99*nv-v;gae=delta+.99*.95*(not terminal)*gae;adv[k]=gae
                targets=torch.tensor(adv+np.array([b[3] for b in buffer]),dtype=torch.float32)
                advantage=torch.tensor(adv,dtype=torch.float32);advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                losses=[];kls=[];clipfractions=[]
                for epoch in range(4):
                    ids=rng.permutation(len(buffer))
                    for start in range(0,len(ids),64):
                        ix=ids[start:start+64];dist=policy.dist(ob[ix]);logp=dist.log_prob(ac[ix]).sum(-1)
                        ratio=torch.exp(logp-old[ix]);surrogate=torch.minimum(ratio*advantage[ix],ratio.clamp(1-clip,1+clip)*advantage[ix])
                        loss=-surrogate.mean()+.5*(policy.value(ob[ix])-targets[ix]).square().mean()
                        if not torch.isfinite(loss):raise FloatingPointError('nonfinite PPO loss')
                        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                        losses.append(float(loss.detach()));kls.append(float((old[ix]-logp).mean().detach()));clipfractions.append(float(((ratio-1).abs()>clip).float().mean()))
                        cost['ppo_optimizer_steps']+=1;cost['ppo_samples']+=len(ix)
                cost['ppo_updates']+=1
                curves.append(dict(decision=decision+1,loss=float(np.mean(losses)),approx_kl=float(np.mean(kls)),clip_fraction=float(np.mean(clipfractions)),
                                   completed_episodes=len(episodes),successful_episodes=sum(e['success'] for e in episodes),mean_batch_reward=float(np.mean([b[4] for b in buffer]))))
                buffer=[];write_json(stage/'training_curve.json',curves);write_json(stage/'episodes.json',episodes)
                print('PPO',seed,arm,decision+1,curves[-1]['successful_episodes'],flush=True)
    except Exception as exc:error=type(exc).__name__+': '+str(exc)
    cost['ppo_wall_s']=time.perf_counter()-train_started;cost['total_wall_s']=time.perf_counter()-started
    payload=dict(policy=policy.state_dict(),seed=seed,arm=arm,d=d,cost=cost,clip=clip,steps_completed=cost['training_interactions'],error=error,
        optimizer=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),batch_rng_state=rng.bit_generator.state,
        path_rng_state=sample_rng.bit_generator.state,unfinished_training_episode=dict(return_value=episode_return,decisions=episode_steps))
    torch.save(payload,stage/'final.pt')
    write_json(stage/'receipt.json',dict(created_utc=utc(),complete=error is None and cost['training_interactions']==steps,error=error,cost=cost,
        model_sha256=sha(stage/'final.pt'),source_unchanged=sha(__file__)==source_hash))
    print('TRAIN_COMPLETE',seed,arm,error,cost,flush=True)


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['policy']
    policy=Policy(saved['seed'],state['actor.centers'].numpy(),state['actor.scale'].numpy(),state['actor.radii'].numpy(),saved['d'],saved['arm'])
    policy.load_state_dict(state);policy.eval()
    return policy,saved


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--arm',choices=['fixed','adaptive'],default='adaptive');p.add_argument('--steps',type=int,default=4096);p.add_argument('--clip',type=float,default=.2);p.add_argument('--tag',default='fullrange')
    a=p.parse_args()
    if a.mode=='prepare':select_and_collect()
    else:train(a.seed,a.arm,a.steps,a.clip,a.tag)
