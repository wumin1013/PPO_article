"""Development-only PPO screening for efficient in-tolerance corner passage.

Three sequential candidates share the 20 ms decision / 1 ms KCM interface:
joint tracking; remove the redundant angular residual; replace interior
tracking/smoothness preferences with travel-time cost. Fixed controls receive
every common change. No test command is available in this development runner.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

import preview_event_study as old
from preview_macro_ppo import PreviewMacroPPO
from terminal_braking import plan_terminal_brake
from terminal_metrics import evaluate_terminal_trajectory

OUT=old.ROOT/'论文项目/revision_20260909_terminal/macro_study'
CANDIDATES=('joint_tracking','preview_tracking','preview_time')
METHODS=('fixed1p2','fixed3','fixed4','adaptive')
DISTANCES={'fixed1p2':1.2,'fixed3':3.,'fixed4':4.,'adaptive':3.}
PERIOD=20


def source_state():
    result=old.source_state()
    for name in ('preview_macro_study.py','preview_macro_ppo.py','terminal_braking.py','terminal_metrics.py'):
        p=Path(__file__).with_name(name)
        result['source_files'][p.relative_to(old.ROOT).as_posix()]=old.sha(p)
    import hashlib,json
    result['source_sha256']=hashlib.sha256(json.dumps(result['source_files'],sort_keys=True).encode()).hexdigest()
    return result


def configuration(candidate,method,seed):
    if candidate not in CANDIDATES or method not in METHODS:raise ValueError('Unregistered candidate/method')
    cfg=old.config_for('state_direct' if method=='adaptive' else 'fixed3',seed)
    distance=DISTANCES[method]
    active=([0] if candidate=='joint_tracking' else [])+[1]+([2] if method=='adaptive' else [])
    cfg['revision'].update(study='macro-preview-development-v1',candidate=candidate,variant=method,
        effective_variant=method,adaptive_preview=method=='adaptive',preview_period_steps=1,
        initial_preview_mm=distance,fixed_distance_mm=distance,active_action_dims=active,
        motion_anchor_beta=0.,macro_period_steps=PERIOD,
        objective='ordered progress minus physical elapsed time' if candidate=='preview_time' else 'prior time-integrated tracking objective',
        terminal_definition='Same finite brake and full 20ms rest audit; complete tail cost included',
        data_status='Prior train/validation only; no held-out access or final performance claim')
    cfg['reward_weights']['lookahead']['distance']=distance
    cfg['ppo'].update(actor_lr=3e-4,critic_lr=3e-4,gamma=math.exp(-.020/10),lmbda=1.)
    return cfg


class MacroEnvironment:
    def __init__(self,cfg,path_id,reference=None):
        self.cfg=cfg;self.reference=old.points(path_id) if reference is None else np.asarray(reference,float)
        self.inner=old.EventPreviewEnvironment(cfg,path_id,reference=self.reference)
        self.state=self.inner.state;self.observation_dim=len(self.state)
        self.trace=[old.sample_row(self.inner,{},0.,[0,0,self.inner.held_preview])]
        self.physics_steps=0;self.command_steps=0;self.done=False;self.result=None
        self.last_info={};self.last_macro_duration_s=0.

    def _audit(self,reason,claimed):
        return evaluate_terminal_trajectory(self.reference,[[r['x_mm'],r['y_mm']] for r in self.trace],
            [r['timestamp_s'] for r in self.trace],tolerance_mm=self.cfg['environment']['epsilon']/2,
            limits=self.cfg['kinematic_constraints'],witness_spacing_mm=.01,
            initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=self.trace[0]['heading_rad'],
            initial_angular_velocity=0.,initial_angular_acceleration=0.,termination_reason=reason,claimed_success=claimed)

    def _finish(self,info):
        reason=info['done_reason'];tail=None
        if reason=='success':
            c=self.trace[-1]
            try:
                tail=plan_terminal_brake([c['x_mm'],c['y_mm']],c['heading_rad'],c['v_mm_s'],c['a_mm_s2'],
                    c['omega_rad_s'],c['alpha_rad_s2'],dt=.001,limits=self.cfg['kinematic_constraints'])
                d=tail['columns'];base_time=c['timestamp_s']
                for i in range(1,len(d['time_s'])):
                    row={k:float(v[i]) for k,v in d.items() if k!='time_s'}
                    row.update(timestamp_s=base_time+d['time_s'][i],lookahead_mm=c['lookahead_mm'],terminal_braking=1.)
                    self.trace.append(row)
            except ValueError as exc:
                info['brake_failure']=str(exc);reason='terminal_brake_failure'
        self.result=self._audit('completed' if tail else reason,bool(tail))
        self.result['internal_terminal_rest']=bool(tail and tail['internal_terminal_rest'])
        info['done_reason']='success' if self.result['success'] else reason if reason!='success' else 'terminal_audit_failure'
        info['terminal_audit']=self.result
        self.done=True
        return 0. if tail is None else tail['total_tail_time_s']

    def step(self,action):
        if self.done:raise RuntimeError('Completed macro episode')
        action=np.asarray(action,float).copy()
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Finite three-vector required')
        if 0 not in self.cfg['revision']['active_action_dims']:action[0]=0.
        if not self.cfg['revision']['adaptive_preview']:action[2]=(self.cfg['revision']['fixed_distance_mm']-.8)/3.8
        reward=0.;duration=0.;discount_tick=math.exp(-.001/10)
        for k in range(PERIOD):
            self.state,old_reward,done,info=self.inner.step(action)
            components=info['reward_components']
            tick_reward=(components['progress_mm']-.001 if self.cfg['revision']['candidate']=='preview_time'
                         else old_reward-components.get('r_terminal',0.))
            reward+=discount_tick**k*tick_reward
            self.physics_steps+=1;duration+=.001
            row=old.sample_row(self.inner,info,tick_reward,action)
            event=float(k==0 and self.cfg['revision']['adaptive_preview'])
            row.update(macro_decision=float(k==0),preview_event=event,
                       macro_phase=float(k),control_status__preview_event=event)
            self.trace.append(row)
            if done:
                tail_time=self._finish(info)
                # Same discounted 1 ms time-cost sum in policy and brake phases.
                tail_ticks=int(round(tail_time/.001))
                tail_cost=.001*(-math.expm1(-tail_ticks*.001/10))/(-math.expm1(-.001/10))
                reward-=math.exp(-duration/10)*tail_cost
                duration+=tail_time
                bonus=10. if self.result['success'] else -20.
                reward+=math.exp(-duration/10)*bonus
                break
        self.command_steps=len(self.trace)-1;self.last_info=info;self.last_macro_duration_s=duration
        return self.state.copy(),float(reward),self.done,info


def make_agent(cfg,env):
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    return PreviewMacroPPO(state_dim=env.observation_dim,device=torch.device('cpu'),
        active_action_dims=cfg['revision']['active_action_dims'],
        fixed_preview=(cfg['revision']['fixed_distance_mm']-.8)/3.8,
        **{k:cfg['ppo'][k] for k in keys})


def prepare():
    old.check_protocol()
    protocol=dict(version='macro-preview-development-v1',source=source_state(),
        manifest_sha256=old.sha(old.DATA/'manifest.json'),candidates=list(CANDIDATES),methods=list(METHODS),
        pilot_seed=101,pilot_macro_decisions=4096,smoke_macro_decisions=32,
        initial_screen='Each candidate: fixed3 and adaptive. Only promising valid candidates proceed to short/medium/long fixed grid.',
        objective_sequence='joint_tracking -> preview_tracking removes angular action only -> preview_time replaces interior tracking/smoothness costs only',
        shared_changes='20ms all-action decisions, 1ms KCM, no narrow motion anchor, initial std [.03,.05,.15], actor/critic lr 3e-4, full terminal braking',
        equal_budget='Equal macro decision counts; report exact physical interactions, generated brake samples and wall time separately',
        test_access=False,success_priority='Complete path and constraints and rest; then paired time on common successes',
        failures='Keep all training episodes, failed policies and validation outcomes',machine_execution_authorized=False)
    path=OUT/'protocol.json'
    if path.exists():
        previous=old.read(path)
        if previous!=protocol:raise ValueError('Registered source/config changed; keep existing study')
    else:old.write(path,protocol)
    return protocol


def train(candidate,method,seed,decisions,stage):
    torch.set_num_threads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    protocol=prepare()
    if stage=='pilot' and (seed!=101 or decisions!=4096):raise ValueError('Registered pilot budget is 4096 decisions, seed101')
    if stage=='smoke' and not 1<=decisions<=64:raise ValueError('Smoke budget 1..64')
    cfg=configuration(candidate,method,seed);run=OUT/stage/f'{candidate}_{method}_seed{seed}'
    run.mkdir(parents=True,exist_ok=True);checkpoint=run/'checkpoint.pt'
    config_path=run/'config.yaml'
    if config_path.exists():
        if yaml.safe_load(config_path.read_text(encoding='utf-8'))!=cfg:raise ValueError('Changed saved config')
        if old.read(run/'provenance.json')['source_sha256']!=protocol['source']['source_sha256']:raise ValueError('Changed saved code')
    else:
        config_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        old.write(run/'provenance.json',dict(**protocol['source'],config_sha256=old.sha(config_path),
            protocol_sha256=old.sha(OUT/'protocol.json'),command='python '+' '.join(sys.argv)))
        with zipfile.ZipFile(run/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in protocol['source']['source_files']:z.write(old.ROOT/name,name)
    ids=cfg['revision']['training_path_ids'];episode=steps=physical_before=commands_before=0;wall_before=0.;actions=[];ep_return=0.
    env=MacroEnvironment(cfg,ids[0]);agent=make_agent(cfg,env)
    if checkpoint.exists():
        c=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for name in ('actor','critic','actor_optimizer','critic_optimizer'):getattr(agent,name).load_state_dict(c[name])
        steps,episode,physical_before,commands_before,wall_before,actions,ep_return=(c[k] for k in
            ('decisions','episode','completed_episode_physics_steps','completed_episode_command_steps','training_wall_s','episode_actions','ep_return'))
        env=MacroEnvironment(cfg,ids[episode%len(ids)])
        for action in actions:env.step(action)
        np.testing.assert_array_equal(env.state,c['observation'])
        torch.set_rng_state(c['torch_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng'])
        agent.logged_steps=steps
    if steps>=decisions:
        _complete_training(run,cfg,steps,episode,physical_before,commands_before,env,wall_before)
        return run
    import json
    started=time.perf_counter()
    keys=('states','actions','next_states','rewards','dones','terminated','old_log_probs','action_masks')
    buffer={key:[] for key in keys}
    def save():
        c=dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),actor_optimizer=agent.actor_optimizer.state_dict(),
            critic_optimizer=agent.critic_optimizer.state_dict(),decisions=steps,episode=episode,
            completed_episode_physics_steps=physical_before,completed_episode_command_steps=commands_before,
            episode_actions=actions,ep_return=ep_return,training_wall_s=wall_before+time.perf_counter()-started,
            observation=env.state,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        tmp=run/'checkpoint.tmp';torch.save(c,tmp);tmp.replace(checkpoint)
    while steps<decisions:
        obs=env.state.copy();action,choice=agent.take_action(obs,return_info=True)
        nxt,reward,done,info=env.step(action);actions.append(action.tolist());ep_return+=reward
        # A physical time-limit is an episode termination in this finite-horizon
        # task; no bootstrap from a partially completed macro transition.
        values=(obs,action,nxt,reward,done,done,choice['old_log_prob'],choice['action_mask'])
        for key,value in zip(keys,values):buffer[key].append(value)
        steps+=1
        if len(buffer['states'])==64 or done or steps==decisions:
            agent.update(buffer)
            with (run/'updates.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(decisions=steps,**agent.last_update_diagnostics),allow_nan=False)+'\n')
            buffer={key:[] for key in keys}
            # A terminal checkpoint is written only after episode accounting
            # and reset, so deterministic replay never resumes a done episode.
            if not done:save()
        if done:
            old.save_trace(run/f'train_episode_{episode:04d}.npz',env.trace)
            with (run/'episodes.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(episode=episode,path_id=ids[episode%len(ids)],decisions=steps,
                    physical_steps=env.physics_steps,command_steps=env.command_steps,reward=ep_return,result=env.result),allow_nan=False)+'\n')
            physical_before+=env.physics_steps;commands_before+=env.command_steps
            episode+=1;actions=[];ep_return=0.;env=MacroEnvironment(cfg,ids[episode%len(ids)])
            save()
        if steps%256==0:print(f'{candidate} {method}: decisions={steps}/{decisions}, episodes={episode}',flush=True)
    save()
    _complete_training(run,cfg,steps,episode,physical_before,commands_before,env,wall_before+time.perf_counter()-started)
    return run


def _complete_training(run,cfg,steps,episode,physical_before,commands_before,env,elapsed):
    """Also finish bookkeeping after a crash just after the final checkpoint."""
    old.save_trace(run/'training_final_partial.npz',env.trace)
    old.write(run/'training_summary.json',dict(candidate=cfg['revision']['candidate'],method=cfg['revision']['variant'],
        seed=cfg['seed'],decisions=steps,episodes=episode,
        physics_steps=physical_before+env.physics_steps,command_steps=commands_before+env.command_steps,
        training_wall_s=elapsed,checkpoint_sha256=old.sha(run/'checkpoint.pt'),
        pretraining_steps=0,convergence_established=False,status='registered_development_budget_complete'))


def evaluate(run,output=None):
    torch.set_num_threads(1);protocol=prepare();run=Path(run).resolve()
    cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    provenance=old.read(run/'provenance.json')
    if provenance['config_sha256']!=old.sha(run/'config.yaml') or provenance['source_sha256']!=protocol['source']['source_sha256']:
        raise ValueError('Evaluation source/config mismatch')
    c=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    out=run/'validation' if output is None else Path(output).resolve()
    if (out/'summary.json').exists():raise FileExistsError('Keep completed validation')
    binding=dict(checkpoint_sha256=old.sha(run/'checkpoint.pt'),config_sha256=old.sha(run/'config.yaml'),
                 source_sha256=protocol['source']['source_sha256'],split='validation',
                 manifest_sha256=old.sha(old.DATA/'manifest.json'))
    if (out/'binding.json').exists() and old.read(out/'binding.json')!=binding:
        raise ValueError('Partial validation belongs to a different model/config/source; use a new --output')
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'binding.json').exists():old.write(out/'binding.json',binding)
    rows=[]
    for path_id in [p['id'] for p in old.manifest()['paths'] if p['split']=='validation']:
        if (out/f'{path_id}.json').exists():
            row=old.read(out/f'{path_id}.json')
            if any(row.get(k)!=binding[k] for k in ('checkpoint_sha256','config_sha256','source_sha256')) or row['trace_sha256']!=old.sha(out/f'{path_id}.npz'):
                raise ValueError('Cached validation row or trace changed')
            rows.append(row);continue
        if (out/f'{path_id}.npz').exists():
            raise FileExistsError('Keep orphaned partial trace; rerun with a new --output directory')
        env=MacroEnvironment(cfg,path_id);agent=make_agent(cfg,env);agent.actor.load_state_dict(c['actor']);agent.actor.eval()
        start=time.perf_counter()
        while not env.done:env.step(agent.take_action(env.state,deterministic=True))
        out.mkdir(exist_ok=True);old.save_trace(out/f'{path_id}.npz',env.trace)
        row=copy.deepcopy(env.result)
        feed=np.array([r['v_mm_s'] for r in env.trace]);distance=np.array([r['lookahead_mm'] for r in env.trace])
        row.update(path_id=path_id,candidate=cfg['revision']['candidate'],method=cfg['revision']['variant'],seed=cfg['seed'],
            training_decisions=c['decisions'],preview_min_mm=float(distance.min()),preview_max_mm=float(distance.max()),
            feed_mean_mm_s=float(feed.mean()),feed_max_mm_s=float(feed.max()),
            checkpoint_sha256=old.sha(run/'checkpoint.pt'),trace_sha256=old.sha(out/f'{path_id}.npz'),
            source_sha256=protocol['source']['source_sha256'],config_sha256=old.sha(run/'config.yaml'),
            evaluation_wall_s=time.perf_counter()-start,data_role='development validation')
        old.write(out/f'{path_id}.json',row);rows.append(row)
        print(f'validation {run.name} {path_id}: {row["success"]}, time={row["elapsed_time_s"]:.3f}, D={row["preview_min_mm"]:.3f}..{row["preview_max_mm"]:.3f}',flush=True)
    old.write(out/'summary.json',dict(rows=rows,summary=old.backend.summarize_runs(rows)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('train');p.add_argument('--candidate',choices=CANDIDATES,required=True)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=101)
    p.add_argument('--decisions',type=int,default=4096);p.add_argument('--stage',choices=('smoke','pilot'),default='pilot')
    p=sub.add_parser('evaluate');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='train':train(args.candidate,args.method,args.seed,args.decisions,args.stage)
    else:evaluate(args.run,args.output)


if __name__=='__main__':main()
