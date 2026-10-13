"""Auditable event-preview PPO study with matched fixed and instant controls.

Every distance decision is a policy sample; holding never substitutes a
geometry rule. All outputs belong to a new version. No automatic test access.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import yaml
import preview_direct_explore as predecessor
import preview_direct_fine as fine

base = predecessor.study
backend = base.backend
ROOT = base.ROOT
OUT = ROOT / '论文项目/revision_20260909_event_preview/study'
DATA = ROOT / 'PPO_project/data/preview_event_v1'
SEEDS = (11,22,33,44,55)
PILOT_STEPS, FORMAL_STEPS = 32768, 262144
FIXED_GRID = {'fixed1p2':1.2,'fixed2p3':2.3,'fixed3':3.,'fixed4':4.}
LEVELS = ('legacy','filter','ordered','finite','std','state')
PILOTS = tuple(f'{level}_{kind}' for level in LEVELS for kind in ('fixed','direct')) + ('hold20','hold50','hold100') + tuple(FIXED_GRID)
FORMAL = ('legacy_direct','state_direct','selected_fixed','selected_hold')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def manifest():return base.v2.read_manifest(DATA)
def points(path_id):return base.v2.read_path(path_id,DATA)


def source_state():
    state=predecessor.source_state()
    for path in [Path(__file__),ROOT/'PPO_project/src/algorithms/preview_event_ppo.py',ROOT/'PPO_project/tools/generate_preview_event_paths.py']:
        state['source_files'][path.relative_to(ROOT).as_posix()]=sha(path)
    state['source_sha256']=hashlib.sha256(json.dumps(state['source_files'],sort_keys=True).encode()).hexdigest()
    return state


def config_for(variant,seed):
    if variant not in (*PILOTS,*FORMAL):raise ValueError(variant)
    chosen=variant
    if variant.startswith('selected_'):
        chosen=read(OUT/'selection.json')[variant]
    if chosen.startswith('hold'):
        level,adaptive,period=5,True,int(chosen[4:])
    elif chosen in FIXED_GRID:
        level,adaptive,period=5,False,1
    else:
        prefix,kind=chosen.split('_')
        level,adaptive,period=LEVELS.index(prefix),kind=='direct',1
    cfg=predecessor.config_for('direct',seed)
    distance=FIXED_GRID.get(chosen,3.)
    cfg['revision'].update(variant=variant,study='event-preview-v1',effective_variant=chosen,
        shared_fix_level=level,adaptive_preview=adaptive,preview_period_steps=period,
        observation_dim=26 if level>=5 else 20,std_mode='std_fix' if level>=4 else 'legacy',
        fixed_distance_mm=distance,initial_preview_mm=distance,
        active_action_dims=[0,1,2] if adaptive else [0,1],
        training_path_ids=[p['id'] for p in manifest()['paths'] if p['split']=='train'],
        extra_event_state=['held_preview_u','steps_to_decision/period','filtered_reference_minus_heading/pi',
                           'angular_acceleration/MAX_ANG_ACC','remaining_arc_mm/10','raw_LOS_minus_heading/pi'],
        initialization='Fresh identical-width 3-output MLP, means [0,.05,(3-.8)/3.8], std [.01,.01,.05]; fixed masks preview',
        likelihood='Physical dimensions every 1ms; preview only on true resampling events; raw latent density',
        preview_loss_normalization='Mean over physical transitions; no inverse-event-rate reweighting',
        terminal_definition='Ordered geometric arrival, no terminal rest requirement; simulation only')
    control=cfg['reward_weights']['control_authority']
    control.update(reference_filter_update_mode='once_per_step' if level>=1 else 'legacy_query',
        turn_projection_mode='ordered' if level>=2 else 'legacy_rtree')
    cfg['reward_weights']['lookahead'].update(distance=distance,finite_episode_turn_scan=level>=3)
    return cfg


class EventPreviewEnvironment:
    def __init__(self,cfg,path_id,reference=None,episode=0):
        self.cfg=cfg;self.adaptive=cfg['revision']['adaptive_preview']
        self.period=int(cfg['revision']['preview_period_steps'])
        if self.period<1:raise ValueError('Positive holding period required')
        self.tick=0;self.held_preview=(cfg['revision']['initial_preview_mm']-.8)/3.8
        local=copy.deepcopy(cfg)
        # The existing direct wrapper always handles the geometry. The outer
        # wrapper decides which sampled event may replace its distance.
        local['revision'].update(variant='direct',active_action_dims=[0,1,2])
        reference=points(path_id) if reference is None else reference
        local['reward_weights']['w_s']=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())*local['revision']['progress_reward_per_mm']
        raw,_=base.v2.BASE_ENV(local,path_id,episode,reference=reference)
        objective=fine.FineTimeObjective(raw,local,reference)
        self.env=base.DirectPreviewEnvironment(objective,local,reference)
        self.env.initialize_observation()
        self.raw=self.env.raw
        self.observation_dim=cfg['revision']['observation_dim']
        self.action_space_dim=3
        self.action_space=self.env.action_space
        self.observation_space=type(self.env.observation_space)(low=-np.inf,high=np.inf,shape=(self.observation_dim,),dtype=np.float32)
        self.last_event=False
        self._observation=self._augment(self.env.state)

    def __getattr__(self,key):return getattr(self.env,key)
    @property
    def preview_decision(self):return self.adaptive and self.tick%self.period==0
    @property
    def state(self):return self._observation.copy()

    def _augment(self,obs):
        if self.observation_dim==20:return np.asarray(obs,np.float32).copy()
        raw=self.raw
        s=float(raw._project_onto_progress_path(raw.current_position)[2])
        remaining=max(0.,float(raw._progress_total_length)-s)
        phase=((self.period-self.tick%self.period)%self.period)/self.period if self.adaptive else 0.
        filtered=float(getattr(raw,'_theta_ref_filtered',raw.heading))
        los=raw._raw_path_direction(raw.current_position,lookahead_distance=.8+3.8*self.held_preview)
        values=[self.held_preview,phase,raw._wrap_angle(filtered-raw.heading)/math.pi,
            raw.angular_acc/raw.MAX_ANG_ACC,remaining/10.,raw._wrap_angle(los-raw.heading)/math.pi]
        return np.r_[obs,values].astype(np.float32)

    def step(self,action):
        action=np.asarray(action,float).copy()
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Three finite latent actions required')
        event=self.preview_decision
        if event:self.held_preview=float(np.clip(action[2],0.,1.))
        if not self.adaptive:self.held_preview=(self.cfg['revision']['fixed_distance_mm']-.8)/3.8
        action[2]=self.held_preview
        obs,reward,done,info=self.env.step(action)
        self.tick+=1;self.last_event=event
        self._observation=self._augment(obs)
        info.setdefault('control_status',{}).update(preview_event=float(event),preview_period_steps=float(self.period),
            preview_held_u=self.held_preview,reference_filter_updates=float(getattr(self.raw,'_theta_ref_filter_update_count',0)))
        return self.state,reward,done,info


def make_agent(cfg,env):
    from src.algorithms.preview_event_ppo import PreviewEventPPO
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    return PreviewEventPPO(state_dim=env.observation_dim,action_dim=3,device=torch.device('cpu'),
        active_action_dims=cfg['revision']['active_action_dims'],std_mode=cfg['revision']['std_mode'],
        anchor_beta=cfg['revision']['motion_anchor_beta'],initial_mean=[0.,.05,(3.-.8)/3.8],
        initial_std=[.01,.01,.05],**{k:cfg['ppo'][k] for k in keys})


def protocol_definition():
    return dict(version='event-preview-v1',source_sha256=source_state()['source_sha256'],
        manifest_sha256=sha(DATA/'manifest.json'),pilot_variants=list(PILOTS),pilot_steps=PILOT_STEPS,pilot_seed=101,
        formal_variants=list(FORMAL),formal_steps=FORMAL_STEPS,formal_seeds=list(SEEDS),
        plan='Sequential shared-fix ablations, then preview duration 20/50/100ms and fixed distance 1.2/2.3/3/4mm; validation selection only',
        formal_budget_role='Twice predecessor physical interaction per model; compare newly retrained matching budgets, retain earlier 131072-step results separately',
        selection='Lexicographic strict validation successes, ordered coverage, then total successful elapsed time; no per-test path selection',
        test_gate='All 20 formal models must reach registered budget; freeze model/config/source before new tests',
        scope='Scalar feed and heading limits, geometric arrival; Cartesian hard limits and terminal rest not implemented',
        forbidden='No machine connection, performance value fabrication, deletion of failed runs, or test-guided tuning')


def check_protocol():
    if read(OUT/'protocol.json')!=protocol_definition():raise ValueError('Source or registered protocol changed')


def prepare():
    target=OUT/'protocol.json';value=protocol_definition()
    if target.exists() and read(target)!=value:raise ValueError('Preserve previous protocol')
    write(target,value);return value


def save_trace(path,rows):backend.save_trace(path,rows)
def sample_row(env,info,reward,latent):
    row=base.sample_row(env.env,info,reward,latent)
    row.update(preview_event=float(env.last_event),preview_phase=float(env.tick%env.period),preview_held_u=env.held_preview)
    return row


def train(variant,seed,stage,total_steps):
    torch.set_num_threads(1)
    if stage!='smoke':
        check_protocol()
        assert (stage=='pilot' and variant in PILOTS and seed==101 and total_steps==PILOT_STEPS) or (stage=='formal' and variant in FORMAL and seed in SEEDS and total_steps==FORMAL_STEPS)
    elif not 0<total_steps<=2048:raise ValueError('Smoke budget 1..2048')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    cfg=config_for(variant,seed);run=OUT/stage/f'{variant}_seed{seed}';run.mkdir(parents=True,exist_ok=True)
    state=source_state();cfg_path=run/'config.yaml';checkpoint=run/'checkpoint.pt'
    if cfg_path.exists():
        if yaml.safe_load(cfg_path.read_text(encoding='utf-8'))!=cfg:raise ValueError('Changed run configuration')
        if read(run/'provenance.json')['source_sha256']!=state['source_sha256']:raise ValueError('Changed run source')
    else:
        cfg_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        write(run/'provenance.json',dict(**state,config_sha256=sha(cfg_path),manifest_sha256=sha(DATA/'manifest.json'),command='python '+' '.join(sys.argv)))
        with zipfile.ZipFile(run/'source.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for name in state['source_files']:archive.write(ROOT/name,name)
    ids=cfg['revision']['training_path_ids']
    env=EventPreviewEnvironment(cfg,ids[0]);agent=make_agent(cfg,env);obs=env.state
    steps=episode=0;actions=[];ep_return=0.;wall_before=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for name in ('actor','critic'):getattr(agent,name).load_state_dict(saved[name])
        for name in ('actor_optimizer','critic_optimizer'):getattr(agent,name).load_state_dict(saved[name])
        steps,episode,actions,ep_return,wall_before=(saved[k] for k in ('steps','episode','episode_actions','ep_return','training_wall_s'))
        env=EventPreviewEnvironment(cfg,ids[episode%len(ids)],episode=episode)
        for action in actions:env.step(action)
        obs=env.state
        if not np.array_equal(obs,saved['observation']):raise ValueError('Resume state/holding phase differs')
        torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        agent.logged_steps=steps
    if steps>=total_steps:return run
    started=time.perf_counter();trace_start=steps
    buffer={k:[] for k in ('states','actions','next_states','rewards','dones','terminated','preview_decisions','old_log_probs')}
    trace=[sample_row(env,{},0.,[0,0,env.held_preview])]
    def checkpoint_save():
        data=dict(actor=agent.actor.state_dict(),critic=agent.critic.state_dict(),actor_optimizer=agent.actor_optimizer.state_dict(),
            critic_optimizer=agent.critic_optimizer.state_dict(),steps=steps,episode=episode,episode_actions=actions,
            ep_return=ep_return,training_wall_s=wall_before+time.perf_counter()-started,observation=env.state,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        temp=run/'checkpoint.tmp';torch.save(data,temp);temp.replace(checkpoint)
        if steps==total_steps:
            version=run/'checkpoints'/f'step_{steps:08d}.pt';version.parent.mkdir(exist_ok=True)
            if version.exists() and sha(version)!=sha(checkpoint):raise ValueError('Preserve previous budget checkpoint')
            if not version.exists():shutil.copyfile(checkpoint,version)
    while steps<total_steps:
        event=env.preview_decision
        latent,decision=agent.take_action(obs,preview_decision=event,held_preview=env.held_preview,return_info=True)
        nxt,reward,done,info=env.step(latent);actions.append(np.asarray(latent).tolist());ep_return+=reward
        terminal=done and info.get('done_reason') not in ('max_steps','timeout','time_limit')
        values=(obs,latent,nxt,reward,done,terminal,event,decision['old_log_prob'])
        for key,value in zip(buffer,values):buffer[key].append(value)
        trace.append(sample_row(env,info,reward,latent));steps+=1;obs=nxt
        if len(buffer['states'])==512 or done or steps==total_steps:
            losses=agent.update(buffer)
            entry=dict(agent.last_update_diagnostics)
            entry.update(step=steps,actor_loss=losses[0],critic_loss=losses[1])
            with (run/'updates.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(entry,allow_nan=False)+'\n')
            buffer={k:[] for k in buffer}
        if done or steps==total_steps:
            save_trace(run/f'train_trace_{trace_start:08d}_{steps:08d}.npz',trace)
            with (run/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(episode=episode,path_id=ids[episode%len(ids)],steps=steps,episode_steps=env.tick,
                    reward=ep_return,done=done,done_reason=info.get('done_reason'),budget_truncation=not done))+'\n')
            if done:
                episode+=1;actions=[];ep_return=0.;env=EventPreviewEnvironment(cfg,ids[episode%len(ids)],episode=episode);obs=env.state
            checkpoint_save();trace_start=steps;trace=[sample_row(env,{},0.,[0,0,env.held_preview])]
            print(f'{stage} {variant} seed={seed} steps={steps}/{total_steps} episodes={episode}',flush=True)
    write(run/'training_summary.json',dict(variant=variant,seed=seed,steps=steps,episodes=episode,
        training_wall_s=wall_before+time.perf_counter()-started,checkpoint_sha256=sha(checkpoint),status='budget_complete',convergence_established=False))
    return run


def evaluate(run,split,regime='nominal'):
    torch.set_num_threads(1);run=Path(run).resolve();run.relative_to(OUT.resolve())
    cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'));provenance=read(run/'provenance.json')
    if provenance['source_sha256']!=source_state()['source_sha256'] or provenance['config_sha256']!=sha(run/'config.yaml') or provenance['manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Changed evaluation source/config/manifest')
    if split=='test':
        frozen=read(OUT/'freeze.json');key=run.relative_to(OUT).as_posix()
        if frozen['models'].get(key)!=sha(run/'checkpoint.pt') or frozen['configs'].get(key)!=sha(run/'config.yaml') or frozen['source_sha256']!=source_state()['source_sha256'] or frozen['manifest_sha256']!=sha(DATA/'manifest.json') or frozen['protocol_sha256']!=sha(OUT/'protocol.json') or frozen['selection_sha256']!=sha(OUT/'selection.json'):raise ValueError('Test freeze mismatch')
    if regime=='tight':cfg['environment']['epsilon']=1.25
    elif regime=='low_jerk':cfg['kinematic_constraints']['MAX_JERK']=31250.
    elif regime!='nominal':raise ValueError(regime)
    out=run/f'evaluation_{split}_{regime}';out.mkdir(exist_ok=True)
    if split=='regression':
        old=yaml.safe_load((ROOT/'论文项目/generated/configs/full_method_snapshot.yaml').read_text(encoding='utf-8'))
        specs=[(p['name'],backend._build_path(p)) for p in old['training']['path_curriculum']['paths']]
    else:specs=[(p['id'],points(p['id'])) for p in manifest()['paths'] if p['split']==split]
    if not specs:raise ValueError('Empty split')
    rows=[];checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    version=run/'checkpoints'/f"step_{checkpoint['steps']:08d}.pt";version.parent.mkdir(exist_ok=True)
    if version.exists() and sha(version)!=sha(run/'checkpoint.pt'):raise ValueError('Preserve evaluated checkpoint')
    if not version.exists():shutil.copyfile(run/'checkpoint.pt',version)
    for path_id,reference in specs:
        dest=out/f'{path_id}.json'
        if dest.exists():
            previous=read(dest)
            validate_result(previous,run,path_id,split,regime)
            rows.append(previous);continue
        env=EventPreviewEnvironment(cfg,path_id,reference=reference);agent=make_agent(cfg,env)
        agent.actor.load_state_dict(checkpoint['actor']);agent.actor.eval()
        obs=env.state;trace=[sample_row(env,{},0.,[0,0,env.held_preview])];latencies=[];done=False
        started=time.perf_counter()
        with torch.no_grad():
            while not done:
                began=time.perf_counter_ns();latent=agent.actor(torch.as_tensor(obs).unsqueeze(0))[0][0].numpy()
                latencies.append((time.perf_counter_ns()-began)/1000.)
                obs,reward,done,info=env.step(latent);trace.append(sample_row(env,info,reward,latent))
        save_trace(out/f'{path_id}.npz',trace)
        metric=backend.evaluate_trajectory(reference,[[r['x_mm'],r['y_mm']] for r in trace],[r['timestamp_s'] for r in trace],
            tolerance_mm=cfg['environment']['epsilon']/2,limits=cfg['kinematic_constraints'],
            termination_reason=info['done_reason'],claimed_success=info['done_reason']=='success',
            initial_velocity=[0.,0.],initial_acceleration=[0.,0.],initial_heading=trace[0]['heading_rad'],
            initial_angular_velocity=0.,initial_angular_acceleration=0.,witness_spacing_mm=.01)
        metric.update(variant=cfg['revision']['variant'],effective_variant=cfg['revision']['effective_variant'],seed=cfg['seed'],
            path_id=path_id,split=split,regime=regime,training_steps=checkpoint['steps'],checkpoint_sha256=sha(run/'checkpoint.pt'),
            config_sha256=sha(run/'config.yaml'),source_sha256=provenance['source_sha256'],manifest_sha256=sha(DATA/'manifest.json'),
            inference_median_us=float(np.median(latencies)),inference_p95_us=float(np.percentile(latencies,95)),
            evaluation_wall_s=time.perf_counter()-started,preview_min_mm=min(r['lookahead_mm'] for r in trace),
            preview_max_mm=max(r['lookahead_mm'] for r in trace),preview_event_count=sum(r['preview_event'] for r in trace),
            terminal_feed_mm_s=trace[-1]['v_mm_s'])
        write(dest,metric);rows.append(metric)
        print(f'eval {run.name} {path_id}: success={metric["success"]} time={metric["elapsed_time_s"]:.3f}',flush=True)
    write(out/'summary.json',dict(rows=rows,summary=backend.summarize_runs(rows)));return rows


def validate_result(row,run,path_id,split='validation',regime='nominal'):
    cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    expected=dict(checkpoint_sha256=sha(run/'checkpoint.pt'),config_sha256=sha(run/'config.yaml'),
        source_sha256=source_state()['source_sha256'],manifest_sha256=sha(DATA/'manifest.json'),
        path_id=path_id,split=split,regime=regime,variant=cfg['revision']['variant'],seed=cfg['seed'],
        training_steps=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)['steps'])
    if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('Stale or misbound evaluation result')


def validation_rows(run):
    rows=read(run/'evaluation_validation_nominal/summary.json')['rows']
    ids={p['id'] for p in manifest()['paths'] if p['split']=='validation'}
    if len(rows)!=len(ids) or {r['path_id'] for r in rows}!=ids:raise ValueError('Validation paths incomplete or duplicated')
    for row in rows:validate_result(row,run,row['path_id'])
    return rows


def rank(variant):
    run=OUT/'pilot'/f'{variant}_seed101'
    verify_training(run,PILOT_STEPS)
    rows=validation_rows(run)
    return dict(variant=variant,score=[sum(r['success'] for r in rows),sum(r['path_completed'] for r in rows),
        sum(r['ordered_progress'] for r in rows),-sum(r['elapsed_time_s'] for r in rows if r['success'])])


def verify_training(run,expected_steps):
    summary=read(run/'training_summary.json');provenance=read(run/'provenance.json')
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    if checkpoint['steps']!=expected_steps or summary['steps']!=expected_steps:raise ValueError('Checkpoint or summary budget incomplete')
    if summary['checkpoint_sha256']!=sha(run/'checkpoint.pt'):raise ValueError('Training summary model hash differs')
    if provenance['source_sha256']!=source_state()['source_sha256'] or provenance['config_sha256']!=sha(run/'config.yaml') or provenance['manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Training source/config/path binding differs')


def select():
    check_protocol();rankings=[rank(v) for v in PILOTS]
    value=dict(selected_fixed=max((r for r in rankings if r['variant'] in FIXED_GRID),key=lambda r:r['score'])['variant'],
        selected_hold=max((r for r in rankings if r['variant'].startswith('hold')),key=lambda r:r['score'])['variant'],
        rankings=rankings,source_sha256=source_state()['source_sha256'],test_used=False)
    target=OUT/'selection.json'
    if target.exists() and read(target)!=value:raise ValueError('Preserve previous selection')
    write(target,value);return value


def freeze():
    check_protocol();models={};configs={}
    for variant in FORMAL:
        for seed in SEEDS:
            run=OUT/'formal'/f'{variant}_seed{seed}'
            verify_training(run,FORMAL_STEPS)
            validation_rows(run)
            key=run.relative_to(OUT).as_posix();models[key]=sha(run/'checkpoint.pt');configs[key]=sha(run/'config.yaml')
    target=OUT/'freeze.json'
    if target.exists():raise FileExistsError('Already frozen')
    value=dict(models=models,configs=configs,source_sha256=source_state()['source_sha256'],manifest_sha256=sha(DATA/'manifest.json'),
        protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection.json'),timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(target,value);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','evaluate','select','freeze'])
    parser.add_argument('--variant',default='hold50');parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--stage',choices=['smoke','pilot','formal'],default='smoke');parser.add_argument('--steps',type=int,default=512)
    parser.add_argument('--run',type=Path);parser.add_argument('--split',choices=['validation','test','regression'],default='validation')
    parser.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='train':train(args.variant,args.seed,args.stage,args.steps)
    elif args.command=='evaluate':evaluate(args.run,args.split,args.regime)
    elif args.command=='select':print(json.dumps(select(),ensure_ascii=False))
    else:freeze()


if __name__=='__main__':main()
