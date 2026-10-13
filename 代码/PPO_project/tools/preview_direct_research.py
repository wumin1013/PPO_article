"""Direct-distance PPO study with a shared physical-action distribution anchor.

Geometry supplies two observation features only. During execution the policy
alone selects D = 0.8 + 3.8 * clip(u, 0, 1); there is no geometry distance prior,
gate, teacher or behaviour-cloning target. Imports never activate a study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import preview_research_v2 as v2
from src.algorithms.ppo import PPOContinuous
from src.utils import rl_utils

backend = v2.backend
ROOT, DATA = v2.ROOT, v2.DATA
OUT = ROOT/'论文项目/revision_20260908/preview_direct'
BASE_SAMPLE_ROW = backend.sample_row
DISTANCES = dict(fixed_short=1.2, fixed_mid=2.3, fixed_center=3., fixed_long=4.)
PILOT_VARIANTS = (*DISTANCES, 'direct', 'direct_no_anchor')
FORMAL_VARIANTS = ('selected_fixed', 'direct', 'direct_no_anchor')
SEEDS = (11, 22, 33, 44, 55)
PILOT_STEPS, FORMAL_STEPS = 32768, 131072
UPDATE_CONTEXT = None


def manifest(): return v2.read_manifest(DATA)
def path_points(path_id): return v2.read_path(path_id, DATA)


def source_state():
    state = v2.source_state()
    for path in (Path(__file__),Path(__file__).with_name('preview_direct_batch.py'),Path(__file__).with_name('preview_v2_batch.py')):
        state['source_files'][str(path.relative_to(ROOT)).replace('\\','/')] = backend.sha(path)
    state['source_sha256'] = hashlib.sha256(json.dumps(state['source_files'], sort_keys=True).encode()).hexdigest()
    return state


class GeometryObservation:
    """Signed turn cancellation and unwrapped tangent range; no control output."""
    def __init__(self, reference, window_mm=4.6):
        points = np.asarray(reference, float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Need finite Nx2 reference')
        vectors = np.diff(points,axis=0)
        lengths = np.linalg.norm(vectors,axis=1)
        if np.any(lengths<=1e-10): raise ValueError('Adjacent repeated reference points')
        headings = np.arctan2(vectors[:,1],vectors[:,0])
        turns = np.arctan2(np.sin(np.diff(headings)),np.cos(np.diff(headings)))
        self.arcs = np.cumsum(lengths)[:-1]
        self.signed = np.r_[0.,np.cumsum(turns)]
        self.absolute = np.r_[0.,np.cumsum(np.abs(turns))]
        self.window_mm = float(window_mm)

    def at(self,s):
        lo,hi = np.searchsorted(self.arcs,[s,s+self.window_mm],side='right')
        signed = float(self.signed[hi]-self.signed[lo])
        total = float(self.absolute[hi]-self.absolute[lo])
        cancellation = max(0.,1.-abs(signed)/(total+1e-12)) if total>1e-8 else 0.
        q = cancellation*min(1.,total/math.pi)
        turns = self.signed[lo:hi+1]
        return dict(q=float(q),range_over_pi=float((np.max(turns)-np.min(turns))/math.pi))


def direct_distance(preview_u):
    return float(.8+3.8*np.clip(preview_u,0.,1.))


def config_for(variant,seed):
    if variant not in (*PILOT_VARIANTS,'selected_fixed'):
        raise ValueError(f'Unknown direct-study variant: {variant}')
    cfg = v2.config_for('pretrain_mid',seed)
    distance = DISTANCES.get(variant,3.)
    if variant=='selected_fixed':
        distance = float(json.loads((OUT/'selection.json').read_text(encoding='utf-8'))['fixed_distance_mm'])
    cfg['reward_weights']['lookahead']['distance'] = distance
    cfg['reward_weights']['lookahead_control'].update(enabled=False,policy_action=False,residual_mode=False,
        straight_dist=distance,corner_dist=distance,action_default=(3.-.8)/3.8)
    cfg['ppo'].update(actor_lr=1e-5,gamma=math.exp(-.001/10.),lmbda=1.)
    cfg['revision'].update(variant=variant,study='direct-distance-motion-anchor-v1',
        active_action_dims=[0,1,2] if variant.startswith('direct') else [0,1],
        fixed_distance_mm=distance,observation_dim=20,base_observation_dim=18,geometry_window_mm=4.6,
        extra_observation_features=['q in [0,1], no additional clipping','unwrapped tangent range/pi; >=0, no clipping'],
        preview_bounds_mm=[.8,4.6],preview_mapping='D_t=0.8+3.8*clip(u_t,0,1); geometry is observation only',
        preview_timing='control and base nextobs use policy D_t; extra features query s_(t+1)',
        initialization='fresh actor means [0,0.05,(3-0.8)/3.8], all std 0.01; no learned weights or supervision',
        initial_preview_mm=3.,initial_preview_origin='documented developer fixed-distance grid; initialization only, not pretraining',
        pretraining_steps=0,teacher_used=False,behavior_cloning_used=False,geometry_distance_rule_used=False,
        motion_anchor_beta=0. if variant=='direct_no_anchor' else 1.,
        motion_anchor_mean=[0.,.05],motion_anchor_std=[.01,.01],
        motion_anchor_definition='KL(current latent Gaussian first two normalized control actions || constant anchor); preview excluded',
        critic_update_mode='eval(): dropout disabled for deterministic TD/GAE; autograd and optimizer remain enabled',
        critic_dropout_change_role='common training correction, shared by all fixed/direct variants; not preview-specific',
        training_path_ids=[p['id'] for p in manifest()['paths'] if p['split']=='train'])
    if variant=='selected_fixed': cfg['revision']['fixed_selection_sha256']=backend.sha(OUT/'selection.json')
    return cfg


class DirectPreviewEnvironment:
    def __init__(self,env,cfg,reference):
        self.env,self.raw,self.cfg = env,env.env,cfg
        self.variant=cfg['revision']['variant']
        self.geometry=GeometryObservation(reference,cfg['revision']['geometry_window_mm'])
        if env.observation_dim!=18: raise ValueError('Requires frozen 18-dimensional base state')
        self.observation_dim=20
        self.action_space_dim=len(cfg['revision']['active_action_dims'])
        self.action_space=type(env.action_space)(low=np.array([-1.,0.,0.],np.float32)[:self.action_space_dim],
            high=np.ones(self.action_space_dim,np.float32),dtype=np.float32)
        self.observation_space=type(env.observation_space)(low=np.r_[env.observation_space.low,[0.,0.]].astype(np.float32),
            high=np.r_[env.observation_space.high,[1.,np.inf]].astype(np.float32),dtype=np.float32)
        self.direct_trace_snapshot={}

    def __getattr__(self,key): return getattr(self.env,key)
    @property
    def state(self): return self._observation.copy()

    def _augment(self,observation):
        began=time.perf_counter_ns()
        s=float(self.raw._project_onto_progress_path(self.raw.current_position)[2])
        feature=self.geometry.at(s)
        self._observation=np.r_[np.asarray(observation,np.float32),[feature['q'],feature['range_over_pi']]].astype(np.float32)
        self.direct_trace_snapshot.update(observation_arc_mm=s,observation_q=feature['q'],
            observation_range_over_pi=feature['range_over_pi'],observation_features_us=(time.perf_counter_ns()-began)/1000.)
        return self._observation.copy()

    def initialize_observation(self):
        distance=self.cfg['revision']['initial_preview_mm'] if self.variant.startswith('direct') else self.cfg['revision']['fixed_distance_mm']
        self.raw.lookahead_dist=float(distance)
        self.raw._resolve_active_lookahead_dist()
        self.raw.state=self.raw._build_state()
        self.direct_trace_snapshot=dict(selected_mm=float(distance),policy_preview_u=(3.-.8)/3.8,mapping_us=0.)
        return self._augment(self.raw.state)

    def reset(self,*args,**kwargs):
        self.env.reset(*args,**kwargs)
        return self.initialize_observation()

    def step(self,action):
        values=np.asarray(action,float)
        if values.shape!=(self.action_space_dim,) or not np.isfinite(values).all():
            raise ValueError('Action dimension or finite-value check failed')
        values=np.clip(values,self.action_space.low,self.action_space.high)
        began=time.perf_counter_ns()
        preview_u=float(values[2]) if self.variant.startswith('direct') else (3.-.8)/3.8
        selected=direct_distance(preview_u) if self.variant.startswith('direct') else self.cfg['revision']['fixed_distance_mm']
        self.raw.lookahead_dist=float(selected)
        self.direct_trace_snapshot=dict(selected_mm=float(selected),policy_preview_u=preview_u,
            mapping_us=(time.perf_counter_ns()-began)/1000.)
        observation,reward,done,info=self.env.step(values[:2])
        observation=self._augment(observation)
        info.setdefault('control_status',{}).update({'direct_'+k:float(v) for k,v in self.direct_trace_snapshot.items()})
        return observation,reward,done,info


class AnchoredPPO(PPOContinuous):
    """PPO latent-action objective plus KL for the first two control heads only."""
    def __init__(self,*args,anchor_beta=1.,anchor_mean=(0.,.05),anchor_std=(.01,.01),update_context=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.anchor_beta=float(anchor_beta)
        self.anchor_mean=torch.tensor(anchor_mean,dtype=torch.float32,device=self.device)
        self.anchor_std=torch.tensor(anchor_std,dtype=torch.float32,device=self.device)
        self.update_context=update_context
        self.logged_steps=int(update_context['checkpoint_start_steps']) if update_context else 0
        self.last_update_diagnostics={}

    def motion_anchor_kl(self,mu,std):
        current=torch.distributions.Normal(mu[:,:2],std[:,:2])
        anchor=torch.distributions.Normal(self.anchor_mean,self.anchor_std)
        return torch.distributions.kl_divergence(current,anchor).sum(dim=1).mean()

    def latent_log_prob(self,mu,std,actions):
        return torch.distributions.Normal(mu,std).log_prob(actions)[:,self.active_action_dims].sum(dim=1,keepdim=True)

    def update(self,transition_dict):
        # eval() disables the frozen ValueNet dropout; it does not disable
        # differentiation or optimizer updates. This is shared by all variants.
        self.critic.eval()
        states=torch.tensor(np.array(transition_dict['states']),dtype=torch.float,device=self.device)
        actions=torch.tensor(np.array(transition_dict['actions']),dtype=torch.float,device=self.device)
        rewards=torch.tensor(transition_dict['rewards'],dtype=torch.float,device=self.device).view(-1,1)
        next_states=torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float,device=self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float,device=self.device).view(-1,1)
        bootstrap_dones=torch.tensor(transition_dict.get('terminated',transition_dict['dones']),
            dtype=torch.float,device=self.device).view(-1,1)
        td_target=rewards+self.gamma*self.critic(next_states)*(1-bootstrap_dones)
        td_delta=td_target-self.critic(states)
        advantage=rl_utils.compute_advantage(self.gamma,self.lmbda,td_delta.cpu(),dones.cpu()).to(self.device)
        if advantage.std(unbiased=False)>1e-6:
            advantage=torch.clamp((advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8),-10.,10.)
        mu,std=self.actor(states)
        old_log_probs=self.latent_log_prob(mu.detach(),std.detach(),actions)
        epoch_rows=[]
        for epoch in range(self.epochs):
            mu,std=self.actor(states)
            distribution=torch.distributions.Normal(mu,std)
            log_probs=self.latent_log_prob(mu,std,actions)
            entropy=distribution.entropy()[:,self.active_action_dims].sum(dim=1).mean()*self.ent_coef
            log_ratio=torch.clamp(log_probs-old_log_probs,-5.,5.)
            ratio=torch.exp(log_ratio)
            policy_loss=-torch.min(ratio*advantage,torch.clamp(ratio,1-self.eps,1+self.eps)*advantage).mean()
            anchor_kl=self.motion_anchor_kl(mu,std)
            actor_loss=policy_loss-entropy+self.anchor_beta*anchor_kl
            critic_loss=F.smooth_l1_loss(self.critic(states),td_target.detach())
            self.actor_optimizer.zero_grad();self.critic_optimizer.zero_grad()
            actor_loss.backward();critic_loss.backward()
            gradients=self.actor.mu_head.weight.grad
            bias_gradients=self.actor.mu_head.bias.grad
            entry=dict(epoch=epoch,actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()),
                policy_loss=float(policy_loss.detach()),entropy_bonus=float(entropy.detach()),motion_anchor_kl=float(anchor_kl.detach()),
                motion_anchor_beta=self.anchor_beta,anchor_loss=float(self.anchor_beta*anchor_kl.detach()),
                preview_mu_weight_gradient_norm=float(gradients[2].norm()),preview_mu_bias_gradient=float(bias_gradients[2]),
                physics_mu_weight_gradient_norm=float(gradients[:2].norm()),
                preview_mu_min=float(mu[:,2].detach().min()),preview_mu_max=float(mu[:,2].detach().max()),
                preview_std_mean=float(std[:,2].detach().mean()),
                physics_mu_mean=mu[:,:2].detach().mean(dim=0).tolist(),physics_std_mean=std[:,:2].detach().mean(dim=0).tolist(),
                latent_log_prob_mean=float(log_probs.detach().mean()),ratio_min=float(ratio.detach().min()),ratio_max=float(ratio.detach().max()))
            entry['actor_gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm))
            entry['critic_gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm))
            self.actor_optimizer.step();self.critic_optimizer.step()
            epoch_rows.append(entry)
        self.logged_steps+=len(states)
        self.last_update_diagnostics=dict(step=self.logged_steps,transition_count=len(states),epochs=epoch_rows,
            active_action_dims=list(self.active_action_dims),preview_latent_min=float(actions[:,2].min()),
            preview_latent_max=float(actions[:,2].max()),preview_latent_clipped_fraction=float(((actions[:,2]<0)|(actions[:,2]>1)).float().mean()),
            likelihood='latent Gaussian z scored before execution clip; preview included only for direct variants')
        if self.update_context:
            record=dict(self.last_update_diagnostics,invocation_id=self.update_context['invocation_id'],
                        checkpoint_start_steps=self.update_context['checkpoint_start_steps'])
            with Path(self.update_context['path']).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,allow_nan=False)+'\n')
        return actor_loss.item(),critic_loss.item()


def make_env(cfg,path_id,episode=0,reference=None):
    points=path_points(path_id) if reference is None else reference
    env,_=v2.make_env(cfg,path_id,episode,reference=points)
    wrapped=DirectPreviewEnvironment(env,cfg,points)
    return wrapped,wrapped.initialize_observation()


def make_agent(cfg,env):
    keys=('hidden_dim','actor_lr','critic_lr','lmbda','epochs','eps','gamma','ent_coef')
    revision=cfg['revision']
    agent=AnchoredPPO(state_dim=20,action_dim=3,device=torch.device('cpu'),
        active_action_dims=revision['active_action_dims'],anchor_beta=revision['motion_anchor_beta'],
        anchor_mean=revision['motion_anchor_mean'],anchor_std=revision['motion_anchor_std'],
        update_context=UPDATE_CONTEXT,**{k:cfg['ppo'][k] for k in keys})
    u=(3.-.8)/3.8
    with torch.no_grad():
        agent.actor.mu_head.weight.zero_()
        agent.actor.mu_head.bias.copy_(torch.tensor([0.,math.log(.05/.95),math.log(u/(1-u))]))
        agent.actor.std_head.weight.zero_()
        agent.actor.std_head.bias.fill_(math.log(math.expm1(.01-.001)))
    return agent


def sample_row(env,info,reward,latent):
    row=BASE_SAMPLE_ROW(env,info,reward,latent)
    row.update({'control_status__direct_'+k:float(v) for k,v in env.direct_trace_snapshot.items()})
    return row


def activate_backend():
    backend.OUT,backend.DEFAULT_DATA_DIR=OUT,DATA
    backend.load_manifest,backend.load_path=manifest,path_points
    backend.config_for,backend.make_env,backend.make_agent=config_for,make_env,make_agent
    backend.source_state,backend.sample_row=source_state,sample_row


def protocol_definition():
    return dict(version='preview-direct-v1',source_sha256=source_state()['source_sha256'],
        manifest_sha256=backend.sha(DATA/'manifest.json'),pilot_variants=list(PILOT_VARIANTS),pilot_seed=101,
        pilot_steps=PILOT_STEPS,formal_variants=list(FORMAL_VARIANTS),formal_seeds=list(SEEDS),formal_steps=FORMAL_STEPS,
        total_pilot_steps=len(PILOT_VARIANTS)*PILOT_STEPS,total_formal_steps=len(FORMAL_VARIANTS)*len(SEEDS)*FORMAL_STEPS,
        method='Direct policy distance D=0.8+3.8*clip(u,0,1); q and range/pi are observations only',
        geometry_distance_rule_used=False,teacher_used=False,behavior_cloning_used=False,
        initialization='fresh means [0,0.05,(3-0.8)/3.8], all std 0.01; 3 mm from developer fixed-grid evidence, not pretraining',
        shared_training=dict(actor_lr=1e-5,gamma=math.exp(-.001/10.),lmbda=1.,rollout_steps=512,
            anchor_beta=1.,anchor_actions=[0,1],anchor_mean=[0.,.05],anchor_std=[.01,.01],
            critic_dropout_during_update=False,critic_gradient_training=True,
            attribution='Common physical-action stabilization shared with independently trained fixed distances'),
        anchor_ablation='direct_no_anchor beta=0; same initial distribution and other training parameters',
        anchor_definition='KL(current latent Gaussian first two normalized physical-control actions || constant initial Gaussian); preview head not included',
        geometry_window_mm=4.6,extra_observations=['q in [0,1], no additional clipping','unwrapped tangent range/pi, nonnegative and unclipped'],
        timing='policy D_t drives control and base next observation; extra geometry features query s_(t+1)',
        unchanged='V2 reward/ordered coverage, KCM, endpoint handling, PPO MLP depth and width',
        critic_noise_diagnosis='Saved same-batch repeated-forward probe identified TD/GAE noise from dropout; all direct-study methods use critic.eval() during update',
        selection='all six pilots retained; uniform fixed distance selected on validation strict success, geometric completion, coverage, success time',
        inference='actor cost in evaluation JSON; mapping and geometry-observation cost separately in trace',
        test_policy='No holdout use until all fifteen formal models/configurations freeze; never automatic evaluation',
        statistics='independent training seed and path; all failures retained, time differences use strict common-success pairs')


def prepare():
    value=protocol_definition();target=OUT/'protocol.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:
        raise ValueError('Protocol/source changed; preserve this version')
    backend.write_json(target,value);return target


def check_protocol():
    value=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    if value!=protocol_definition():raise ValueError('Protocol/source changed since preparation')
    return value


def train(variant,seed,stage,steps):
    global UPDATE_CONTEXT
    check_protocol()
    if stage=='pilot':valid=variant in PILOT_VARIANTS and seed==101 and 0<steps<=PILOT_STEPS
    elif stage=='formal':
        valid=variant in FORMAL_VARIANTS and seed in SEEDS and 0<steps<=FORMAL_STEPS
        if not (OUT/'selection.json').exists():raise ValueError('Select fixed distance before formal training')
    elif stage=='smoke':valid=variant in PILOT_VARIANTS and 0<steps<=512
    else:valid=False
    if not valid:raise ValueError('Unregistered variant, seed, stage or training budget')
    run=OUT/stage/f'{variant}_seed{seed}'
    checkpoint=run/'checkpoint.pt'
    start=int(torch.load(checkpoint,map_location='cpu',weights_only=False)['steps']) if checkpoint.exists() else 0
    previous=UPDATE_CONTEXT
    UPDATE_CONTEXT=dict(path=str(run/'actor_update_diagnostics.jsonl'),checkpoint_start_steps=start,invocation_id=str(time.time_ns()))
    try:return backend.train(variant,seed,stage,steps)
    finally:UPDATE_CONTEXT=previous


def evaluate(run,split,regime='nominal'):
    check_protocol();run=Path(run).resolve();run.relative_to(OUT.resolve())
    if split=='test':
        frozen=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
        if frozen['protocol_sha256']!=backend.sha(OUT/'protocol.json') or frozen['selection_sha256']!=backend.sha(OUT/'selection.json'):
            raise ValueError('Frozen protocol or selection changed')
    return backend.evaluate(run,split,regime)


def _validated_run(stage,variant,seed,budget):
    run=OUT/stage/f'{variant}_seed{seed}'
    cfg=yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8'))
    provenance=json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'training_summary.json').read_text(encoding='utf-8'))
    source=source_state()['source_sha256'];model_hash=backend.sha(run/'checkpoint.pt');config_hash=backend.sha(run/'config.yaml')
    if cfg!=config_for(variant,seed) or provenance['config_sha256']!=config_hash:
        raise ValueError('Run configuration differs from registered variant')
    if provenance['source_sha256']!=source or provenance['path_manifest_sha256']!=backend.sha(DATA/'manifest.json'):
        raise ValueError('Run source/manifest mismatch')
    checkpoint=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=False)
    if summary['steps']!=budget or checkpoint['steps']!=budget or summary['checkpoint_sha256']!=model_hash:
        raise ValueError('Exact registered training budget/checkpoint required')
    evaluation=run/'evaluation_validation_nominal'
    rows=json.loads((evaluation/'summary.json').read_text(encoding='utf-8'))['rows']
    ids={p['id'] for p in manifest()['paths'] if p['split']=='validation'}
    if len(rows)!=len(ids) or {r['path_id'] for r in rows}!=ids:raise ValueError('Missing/duplicated validation paths')
    for r in rows:
        if (r['split']!='validation' or r['regime']!='nominal' or r['checkpoint_sha256']!=model_hash
            or r['config_sha256']!=config_hash or r['code_sha256']!=source
            or r['effective_config_sha256']!=backend.sha(evaluation/'effective_config.yaml')
            or r!=json.loads((evaluation/(r['path_id']+'.json')).read_text(encoding='utf-8'))):
            raise ValueError('Stale or inconsistent validation rows')
    return run,rows


def _rank(variant,rows):
    times=[r['elapsed_time_s'] for r in rows if r['success']]
    return dict(variant=variant,score=[sum(r['success'] for r in rows),
        sum(r['path_completed'] and r['geometry_feasible'] for r in rows),
        float(np.mean([r['ordered_progress'] for r in rows])),-float(np.mean(times)) if times else -1e9])


def select():
    check_protocol();target=OUT/'selection.json'
    if target.exists():raise FileExistsError('Preserve selected fixed distance')
    ranking=[]
    for variant in PILOT_VARIANTS:
        _,rows=_validated_run('pilot',variant,101,PILOT_STEPS);ranking.append(_rank(variant,rows))
    chosen=max((r for r in ranking if r['variant'] in DISTANCES),key=lambda r:r['score'])['variant']
    backend.write_json(target,dict(selected_fixed=chosen,fixed_distance_mm=DISTANCES[chosen],rankings=ranking,
        source_sha256=source_state()['source_sha256'],protocol_sha256=backend.sha(OUT/'protocol.json'),
        test_results_used_for_selection=False,timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return target


def freeze():
    check_protocol();target=OUT/'freeze.json'
    if target.exists():raise FileExistsError('Preserve existing test freeze')
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    if selection['source_sha256']!=source_state()['source_sha256'] or selection['protocol_sha256']!=backend.sha(OUT/'protocol.json'):
        raise ValueError('Selection source/protocol mismatch')
    models,configs,rankings={},{},[]
    for variant in FORMAL_VARIANTS:
        combined=[]
        for seed in SEEDS:
            run,rows=_validated_run('formal',variant,seed,FORMAL_STEPS)
            key=str(run.relative_to(OUT)).replace('\\','/')
            models[key],configs[key]=backend.sha(run/'checkpoint.pt'),backend.sha(run/'config.yaml');combined.extend(rows)
        rankings.append(_rank(variant,combined))
    backend.write_json(target,dict(models=models,configs=configs,validation_ranking=rankings,
        source_sha256=source_state()['source_sha256'],path_manifest_sha256=backend.sha(DATA/'manifest.json'),
        protocol_sha256=backend.sha(OUT/'protocol.json'),selection_sha256=backend.sha(OUT/'selection.json'),selection=selection,
        primary_adaptive_candidate='direct',anchor_ablation='direct_no_anchor',test_results_used_for_selection=False,
        timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','select','freeze'):commands.add_parser(name)
    p=commands.add_parser('train');p.add_argument('--variant',choices=[*PILOT_VARIANTS,'selected_fixed'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['smoke','pilot','formal'],default='pilot')
    p.add_argument('--steps',type=int,default=PILOT_STEPS)
    for name in ('evaluate','eval'):
        p=commands.add_parser(name);p.add_argument('--run',type=Path,required=True)
        p.add_argument('--split',choices=['validation','test','regression'],required=True)
        p.add_argument('--regime',choices=['nominal','tight','low_jerk'],default='nominal')
    args=parser.parse_args();activate_backend()
    if args.command=='train':print(train(args.variant,args.seed,args.stage,args.steps),flush=True)
    elif args.command in ('eval','evaluate'):evaluate(args.run,args.split,args.regime)
    else:print(globals()[args.command](),flush=True)


if __name__=='__main__':main()
