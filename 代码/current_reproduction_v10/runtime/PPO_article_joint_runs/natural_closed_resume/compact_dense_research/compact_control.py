"""Explicit compact observations and a single memoryless PPO Actor/Critic.

No inherited model, fitted motion residual, cruise override, route identifier,
clock input, or unreported policy memory. The common physical executor is reused.
"""
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(BASE),str(OUT)]
import joint_runtime
from joint_runtime_v5 import JointRuntimeV5
from continuous_gate_guidance import continuous_gate_target,active_gate_transitions
from turn_aware_projector import TurnAwareProjector

SENSORS={14:(2.,6.,15.),12:(2.,6.),10:(6.,)}

def feature_names(dimension=14):
    return ['cross_track','velocity_heading_x','velocity_heading_y','acceleration_heading_x',
        'acceleration_heading_y','reference_heading_cos','reference_heading_sin']+[
        f'forward_{d:g}mm_{axis}' for d in SENSORS[dimension] for axis in ['x','y']]+['terminal_distance_30mm']

def local_progress(env):
    active=active_gate_transitions(env,2.,require_channel=False)
    if len(active)!=1:return env.progress()[0]
    gate,distance=active[0];c=env.corridor;values=[]
    for i in [gate.vertex_index-1,gate.vertex_index]:
        along=np.clip((env.position-c.starts[i])@c.tangents[i],0.,c.lengths[i])
        values.append(c.arcs[i]+along)
    z=np.clip(.5+distance/2.,0.,1.);w=z*z*(3.-2.*z)
    return float((1.-w)*values[0]+w*values[1])

class CompactRuntime(JointRuntimeV5):
    def __init__(self,reference,*,dimension=14,action_dim=3,preview_rule_mm=1.5,allow_swing=True,max_time=None):
        if dimension not in SENSORS or action_dim not in (2,3):raise ValueError('Unsupported compact interface')
        self.dimension=dimension;self.action_dim=action_dim;self.allow_swing=allow_swing
        self.preview_rule_mm=float(preview_rule_mm)
        length=float(np.linalg.norm(np.diff(reference,axis=0),axis=1).sum())
        super().__init__(reference,fixed_d=None,max_time=max_time or max(5.,length/20.+1.),
            closed=bool(np.linalg.norm(reference[0]-reference[-1])<1e-9))
        self.experimental_projector=TurnAwareProjector(self)
        self.extra_failure=None
        tangent=self.corridor.tangents
        change=np.arctan2(tangent[:-1,0]*tangent[1:,1]-tangent[:-1,1]*tangent[1:,0],
                          (tangent[:-1]*tangent[1:]).sum(1))
        self.corner_vertices=np.flatnonzero(abs(change)>np.deg2rad(10.))+1
        self.corner_sign=np.sign(change[self.corner_vertices-1])
        original_update=self.tracker.update
        def update(position):
            count=original_update(position)
            if not self.allow_swing and len(self.corner_vertices):
                j=int(np.searchsorted(self.corner_vertices,self.tracker.frontier,side='right')-1)
                if j>=0:
                    k=int(self.corner_vertices[j]);s,_=self.progress()
                    next_s=(self.corridor.arcs[self.corner_vertices[j+1]] if j+1<len(self.corner_vertices) else self.corridor.arcs[-1])
                    end=min(self.corridor.arcs[k]+50.,.5*(self.corridor.arcs[k]+next_s))
                    outgoing=tangent[k];normal=self.corner_sign[j]*np.array([-outgoing[1],outgoing[0]])
                    e=float((np.asarray(position)-self.corridor.reference[k])@normal)
                    if s<=end+1e-9 and e < -1e-7:
                        self.extra_failure='outward_prior_violation';self.done=True;self.reason=self.extra_failure
            return count
        self.tracker.update=update

    def decode_action(self,latent,fixed_d=None):
        raw=np.asarray(latent,float)
        if raw.shape!=(self.action_dim,) or not np.isfinite(raw).all():raise ValueError('Wrong action dimension')
        action=np.clip(raw,-1.,1.)
        preview=.8+5.6*(1.+action[2]) if self.action_dim==3 else self.preview_rule_mm
        return np.array([math.pi/2*action[0],50.*(1.+action[1]),preview])

    def guidance_target(self):return continuous_gate_target(self)

    def observation(self):
        s=local_progress(self);total=self.corridor.arcs[-1]
        point,_=self.sample_reference(s)
        low,_=self.sample_reference(max(0.,s-1.));high,_=self.sample_reference(min(total,s+1.))
        t=high-low;t/=max(float(np.linalg.norm(t)),1e-12)
        normal=np.array([-t[1],t[0]])
        h=np.array([math.cos(self.heading),math.sin(self.heading)])
        frame=np.stack([h,[-h[1],h[0]]],axis=1)
        points,_=self.sample_reference(s+np.asarray(SENSORS[self.dimension]))
        result=np.r_[float((self.position-point)@normal)/1.25,self.velocity@frame/100.,
            self.acceleration@frame/2000.,t@frame,((points-self.position)@frame/15.).ravel(),
            np.clip((total-s)/30.,0.,1.)].astype(np.float32)
        assert result.shape==(self.dimension,) and np.isfinite(result).all()
        return result

    def step(self,latent):
        previous=joint_runtime.projector
        joint_runtime.projector=self.experimental_projector
        try:
            result=super().step(latent)
        finally:joint_runtime.projector=previous
        if self.extra_failure:
            result[3]['reason']=self.extra_failure
        return result

    def state(self):
        value=super().state()
        value['tracker'].pop('update',None)
        value['extra_failure']=self.extra_failure
        return value

class CompactPolicy(nn.Module):
    def __init__(self,dimension=14,action_dim=3,seed=922301):
        super().__init__();self.dimension=dimension;self.action_dim=action_dim
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            def network(n):return nn.Sequential(nn.Linear(dimension,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,n))
            self.actor=network(action_dim);self.critic=network(1)
            # Matched seed streams for common layers across2/3-output actors.
            for head_idx,head in enumerate([self.actor,self.critic]):
                for i,layer in enumerate(head):
                    if isinstance(layer,nn.Linear):
                        torch.manual_seed(seed+1000*head_idx+10*i);layer.reset_parameters()
            nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)
            with torch.no_grad():
                self.actor[-1].bias.copy_(torch.tensor([0.,-.4,(1.5-6.4)/5.6][:action_dim]))
        self.log_std=nn.Parameter(torch.log(torch.tensor([.025,.10,.08][:action_dim])))

    def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-5.,-.3).exp())
    def value(self,x):return self.critic(x).squeeze(-1)
    def evaluate(self,x,latent):
        d=self.distribution(x)
        return d.log_prob(latent).sum(-1),d.entropy().sum(-1),self.value(x)
    @torch.no_grad()
    def act(self,x,deterministic=False):
        x=torch.as_tensor(x,dtype=torch.float32);d=self.distribution(x)
        latent=d.mean if deterministic else d.sample()
        return latent.numpy(),float(d.log_prob(latent).sum()),float(self.value(x))
