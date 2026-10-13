"""Sample-request PPO interface over the byte-frozen recoverable executor.

Policy likelihood belongs to the original Gaussian sample, never to a corrected
request, geometric action or stored stop. The filter is part of the environment.
"""
import copy
import inspect
import math
import textwrap
import time
import numpy as np
import torch
from common import sha
from strong_ppo_v4 import StrongPPOV4
from strong_policy_v4 import load_strong_policy,tensor_hash,STATIC_KEYS,StrongPolicyV4
import recoverable_action_candidate_v5 as frozen
from geometric_resumption_candidate_v5 import geometric_action as adaptive_geometric_action
from geometric_resolution_check_v5 import configure,install_checked
from predictive_action_candidate_v5 import new_environment
from physical_teacher_v5 import guide_angle

VERSION='safe_request_ppo_v7_1'
DT=.001
CONTROL_PERIOD=.02
FIXED_PREVIEW=3.
STD=np.array([.008,.08,.02],dtype=np.float32)


class SafePPOPolicy(StrongPPOV4):
    def dist(self,x):
        mean=self.actor(x)[...,:self.action_dim]
        base=self.log_std[:self.action_dim].clamp(-8.,-2.).exp()
        mask=torch.tensor([0.,1.,0.],dtype=mean.dtype,device=mean.device)[:self.action_dim]
        scale=base+mask*torch.relu(mean.abs()-1.)
        return torch.distributions.Normal(mean,scale)


def from_fit(path):
    fitted,metadata=load_strong_policy(path)
    policy=SafePPOPolicy(fitted)
    with torch.no_grad():policy.log_std.copy_(torch.log(torch.tensor(STD)))
    return policy,metadata


def load_policy(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['format']=='safe_request_ppo_v7'
    assets=dict(tensors={key:saved['policy']['actor.'+key] for key in STATIC_KEYS})
    policy=SafePPOPolicy(StrongPolicyV4(saved['seed'],assets,saved['arm'],FIXED_PREVIEW))
    policy.load_state_dict(saved['policy'],strict=True)
    policy.action_generator.set_state(saved['action_rng_state'])
    assert tensor_hash(dict(policy.named_parameters()))==saved['trainable_tensor_sha256']
    return policy,saved


def fixed_geometric_action(env):
    if env.fixed_d is None:return adaptive_geometric_action(env)
    assert env.fixed_d==FIXED_PREVIEW
    target=(env.corridor.gates[env.tracker.frontier].point if
            env.tracker.frontier<len(env.corridor.gates) else env.corridor.reference[-1])
    delta=target-env.position;bearing=math.atan2(delta[1],delta[0])
    wrap=lambda value:(value+math.pi)%(2*math.pi)-math.pi
    beta=guide_angle(env.env,0.,20.,FIXED_PREVIEW)
    steering=np.clip(wrap(bearing-beta),-math.pi/2,math.pi/2)
    feed=0. if abs(wrap(bearing-env.heading))>.05 else 20.
    return np.array([steering/(math.pi/2),feed/50.-1.,(FIXED_PREVIEW-6.4)/5.6])


class SampleRequest:
    def __init__(self,arm):self.arm=arm;self.calls=0;self.observation=None;self.sample=None
    def begin(self,observation,sample):
        self.observation=np.asarray(observation).copy();self.sample=np.asarray(sample).copy();self.calls=0
        assert self.sample.shape==((3,) if self.arm=='adaptive' else (2,))
    def mean(self,observation):
        assert self.calls==0 and np.array_equal(observation,self.observation),'One cached policy request per real decision'
        self.calls+=1
        request=(self.sample if self.arm=='adaptive' else np.r_[self.sample,(FIXED_PREVIEW-6.4)/5.6])
        return torch.tensor(request)


def injected_function(function,replacements,extra):
    source=textwrap.dedent(inspect.getsource(function))
    for old,new in replacements:
        assert source.count(old)==1,(function.__name__,old)
        source=source.replace(old,new)
    namespace=dict(frozen.__dict__);namespace.update(extra);output={}
    exec(compile(source,__file__,'exec'),namespace,output)
    return output[function.__name__]


def constructor(provider):
    return injected_function(frozen.RecoverableActionCandidate.__init__,
        [('self.teacher = FrozenPolicy(model)','self.teacher = request_provider')],dict(request_provider=provider))


def controller_step():
    old="            source = 'certified_stop'\n"
    new="""            restore(self.clone, state)
            geometric = evaluate(geometric_action(self.clone), 25)
            if geometric[2]['feasible']:
                selected = geometric
                source = 'geometric_resumption'
            else:
                source = 'certified_stop'
"""
    return injected_function(frozen.RecoverableActionCandidate._step,[(old,new)],dict(geometric_action=fixed_geometric_action))


class RequestController(frozen.RecoverableActionCandidate):
    _step=controller_step()
    def __init__(self,initial_fit,output,provider,deadline):
        constructor(provider)(self,initial_fit,output,deadline)
        self.counts['geometric_resumption']=0
    def close(self):
        result=super().close();result.update(version=VERSION,
            model_sha256_role='Supervised initialization file; actual current requests and PPO snapshot identities are logged separately',
            likelihood_action='Original sampled Gaussian request, not executed correction or stop')
        return result


class SafeEpisode:
    def __init__(self,reference,arm,fit,output,*,tolerance=1.25,deadline=None):
        configure(DT);self.settings=install_checked(dt=DT,tolerance=tolerance)
        self.reference=np.asarray(reference);self.arm=arm
        self.env=new_environment(reference)
        if arm=='fixed':self.env.env.fixed_d=FIXED_PREVIEW;self.env.env.held[2]=FIXED_PREVIEW
        self.env.env.max_ticks=round((5.+np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()/5.)/DT)
        self.positions=[reference[0].copy()];self.held_previews=[];self.rows=[];self.transitions=[]
        update=self.env.tracker.update
        def record(point):
            self.positions.append(np.asarray(point).copy());self.held_previews.append(float(self.env.held[2]));return update(point)
        self.env.tracker.update=record
        self.provider=SampleRequest(arm)
        self.controller=RequestController(fit,output,self.provider,deadline)
        self.ended=False

    def advance(self,observation,sample):
        assert not self.ended and np.array_equal(observation,self.env.observation())
        self.provider.begin(observation,sample)
        tick_before=self.env.ticks;phi_before=float(self.env.potential());started=time.perf_counter()
        error=None;truncated=False;submitted=None;decision=None
        try:
            submitted,rows,decision=self.controller.step(self.env)
            terminated=bool(self.env.done);success=bool(self.env.success)
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            rows=np.asarray(self.controller.actual_partial_rows).reshape(-1,13)
            # A computational wall cap is external truncation, not an MDP failure.
            truncated=isinstance(exc,TimeoutError) and not self.env.done
            terminated=bool(self.env.done or not truncated);success=False
            decision=self.controller.active_record.copy()
        elapsed=(self.env.ticks-tick_before)*DT
        assert len(rows)==self.env.ticks-tick_before
        assert len(self.positions)-1==self.env.ticks
        if self.arm=='fixed':
            assert self.env.fixed_d==FIXED_PREVIEW and self.env.held[2]==FIXED_PREVIEW
            assert all(value==FIXED_PREVIEW for value in self.held_previews)
        phi_after=0. if terminated else float(self.env.potential())
        penalty=0.
        if terminated and not success:
            remaining=max(0.,self.env.corridor.arcs[-1]-self.env.progress()[0])
            penalty=self.env.max_ticks*DT+1.+remaining/20.
        reward=-elapsed+phi_after-phi_before-penalty
        result=dict(elapsed_s=elapsed,tick_before=tick_before,tick_after=self.env.ticks,
            phi_before=phi_before,phi_after=phi_after,failure_penalty=penalty,reward=reward,
            terminated=terminated,truncated=truncated,success=success,error=error,
            source=decision['source'],submitted_request=None if submitted is None else submitted.tolist(),
            policy_sample=np.asarray(sample).tolist(),cached_request_calls=self.provider.calls,
            reason=self.env.reason or error,executor_wall_s=time.perf_counter()-started,
            likelihood_uses_original_sample=True)
        self.rows.append(rows);self.transitions.append(result);self.ended=terminated or truncated
        return self.env.observation().copy(),result

    def close(self):return self.controller.close()


def physical_time_gae(records,gamma=1.,lam=.95):
    """Undiscounted episodic time objective; eligibility decay per20ms physical time."""
    assert gamma==1.,'This version registers an undiscounted complete-episode time objective'
    advantages=np.empty(len(records),dtype=np.float32);groups={}
    for i,row in enumerate(records):groups.setdefault((row['path'],row['episode']),[]).append(i)
    for indices in groups.values():
        carry=0.
        for n,index in reversed(list(enumerate(indices))):
            row=records[index]
            if n+1<len(indices):
                following=records[indices[n+1]]
                assert row['tick_after']==following['tick_before'] and not row['terminated'] and not row['truncated']
            live=1.-float(row['terminated'])
            delta=row['reward']+live*row['next_value']-row['value']
            trace=lam**(row['elapsed_s']/CONTROL_PERIOD)
            carry=delta+live*trace*carry;advantages[index]=carry
    returns=advantages+np.asarray([r['value'] for r in records],dtype=np.float32)
    return advantages,returns
