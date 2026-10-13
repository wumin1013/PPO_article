"""Finite-horizon action search for diagnosing dense-path dynamic viability.

This is online model-based parameter selection, NOT PPO training or a new
learned policy. A forecast holds a candidate for120ms and is not a recursive
safety certificate. The real environment retains all original failure checks.
"""
import time
import numpy as np
from turn_aware_projector import environment_factory


def encode(action):
    unit=np.asarray(action,dtype=float)/np.array([np.pi/2,50.,5.6])-np.array([0.,1.,6.4/5.6])
    return np.arctanh(np.clip(unit,-1.+1e-10,1.-1e-10))


class PredictiveActionDiagnostic:
    def __init__(self,base,deadline,horizon_steps=6):
        self.base=base
        self.deadline=deadline
        self.horizon_steps=horizon_steps
        self.records=[]
        self.forecast_ticks=0

    def bind(self,env):
        self.env=env
        self.prediction=environment_factory(env.corridor.reference)
        self.prediction.fixed_d=env.fixed_d
        self.base.bind(env)
        self.records=[]
        return env

    def forecast(self,state,latent):
        if time.perf_counter()>=self.deadline:
            raise TimeoutError('Registered predictive diagnostic wall limit')
        p=self.prediction
        p.restore(state)
        p.pending_pulse=np.zeros(3)
        largest=0.
        steps=0
        initial_s=p.progress()[0]
        while steps<self.horizon_steps and not p.done:
            _,_,_,info=p.step(latent)
            self.forecast_ticks+=len(info['rows'])
            steps+=1
            for point in info['rows'][:,:2]:
                largest=max(largest,p.corridor.point_distance(point))
        physical_ok=not p.done or p.success
        return dict(feasible=bool(physical_ok and largest<=1.24),
            physical_ok=bool(physical_ok),max_distance_mm=float(largest),
            forecast_ticks=p.ticks-state['ticks'],reason=p.reason,
            progress_mm=float(p.progress()[0]-initial_s),end_speed=float(np.linalg.norm(p.velocity)))

    def act(self,observation,deterministic=True):
        nominal=self.base.act(observation,deterministic)[0]
        assert not np.any(self.env.pending_pulse)
        state=self.env.state()
        nominal_action=self.env.decode_action(nominal,self.env.fixed_d)
        records=[dict(label='nominal',action=nominal_action.tolist(),latent=nominal,
                      intervention=0.,**self.forecast(state,nominal))]
        if not records[0]['feasible']:
            candidates=[]
            def add(label,steering=None,feed=None,preview=None):
                a=nominal_action.copy()
                if steering is not None:a[0]=steering
                if feed is not None:a[1]=feed
                if preview is not None:a[2]=preview
                a=np.clip(a,[-np.pi/2,0.,.8],[np.pi/2,100.,12.])
                if not any(np.max(np.abs(a-b))<1e-8 for _,b in candidates):candidates.append((label,a))
            shift=np.deg2rad(5.)
            add('zero_steer',steering=0.)
            for sign in [-1,1]:add(f'steer_{sign}',steering=nominal_action[0]+sign*shift)
            for f in [.8,.6]:add(f'feed_{f}',feed=f*nominal_action[1])
            for d in [2.,4.,6.]:add(f'preview_{d}',preview=d)
            add('joint_feed_preview',feed=.8*nominal_action[1],preview=4.)
            for sign in [-1,1]:add(f'joint_{sign}',steering=nominal_action[0]+sign*shift,feed=.8*nominal_action[1],preview=4.)
            for label,a in candidates:
                latent=encode(a)
                intervention=float(np.sum(((a-nominal_action)/np.array([shift,20.,2.]))**2))
                records.append(dict(label=label,action=a.tolist(),latent=latent,intervention=intervention,
                                    **self.forecast(state,latent)))
        feasible=[r for r in records if r['feasible']]
        if feasible:
            selected=min(feasible,key=lambda r:(r['intervention'],-r['progress_mm']))
        else:
            selected=max(records,key=lambda r:(r['forecast_ticks'],-r['max_distance_mm'],-r['intervention']))
        self.records.append(dict(decision=self.env.decisions,physical_time_s=self.env.ticks*.001,
            selected=selected['label'],feasible_candidates=len(feasible),
            candidates=[{k:v for k,v in r.items() if k!='latent'} for r in records]))
        return selected['latent'].copy(),0.,0.
