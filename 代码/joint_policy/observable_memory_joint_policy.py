"""Three-action feedback with learned weights on causal observation traces."""
import numpy as np
from joint_teacher_state_policy import JointTeacherStatePolicy,features,OUTPUT_SCALE,motion_geometry_context
from decision_innovation_policy import innovation_features


def dynamic_context(observations):
    """Reflection-even speed and tangential acceleration in the observed local frame."""
    x=np.asarray(observations);v=100.*x[...,2:4];a=2000.*x[...,4:6]
    speed=np.linalg.norm(v,axis=-1)
    tangent_acceleration=np.sum(v*a,axis=-1)/np.maximum(speed,5.)
    return np.stack([speed,tangent_acceleration],axis=-1)


def future_bend_context(observations):
    """Reflection-even bend visible in the existing reference-tangent sensors."""
    x=np.asarray(observations)
    directions=x[...,42:62].reshape(*x.shape[:-1],10,2).astype(float)
    directions/=np.maximum(np.linalg.norm(directions,axis=-1,keepdims=True),1e-12)
    dot=np.sum(directions*directions[...,:1,:],axis=-1)
    return np.max(.5*(1.-np.clip(dot,-1.,1.))*np.clip(x[...,62:72],0.,1.),axis=-1)


def coefficient_context(observations,mode='dynamic'):
    if mode=='dynamic':
        return dynamic_context(observations)
    if mode=='geometry':
        x=np.asarray(observations)
        return np.stack([30.*x[...,11],30.*np.abs(x[...,12])],axis=-1)
    raise ValueError('Unknown coefficient context: '+mode)


def actor_features(metadata,observation,innovation):
    """Shared runtime/replay feature arithmetic; distance support stays geometric."""
    f=features(metadata,observation,innovation)
    context_count=metadata.get('dynamic_context_count',0)
    if context_count:
        assert not metadata.get('context_memory'), 'Use one declared context representation'
        context=np.tanh((coefficient_context(observation,metadata.get('coefficient_context','dynamic'))[:context_count]-
            metadata.get('per_center_context_mean',metadata['dynamic_context_mean']))/
            metadata.get('per_center_context_scale',metadata['dynamic_context_scale']))
        context=np.broadcast_to(context,(len(f),context_count))
        basis=metadata.get('dynamic_context_basis','linear')
        if basis!='linear':
            v,a=context[:,0:1],context[:,1:2]
            extra=[v*a] if basis=='bilinear' else [v*a,v*v,a*a]
            context=np.concatenate([context]+extra,axis=1)
        geometry_count=metadata.get('geometry_context_terms',0)
        if geometry_count:
            geom=np.tanh((future_bend_context(observation)-metadata['per_center_geometry_mean'])/
                         metadata['per_center_geometry_scale'])[:,None]
            extra=[geom,v*geom,a*geom,geom*geom][:geometry_count]
            context=np.concatenate([context]+extra,axis=1)
        f=np.concatenate([f]+[f*context[:,k,None] for k in range(context.shape[1])],axis=0)
    return f


def memory_correction(f,memory,coefficients):
    """Shared exact arithmetic for execution and saved-state action identity."""
    count=len(f)
    correction=(f*coefficients[:count]).sum(axis=-2)*OUTPUT_SCALE
    for index,trace in enumerate(memory):
        block=coefficients[(index+1)*count:(index+2)*count]
        if np.any(block):correction+=(trace*block).sum(axis=-2)*OUTPUT_SCALE
    return correction


class ObservableMemoryJointPolicy(JointTeacherStatePolicy):
    def bind(self,env):
        super().bind(env)
        context_count=self.metadata.get('dynamic_context_count',0)
        extra={'linear':0,'bilinear':1,'quadratic':3}[self.metadata.get('dynamic_context_basis','linear')]
        assert not extra or context_count==2
        geometry_count=self.metadata.get('geometry_context_terms',0)
        assert geometry_count in (0,1,3,4)
        assert not geometry_count or self.metadata.get('dynamic_context_basis')=='quadratic'
        count=len(self.metadata['centers'])*(1+context_count+extra+geometry_count)
        self.decays=np.asarray(self.metadata['memory_decays'])
        assert np.all((self.decays>0.)&(self.decays<1.))
        self.memory=np.zeros((len(self.decays),count,3))
        self.previous_features=(np.zeros_like(self.memory) if self.metadata.get('context_memory')
                                else np.zeros((count,3)))
        self.last_decision=None
        return env

    def act(self,observation,deterministic=True):
        original=self.base.act(observation,deterministic)[0]
        _,innovation=innovation_features(self.env,original)
        f=actor_features(self.metadata,observation,innovation)
        if self.last_decision!=self.env.decisions:
            if self.last_decision is None:
                assert self.env.decisions==0,'Replay policy history before a mid-route memory intervention'
            else:
                assert self.env.decisions==self.last_decision+1,'Memory requires every decision in order'
            a=self.decays[:,None,None]
            self.memory=a*self.memory+(1.-a)*self.previous_features
            if self.metadata.get('context_memory'):
                context=np.tanh((motion_geometry_context(observation)-self.metadata['center_context'])/
                                self.metadata['context_scale'])
                assert len(self.decays)==4
                self.previous_features=np.stack([f,f,f*context[:,0,None],f*context[:,1,None]])
            else:self.previous_features=f.copy()
            self.last_decision=self.env.decisions
        # Expose the exact sufficient statistics for matched on-policy credit updates.
        self.last_instant_features=f.copy()
        correction=memory_correction(f,self.memory,self.coefficients)
        self.last_memory_features=self.memory.copy()
        if self.metadata.get('residual_output_precision')=='float64':
            return original.astype(np.float64)+correction,0.,0.
        return (original+correction).astype(original.dtype),0.,0.
