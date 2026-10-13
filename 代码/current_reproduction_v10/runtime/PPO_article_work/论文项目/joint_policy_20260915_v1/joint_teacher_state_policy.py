"""Compact observable-state joint residual learned from complete-return teachers."""
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from raw_state_joint_policy import LOCAL_INDICES
from reflection_policy import reflect_observation
from decision_innovation_policy import innovation_features
from frame_feedback_policy import PAIRS,NORMAL_PAIRS


INDICES=LOCAL_INDICES+[8,9,10]
OUTPUT_SCALE=np.array([.12,.25,.25])


def reference_oriented_observation(observation):
    """Use only the observed orientation, including the declared gate blend."""
    x=np.asarray(observation,dtype=float);out=x.copy()
    norm=np.sqrt(x[...,6:7]**2+x[...,7:8]**2+1e-12)
    c=x[...,6:7]/norm;s=x[...,7:8]/norm
    out[...,PAIRS]=c*x[...,PAIRS]-s*x[...,NORMAL_PAIRS]
    out[...,NORMAL_PAIRS]=s*x[...,PAIRS]+c*x[...,NORMAL_PAIRS]
    return out


def motion_geometry_context(observations):
    x=np.asarray(observations)
    return np.stack([100*np.linalg.norm(x[...,2:4],axis=-1),30*x[...,11]],axis=-1)


def state_pair(observations,innovations):
    obs=np.asarray(observations)
    mirror=reflect_observation(torch.as_tensor(obs)).numpy()
    return (np.concatenate([obs[...,INDICES],innovations],axis=-1),
            np.concatenate([mirror[...,INDICES],-np.asarray(innovations)],axis=-1))


def features(spec,observations,innovations):
    pair=state_pair(observations,innovations)
    reference_pair=(state_pair(reference_oriented_observation(observations),innovations)
                    if 'per_center_reference_frame' in spec else None)
    bend_distance=0.
    if 'observed_bend_metric_weights' in spec:
        # A learned observation metric, shared across every route and action.
        obs=np.asarray(observations)
        directions=obs[...,42:62].reshape(*obs.shape[:-1],10,2).astype(float)
        directions/=np.maximum(np.linalg.norm(directions,axis=-1,keepdims=True),1e-12)
        dot=np.sum(directions*directions[...,:1,:],axis=-1)
        bend=np.max(.5*(1.-np.clip(dot,-1.,1.))*np.clip(obs[...,62:72],0.,1.),axis=-1)
        weights=np.asarray(spec['observed_bend_metric_weights'])
        assert np.all(weights>=0.)
        bend_distance=(bend[...,None]-spec['observed_bend_metric_centers'])**2*weights
    kernels=[]
    for pair_index,raw in enumerate(pair):
        if reference_pair is not None:
            mask=np.asarray(spec['per_center_reference_frame'],dtype=bool)
            assert mask.shape==(len(spec['centers']),)
            per_center_raw=np.where(mask[:,None],reference_pair[pair_index][...,None,:],raw[...,None,:])
            x=(per_center_raw-spec.get('per_center_mean',spec['mean']))/spec.get('per_center_std',spec['std'])
        elif 'per_center_mean' in spec:
            x=(raw[...,None,:]-spec['per_center_mean'])/spec['per_center_std']
        else:
            x=((raw-spec['mean'])/spec['std'])[...,None,:]
        squared=(x-spec['centers'])**2
        if 'per_center_distance_weights' in spec:
            weights=np.asarray(spec['per_center_distance_weights'])
            assert np.all(weights>=0.)
            squared=squared*weights
        distance=np.sqrt(np.mean(squared,axis=-1)+bend_distance)/spec['radii']
        kernels.append(np.maximum(1-distance,0.)**4*(1+4*distance))
    a,b=kernels
    return np.stack([.5*(a-b),.5*(a+b),.5*(a+b)],axis=-1)


class JointTeacherStatePolicy:
    def __init__(self,metadata):
        self.metadata=metadata
        self.base=load_credit_policy(metadata['parent'])
        self.env=None

    @property
    def coefficients(self):
        return self.metadata['coefficients']

    @coefficients.setter
    def coefficients(self,value):
        self.metadata['coefficients']=np.asarray(value).reshape(-1,3)

    def bind(self,env):
        if self.metadata.get('temporal_mode')=='quadratic_decay_pulse':
            from learned_action_pulse_policy import LearnedActionPulseRuntime
            if not isinstance(env,LearnedActionPulseRuntime):
                raise TypeError('Learned temporal residual requires the declared 1ms pulse runtime')
        self.env=env
        if hasattr(self.base,'bind'):self.base.bind(env)
        return env

    def act(self,observation,deterministic=True):
        original=self.base.act(observation,deterministic)[0]
        _,innovation=innovation_features(self.env,original)
        f=features(self.metadata,observation,innovation)
        coefficients=self.metadata['coefficients'];count=len(self.metadata['centers'])
        correction=(f*coefficients[:count]).sum(axis=-2)*OUTPUT_SCALE
        if self.metadata.get('local_affine_context') and np.any(coefficients[count:]):
            context=np.tanh((motion_geometry_context(observation)-self.metadata['center_context'])/
                            self.metadata['context_scale'])
            for index in range(context.shape[-1]):
                block=coefficients[(index+1)*count:(index+2)*count]
                correction+=(f*context[:,index,None]*block).sum(axis=-2)*OUTPUT_SCALE
        if self.metadata.get('temporal_mode')=='quadratic_decay_pulse':
            self.env.pending_pulse=correction
            return original,0.,0.
        return (original+correction).astype(original.dtype),0.,0.
