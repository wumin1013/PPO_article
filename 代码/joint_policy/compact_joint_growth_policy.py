"""Smooth compact state features for joint growth from the retained actor."""
import numpy as np
import torch
from retained_joint_growth_policy import RetainedJointGrowthPolicy
from raw_state_joint_policy import construct_raw_basis, LOCAL_INDICES
from reflection_policy import reflect_observation


class CompactJointGrowthPolicy(RetainedJointGrowthPolicy):
    def raw_features(self, observation):
        def kernel(x):
            x = (x[..., LOCAL_INDICES]-self.mean)/self.std
            r = torch.sqrt((x.unsqueeze(-2)-self.centers).square().mean(-1))/self.radius
            # Wendland C2 basis: compact support is a continuous state-function
            # architecture, not a path phase, angle limiter or runtime switch.
            return torch.clamp(1-r, min=0.).pow(4)*(4*r+1)
        a,b = kernel(observation),kernel(reflect_observation(observation))
        odd,even = .5*(a-b),.5*(a+b)
        n=len(self.centers)
        result=observation.new_zeros(observation.shape[:-1]+(3,3*n))
        result[...,0,:n]=odd
        result[...,1,n:2*n]=even*self.feed_source.feed_correction(observation).unsqueeze(-1)
        result[...,2,2*n:]=even
        return result


def construct_compact_growth(retained,feed,observations):
    basis,coverage=construct_raw_basis(observations,count=16)
    x=(observations[:,LOCAL_INDICES]-basis['mean'])/basis['std']
    mirror=reflect_observation(torch.as_tensor(observations,dtype=torch.float32)).numpy()
    y=(mirror[:,LOCAL_INDICES]-basis['mean'])/basis['std']
    centers=basis['centers']
    distance=np.sqrt(((centers[:,None]-centers[None,:])**2).mean(-1))
    distance[np.eye(len(centers),dtype=bool)]=np.inf
    nearest=np.minimum(np.min(np.sqrt(((x[:,None]-centers[None,:])**2).mean(-1)),axis=1),
                       np.min(np.sqrt(((y[:,None]-centers[None,:])**2).mean(-1)),axis=1))
    radius=max(float(np.median(distance.min(1))*1.5),float(np.quantile(nearest,.99)*1.3),.2)
    basis['radius']=np.array(radius)
    policy=CompactJointGrowthPolicy(retained,feed,basis,np.zeros((48,1)))
    with torch.no_grad():
        features=policy.raw_features(torch.as_tensor(observations,dtype=torch.float32)).numpy()
    maximum=abs(features).max(axis=(0,1))
    keep=maximum>1e-7
    limits=np.repeat([.02,1.5,.1],16)
    transform=np.eye(48)[:,keep]*(limits[keep]/maximum[keep])[None,:]
    return basis,transform,dict(coverage=coverage,radius=radius,dimensions=int(keep.sum()),
        maximum_nearest_center_distance=float(nearest.max()),
        fraction_with_any_compact_support=float((nearest<radius).mean()),
        mean_active_functions=float((abs(features).sum(1)>1e-9).sum(1).mean()),
        omitted_identically_negligible_columns=np.flatnonzero(~keep).tolist(),
        function='Wendland C2 in existing continuous physical/geometric state, reflection-equivariant.',
        learned_outputs='All steering, nonlinear-feed amplitude and preview coefficients; no labels or deployed path/angle rules.')
