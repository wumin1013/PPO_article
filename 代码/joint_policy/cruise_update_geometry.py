"""Saved-state update geometry; no deployed action or trajectory rules."""
import numpy as np
from search_complete_return import read
from study_io import sha


def capsule_endpoint_facets(model,x,indices=None):
    """Expose redundant smooth endpoint constraints without changing capsules."""
    xy=model.nominal.copy()
    xy[1:]+=np.einsum('tcd,d->tc',model.gradient,x)
    points=(1.-model.alpha[:,None])*xy[model.point_index]+model.alpha[:,None]*xy[model.right_index]
    offsets=points-model.starts
    projection=np.clip(np.sum(offsets*model.vectors,axis=1)/model.squared_length,0.,1.)
    residual=offsets-projection[:,None]*model.vectors
    distance=np.linalg.norm(residual,axis=1)
    if indices is None:return distance
    jac=[]
    for k in indices:
        point=int(model.point_index[k]);alpha=model.alpha[k]
        normal=residual[k]/max(distance[k],1e-30);row=np.zeros(model.gradient.shape[-1])
        if point:row+=(1.-alpha)*(normal@model.gradient[point-1])
        if alpha:row+=alpha*(normal@model.gradient[model.right_index[k]-1])
        jac.append(row)
    return distance[indices],np.asarray(jac)


class CruiseUpdateGeometry:
    def __init__(self, directory, arm, parent_sha):
        assert arm in ('latent', 'physical')
        assert read(directory/'receipt.json')['status']=='prepared_matched_physical_and_latent_update_geometries'
        metadata=read(directory/'results.json')
        assert metadata['parent_sha256']==parent_sha
        assert metadata['nominal_runtime_decoder_max_error']<1e-10
        assert metadata['perturbed_runtime_decoder_max_error']<1e-10
        with np.load(directory/'geometry.npz') as z:
            self.metric=z['metric_'+arm].copy()
            self.support=z['support'].copy()
            self.lower=z['lower_'+arm].copy()
            self.upper=z['upper_'+arm].copy()
            self.features=z['cruise_features'].copy()
            self.preservation_features=(z['cruise_preservation_features'].copy()
                if 'cruise_preservation_features' in z else self.features.copy())
            self.latent=z['cruise_latent'].copy()
            self.weight=z['cruise_weight'].copy()
            cruise_indices=z['cruise_state_indices'].copy()
        assert self.metric.shape==(768,768)
        assert np.all(self.lower<0.) and np.all(self.upper>0.)
        assert abs(self.weight.sum()-1.)<1e-10
        self.positive_cruise_latent_domain=bool(np.all(self.latent+self.lower.reshape(-1,3)[cruise_indices,1]>0.))
        self.identity=dict(directory=directory.name,arm=arm,sha256=sha(directory/'geometry.npz'))

    def support_constraint(self,x):
        """Largest normalized violation of the exact saved-action halfspaces."""
        delta=self.support@x
        best=(-np.inf,None)
        for sign,bound in ((1.,self.upper),(-1.,-self.lower)):
            index=np.flatnonzero(np.isfinite(bound))
            value=sign*delta[index]/bound[index]-1.
            k=int(np.argmax(value));row=index[k]
            if value[k]>best[0]:best=(float(value[k]),sign*self.support[row]/bound[row])
        return best

    def objective(self,x):
        z=self.latent+self.features@x[256:512]
        t=np.tanh(z);deficit=np.maximum(99.5-50.*(1.+t),0.)
        loss=.5*np.sum(self.weight*deficit**2)
        gradient=np.zeros(768)
        gradient[256:512]=self.features.T@(-self.weight*deficit*50.*(1.-t*t))
        return float(loss),gradient
