"""Supporting affine bounds for convex shape quantities of a local waveform."""
import numpy as np

SIGNED_SHAPE_CHANNELS={'in_edge_outward_mm':(0,-1.),'out_edge_outward_mm':(1,-1.),
                       'negative_signed_heading_deg':(2,-1.),'heading_overshoot_deg':(2,1.)}


def corrected_signed_facet(nominal,gradient,direction,metric_direction,norm_squared,
                           rate,valid,channel,sign,offset=0.):
    """Apply measured signed error before the nonnegative maximum."""
    values=sign*(nominal[:,channel]+gradient[:,channel]@direction)+offset+rate*norm_squared
    ids=np.flatnonzero(valid)
    k=int(ids[np.argmax(values[ids])]);value=float(max(0.,values[k]))
    row=(sign*gradient[k,channel]+2*rate[k]*metric_direction if value>0. else np.zeros(len(direction)))
    return value,row,0.


def waveform_facets(nominal,gradient,valid,turn_deg,direction):
    """Return value, affine gradient and intercept, exact at direction.

    Each affine function is a global lower bound on its named convex local
    waveform quantity. Repeated separation can enforce the complete max/min
    or negative-rate sum without fixing the active times at the parent policy.
    This is exact only for the supplied affine waveform, not real new physics.
    """
    value=nominal+np.einsum('tcd,d->tc',gradient,direction)
    dimension=gradient.shape[-1];facets={}
    def maximum(name,channel,sign=1.,offset=0.,nonnegative=False):
        ids=np.flatnonzero(valid[:,channel])
        tick=int(ids[np.argmax(sign*value[ids,channel]+offset)])
        maximum=float(sign*value[tick,channel]+offset)
        if nonnegative and maximum<=0:
            facets[name]=(0.,np.zeros(dimension),0.)
        else:
            facets[name]=(maximum,sign*gradient[tick,channel],float(sign*nominal[tick,channel]+offset))
    maximum('in_edge_outward_mm',0,-1.,nonnegative=True)
    maximum('out_edge_outward_mm',1,-1.,nonnegative=True)
    maximum('negative_signed_heading_deg',2,-1.,nonnegative=True)
    maximum('heading_overshoot_deg',2,1.,-abs(turn_deg),True)
    for channel,name in ((3,'full_xy_peak_curvature_mm_inv'),(4,'full_physical_peak_curvature_mm_inv')):
        ids=np.flatnonzero(valid[:,channel]);tick=int(ids[abs(value[ids,channel]).argmax()])
        sign=float(np.sign(value[tick,channel]))
        facets[name]=(float(abs(value[tick,channel])),sign*gradient[tick,channel],float(sign*nominal[tick,channel]))
    active=valid[:,5]&(value[:,5]<0.)
    facets['full_physical_reverse_deg']=(float(-value[active,5].sum()*.001),
        -gradient[active,5].sum(0)*.001,float(-nominal[active,5].sum()*.001))
    return facets
