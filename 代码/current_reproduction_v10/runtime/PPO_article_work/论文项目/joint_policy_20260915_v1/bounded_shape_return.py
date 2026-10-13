"""Bounded shape cost with spatial energy assigned back to decision bins.

C=1-exp(-Q) lies in [0,1]. With the unchanged terminal failure charge
Tmax+1+remaining/20, every successful episode (Ts<=Tmax) outranks every
failed episode on the same path, even after adding this cost. The original
failure penalty is never reduced. Multi-corner paths retain base time reward
in this isolated-corner mechanism pilot.
"""
import numpy as np
from geometric_turn_metrics import spatial_turn_profile
from isolated_corner_shape import corner_shape


def bounded_shape_cost(reference, rows, actions, *, discrete_first_label=True, two_edges=False):
    reference=np.asarray(reference);rows=np.asarray(rows);actions=np.asarray(actions)
    pieces=np.zeros(len(actions),float)
    if len(reference)!=3 or len(rows)<22 or np.linalg.norm(rows[:,2:4],axis=1).max()<=5:
        return pieces,dict(cost=0.,quality=0.,reason='no_isolated_resolvable_turn')
    try:profile=spatial_turn_profile(np.vstack([reference[0],rows[:,:2]]),.1)
    except ValueError:return pieces,dict(cost=0.,quality=0.,reason='insufficient_spatial_support')
    xy,arc,grid,turn=(profile[k] for k in ('xy','arc','grid','turn'))
    u,v=np.diff(reference,axis=0);angle=np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)
    apex_s=arc[np.argmin(np.linalg.norm(xy-reference[1],axis=1))]
    locations=grid[1:-1];window=(locations>=apex_s-12)&(locations<=apex_s+12)
    energy_components=1.25*turn[window]**2/(.1*angle**2)
    ticks=np.interp(locations[window],arc,profile['ticks'])
    decisions=np.clip(np.floor(np.maximum(0,ticks-1)/20).astype(int),0,len(actions)-1)
    np.add.at(pieces,decisions,energy_components)
    shape=corner_shape(reference,rows,actions)
    outward=shape['outward_excursion_mm']
    if two_edges:
        # A general half-plane diagnostic for a single turn, not a target curve.
        sign=np.sign(angle);u=u/np.linalg.norm(u);v=v/np.linalg.norm(v)
        relative=rows[:,:2]-reference[1]
        inside_in=relative@(sign*np.array([-u[1],u[0]]))
        inside_out=relative@(sign*np.array([-v[1],v[0]]))
        outward=max(0.,-inside_in.min(),-inside_out.min())
    incoming=max(outward-.05,0)/1.25
    if discrete_first_label:incoming+=.1*(shape['first_physical_turn']!='inside')
    reverse=max(shape['reverse_turning_total_deg']-5.,0)/abs(shape['turn_deg'])
    pieces[-1]+=5*incoming+2*reverse
    quality=float(pieces.sum());cost=float(-np.expm1(-quality))
    if quality>0:pieces*=cost/quality
    assert 0<=cost<=1 and abs(pieces.sum()-cost)<1e-12
    return pieces,dict(cost=cost,quality=quality,normalized_energy=float(energy_components.sum()),incoming=incoming,reverse=reverse,
        energy_credit='0.1mm spatial vertices mapped to the generating20ms decision bin; incoming/reversal components charged at terminal decision')


def terminal_shape_cost(reference, rows, actions, *, discrete_first_label=True, two_edges=False):
    """Causal accounting for the same non-additive complete-trajectory cost.

    Spatial allocation above uses a full-episode normalization and a retrospectively
    selected window. Its early entries can depend on future actions, so ordinary
    reward-to-go cannot discard them. Charge the unchanged scalar at termination.
    """
    old,details=bounded_shape_cost(reference,rows,actions,discrete_first_label=discrete_first_label,two_edges=two_edges)
    pieces=np.zeros_like(old)
    pieces[-1]=old.sum()
    details=dict(details,energy_credit='unchanged complete-trajectory cost charged at termination; every earlier action receives its full cost in MC return')
    return pieces,details
