"""Optimistic cruise-command bounds inside the existing learned-update domain."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    source=BASE/'canonical_gate_band_current_hold_geometry_20260918'
    metadata=read(source/'results.json')
    stage=register('canonical_gate_band_current_cruise_search_capacity_20260918',
        [__file__,source/'geometry.npz',source/'results.json',source/'receipt.json',BASE/'cruise_update_geometry.py'],
        dict(new_physics_ticks=0,new_actor_decisions=0,states=1175,wall_s=60,metric_radius=.1),
        'Before proposing more cruise updates, bound each target-state feed '
        'command under the exact existing halfspace and metric-radius limits. '
        'Compute the ellipsoid support function and the single-state action '
        'ceiling; their minimum is only an optimistic per-state upper bound '
        'because other states, all shape constraints and state validity cuts '
        'are omitted. If this upper bound is below99 or99.5mm/s, that request '
        'is impossible within the current one-step optimization domain. This '
        'is a search-domain diagnosis, not a physical speed limit, proof of '
        'long-run learning impossibility or complete-route cruise certificate.')
    start=time.perf_counter()
    with np.load(source/'geometry.npz') as z:d={k:z[k].copy() for k in z.files}
    ids=d['cruise_state_indices'];latent=d['cruise_latent'];features=d['cruise_features']
    rows=np.zeros((len(ids),768));rows[:,256:512]=features
    inverse=cho_solve(cho_factor(d['metric_physical'],lower=True),rows.T)
    dual=np.sqrt(np.maximum(np.einsum('ij,ji->i',rows,inverse),0.))
    lower=d['lower_physical'].reshape(-1,3)[ids,1];upper=d['upper_physical'].reshape(-1,3)[ids,1]
    increment=np.minimum(.1*dual,upper)
    center=d['action_centers'][1];extent=d['action_ranges'][1]
    nominal=center+extent*np.tanh(latent)
    action_bound=center+extent*np.tanh(latent+upper)
    optimistic=center+extent*np.tanh(latent+increment)
    summary=dict(parent_sha256=metadata['parent_sha256'],states=len(ids),
        nominal_feed_mm_s={k:float(v) for k,v in zip(('min','median','max'),np.quantile(nominal,[0,.5,1]))},
        optimistic_feed_mm_s={k:float(v) for k,v in zip(('min','median','max'),np.quantile(optimistic,[0,.5,1]))},
        excluded_by_action_bound={str(t):int((action_bound<t).sum()) for t in (99.,99.5)},
        excluded_by_optimistic_joint_bound={str(t):int((optimistic<t).sum()) for t in (99.,99.5)},
        optimism_ignores_other_state_and_shape_constraints=True,physical_impossibility_claim=False)
    per_path=[]
    for path in np.unique(d['state_path'][ids]):
        mask=d['state_path'][ids]==path
        per_path.append(dict(path=int(path),states=int(mask.sum()),
            cannot_request99=int((optimistic[mask]<99.).sum()),
            nominal_mean_mm_s=float(nominal[mask].mean()),optimistic_max_mm_s=float(optimistic[mask].max())))
    write_json(stage/'results.json',dict(summary=summary,per_path=per_path,goal_complete=False))
    np.savez_compressed(stage/'bounds.npz',state_index=ids,state_path=d['state_path'][ids],
        nominal_feed=nominal,action_bound=action_bound,optimistic_bound=optimistic,
        latent_increment=increment,dual_norm=dual)
    assert time.perf_counter()-start<60
    receipt(stage,dict(status='completed_cruise_search_domain_bound',new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start,goal_complete=False))
    print('CRUISE_SEARCH_CAPACITY',summary,flush=True)


if __name__=='__main__':main()
