"""Choose a shared observable state before the obstructing raw curvature peak."""
import numpy as np
import torch
from joint_teacher_state_policy import features, state_pair
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920'
    capacity=read(stage/'conditioned_route_capacity.json')['models']
    blocked=[r['path'] for r in capacity if r['radius']==4. and not r['success']]
    assert blocked==[104]
    history=read(stage/'state_joint/results.json')['candidates']
    anchor=next(c for c in history if c['tag']=='approach_joint_18')
    record=next(r for r in anchor['records'] if r['path']==104)
    with np.load(record['trace']) as z:rows=z['rows'].copy()
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    curvature=np.abs(rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4])/np.maximum(speed,5.)**3
    curvature[(speed<=5.)|(rows[:,12]!=0)]=0.
    peak=int(curvature.argmax());decision=max(0,(peak-20)//20)
    with np.load(stage/'replayed_states.npz') as z:original={k:z[k].copy() for k in z.files}
    with np.load(stage/'conditioned_states.npz') as z:update={k:z[k].copy() for k in z.files}
    keep=~np.isin(original['path_ids'],update['path_ids'])
    training={k:np.concatenate([v[keep],update[k]]) for k,v in original.items()}
    a,b=state_pair(training['observations'],training['innovations'])
    both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05)
    a,b=(a-mean)/std,(b-mean)/std
    index=np.flatnonzero((training['path_ids']==104)&(training['decisions']==decision)).item()
    center=a[index]
    distance=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
    target_paths=[14,104,105,204,205,304,305]
    radius=1.2*max(float(distance[training['path_ids']==p].min()) for p in target_paths)
    spec=dict(mean=mean,std=std,centers=center[None],radii=np.array([radius]))
    f=features(spec,training['observations'],training['innovations']);active=np.any(f!=0.,axis=(1,2))
    paths=np.unique(training['path_ids'][active]).tolist()
    support=[dict(path=p,decisions_ms=(20*training['decisions'][active&(training['path_ids']==p)]).tolist(),
                  maximum_feature=float(abs(f[training['path_ids']==p]).max())) for p in paths]
    write_json(stage/'main_turn_support_audit.json',dict(training_only_anchor_path=104,
        raw_physical_curvature_peak_ms=peak,raw_physical_curvature_peak_mm_inv=float(curvature[peak]),
        training_only_center_ms=20*decision,mean=mean.tolist(),std=std.tolist(),center=center.tolist(),radius=radius,
        paths=paths,support=support,trace_sha256=sha(record['trace']),new_physical_runs=0,
        selection='State at least20ms before obstructing measured physical curvature peak; runtime receives observation only.'))
    np.savez_compressed(stage/'conditioned_full_states.npz',**training)
    print('MAIN_TURN_SHARED_SUPPORT',peak,20*decision,radius,support,flush=True)
    target_indices=[np.flatnonzero((training['path_ids']==p)&(training['decisions']==decision)).item() for p in target_paths]
    second=max(target_indices,key=lambda i:distance[i])
    centers=np.stack([center,a[second]])
    distances=np.stack([np.minimum(np.sqrt(np.mean((a-c)**2,axis=1)),np.sqrt(np.mean((b-c)**2,axis=1))) for c in centers],axis=1)
    assignment=distances[target_indices].argmin(1)
    # A singleton cluster needs a positive neighborhood, not a zero radius.
    radii=np.asarray([max(1.2*max(distances[i,k] for i,owner in zip(target_indices,assignment) if owner==k),
                          .5*float(distances[distances[:,k]>1e-12,k].min())) for k in range(2)])
    assert np.all(np.isfinite(radii)&(radii>0.))
    spec.update(centers=centers,radii=radii)
    clustered=features(spec,training['observations'],training['innovations'])
    assert np.isfinite(clustered).all()
    active=np.any(clustered!=0.,axis=(1,2));cluster_paths=np.unique(training['path_ids'][active]).tolist()
    clusters=[dict(center_training_path=int(training['path_ids'][i]),radius=float(radii[k]),
        assigned_training_paths=[p for p,owner in zip(target_paths,assignment) if owner==k],
        affected_paths=np.unique(training['path_ids'][np.any(clustered[:,k]!=0.,axis=1)]).tolist()) for k,i in enumerate((index,second))]
    output=dict(mean=mean.tolist(),std=std.tolist(),centers=centers.tolist(),radii=radii.tolist(),paths=cluster_paths,
        clusters=clusters,active_decisions=int(active.sum()),training_only_center_ms=20*decision,
        state_support=[dict(path=p,decisions_ms=(20*training['decisions'][active&(training['path_ids']==p)]).tolist()) for p in cluster_paths],
        new_physical_runs=0,selection='Two farthest-point observation centers separate the two measured approach states; no runtime path/time/angle switch.')
    write_json(stage/'clustered_main_turn_support_audit.json',output)
    print('CLUSTERED_MAIN_TURN_SUPPORT',clusters,'decisions',int(active.sum()),flush=True)


if __name__=='__main__':main()
