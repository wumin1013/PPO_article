"""Offline diagnostic of heading dependence in the late geometry support."""
import numpy as np
import torch
from joint_teacher_state_policy import features,state_pair,reference_oriented_observation
from study_io import BASE,write_json


def main():
    source=BASE/'canonical_gate_band_postpeak_bounded_joint_recovery_20260920'
    md=torch.load(source/'initial_policy.pt',map_location='cpu',weights_only=False)
    one=dict(md)
    for key in ('centers','per_center_mean','per_center_std','radii'):
        one[key]=np.asarray(md[key])[[5]].copy()
    one['per_center_distance_weights']=md['per_center_distance_weights'][[2]].copy()
    with np.load(source/'radius_0.97_replay.npz') as z:
        obs=reference_oriented_observation(z['observations']);raw,_=state_pair(obs,z['innovations'])
        one['centers']=(raw[[17]]-one['per_center_mean'])/one['per_center_std']
    rows=[]
    for tag in ('radius_0.97','bounded_50','bounded_51'):
        with np.load(source/(tag+'_replay.npz')) as z:
            f=features(one,reference_oriented_observation(z['observations']),z['innovations'])
            rows.append(dict(tag=tag,activation_340ms=float(abs(f[17]).max())))
    data=BASE/'canonical_gate_band_reversal_window_local_recovery_20260920/active_window_training_states.npz'
    with np.load(data) as z:
        f=features(one,reference_oriented_observation(z['observations']),z['innovations'])
        paths=sorted(np.unique(z['path_ids'][np.any(f!=0.,axis=(1,2))]).tolist())
    write_json(source/'reference_oriented_support_audit.json',dict(records=rows,baseline_support_paths=paths,
        new_physical_runs=0,representation_only=True,interpretation='Invert the observed heading rotation '
        'for this geometry-distance diagnostic. The normalized observed orientation is used even in '
        'a blended gate state; no unobserved true frame or runtime path/time index is supplied. '
        'Activation stability alone cannot pass closed-loop policy acceptance.'))
    print('REFERENCE_ORIENTED_SUPPORT',rows,paths,flush=True)


if __name__=='__main__':main()
