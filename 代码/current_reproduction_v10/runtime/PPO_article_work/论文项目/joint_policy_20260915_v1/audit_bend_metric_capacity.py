"""Check whether a continuous observed-geometry metric separates conflicting support."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import argparse
import torch
from joint_teacher_state_policy import state_pair, features
from observable_memory_joint_policy import future_bend_context
from study_io import BASE, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target-centers',action='store_true')
    args=parser.parse_args()
    stage=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    with np.load(stage/'geometry_secant_17_replay.npz') as z:current={k:z[k].copy() for k in z.files}
    with np.load(stage/'replayed_states.npz') as z:parent={k:z[k].copy() for k in z.files}
    centers=np.asarray(md['per_center_geometry_mean']).copy()
    selected_kernels=[0,3,4];conflict_paths=[12,102,103,203,302,303]
    if args.target_centers:
        f=features(md,current['observations'],current['innovations'])
        g=future_bend_context(current['observations'])
        target=np.isin(current['path_ids'],[14,104,105,204,205,304,305])
        for k in selected_kernels:
            mask=target&np.any(f[:,k]!=0.,axis=1)
            centers[k]=g[mask].mean()
    def distances(data):
        result=[]
        for raw in state_pair(data['observations'],data['innovations']):
            x=(raw[:,None,:]-md['per_center_mean'])/md['per_center_std']
            result.append(np.mean((x-md['centers'])**2,axis=-1))
        return np.asarray(result)
    d=distances(current);bend=future_bend_context(current['observations'])
    mask=np.isin(current['path_ids'],conflict_paths)
    required=[]
    for k in selected_kernels:
        delta=(bend-centers[k])**2
        for mirror in range(2):
            active=mask&(d[mirror,:,k]<md['radii'][k]**2)
            if np.any(active):
                assert np.all(delta[active]>1e-12)
                required.extend(((md['radii'][k]**2-d[mirror,active,k])/delta[active]).tolist())
    base=max(required)*1.01
    choices=[]
    for factor in (1.,2.,4.):
        weights=np.zeros(len(centers));weights[selected_kernels]=base*factor
        datasets=[]
        for label,data in [('current',current),('parent',parent)]:
            squared=distances(data);g=future_bend_context(data['observations'])
            squared+=((g[:,None]-centers)**2*weights)[None]
            active=np.any(squared<np.asarray(md['radii'])[None,None,:]**2,axis=(0,2))
            paths=sorted(np.unique(data['path_ids'][active]).tolist())
            datasets.append(dict(label=label,paths=paths,decisions=int(active.sum()),
                excluded_conflict_paths=sorted(set(conflict_paths)-set(paths))))
        choices.append(dict(factor=factor,weights=weights.tolist(),support=datasets))
        print('BEND_METRIC_CAPACITY',factor,weights.tolist(),datasets,flush=True)
    write_json(stage/('centered_bend_metric_capacity.json' if args.target_centers else 'bend_metric_capacity.json'),dict(centers=centers.tolist(),choices=choices,
        selected_kernels=selected_kernels,training_conflict_paths=conflict_paths,
        centers_from_actual_target_support=args.target_centers,
        new_physical_runs=0,interpretation='A nonnegative continuous observed-bend term only '
        'shrinks existing radial support. Its weight is estimated from recorded conflicting '
        'states; no angle/speed table or runtime path/time selector is introduced. Exact '
        'decoded-action identity and actual closed-loop effects still require verification.'))


if __name__=='__main__':main()
