"""Audit real state support and exact actions before changing the joint representation."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from consolidate_joint_teacher_policy import collect_data
from observable_memory_joint_policy import dynamic_context, future_bend_context
from joint_teacher_state_policy import features, OUTPUT_SCALE
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    stage=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    history=read(stage/'state_joint/results.json')['candidates']
    c=next(c for c in history if c['tag']=='geometry_secant_17')
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md['coefficients']=np.asarray(c['coefficients']).reshape(-1,3)
    replay=collect_data(md,md['parent'],c['records'],include_latents=True)
    obs=replay['observations'];f=features(md,obs,replay['innovations'])
    bend=future_bend_context(obs);rows=[]
    for path in np.unique(replay['path_ids']):
        mask=replay['path_ids']==path
        centers=[]
        for k in range(f.shape[1]):
            active=mask&np.any(f[:,k]!=0.,axis=1)
            centers.append(dict(kernel=k,decisions=replay['decisions'][active].tolist(),
                bend=bend[active].tolist(),feature_max=float(abs(f[active,k]).max()) if np.any(active) else 0.))
        delta=replay['current_latents'][mask]-replay['parent_latents'][mask]
        row=dict(path=int(path),active_centers=centers,action_delta_max=np.max(abs(delta),axis=0).tolist())
        rows.append(row)
        print('GEOMETRY_SUPPORT',int(path),[(r['kernel'],r['decisions'],r['bend']) for r in centers if r['decisions']],flush=True)
    np.savez_compressed(stage/'geometry_secant_17_replay.npz',**replay)
    write_json(stage/'geometry_support_audit.json',dict(records=rows,exact_decoded_action_replay=True,
        new_physical_runs=0,source_sha256=sha(stage/'state_joint/results.json')))


if __name__=='__main__':main()
