"""Locate remaining shape events relative to actual observable-kernel support."""
import numpy as np
import torch
from joint_teacher_state_policy import features
from localize_joint_shape_residual import characterize, intervals
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    data=read(stage/'baseline.json')
    with np.load(stage/'replayed_states.npz') as z:
        training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations'])
    output=[]
    for p in (14,104,105,204,205,304,305):
        record=next(r for r in data['baseline']['records'] if r['path']==p)
        with np.load(record['trace']) as z:
            ref,rows=z['reference'].copy(),z['rows'].copy()
        u,v=np.diff(ref,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        sign=np.sign(u[0]*v[1]-u[1]*v[0]);ni=sign*np.array([-u[1],u[0]])
        speed=np.linalg.norm(rows[:,2:4],axis=1)
        heading=np.unwrap(np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u))
        apex=np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin()
        mask=(np.arange(len(rows))<apex)&(speed>5.)&(heading<-np.deg2rad(.05))&(rows[:,12]==0)
        lateral=intervals(mask)
        active=[]
        for kernel in (0,1):
            ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,kernel]!=0.,axis=1))
            active.append((training['decisions'][ids]*20).tolist())
        info=characterize(record['trace'])
        row=dict(path=p,lateral_intervals_ms=[list(map(int,x)) for x in lateral],
                 state_kernel_active_ms=active,apex_ms=int(apex),
                 outgoing_maximum_ms=info['outgoing_min_ms'],
                 largest_target_jump_ms=info['target_jumps'][0]['time_ms'],
                 trace_sha256=sha(record['trace']))
        output.append(row)
        print('SHAPE_SUPPORT_TIMING',{k:v for k,v in row.items() if k!='trace_sha256'},flush=True)
    write_json(stage/'shape_control_timing.json',dict(records=output,new_physical_rollouts=0,
        interpretation='Times are diagnostic only. A residual with zero support before an event '
                       'cannot change that event in this deterministic causal execution chain. '
                       'This does not show that all other policy parameters lack earlier support.'))


if __name__=='__main__':main()
