"""Select an observable approach support before the remaining lateral reversal."""
import numpy as np
import torch
from joint_teacher_state_policy import state_pair, features
from study_io import BASE, write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    with np.load(stage/'replayed_states.npz') as z:
        training={k:z[k].copy() for k in z.files}
    baseline=read(stage/'baseline.json')['baseline']['records']
    a,b=state_pair(training['observations'],training['innovations'])
    both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05)
    a,b=(a-mean)/std,(b-mean)/std
    onset=read(stage/'shape_control_timing.json')['records']
    onset=next(r for r in onset if r['path']==304)['lateral_intervals_ms'][0][0]
    decision=(onset-1)//20
    index=np.flatnonzero((training['path_ids']==304)&(training['decisions']==decision)).item()
    center=a[index]
    distance=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
    radius=1.2*max(float(distance[training['path_ids']==p].min()) for p in (14,104,204,304))
    spec=dict(mean=mean,std=std,centers=center[None],radii=np.array([radius]))
    f=features(spec,training['observations'],training['innovations'])
    active=np.any(f!=0.,axis=(1,2));paths=np.unique(training['path_ids'][active]).tolist()
    support=[dict(path=p,decisions_ms=(20*training['decisions'][active&(training['path_ids']==p)]).tolist(),
                  maximum_feature=float(abs(f[training['path_ids']==p]).max())) for p in paths]
    report=dict(training_only_center_path=304,training_only_center_ms=20*decision,
        radius=radius,paths=paths,active_decisions=int(active.sum()),
        all_corners=all('shape' in r for r in baseline if r['path'] in paths),support=support,
        mean=mean.tolist(),std=std.tolist(),center=center.tolist(),new_physical_runs=0,
        selection='Observation preceding the measured wrong-way onset; radius covers four existing training geometries. '
                  'No time, angle or path identifier is used in runtime features.')
    write_json(stage/'approach_joint_support.json',report)
    print('APPROACH_SUPPORT',{k:v for k,v in report.items() if k not in ('mean','std','center','selection')},flush=True)


if __name__=='__main__':main()
