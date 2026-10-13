"""Select earlier observable support; time/path indices are training selectors only."""
import numpy as np
import argparse
import torch
from joint_teacher_state_policy import state_pair,features
from observable_memory_joint_policy import dynamic_context,future_bend_context
from study_io import BASE,write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clustered',action='store_true')
    args=parser.parse_args()
    source=BASE/'canonical_gate_band_signed_heading_joint_recovery_20260920'
    target_paths=[14,104,105,204,205,304,305]
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    baseline=read(source/'baseline.json')['baseline']['records']
    a,b=state_pair(training['observations'],training['innovations'])
    both=np.vstack([a,b]);mean=both.mean(0);std=np.maximum(both.std(0),.05)
    a,b=(a-mean)/std,(b-mean)/std
    bend=future_bend_context(training['observations']);reports=[]
    choices_of_center=([(5,304,[p for p in target_paths if p!=104]),(5,104,[104])] if args.clustered else
                       [(5,304,target_paths),(9,304,target_paths)])
    previous_radius=None
    for decision,center_path,group in choices_of_center:
        seed=np.flatnonzero((training['path_ids']==center_path)&(training['decisions']==decision)).item()
        desired=np.flatnonzero(np.isin(training['path_ids'],group)&(training['decisions']==decision))
        center=a[seed];distance=np.minimum(np.sqrt(np.mean((a-center)**2,axis=1)),np.sqrt(np.mean((b-center)**2,axis=1)))
        radius=(2. if args.clustered else 1.2)*max(distance[desired])
        if radius==0.:
            assert previous_radius is not None
            radius=min(previous_radius,.5*float(distance[distance>1e-10].min()))
        assert radius>0.
        previous_radius=radius;geom_center=float(bend[desired].mean())
        choices=[]
        for weight in (0.,36.516875105279546,150.):
            spec=dict(mean=mean,std=std,centers=center[None],radii=np.array([radius]),
                observed_bend_metric_centers=np.array([geom_center]),observed_bend_metric_weights=np.array([weight]))
            f=features(spec,training['observations'],training['innovations'])
            active=np.any(f!=0.,axis=(1,2));paths=sorted(np.unique(training['path_ids'][active]).tolist())
            support=[dict(path=p,decisions_ms=(20*training['decisions'][active&(training['path_ids']==p)]).tolist(),
                          maximum_feature=float(abs(f[training['path_ids']==p]).max())) for p in paths]
            choices.append(dict(weight=weight,paths=paths,all_corners=all('shape' in r for r in baseline if r['path'] in paths),support=support))
        raw=dynamic_context(training['observations'][desired]);geom=bend[desired]
        row=dict(training_only_center_path=center_path,training_only_center_ms=20*decision,target_group=group,
            mean=mean.tolist(),std=std.tolist(),center=center.tolist(),radius=float(radius),
            bend_center=geom_center,context_mean=raw.mean(0).tolist(),
            context_scale=np.maximum(raw.std(0),[5.,100.]).tolist(),
            geometry_mean=float(geom.mean()),geometry_scale=float(max(geom.std(),.02)),choices=choices)
        reports.append(row)
        print('PREEMPTIVE_SUPPORT',20*decision,'radius',radius,
              [(q['weight'],q['paths'],q['all_corners'],q['support']) for q in choices],flush=True)
    write_json(source/('preemptive_cluster_support_audit.json' if args.clustered else 'preemptive_support_audit.json'),dict(records=reports,parent_paths=len(baseline),
        new_physical_runs=0,runtime_inputs_are_observable_states_only=True))


if __name__=='__main__':main()
