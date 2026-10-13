"""Inspect the remaining real guard violations and unused observable kernels."""
import numpy as np
import torch
from joint_teacher_state_policy import features
from audit_preemptive_joint_credit import constraint_names
from intrinsic_curve_metrics import intrinsic_metrics
from study_io import BASE,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    stage=BASE/'canonical_gate_band_event_recentered_preemptive_credit_20260920'
    md=torch.load(BASE/'canonical_gate_band_preemptive_joint_credit_learning_20260920/initial_policy.pt',map_location='cpu',weights_only=False)
    data=read(stage/'state_joint/results.json');index={r['tag']:r for r in data['candidates']}
    near=index[read(stage/'recovery.json')['nearest']];zero=data['candidates'][0]
    with np.load(stage/'anchor_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    f=features(md,replay['observations'],replay['innovations']);supports=[]
    for r in near['records']:
        p=r['path'];mask=replay['path_ids']==p;entries=[]
        for k in range(7):
            touched=mask&np.any(f[:,k]!=0.,axis=1)
            entries.append(dict(kernel=k,active_ms=(20*replay['decisions'][touched]).tolist(),
                feature_max=float(abs(f[touched,k]).max()) if np.any(touched) else 0.))
        with np.load(r['trace']) as z:physical=intrinsic_metrics(z['rows'])
        supports.append(dict(path=p,kernels=entries,nearest_actual_peak_ms=physical['peak_ms']))
    names=constraint_names([r['path'] for r in near['records']]);g=np.asarray(near['augmented_measured_inequalities'])
    violations=[dict(guard=names[i],value=float(g[i])) for i in np.flatnonzero(g < -1e-8)]
    print('NEAREST',near['tag'],'violations',violations,flush=True)
    for row in supports:print('AVAILABLE_SUPPORT',row,flush=True)
    write_json(stage/'remaining_support_audit.json',dict(nearest=near['tag'],violations=violations,
        supports=supports,support_replay_source='preemptive_joint_84',new_physical_runs=0,
        base_lateral_mm=sum(r['shape']['reverse_lateral_significant_mm'] for r in zero['records']),
        nearest_lateral_mm=sum(r['shape']['reverse_lateral_significant_mm'] for r in near['records']),
        interpretation='Kernel availability is from the exact joint84 replay, not claimed '
        'identical at the later exploratory78 state. Physical peak locations use78 raw V/A. '
        'This selects a next measurable action-support hypothesis, not an acceptance decision.'))


if __name__=='__main__':main()
