"""Saved-state support and measured constraint conflicts; no new rollouts."""
import argparse
import copy
import numpy as np
import torch
from joint_teacher_state_policy import features
from audit_saved_lateral_motion import lateral_motion
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='canonical_gate_band_early_local_affine_joint_learning_20260920')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();stage=BASE/args.stage
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    with np.load(stage/'replayed_states.npz') as z:data={k:z[k].copy() for k in z.files}
    f=features(md,data['observations'],data['innovations'])
    baseline=read(stage/'baseline.json')['baseline']['records']
    support=[]
    for r in baseline:
        mask=data['path_ids']==r['path'];local=f[mask];active=np.any(local!=0.,axis=(1,2))
        if not np.any(active):continue
        decisions=data['decisions'][mask][active]
        row=dict(path=r['path'],active_decision_ms=(20*decisions).tolist(),
            max_abs_action_features=np.max(abs(local),axis=0).tolist())
        if 'shape' in r:
            intervals=lateral_motion(r['trace'])['intervals']
            row['significant_lateral_intervals']=intervals
            row['active_during_lateral_ms']=[int(t) for t in 20*decisions if any(i['start_ms']<=t<i['stop_ms'] for i in intervals)]
        support.append(row)
    result=read(stage/'state_joint/results.json');zero=result['candidates'][0]
    keys=['in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
          'full_normalized_energy','full_spatial_peak_curvature_mm_inv',
          'reverse_lateral_total_mm','reverse_lateral_significant_duration_s']
    comparisons=[]
    for c in sorted(result['candidates'][1:],key=lambda c:c['score'])[:5]:
        changed=[]
        for r,b in zip(c['records'],zero['records']):
            if 'shape' not in r or 'shape' not in b:continue
            delta={k:r['shape'][k]-b['shape'][k] for k in keys}
            if any(abs(v)>1e-10 for v in delta.values()):changed.append(dict(path=r['path'],delta=delta))
        comparisons.append(dict(tag=c['tag'],score=c['score'],final_guard_minimum=min(c['measured_inequalities']),
                                coefficients=c['coefficients'],changes=changed))
    actual_support=[]
    if args.replay:
        from consolidate_joint_teacher_policy import collect_data
        for c in sorted(result['candidates'][1:],key=lambda c:c['score'])[:2]:
            current=copy.deepcopy(md);current['coefficients']=np.asarray(c['coefficients']).reshape(-1,3)
            records=[r for r in c['records'] if r['path'] in (14,206,304)]
            replay=collect_data(current,md['parent'],records)
            ff=features(md,replay['observations'],replay['innovations'])
            for r in records:
                mask=replay['path_ids']==r['path'];active=np.any(ff[mask]!=0.,axis=(1,2))
                actual_support.append(dict(tag=c['tag'],path=r['path'],decoded_actions_bitwise_reproduced=True,
                    active_decision_ms=(20*replay['decisions'][mask][active]).tolist(),
                    significant_lateral_intervals=lateral_motion(r['trace'])['intervals']))
        print('ACTUAL_SUPPORT',actual_support,flush=True)
    write_json(stage/'early_support_response_audit.json',dict(model_sha256=sha(stage/'state_joint/candidate.pt'),
        baseline_score=zero['score'],support=support,best_measured_loss_vectors=comparisons,
        replayed_actual_candidate_support=actual_support,
        physical_reruns=0,interpretation='Support is evaluated on saved PARENT states. It establishes '
        'where the first perturbation can enter, not the support on a changed trajectory or '
        'a proof that finite support is the only failure cause. Constraints and reported '
        'responses come from saved complete physical trajectories.'))
    print('SUPPORT',[(r['path'],r['active_decision_ms'],r.get('active_during_lateral_ms')) for r in support])
    print('LOWEST_LOSS',[(r['tag'],r['score'],r['final_guard_minimum']) for r in comparisons])


if __name__=='__main__':main()
