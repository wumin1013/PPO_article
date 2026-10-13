"""Check whether an observable-state residual can act before saved shape errors."""
import numpy as np
import torch
from audit_saved_lateral_motion import lateral_motion
from joint_teacher_state_policy import features
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_direct_observable_joint_learning_20260920'
    md=torch.load(stage/'research_candidate.pt',map_location='cpu',weights_only=False)
    with np.load(stage/'replayed_states.npz') as z:data={k:z[k].copy() for k in z.files}
    f=features(md,data['observations'],data['innovations'])
    active=np.any(f!=0.,axis=(1,2))
    records=read(stage/'baseline.json')['baseline']['records'];output=[]
    for path in (206,304):
        r=next(r for r in records if r['path']==path)
        lateral=lateral_motion(r['trace'])
        decisions=data['decisions'][(data['path_ids']==path)&active]
        earliest=int(decisions.min()*20)
        ended=[p for p in lateral['intervals'] if p['stop_ms']<=earliest]
        item=dict(path=path,any_coefficient_support_decisions_ms=(decisions*20).tolist(),
            earliest_possible_action_change_ms=earliest,
            prior_significant_lateral_intervals=ended,
            prior_wrong_way_mm=sum(p['reverse_lateral_mm'] for p in ended),
            prior_wrong_way_duration_ms=sum(p['stop_ms']-p['start_ms'] for p in ended))
        output.append(item);print('COMPACT_CAUSAL_SUPPORT',item,flush=True)
    write_json(stage/'temporal_support_audit.json',dict(records=output,physical_reruns=0,
        interpretation='Before the first nonzero kernel, every coefficient vector executes '
        'the exact parent actions by induction. Lateral motion already accumulated before '
        'that point cannot be removed by optimizing this basis. This is a specific '
        'representation/support limitation, not a claim about the PPO estimator.'))


if __name__=='__main__':main()
