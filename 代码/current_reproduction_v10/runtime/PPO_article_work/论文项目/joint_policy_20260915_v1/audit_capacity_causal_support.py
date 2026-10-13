"""Check which wrong-way motion is causally upstream of new policy support."""
import numpy as np
import torch
from joint_teacher_state_policy import features
from audit_saved_lateral_motion import lateral_motion
from study_io import BASE,write_json,sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_action_capacity_joint_training_20260920'
    md=torch.load(stage/'initial_policy.pt',map_location='cpu',weights_only=False)
    with np.load(stage/'affected_replay.npz') as z:replay={k:z[k].copy() for k in z.files}
    f=features(md,replay['observations'],replay['innovations'])
    records=read(stage/'baseline.json')['baseline']['records'];out=[]
    for p in np.unique(replay['path_ids']):
        ids=np.flatnonzero(replay['path_ids']==p);active=np.flatnonzero(np.any(f[ids]!=0.,axis=(1,2)))
        first=int(active[0]*20);r=next(r for r in records if r['path']==p)
        lateral=lateral_motion(r['trace'])
        with np.load(r['trace']) as z:
            rows=z['rows'];ref=z['reference'];u,v=np.diff(ref,axis=0);u=u/np.linalg.norm(u)
            sign=np.sign(u[0]*v[1]-u[1]*v[0]);normal=sign*np.array([-u[1],u[0]])
            displacement=np.diff(np.vstack([ref[0],rows[:,:2]]),axis=0)@normal
            before=(np.arange(len(rows))<lateral['apex_ms'])&(np.linalg.norm(rows[:,2:4],axis=1)>5)&(rows[:,12]==0)
            reverse=before&(displacement<0)
            untouched=float(-displacement[reverse&(np.arange(len(rows))<first)].sum())
            all_negative=float(-displacement[reverse].sum())
            last_negative=int(np.flatnonzero(reverse)[-1]) if np.any(reverse) else None
        row=dict(path=int(p),first_possible_effect_ms=first,last_incoming_negative_tick=last_negative,
            incoming_reverse_mm=all_negative,unmodifiable_prefix_reverse_mm=untouched,
            every_negative_tick_upstream=bool(last_negative is None or last_negative<first),
            significant_intervals=lateral['intervals'])
        out.append(row);print('CAPACITY_CAUSAL_SUPPORT',row,flush=True)
    write_json(stage/'causal_support_audit.json',dict(records=out,new_physical_runs=0,
        initial_policy_sha256=sha(stage/'initial_policy.pt'),
        conclusion='Before the first nonzero observed kernel every coefficient vector gives the protected parent action: '
        'causal memories start at zero. Deterministic induction fixes that physical prefix. This proves only the '
        'listed prefix is outside this new layer action support; not a universal policy-class or PPO limitation.'))


if __name__=='__main__':main()
