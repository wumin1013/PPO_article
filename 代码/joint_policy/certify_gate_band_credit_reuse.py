"""Sufficient trajectory-wise equivalence certificate for state-support repair."""
import time
import numpy as np
import torch
from geometry_spec import Corridor
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def support_unchanged(corridor,positions):
    """Conservative: inspect every gate, including non-adjacent ones.

    If a decision point is in a gate's longitudinal band, require its plane
    projection to lie strictly inside both adjacent capsules. Then removing
    the lateral membership check cannot add any active gate to that decision.
    Coverage, guide and dynamics are identical; induction gives the same path.
    """
    for gate in corridor.gates:
        h=(positions-gate.point)@gate.normal
        points=positions[abs(h)<1.+1e-8]-h[abs(h)<1.+1e-8,None]*gate.normal
        for index in (gate.vertex_index-1,gate.vertex_index):
            tangent=corridor.tangents[index]
            along=np.clip((points-corridor.starts[index])@tangent,0.,corridor.lengths[index])
            distance=np.linalg.norm(points-corridor.starts[index]-along[:,None]*tangent,axis=1)
            if np.any(distance>corridor.epsilon-1e-8):return False
    return True


def main():
    old=BASE/'canonical_recentered_complete_credit_20260917'
    parent=BASE/'canonical_gate_band_matched_probe_20260917'
    raw=read(old/'results.json');selected=read(parent/'results.json')['candidates'][0]
    model=parent/selected['model'];old_model=BASE/'canonical_gate_waveform_learned_update_20260917/candidate_1.pt'
    before=torch.load(old_model,map_location='cpu',weights_only=False)['policy']
    after=torch.load(model,map_location='cpu',weights_only=False)['policy']
    assert before.keys()==after.keys() and all(torch.equal(before[k],after[k]) for k in before)
    stage=register('canonical_gate_band_credit_reuse_certificate_20260917',
        [__file__,old/'results.json',old/'receipt.json',old/'queries.npz',parent/'results.json',
         parent/'receipt.json',model,old_model,BASE/'geometry_spec.py',BASE/'gate_band_state.py',
         BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py'],
        dict(saved_suffixes=len(raw['records']),new_actor_decisions=0,new_physics_ticks=0,wall_s=120),
        'Certify reusable complete one-decision consequences after ONLY the state '
        'support repair. Require identical policy tensors, identical nominal '
        'prefix states/actions up to the branch anchor, and no decision position '
        'in any gate band with a lateral projection outside either adjacent '
        'capsule. Checking all gates is conservative. Under these conditions '
        'the observation is identical at every recorded policy call; the guide, '
        'execution, coverage and stopping remain identical, so induction certifies '
        'the saved branch for the new runtime. Otherwise reject reuse and collect '
        'fresh physics. Bind every accepted suffix by hash. No approximate '
        'trajectory equality or old-gradient substitution is allowed.')
    start=time.perf_counter();cache={};reports=[]
    for b in raw['baselines']:
        path=b['path'];j=next(i for i,r in enumerate(selected['results']) if r['original_path']==path)
        with np.load(BASE/b['trace']) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        with np.load(BASE/selected['traces'][j]) as z:newrows,newactions=z['rows'].copy(),z['actions'].copy()
        n=min(len(rows),len(newrows));diff=np.flatnonzero(np.any(rows[:n]!=newrows[:n],axis=1))
        first_tick=int(diff[0]) if len(diff) else n
        n=min(len(actions),len(newactions));diff=np.flatnonzero(np.any(actions[:n]!=newactions[:n],axis=1))
        first_action=int(diff[0]) if len(diff) else n
        cache[path]=(Corridor(ref),rows,ref,first_tick,first_action)
        reports.append(dict(path=path,first_different_tick=first_tick,first_different_action=first_action,
            source_trace=b['trace'],current_trace=selected['traces'][j]))
    records=[]
    for r in raw['records']:
        assert time.perf_counter()-start<120
        corridor,rows,ref,first_tick,first_action=cache[r['path']];anchor=r['anchor']
        reused=False;reason='nominal_prefix_changed'
        if anchor*20<=first_tick and anchor<=first_action:
            with np.load(BASE/r['trace']) as z:
                full=np.vstack([rows[:anchor*20],z['suffix_rows']]);count=anchor+len(z['suffix_actions'])
            positions=np.vstack([ref[0],full[:-1,:2]])[np.arange(count)*20]
            reused=support_unchanged(corridor,positions)
            reason='exact_runtime_equivalence' if reused else 'possible_state_support_change'
        records.append(dict(path=r['path'],anchor=anchor,axis=r['axis'],direction=r['direction'],
            reusable=reused,reason=reason,trace=r['trace'],trace_sha256=sha(BASE/r['trace']) if reused else None))
    result=dict(source=old.name,parent=parent.name,arm='parent',parent_sha256=sha(model),
        same_policy_tensors=True,records=records,paths=reports,
        reusable_suffixes=sum(r['reusable'] for r in records),fresh_suffixes=sum(not r['reusable'] for r in records),
        goal_complete=False)
    write_json(stage/'results.json',result)
    receipt(stage,dict(status='certified_exact_gate_band_suffix_reuse',new_actor_decisions=0,
        new_physics_ticks=0,reused=result['reusable_suffixes'],fresh=result['fresh_suffixes'],
        wall_s=time.perf_counter()-start))
    print('GATE_BAND_REUSE',result['reusable_suffixes'],result['fresh_suffixes'],flush=True)


if __name__=='__main__':main()
