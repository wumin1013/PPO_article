"""Localize actual combined-target changes with principal-angle differences, from saved data."""
import numpy as np
from joint_runtime import FILTER_TAU
from localize_joint_shape_residual import characterize
from study_io import BASE,write_json,sha
from search_complete_return import read


def target_changes(trace):
    with np.load(trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
    origin=np.arctan2(*(ref[1]-ref[0])[::-1])
    filtered=np.arctan2(rows[:,9],rows[:,8]);previous=np.r_[origin,filtered[:-1]]
    target=previous+rows[:,10]*.001/(1.-np.exp(-.001/FILTER_TAU))
    delta=np.degrees(np.arctan2(np.sin(np.diff(target)),np.cos(np.diff(target))))
    eligible=(rows[:-1,12]==0)&(rows[1:,12]==0)
    ids=np.flatnonzero(eligible);rank=ids[np.argsort(abs(delta[ids]))[-6:][::-1]]
    values=[]
    for i in rank:
        before=actions[min(i//20,len(actions)-1)];after=actions[min((i+1)//20,len(actions)-1)]
        values.append(dict(tick=int(i+1),decision_boundary=bool((i+1)%20==0),
            target_change_deg=float(delta[i]),steering_change_deg=float(np.degrees(after[0]-before[0])),
            feed_change_mm_s=float(after[1]-before[1]),preview_change_mm=float(after[2]-before[2])))
    return dict(trace_sha256=sha(trace),largest_changes=values,
        max_decision_jump_deg=float(np.max(abs(delta[ids[(ids+1)%20==0]]),initial=0.)),
        max_between_decision_jump_deg=float(np.max(abs(delta[ids[(ids+1)%20!=0]]),initial=0.)))


def main():
    stage=BASE/'canonical_gate_band_context_coherent_parameter_training_20260920'
    results=read(stage/'training_arm_results.json')['arms']
    baseline=read(stage/'baseline.json')['baseline']['records']
    paths=read(stage/'support_audit.json')['selected']['paths'];rows=[]
    for name,records in [('protected_parent',baseline)]+[(a['arm'],a['records']) for a in results]:
        for r in records:
            if r['path'] not in paths:continue
            c=target_changes(r['trace']);shape=characterize(r['trace'])
            rows.append(dict(arm=name,path=r['path'],**c,negative_turn_intervals=shape['negative_turn_intervals'],
                apex_ms=shape['apex_ms'],outgoing_min_ms=shape['outgoing_min_ms']))
            print('TARGET_CHANGE',name,r['path'],round(c['max_decision_jump_deg'],4),
                round(c['max_between_decision_jump_deg'],4),flush=True)
    write_json(stage/'current_target_change_audit.json',dict(records=rows,new_physical_runs=0,
        angular_difference='atan2(sin(delta),cos(delta)); no 360-degree wrap artifact',
        scope='Observed complete trajectories; event coincidence is not a causal intervention or universal cause proof.'))


if __name__=='__main__':main()
