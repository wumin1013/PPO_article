"""Threshold-disclosed first-turn and outgoing-edge crossing diagnostics."""
import argparse
import time
import numpy as np
from prepare_current_waveform_credit import wave_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def onset_metrics(reference,rows):
    value,valid=wave_signals(reference,rows)
    u,v=np.diff(reference,axis=0);u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0]);no=sign*np.array([-v[1],v[0]])
    onset=[]
    for threshold in (.05,.1,.5):
        ids=np.flatnonzero(valid[:,2]&(abs(value[:,2])>=threshold))
        k=int(ids[0]) if len(ids) else None
        onset.append(dict(threshold_deg=threshold,tick=k,
            signed_heading_deg=float(value[k,2]) if k is not None else None,
            signed_entry_offset_mm=float(value[k,0]) if k is not None else None))
    lateral=[]
    for threshold in (.001,.005,.02):
        ids=np.flatnonzero(abs(value[:,0])>=threshold);k=int(ids[0]) if len(ids) else None
        lateral.append(dict(threshold_mm=threshold,tick=k,
            signed_entry_offset_mm=float(value[k,0]) if k is not None else None))
    reverse=valid[:,5]&(value[:,5]<0.)
    edges=np.diff(np.r_[False,reverse,False].astype(int))
    episodes=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        episodes.append(dict(start_tick=int(a),end_tick=int(b),duration_s=float((b-a)*.001),
            reverse_deg=float(-value[a:b,5].sum()*.001),
            heading_start_deg=float(value[a,2]),heading_end_deg=float(value[b-1,2])))
    crossings=np.flatnonzero((value[:-1,1]>=0.)&(value[1:,1]<0.))+1
    return dict(first_heading=onset,first_lateral=lateral,
        negative_heading_duration_gt01_s=float((valid[:,2]&(value[:,2]<-.1)).sum()*.001),
        outside_entry_gt005_s=float((value[:,0]<-.005).sum()*.001),
        maximum_outside_entry_mm=float(max(0.,-value[:,0].min())),
        maximum_outside_exit_mm=float(max(0.,-value[:,1].min())),
        reverse_episodes=sorted(episodes,key=lambda x:-x['reverse_deg'])[:4],
        outgoing_crossings=[dict(tick=int(k),normal_velocity_mm_s=float(rows[k,2:4]@no),
            normal_acceleration_mm_s2=float(rows[k,4:6]@no),signed_heading_deg=float(value[k,2]))
            for k in crossings])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True)
    parser.add_argument('--stage',required=True);args=parser.parse_args()
    parent=BASE/args.parent;data=read(parent/'training_selection.json')
    stage=register(args.stage,[__file__,parent/'training_selection.json',parent/'receipt.json',
        parent/data['model'],BASE/'prepare_current_waveform_credit.py'],
        dict(saved_paths=16,new_actor_decisions=0,new_physics_ticks=0,wall_s=30),
        'Audit the first actual XY departure and velocity-heading change on all16 '
        'isolated training turns of the retained research model. Disclose three '
        'thresholds rather than hiding direction under one onset threshold. '
        'Report negative-heading and outside duration, the largest physical '
        'reverse-turn episodes, and outgoing-edge crossing velocity/acceleration. '
        'These are saved-data diagnostics, not new acceptance thresholds or '
        'evidence that tangency has been achieved.')
    start=time.perf_counter();records=[]
    for path in range(16):
        j=next(i for i,r in enumerate(data['results']) if r['original_path']==path)
        with np.load(BASE/data['traces'][j]) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
        record=dict(path=path,**onset_metrics(ref,rows));records.append(record)
        print('TURN_ONSET',path,'heading',record['first_heading'][1],
              'entry',record['maximum_outside_entry_mm'],'exit',record['maximum_outside_exit_mm'],flush=True)
    write_json(stage/'results.json',dict(records=records,goal_complete=False))
    receipt(stage,dict(status='completed_saved_first_turn_and_exit_crossing_audit',paths=16,
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
