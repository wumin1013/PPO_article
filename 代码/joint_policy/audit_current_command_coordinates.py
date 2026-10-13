"""Decompose saved current trajectories before changing action coordinates."""
import math
import time
import argparse
import numpy as np
from joint_runtime import FILTER_TAU, wrap
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def command_signals(ref, rows, actions):
    incoming,outgoing=np.diff(ref,axis=0)
    incoming/=np.linalg.norm(incoming);outgoing/=np.linalg.norm(outgoing)
    initial=math.atan2(incoming[1],incoming[0])
    sign=np.sign(incoming[0]*outgoing[1]-incoming[1]*outgoing[0])
    heading=np.arctan2(rows[:,9],rows[:,8])
    previous=np.r_[initial,heading[:-1]]
    combined=np.array([wrap(v) for v in previous+rows[:,10]*.001/(1-math.exp(-.001/FILTER_TAU))])
    held=actions[np.minimum(np.arange(len(rows))//20,len(actions)-1)]
    los=np.array([wrap(v) for v in combined-held[:,0]])
    signed=lambda a:sign*np.array([wrap(v-initial) for v in a])*180/np.pi
    return dict(combined_deg=signed(combined),los_deg=signed(los),
        residual_deg=sign*held[:,0]*180/np.pi,desired_heading_deg=signed(heading),
        physical_heading_deg=signed(np.arctan2(rows[:,3],rows[:,2])),preview_mm=held[:,2])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_band_monotone_block_learning_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_current_command_coordinates_20260917')
    args=parser.parse_args()
    parent=BASE/args.parent
    data=read(parent/'results.json');selected=data['full_candidate']
    model=parent/data['chosen']['model'] if data.get('chosen') else BASE/data['model']
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',BASE/'joint_runtime.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,training_corners=16),
        'Invert the recorded heading filter on every latest learned isolated corner. '
        'Separate LOS, held steering residual, combined target and physical heading. '
        'Measure opposing contributions and within-decision target changes; do not '
        'infer historical learning causality from correlation. Exclude terminal tail '
        'where the ordinary guide filter is not active. Reuse real XY without '
        'smoothing; preserve prior shape metrics. This audit selects a justified '
        'action-coordinate mechanism test, not a hand-crafted turn rule.')
    start=time.perf_counter();records=[]
    for p in range(16):
        j=next(j for j,r in enumerate(selected['results']) if r['original_path']==p)
        with np.load(BASE/selected['traces'][j]) as z:
            ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        s=command_signals(ref,rows,actions)
        apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
        pre=(np.arange(len(rows))<apex)&(rows[:,12]==0)
        negative=pre&(s['combined_deg']<-.05)
        macro_start=np.arange(len(rows))//20*20
        born_inside=negative&(s['combined_deg'][macro_start]>=-.05)
        moving=np.linalg.norm(rows[:,2:4],axis=1)>5.
        onset=[]
        for threshold in (.05,.5):
            initial_ids=np.flatnonzero(pre&moving&(abs(s['physical_heading_deg'])>threshold))
            onset.append(dict(threshold_deg=threshold,
                first_direction=('inside' if s['physical_heading_deg'][initial_ids[0]]>0 else 'outside') if len(initial_ids) else None,
                first_time_s=float((initial_ids[0]+1)*.001) if len(initial_ids) else None,
                reverse_heading_duration_s=float((pre&moving&(s['physical_heading_deg']<-threshold)).sum()*.001)))
        ids=np.flatnonzero(negative)
        first=int(ids[0]) if len(ids) else None
        macro=[]
        for k in range(len(actions)):
            a,b=k*20,min((k+1)*20,len(rows))
            if b-a!=20 or rows[a:b,12].any():continue
            change=s['combined_deg'][b-1]-s['combined_deg'][a]
            macro.append(dict(decision=k,time_s=a*.001,
                within_decision_target_change_deg=float(change),
                start_target_deg=float(s['combined_deg'][a]),
                end_target_deg=float(s['combined_deg'][b-1]),preview_mm=float(actions[k,2])))
        macro.sort(key=lambda r:abs(r['within_decision_target_change_deg']),reverse=True)
        shape=next(r for r in selected['summary']['shapes'] if r['path']==p)
        record=dict(path=p,turn_deg=shape['turn_deg'],trace=selected['traces'][j],
            actual_heading_onset=onset,
            negative_target_born_inside_decision_duration_s=float(born_inside.sum()*.001),
            negative_target_duration_s=float(negative.sum()*.001),
            first_negative_target=None if first is None else dict(tick=first,
                time_s=(first+1)*.001,decision=first//20,
                combined_deg=float(s['combined_deg'][first]),los_deg=float(s['los_deg'][first]),
                residual_deg=float(s['residual_deg'][first])),
            negative_target_with_negative_residual_s=float((negative&(s['residual_deg']<0)).sum()*.001),
            negative_target_with_negative_los_s=float((negative&(s['los_deg']<0)).sum()*.001),
            largest_intradecision_changes=macro[:3],shape=shape)
        records.append(record)
        np.savez_compressed(stage/f'path{p}_commands.npz',**s,rows=rows,reference=ref,actions=actions)
        print('CURRENT_COORDINATES',p,record['first_negative_target'],
            'largest_macro_change',round(macro[0]['within_decision_target_change_deg'],3),flush=True)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),records=records,
        new_causal_claim=False,goal_complete=False))
    receipt(stage,dict(status='completed_current_command_coordinate_audit',
        new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))


if __name__=='__main__':main()
