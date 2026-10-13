"""Explain current speed losses using saved physical curvature and feed demand."""
import argparse
import time
import numpy as np
from audit_current_command_coordinates import command_signals
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_band_transition_measured_margin_learning_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_current_curvature_speed_audit_20260917')
    parser.add_argument('--step',type=float)
    args=parser.parse_args()
    parent=BASE/args.parent
    data=read(parent/'results.json')
    candidate=(next(c for c in data['candidates'] if c['step']==args.step)
        if args.step is not None else data['full_candidate'])
    model=parent/(candidate['model'] if args.step is not None else data['chosen']['model'])
    assert candidate['summary']['successes']==len(candidate['results'])
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',BASE/'audit_current_command_coordinates.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=60,training_paths=len(candidate['results']),
            candidate_step=args.step,candidate_passed_learning_screen=candidate['checks']['eligible']),
        'Use saved current learned complete paths to separate speed demanded by '
        'the actor from speed lost in concentrated curvature. At each isolated '
        'corner curvature peak, compute the necessary vector-acceleration bound '
        'v<=sqrt(2000/abs(kappa)), actual normal acceleration, total acceleration '
        'and jerk; this acceleration bound alone is not a jerk-feasibility or '
        'optimality proof. Retain actual before/after-apex speeds and turning '
        'width. On truly straight reference paths, examine nonterminal samples '
        'at least6.2mm from both endpoints: this exceeds the4.1mm ideal rest-to100 '
        'acceleration distance plus2mm decision travel under the original V/A/J '
        'limits, as an explicitly labeled diagnostic window. Compare actor feed '
        'demand, physical speed and active limits without constructing a deployed '
        'speed or angle rule. No physical probe, reward change or final acceptance.')
    start=time.perf_counter();corners=[];straights=[]
    shapes={s['path']:s for s in candidate['summary']['shapes']}
    for r,trace in zip(candidate['results'],candidate['traces']):
        path=r['original_path']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        speed=np.linalg.norm(rows[:,2:4],axis=1);acc=np.linalg.norm(rows[:,4:6],axis=1)
        jerk=np.linalg.norm(rows[:,6:8],axis=1);ordinary=rows[:,12]==0
        feed=actions[np.minimum(np.arange(len(rows))//20,len(actions)-1),1]
        if path in shapes:
            moving=speed>5.
            curvature=np.zeros(len(rows));curvature[moving]=(rows[moving,2]*rows[moving,5]-rows[moving,3]*rows[moving,4])/speed[moving]**3
            k=int(abs(curvature).argmax());cap=min(100.,float(np.sqrt(2000/abs(curvature[k]))))
            control=command_signals(ref,rows,actions)
            item=dict(path=path,turn_deg=shapes[path]['turn_deg'],peak_tick=k,
                peak_curvature_mm_inv=float(abs(curvature[k])),speed_at_peak_mm_s=float(speed[k]),
                actor_feed_at_peak_mm_s=float(feed[k]),necessary_acceleration_speed_cap_mm_s=cap,
                normal_acceleration_at_peak_mm_s2=float(speed[k]**2*abs(curvature[k])),
                total_acceleration_at_peak_mm_s2=float(acc[k]),jerk_at_peak_mm_s3=float(jerk[k]),
                apex_speed_mm_s=shapes[path]['apex_speed_mm_s'],
                middle80_turn_arc_mm=shapes[path]['full_middle80_turn_arc_mm'],
                full_reverse_deg=shapes[path]['full_physical_reverse_deg'],
                requested_feed_range_mm_s=[float(feed[ordinary].min()),float(feed[ordinary].max())])
            corners.append(item)
        if len(ref)==2:
            unit=(ref[-1]-ref[0])/np.linalg.norm(ref[-1]-ref[0])
            along=(rows[:,:2]-ref[0])@unit;length=float(np.linalg.norm(ref[-1]-ref[0]))
            mask=ordinary&(along>=6.2)&(along<=length-6.2)
            if not mask.any():continue
            straights.append(dict(path=path,length_mm=length,diagnostic_window_s=float(mask.sum()*.001),
                requested_feed_quantiles_mm_s=np.quantile(feed[mask],[0,.5,1]).tolist(),
                physical_speed_quantiles_mm_s=np.quantile(speed[mask],[0,.5,1]).tolist(),
                fraction_request_below99=float(np.mean(feed[mask]<99)),
                fraction_physical_ge99=float(np.mean(speed[mask]>=99)),
                fraction_acceleration_ge95pct=float(np.mean(acc[mask]>=1900)),
                fraction_jerk_ge95pct=float(np.mean(jerk[mask]>=59375)),
                median_absolute_demand_gap_mm_s=float(np.median(abs(feed[mask]-speed[mask])))))
    write_json(stage/'results.json',dict(parent_sha256=sha(model),corners=corners,straights=straights,
        complete_time_s=candidate['summary']['full_time_s'],
        complete_cruise_s=candidate['summary']['complete_cruise_s'],goal_complete=False))
    receipt(stage,dict(status='completed_current_curvature_and_cruise_audit',corners=len(corners),
        straight_windows=len(straights),new_actor_decisions=0,new_physics_ticks=0,wall_s=time.perf_counter()-start))
    for r in corners:
        if r['path'] in (0,7,14,15):print('CURVATURE_SPEED',r,flush=True)
    for r in straights:print('STRAIGHT_DEMAND',r,flush=True)


if __name__=='__main__':main()
