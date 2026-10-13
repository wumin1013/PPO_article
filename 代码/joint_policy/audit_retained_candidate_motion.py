"""Verify full saved paths and distinguish raw, filtered and physical headings."""
import argparse
import time
import numpy as np
from audit_current_command_coordinates import command_signals
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--stage',required=True)
    args=parser.parse_args()
    folders=[BASE/args.parent,BASE/args.candidate]
    data=[read(f/'results.json') for f in folders]
    assert all(d['full_candidate']['checks']['eligible'] for d in data)
    stage=register(args.stage,[__file__,BASE/'validate_actions.py',BASE/'audit_current_command_coordinates.py']+
        [f/n for f,d in zip(folders,data) for n in ('results.json','receipt.json',d['chosen']['model'])],
        dict(new_actor_decisions=0,new_physics_ticks=0,saved_full_paths=128,wall_s=60),
        'Independently finite-difference the raw saved XY on all64 complete paths '
        'for both retained parent and candidate. Check exact V/A/J limits and '
        'at least20ms rest. Split actual>=99mm/s cruise into pure-straight, '
        'isolated-corner and multisegment routes. On all16 isolated corners '
        'separately count raw combined target, filtered desired and physical '
        'negative headings before the apex at speed>5mm/s and threshold0.05deg; '
        'also verify the first significant turn direction. Earlier field '
        'negative_command_ms meant filtered desired heading, not raw combined '
        'target. Keep these quantities explicitly separate. No new simulation, '
        'smoothing, test access or final ideal-shape claim.')
    start=time.perf_counter();records=[];groups=[]
    for label,folder,d in zip(('parent','candidate'),folders,data):
        full=d['full_candidate'];model=folder/d['chosen']['model']
        assert sorted(r['original_path'] for r in full['results'])==list(range(64))
        assert full['summary']['successes']==64
        peaks=np.zeros(3);cruise={k:0. for k in ('straight','isolated_corner','multisegment')}
        corners=[];checks=[]
        for r,t in zip(full['results'],full['traces']):
            with np.load(BASE/t) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
            audit=difference_audit(ref[0],rows)
            strict=bool(np.all(np.asarray(audit['peaks'])<=np.array([100.,2000.,62500.])))
            assert r['success'] and audit['rest_pass'] and strict,(label,r['original_path'],audit)
            peaks=np.maximum(peaks,audit['peaks'])
            speed=np.linalg.norm(rows[:,2:4],axis=1)
            kind='straight' if len(ref)==2 else 'isolated_corner' if len(ref)==3 else 'multisegment'
            cruise[kind]+=float((speed>=99.).sum()*.001)
            checks.append(dict(path=r['original_path'],strict_vector_pass=strict,rest_pass=audit['rest_pass']))
            if len(ref)!=3:continue
            s=command_signals(ref,rows,actions)
            apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
            pre=(np.arange(len(rows))<apex)&(rows[:,12]==0)&(speed>5.)
            initial=np.flatnonzero(pre&(abs(s['physical_heading_deg'])>.05))
            item=dict(label=label,path=r['original_path'],
                first_direction=('inside' if s['physical_heading_deg'][initial[0]]>0 else 'outside') if len(initial) else None,
                combined_negative_target_ms=int((pre&(s['combined_deg']<-.05)).sum()),
                filtered_desired_negative_heading_ms=int((pre&(s['desired_heading_deg']<-.05)).sum()),
                physical_negative_heading_ms=int((pre&(s['physical_heading_deg']<-.05)).sum()),
                minimum_heading_deg=float(s['physical_heading_deg'][pre].min()))
            records.append(item);corners.append(item)
        groups.append(dict(label=label,model=str(model.relative_to(BASE)),sha256=sha(model),
            cruise99_s=cruise,independent_vector_peaks=peaks.tolist(),all64_strict_limits_and_rest=True,
            all_first_turns_inside=all(c['first_direction']=='inside' for c in corners),
            combined_negative_target_ms=sum(c['combined_negative_target_ms'] for c in corners),
            filtered_desired_negative_heading_ms=sum(c['filtered_desired_negative_heading_ms'] for c in corners),
            physical_negative_heading_ms=sum(c['physical_negative_heading_ms'] for c in corners),
            full_time_s=full['summary']['full_time_s'],full_successes=64,checks=checks))
    write_json(stage/'results.json',dict(groups=groups,records=records,threshold_deg=.05,
        moving_speed_threshold_mm_s=5.,generalization_verified=False,goal_complete=False))
    receipt(stage,dict(status='verified_saved_full_motion_and_heading_phases',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))
    for g in groups:print('RETAINED_MOTION', {k:v for k,v in g.items() if k!='checks'},flush=True)


if __name__=='__main__':
    main()
