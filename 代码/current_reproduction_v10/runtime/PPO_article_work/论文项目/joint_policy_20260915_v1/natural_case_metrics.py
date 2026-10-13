"""One source of full-lap, all-corner and high-speed statistics for paper figures."""
import numpy as np
from scipy.spatial import cKDTree


def ordered_projection(reference, xy):
    """Nearest adjacent segment on this simple, widely separated contour.

    Full ordered-gate completion is a prerequisite; projection is only a plotting
    and window index, and never substitutes for the independent coverage audit.
    """
    edge=np.diff(reference,axis=0);length=np.linalg.norm(edge,axis=1)
    arc=np.r_[0.,np.cumsum(length)]
    _,vertex=cKDTree(reference).query(xy)
    early=np.arange(len(xy))<len(xy)//2
    vertex[(vertex==len(reference)-1)&early]=0
    vertex[(vertex==0)&~early]=len(reference)-1
    candidates=np.stack([np.clip(vertex-1,0,len(edge)-1),np.clip(vertex,0,len(edge)-1)],axis=1)
    d=xy[:,None,:]-reference[candidates]
    u=np.clip((d*edge[candidates]).sum(2)/length[candidates]**2,0.,1.)
    distance=np.linalg.norm(d-u[:,:,None]*edge[candidates],axis=2)
    idx=distance.argmin(1);seg=candidates[np.arange(len(xy)),idx]
    progress=arc[seg]+u[np.arange(len(xy)),idx]*length[seg]
    # The duplicated start/end is resolved using temporal order, not nearest XY.
    seam=np.linalg.norm(xy-reference[0],axis=1)<2.
    progress[seam & (np.arange(len(xy))<len(xy)//2)]=np.minimum(progress[seam & (np.arange(len(xy))<len(xy)//2)],2.)
    progress[seam & (np.arange(len(xy))>=len(xy)//2)]=np.maximum(progress[seam & (np.arange(len(xy))>=len(xy)//2)],arc[-1]-2.)
    return progress


def cruise_metrics(speed, dt=.001):
    result={}
    distance=float(speed.sum()*dt)
    for threshold in (95,98,99):
        mask=speed>=threshold
        changes=np.diff(np.r_[False,mask,False].astype(int))
        starts=np.flatnonzero(changes==1);stops=np.flatnonzero(changes==-1)
        lengths=stops-starts
        j=int(np.argmax(lengths)) if len(lengths) else None
        result[str(threshold)]=dict(time_s=float(mask.sum()*dt),time_fraction=float(mask.mean()),
            distance_mm=float(speed[mask].sum()*dt),distance_fraction=float(speed[mask].sum()*dt/max(distance,1e-12)),
            longest_time_s=float(lengths[j]*dt) if j is not None else 0.,
            longest_distance_mm=float(speed[starts[j]:stops[j]].sum()*dt) if j is not None else 0.)
    return result


def metrics(reference,rows,geometry):
    dt=.001;xy=rows[:,:2];v=rows[:,2:4];a=rows[:,4:6]
    speed=np.linalg.norm(v,axis=1)
    progress=ordered_projection(reference,xy)
    corners=[];previous_windows=np.zeros(len(rows),bool)
    for corner in geometry['sharp_corners']:
        k=corner['reference_vertex_index'];vertex=reference[k]
        incoming=reference[k]-reference[k-1];incoming/=np.linalg.norm(incoming)
        outgoing=reference[k+1]-reference[k];outgoing/=np.linalg.norm(outgoing)
        sign=np.sign(corner['signed_heading_change_deg']);normal=sign*np.array([-incoming[1],incoming[0]])
        outnormal=sign*np.array([-outgoing[1],outgoing[0]])
        lo=corner['reference_s_mm']-corner['window_before_mm']
        hi=corner['reference_s_mm']+corner['window_after_mm']
        mask=(progress>=lo)&(progress<=hi)
        assert not np.any(mask & previous_windows)
        previous_windows|=mask
        indices=np.flatnonzero(mask)
        if len(indices)<3:
            corners.append(dict(id=corner['id'],complete_window=False));continue
        # Fixed ordered gate crossing, never speed minimum or closest-approach alignment.
        gate_normal=incoming+outgoing;gate_normal/=np.linalg.norm(gate_normal)
        q=(xy[indices]-vertex)@gate_normal
        crossing=np.flatnonzero((q[:-1]<=0)&(q[1:]>=0))
        if len(crossing):
            ci=int(crossing[0]);fraction=-q[ci]/max(q[ci+1]-q[ci],1e-15)
            tc=(indices[ci]+1+fraction)*dt
        else:
            tc=None
        rr=rows[indices];sv=speed[indices];rel=rr[:,:2]-vertex
        heading=np.unwrap(np.arctan2(rr[:,2:4]@normal,rr[:,2:4]@incoming))
        moving=sv>5
        omega=sign*(rr[:,2]*rr[:,5]-rr[:,3]*rr[:,4])/np.maximum(sv**2,1e-12)
        # Significant first turn: incoming-oriented heading exceeds 0.05 degrees.
        significant=np.flatnonzero(moving & (np.abs(heading)>np.deg2rad(.05)))
        first=int(significant[0]) if len(significant) else None
        before=(indices+1)*dt<tc if tc is not None else np.zeros(len(indices),bool)
        reversal=before & moving & (heading < -np.deg2rad(.05))
        lateral=rr[:,2:4]@normal
        record=dict(id=corner['id'],category=corner['category'],included_angle_deg=corner['interior_angle_deg'],
            signed_turn_deg=corner['signed_heading_change_deg'],complete_window=bool(tc is not None),
            first_index=int(indices[0]),last_index=int(indices[-1]),crossing_time_s=tc,
            duration_s=float(len(indices)*dt),minimum_speed_mm_s=float(sv.min()),mean_speed_mm_s=float(sv.mean()),
            time_below95_s=float((sv<95).sum()*dt),speed_deficit_mm=float(np.maximum(100-sv,0).sum()*dt),
            first_turn_inside=bool(heading[first]>0) if first is not None else None,
            first_turn_relative_time_s=float((indices[first]+1)*dt-tc) if first is not None and tc is not None else None,
            first_turn_incoming_distance_mm=float(-(rel[first]@incoming)) if first is not None else None,
            reverse_lateral_mm=float(-np.minimum(lateral[before],0).sum()*dt),
            significant_reverse_lateral_mm=float(-np.minimum(lateral[reversal],0).sum()*dt),
            reverse_lateral_duration_s=float(reversal.sum()*dt),
            reverse_heading_deg=float(np.rad2deg(np.maximum(-omega[moving],0).sum()*dt)),
            incoming_outside_mm=float(max(0,-(rel@normal).min())),
            outgoing_overshoot_mm=float(max(0,-(rel@outnormal).min())),
            heading_entry_deg=float(np.rad2deg(heading[0])),
            heading_exit_error_deg=float(np.rad2deg(heading[-1])-abs(corner['signed_heading_change_deg'])),
            peak_curvature_mm_inv=float(np.max(np.abs(omega[moving]/sv[moving]))) if moving.any() else None)
        # Earliest detectable turn is checked over the complete visible approach
        # (30mm sensing horizon), separately from the registered20mm speed window.
        approach_ids=np.flatnonzero((progress>=corner['reference_s_mm']-30.) &
                                    (progress<corner['reference_s_mm']))
        ah=np.arctan2(v[approach_ids]@normal,v[approach_ids]@incoming)
        ai=np.flatnonzero((speed[approach_ids]>5)&(np.abs(ah)>np.deg2rad(.05)))
        if len(ai):
            j=int(ai[0]);tick=int(approach_ids[j]);previous=int(approach_ids[max(0,j-1)])
            record.update(first_turn_30mm_audit_inside=bool(ah[j]>0),
                first_turn_30mm_audit_left_censored=j==0,
                first_turn_30mm_time_s=float((tick+1)*dt),
                first_turn_30mm_relative_time_s=float((tick+1)*dt-tc) if tc is not None else None,
                first_turn_30mm_distance_before_mm=float(-((xy[tick]-vertex)@incoming)),
                first_turn_30mm_time_bracket_s=[float((previous+1)*dt),float((tick+1)*dt)])
        else:record.update(first_turn_30mm_audit_inside=None,first_turn_30mm_audit_left_censored=False)
        corners.append(record)
    sections=[]
    for i,p in enumerate(geometry['primitives']):
        # Source arc coordinates differ from chord coordinates by <=0.003 mm.
        starts=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
        mask=(progress>=starts[p['start_reference_index']])&(progress<starts[p['end_reference_index']])
        ss=speed[mask]
        sections.append(dict(index=i,label=p['label'],kind=p['kind'],length_mm=p['length_mm'],
            time_s=float(mask.sum()*dt),cruise99_s=float((ss>=99).sum()*dt),
            travel_distance_mm=float(ss.sum()*dt),
            cruise99_distance_mm=float(ss[ss>=99].sum()*dt)))
    return dict(complete_recorded_time_s=len(rows)*dt,travel_distance_mm=float(speed.sum()*dt),
        cruise=cruise_metrics(speed),corners=corners,sections=sections,
        all34_windows_complete=len(corners)==34 and all(x['complete_window'] for x in corners),
        corner_total_speed_deficit_mm=sum(c.get('speed_deficit_mm',0) for c in corners),
        corner_total_below95_s=sum(c.get('time_below95_s',0) for c in corners),
        projection_raw_backward_steps=int((np.diff(progress)<-1e-7).sum()),
        definitions=dict(window='registered +/-10 mm reference-s windows, all34; no overlap',
                         first_turn='first |heading|>0.05 degree at speed>5 mm/s inside registered window',
                         crossing='first forward crossing of registered corner bisector gate',
                         time='from rest through braking and stationary tail',
                         curvature='cross(v,a)/|v|^3 on original1ms rows at |v|>5 mm/s'))
