"""Raw per-corner diagnostics on declared ordered-progress midpoint windows."""
import numpy as np
from geometry_spec import Corridor,GateTracker
from complete_corner_metrics import complete_corner_metrics
from probe_linear_action_hold import target_motion
from learn_complete_joint_teachers import teacher_score


KEYS=(('in_edge_outward_mm',.002),('out_edge_outward_mm',.005),
      ('full_physical_reverse_deg',.5),('full_normalized_energy',.02),
      ('full_spatial_peak_curvature_mm_inv',.2),('reverse_lateral_total_mm',.002))


def augment(record):
    if 'shape' in record or not record['success'] or 'multi_corner_windows' in record:return record
    with np.load(record['trace']) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    assert np.linalg.norm(ref[0]-ref[-1])>1e-9,'Closed-path seam needs a separate window definition'
    corridor=Corridor(ref);tracker=GateTracker(corridor,ref[0]);progress=[]
    for xy in rows[:,:2]:
        tracker.update(xy);i=min(tracker.frontier,len(corridor.lengths)-1)
        offset=np.clip((xy-corridor.starts[i])@corridor.tangents[i],0.,corridor.lengths[i])
        progress.append(corridor.arcs[i]+offset)
    assert tracker.snapshot(1.25)['ordered_complete']
    progress=np.maximum.accumulate(progress)
    windows=[]
    for i in range(1,len(ref)-1):
        u,v=corridor.tangents[i-1:i+1]
        if abs(u[0]*v[1]-u[1]*v[0])<1e-10:continue
        left=.5*(corridor.arcs[i-1]+corridor.arcs[i]);right=.5*(corridor.arcs[i]+corridor.arcs[i+1])
        lo=0 if i==1 else int(np.searchsorted(progress,left))
        hi=len(rows) if i==len(ref)-2 else min(len(rows),int(np.searchsorted(progress,right))+1)
        assert hi-lo>4,(record['path'],i,lo,hi)
        initial=ref[0] if lo==0 else rows[lo-1,:2]
        shape=complete_corner_metrics(ref[i-1:i+2],rows[lo:hi],initial_xy=initial)
        coarse=complete_corner_metrics(ref[i-1:i+2],rows[lo:hi],spacing=.2,initial_xy=initial)
        windows.append(dict(vertex=i,start_ms=lo,stop_ms=hi,ordered_progress_bounds_mm=[float(left),float(right)],
            shape=shape,target_motion=target_motion(ref[i-1:i+2],rows[lo:hi]),
            curvature_02mm=coarse['full_spatial_peak_curvature_mm_inv'],energy_02mm=coarse['full_normalized_energy']))
    record['multi_corner_windows']=windows
    record['multi_corner_definition']='Original edges; raw1ms samples between adjacent reference-leg midpoint progress, extended to path start/end for the first/last corner so no endpoint prefix/tail is omitted; gate frontier disambiguates ordered projection, with monotone accumulation only for window selection. Initial XY is the actual preceding tick, never a synthetic chord. Local first-turn labels include incoming residual motion and do not establish ideal tangency.'
    return record


def guards(record,baseline):
    size=4+11*len(baseline['multi_corner_windows'])
    if not record['success']:return np.full(size,-100.)
    assert len(record['multi_corner_windows'])==len(baseline['multi_corner_windows'])
    values=[(baseline['complete_time_s']+.02-record['complete_time_s'])/.02,
            (record['motion']['cruise_ge99_s']-baseline['motion']['cruise_ge99_s']+.02)/.05,
            record['corridor']['margin_lower_mm']/.01,1.]
    for a,b in zip(record['multi_corner_windows'],baseline['multi_corner_windows']):
        assert a['vertex']==b['vertex']
        values.extend((b['shape'][key]-a['shape'][key])/scale for key,scale in KEYS)
        values.extend([(a['shape']['apex_speed_mm_s']-b['shape']['apex_speed_mm_s'])/2.,
            1. if not b['target_motion']['first_turn_inside'] or a['target_motion']['first_turn_inside'] else -1.,
            (b['curvature_02mm']-a['curvature_02mm'])/.2,(b['energy_02mm']-a['energy_02mm'])/.02,
            (b['shape']['reverse_lateral_significant_duration_s']-a['shape']['reverse_lateral_significant_duration_s'])/.02])
    assert len(values)==size
    return np.asarray(values)


def objective(record,baseline):
    if not record['success']:return 1e6
    result=0.
    for a,b in zip(record['multi_corner_windows'],baseline['multi_corner_windows']):
        ra=dict(record,shape=a['shape'],target_motion=a['target_motion'])
        rb=dict(baseline,shape=b['shape'],target_motion=b['target_motion'])
        result+=teacher_score(ra,rb,b['shape'],barrier=False)
    return result
