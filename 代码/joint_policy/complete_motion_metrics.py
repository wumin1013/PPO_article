"""Complete-path speed and sustained-cruise diagnostics from original1ms rows."""
import numpy as np


def complete_motion_metrics(reference,rows):
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    arc=np.cumsum(speed*.001)
    result=dict(recorded_time_s=len(rows)*.001,travelled_distance_mm=float(arc[-1]),
        peak_speed_mm_s=float(speed.max()),mean_complete_speed_mm_s=float(speed.mean()))
    for threshold in (95,98,99):
        moving=speed>=threshold
        edge=np.diff(np.r_[False,moving,False].astype(int))
        durations=(np.flatnonzero(edge==-1)-np.flatnonzero(edge==1))*.001
        result[f'cruise_ge{threshold}_s']=float(moving.sum()*.001)
        result[f'longest_cruise_ge{threshold}_s']=float(durations.max()) if len(durations) else 0.
    if len(reference)==3:
        apex=int(np.argmin(np.linalg.norm(rows[:,:2]-reference[1],axis=1)))
        for label,offset in (('before_apex_5mm',-5.),('apex',0.),('after_apex_5mm',5.)):
            location=arc[apex]+offset
            result[label+'_speed_mm_s']=(float(np.interp(location,arc,speed))
                if arc[0]<=location<=arc[-1] else None)
    return result
