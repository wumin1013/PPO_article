"""Supplement stored XY audits with wrong-way lateral motion, without new rollouts."""
import argparse
import numpy as np
from complete_corner_metrics import complete_corner_metrics
from localize_joint_shape_residual import intervals
from study_io import BASE,write_json,sha
from search_complete_return import read


def lateral_motion(trace):
    with np.load(trace) as z:ref,rows=z['reference'].copy(),z['rows'].copy()
    u,v=np.diff(ref,axis=0)
    u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    sign=np.sign(u[0]*v[1]-u[1]*v[0])
    ni=sign*np.array([-u[1],u[0]])
    xy=np.vstack([ref[0],rows[:,:2]])
    displacement=np.diff(xy,axis=0)@ni
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    apex=int(np.linalg.norm(rows[:,:2]-ref[1],axis=1).argmin())
    heading=np.arctan2(rows[:,2:4]@ni,rows[:,2:4]@u)
    before=(np.arange(len(rows))<apex)&(speed>5.)&(rows[:,12]==0)
    negative=before&(heading < -np.deg2rad(.05))
    periods=intervals(negative)
    out=dict(apex_ms=apex,
        significant_reverse_lateral_total_mm=float(-np.minimum(displacement[negative],0).sum()),
        significant_reverse_lateral_duration_ms=int(negative.sum()),
        significant_reverse_lateral_max_interval_mm=max(
            (float(-np.minimum(displacement[lo:hi],0).sum()) for lo,hi in periods),default=0.),
        intervals=[dict(start_ms=int(lo),stop_ms=int(hi),
            reverse_lateral_mm=float(-np.minimum(displacement[lo:hi],0).sum())) for lo,hi in periods],
        any_reverse_lateral_total_mm=float(-np.minimum(displacement[before],0).sum()),
        trace_sha256=sha(trace))
    coarse=complete_corner_metrics(ref,rows,spacing=.2)
    out['curvature_at_02mm_inv']=coarse['full_spatial_peak_curvature_mm_inv']
    out['energy_at_02mm']=coarse['full_normalized_energy']
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    result=read(stage/'results.json')
    chosen=next(c for c in result['candidates'] if c['tag']==result['best'])
    old={r['path']:r for r in read(stage/'baseline.json')['baseline']['records']}
    records=[]
    for r in chosen['records']:
        if 'shape' not in r:continue
        records.append(dict(path=r['path'],before=lateral_motion(old[r['path']]['trace']),
                            after=lateral_motion(stage/r['trace'])))
    write_json(stage/'saved_lateral_motion_audit.json',dict(records=records,new_rollouts=0,
        model_sha256=sha(stage/'research_candidate.pt'),
        definition='Raw1ms XY increments projected on the signed incoming normal before '
        'closest vertex approach. Significant means physical heading below-0.05deg at '
        'speed>5mm/s, as in the existing first-turn audit. Also report all negative '
        'increments without this angular threshold. Inward-side position does not '
        'imply absence of wrong-way movement. No acceptance tolerance was changed.',
        ideal_tangency_accepted=False))
    print('SAVED_LATERAL_AUDIT',len(records),'no new rollout',
          'total_mm',sum(r['after']['significant_reverse_lateral_total_mm'] for r in records),
          'max_duration_ms',max(r['after']['significant_reverse_lateral_duration_ms'] for r in records))


if __name__=='__main__':main()
