"""Geometry-mechanism audit of actual saved local executions; no new physics."""
from pathlib import Path
import sys,json,pickle,argparse
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path.insert(0,str(BASE))
from natural_case_metrics import ordered_projection
from study_io import write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--arm',choices=['joint','fixed'],default='joint');args=ap.parse_args()
    root=OUT/'all34_dominance_research';stage=root/args.stage
    with (root/('selected_causal_contexts' if args.arm=='joint' else 'fixed_causal_contexts')/'contexts.pkl').open('rb') as f:contexts=pickle.load(f)
    records=[]
    for c in contexts:
        file=stage/('screen_'+c['corner']['id']+'.npz');z=np.load(file);rows=z['rows'];ref=c['reference'];k=c['corner']['reference_vertex_index']
        v=rows[:,2:4];speed=np.linalg.norm(v,axis=1);s=ordered_projection(ref,rows[:,:2])-c['corner']['reference_s_mm']
        incoming=ref[k]-ref[k-1];incoming/=np.linalg.norm(incoming);sign=np.sign(c['corner']['signed_heading_change_deg']);normal=sign*np.array([-incoming[1],incoming[0]])
        h=np.unwrap(np.arctan2(v@normal,v@incoming));visible=(s>=-30)&(s<0)&(speed>5)
        ids=np.flatnonzero(visible&(abs(h)>np.deg2rad(.05)))
        corner=(s>=-10)&(s<=10);hwin=h[corner];rr=rows[corner];sv=speed[corner]
        curvature=sign*(rr[:,2]*rr[:,5]-rr[:,3]*rr[:,4])/np.maximum(sv,1.)**3
        records.append(dict(id=c['corner']['id'],angle=c['corner']['interior_angle_deg'],
            first_turn_30mm_inside=bool(h[ids[0]]>0) if len(ids) else None,
            first_turn_distance_before_mm=float(-s[ids[0]]) if len(ids) else None,
            first_turn_heading_deg=float(np.rad2deg(h[ids[0]])) if len(ids) else None,
            heading_entry_deg=float(np.rad2deg(hwin[0])),
            heading_exit_error_deg=float(np.rad2deg(hwin[-1])-abs(c['corner']['signed_heading_change_deg'])),
            minimum_speed_mm_s=float(sv.min()),peak_curvature_mm_inv=float(abs(curvature).max()),
            reverse_heading_within_window_deg=float(np.rad2deg(np.maximum(-hwin,0).max())),trace_sha256=sha(file)))
    result=dict(kind='actual_local_trajectory_mechanism_audit',physics_ticks=0,ppo_updates=0,full_lap_evidence=False,records=records,
        inside_count=sum(r['first_turn_30mm_inside'] is True for r in records),
        outside_ids=[r['id'] for r in records if r['first_turn_30mm_inside'] is False],
        max_exit_heading_error_deg=max(abs(r['heading_exit_error_deg']) for r in records))
    write_json(stage/'screen_mechanism.json',result)
    print('SCREEN_MECHANISM',args.arm,result['inside_count'],result['outside_ids'],'max_exit_error',result['max_exit_heading_error_deg'],flush=True)

if __name__=='__main__':main()
