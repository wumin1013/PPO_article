"""Decompose bound complete traces without additional physics or model calls."""
from pathlib import Path
import argparse,json,sys
import numpy as np
OUT=Path(__file__).resolve().parent
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
sys.path[:0]=[str(BASE),str(OUT)]
from study_io import write_json,sha
from natural_case_metrics import ordered_projection


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate');ap.add_argument('fixed');args=ap.parse_args()
    joint=Path(args.candidate);fixed=Path(args.fixed);case=BASE/'paper_evidence_review/natural_closed_case'
    stage=joint/'complete_efficiency_decomposition';stage.mkdir(exist_ok=False)
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    corners=np.array([c['reference_s_mm'] for c in geometry['sharp_corners']])
    traces={'Joint':joint/'full_trace.npz','Fixed':fixed/'full_trace.npz',
            'S2':case/'s2_complete/trace.npz','HistoricalJoint':case/'support_calibration/remaining_median.npz'}
    reports={}
    for name,path in traces.items():
        z=np.load(path);rows=z['rows'];ref=z['reference'];speed=np.linalg.norm(rows[:,2:4],axis=1)
        progress=ordered_projection(ref,rows[:,:2]);length=np.linalg.norm(np.diff(ref,axis=0),axis=1).sum()
        near=np.minimum.reduce([abs(progress-c) for c in corners])<=30.
        ds=np.linalg.norm(np.diff(np.vstack([ref[0],rows[:,:2]]),axis=0),axis=1)
        def summary(mask):
            v=speed[mask];travel=float(ds[mask].sum());deficit=float(np.maximum(100.-v,0).sum()*.001)
            return dict(time_s=float(mask.sum()*.001),travel_distance_mm=travel,speed_deficit_mm=deficit,
                deficit_equivalent_s=deficit/100.,cruise99_s=float((v>=99.).sum()*.001),
                mean_speed_mm_s=float(v.mean()) if len(v) else None)
        report=dict(full=summary(np.ones(len(rows),dtype=bool)),near30=summary(near),far30=summary(~near),
            first_half=summary(progress<length/2),second_half=summary(progress>=length/2),
            quarters=[summary((progress>=length*i/4)&(progress<length*(i+1)/4)) for i in range(4)])
        full=report['full'];report['identity_residual_s']=full['time_s']-(full['travel_distance_mm']+full['speed_deficit_mm'])/100.
        if 'actions' in z.files:
            a=z['actions'];indices=np.minimum(np.arange(len(a))*20,len(rows)-1);far=~near[indices]
            report['far_request_feed_quantiles']=np.quantile(a[far,1],[0,.1,.5,.9,1]).tolist()
            report['far_request_below99_fraction']=float(np.mean(a[far,1]<99.))
            report['far_request_feed_mean_quarters']=[float(np.mean(a[far & (progress[indices]>=length*i/4)&(progress[indices]<length*(i+1)/4),1])) for i in range(4)]
        reports[name]=report
    comparisons={}
    for name in ['Fixed','S2','HistoricalJoint']:
        j=reports['Joint']['full'];b=reports[name]['full']
        comparisons[name]=dict(time_gain_s=b['time_s']-j['time_s'],
            path_length_contribution_s=(b['travel_distance_mm']-j['travel_distance_mm'])/100.,
            speed_deficit_contribution_s=(b['speed_deficit_mm']-j['speed_deficit_mm'])/100.)
    write_json(stage/'audit.json',dict(goal_complete=False,physics_ticks=0,policy_calls=0,methods=reports,
        comparisons=comparisons,source_sha256={str(p):sha(p) for p in list(traces.values())+[Path(__file__),case/'geometry.json']},
        definitions='Actual XY command distance and integrated positive100-v deficit; dt1ms; halves and quarters use registered reference progress. Near30 is explanatory only and does not change registered+/-10mm corner acceptance.',
        attribution='Historical-to-current change includes shared executor and parameter refinement; decomposition is descriptive and cannot isolate a PPO causal effect.'))
    print('COMPLETE_EFFICIENCY',comparisons,flush=True)


if __name__=='__main__':main()
