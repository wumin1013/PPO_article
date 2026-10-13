"""Read-only common-coordinate audit of complete saved trajectories."""
from pathlib import Path
import sys,json,argparse
import numpy as np
BASE=Path(r'D:\PPO_article_work\论文项目\joint_policy_20260915_v1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from natural_case_metrics import ordered_projection
from study_io import sha,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--joint',required=True);ap.add_argument('--fixed',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    target=Path(args.output);assert not target.exists(), 'Preserve prior diagnostic outputs'
    case=BASE/'paper_evidence_review/natural_closed_case'
    binding=json.loads((case/'paper_result_binding.json').read_text(encoding='utf-8'))
    geometry=json.loads((case/'geometry.json').read_text(encoding='utf-8'))
    historical={key:BASE/value['trace'] for key,value in binding['primary_methods'].items()}
    methods={key:path for key,path in historical.items() if 's2' in key.lower()}
    methods.update(joint=Path(args.joint)/'full_trace.npz',fixed=Path(args.fixed)/'full_trace.npz')
    corners=np.array([c['reference_s_mm'] for c in geometry['sharp_corners']]);data={}
    for name,path in methods.items():
        z=np.load(path);rows=z['rows'];speed=np.linalg.norm(rows[:,2:4],axis=1)
        progress=ordered_projection(z['reference'],rows[:,:2]);near=np.min(np.abs(progress[:,None]-corners),axis=1)<=30.
        def summary(mask):
            v=speed[mask]
            return dict(time_s=float(mask.sum()*.001),travel_mm=float(v.sum()*.001),
                deficit_equivalent_s=float(np.maximum(100.-v,0.).sum()*.00001),cruise99_s=float((v>=99).sum()*.001))
        d=dict(trace=str(path),sha256=sha(path),all=summary(np.ones(len(rows),bool)),near30=summary(near),far30=summary(~near),
            corners30=[dict(id=c['id'],**summary(np.abs(progress-c['reference_s_mm'])<=30.)) for c in geometry['sharp_corners']])
        length=np.linalg.norm(np.diff(z['reference'],axis=0),axis=1).sum()
        d['quarters']=[summary((progress>=length*q/4)&(progress<length*(q+1)/4)) for q in range(4)]
        if name in ('joint','fixed'):
            metrics=json.loads(path.with_name('full_metrics.json').read_text(encoding='utf-8'))
            d['corners']=metrics['corners'];d['arcs']=[s for s in metrics['sections'] if s['kind']=='arc']
            fields=('minimum_speed_mm_s','reverse_heading_deg','incoming_outside_mm','outgoing_overshoot_mm','heading_entry_deg','heading_exit_error_deg','peak_curvature_mm_inv')
            d['all34_summary']={f:dict(zip(('min','median','max'),np.quantile([c[f] for c in metrics['corners']],[0,.5,1]).tolist())) for f in fields}
            d['all34_summary'].update(first_inward_count=sum(c['first_turn_30mm_audit_inside'] is True for c in metrics['corners']),left_censored_count=sum(c['first_turn_30mm_audit_left_censored'] for c in metrics['corners']))
        data[name]=d
    joint,fixed=data['joint'],data['fixed']
    differences={region:{k:joint[region][k]-fixed[region][k] for k in joint[region]} for region in ('all','near30','far30')}
    differences['corners30']=[dict(id=a['id'],joint_minus_fixed_s=a['time_s']-b['time_s']) for a,b in zip(joint['corners30'],fixed['corners30'])]
    write_json(target,dict(methods=data,joint_minus_fixed=differences,goal_complete=False))
    print(json.dumps(dict(methods={k:{q:v[q] for q in ('all','near30','far30')} for k,v in data.items()},joint_minus_fixed=differences),indent=2))


if __name__=='__main__':main()
