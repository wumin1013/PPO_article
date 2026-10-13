"""Check complete-decision metric credit against already measured policy updates."""
import numpy as np
import torch
from deep_joint_credit_policy import load_credit_policy
from study_io import BASE,write_json,sha
from search_complete_return import read


KEYS=('out_edge_outward_mm','full_physical_reverse_deg','full_normalized_energy',
      'full_spatial_peak_curvature_mm_inv','apex_speed_mm_s')


def shape_derivative(query, baseline, amplitude):
    result=np.zeros((len(KEYS),3))
    if 'shape' not in baseline:return result
    b=np.array([baseline['shape'][k] for k in KEYS])
    for head in range(3):
        lo,hi=query['records'][2*head:2*head+2]
        low=np.array([lo['shape'][k] for k in KEYS]) if lo['success'] else None
        high=np.array([hi['shape'][k] for k in KEYS]) if hi['success'] else None
        if low is not None and high is not None:result[:,head]=(high-low)/(2*amplitude[head])
        elif high is not None:result[:,head]=(high-b)/amplitude[head]
        elif low is not None:result[:,head]=(b-low)/amplitude[head]
    return result


def main():
    torch.set_num_threads(1)
    sampled=BASE/'canonical_gate_band_complete_current_suffix_credit_20260919'
    evaluated=BASE/'canonical_gate_band_complete_decision_joint_learning_20260919'
    plan=read(sampled/'query_plan.json');baseline={r['path']:r for r in plan['baseline']['records']}
    queries=sorted(read(sampled/'query_results.json')['queries'],key=lambda q:q['query'])
    results=read(evaluated/'results.json')['candidates'];output=[]
    for c in results:
        if not c['summary']['successes']==20:continue
        md=torch.load(evaluated/(c['tag']+'.pt'),map_location='cpu',weights_only=False)
        p=load_credit_policy(md)
        delta=p.correction(torch.tensor([q['observation'] for q in queries],dtype=torch.float32)).detach().numpy()
        changes={path:np.zeros(len(KEYS)) for path in range(16)}
        for q,d in zip(queries,delta):
            if q['path'] in changes:changes[q['path']]+=shape_derivative(q,baseline[q['path']],np.array(plan['amplitude']))@d
        metrics=[]
        for r in c['records']:
            if 'shape' not in r:continue
            for index,key in enumerate(KEYS):
                b=baseline[r['path']]['shape'][key];prediction=b+changes[r['path']][index]
                metrics.append(dict(path=r['path'],metric=key,baseline=b,predicted=float(prediction),
                    actual=r['shape'][key],error=float(prediction-r['shape'][key])))
        output.append(dict(tag=c['tag'],model_sha256=sha(evaluated/(c['tag']+'.pt')),metrics=metrics))
        print('DENSE_SHAPE_PREDICTION',c['tag'],{k:dict(maximum_error=max(abs(r['error']) for r in metrics if r['metric']==k),
            predicted_change=sum(r['predicted']-r['baseline'] for r in metrics if r['metric']==k),
            actual_change=sum(r['actual']-r['baseline'] for r in metrics if r['metric']==k)) for k in KEYS},flush=True)
    write_json(evaluated/'per_corner_credit_prediction.json',dict(candidates=output,
        query_results_sha256=sha(sampled/'query_results.json'),audit_source_sha256=sha(__file__),new_physics_ticks=0))


if __name__=='__main__':main()
