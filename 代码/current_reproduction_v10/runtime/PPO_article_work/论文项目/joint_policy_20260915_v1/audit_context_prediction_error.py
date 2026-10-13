"""Separate local credit approximation error from the added narrow-candidate guards."""
import time
import numpy as np
import torch
from learn_early_memory_joint_policy import metric_functions
from duration_joint_metrics import guards,guard_names
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_context_postpeak_memory_learning_20260920'
    assert read(stage/'receipt.json')['status'].startswith('completed')
    history=read(stage/'state_joint/results.json')['candidates'];zero=history[0]
    model=read(stage/'local_credit_model.json');jac=np.array(model['central_constraints'])
    steps=np.array(model['steps']);c0=np.array(zero['measured_inequalities'])
    names=[str(p)+':'+k for p in (14,104,204,304) for k in
        ('incoming','outgoing','reverse','energy','peak','apex_speed','time_old','cruise_old','first_turn','corridor','lateral')]
    names += [str(p)+':'+k for p in (14,104,204,304) for k in guard_names()]
    blocks=read(stage/'optimization_blocks.json')
    tags=[b['tag'] for b in blocks if b.get('method') in ('linear','diagonal_conservative')]
    nearest=min(history[20:],key=lambda c:np.square(np.minimum(c['measured_inequalities'],0.)).sum())
    if nearest['tag'] not in tags:tags.append(nearest['tag'])
    residual=[]
    for tag in tags:
        c=next(c for c in history if c['tag']==tag)
        z=np.asarray(c['coefficients'])/steps;prediction=c0+jac@z
        actual=np.asarray(c['measured_inequalities']);error=actual-prediction
        residual.append(dict(tag=tag,score=c['score'],actual_guard_minimum=float(actual.min()),
            predicted_guard_minimum=float(prediction.min()),
            failures=[dict(name=n,actual=float(a),predicted=float(p),error=float(e))
                for n,a,p,e in zip(names,actual,prediction,error) if a < -1e-8],
            largest_errors=[dict(name=names[i],error=float(error[i])) for i in np.argsort(abs(error))[-5:][::-1]]))
    data=read(stage/'zero_function_baseline.json')
    selected=[r for r in data['baseline']['records'] if r['path'] in (14,104,204,304)]
    md=torch.load(stage/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    _,old=metric_functions(data,selected,stage,md['parent'],time.perf_counter()+30.)
    alternate=[]
    for c in history:
        values=np.concatenate([old(c['records'])]+[guards(r,b) for r,b in zip(c['records'],selected)])
        if values.min()>=-1e-8:alternate.append(dict(tag=c['tag'],score=c['score'],minimum=float(values.min())))
    output=dict(new_physical_rollouts=0,reference_unchanged_for_actual_selection=True,
        tested_counterfactual_reference='ab5 instead of the additional tiny narrow304 update; physics and original64 unchanged',
        alternate_feasible_candidates=alternate,composition_errors=residual,
        selected_recalibration_anchor=nearest['tag'],anchor_coefficients=nearest['coefficients'],
        interpretation='The reference comparison is diagnostic only. Both local prediction '
        'error and actual violations are reported; no candidate is promoted by changing '
        'this audit reference and no inference about PPO bias is made.')
    write_json(stage/'prediction_error_audit.json',output)
    print('ALTERNATE_REFERENCE',[c['tag'] for c in alternate],flush=True)
    for r in residual:print('COMPOSED_ERROR',r['tag'],r['actual_guard_minimum'],r['largest_errors'],flush=True)


if __name__=='__main__':main()
