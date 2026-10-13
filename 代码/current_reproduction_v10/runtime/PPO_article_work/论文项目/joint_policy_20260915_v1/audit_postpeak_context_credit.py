"""Compare shared physical-state credit models using the same measured responses."""
import numpy as np
from scipy.optimize import linprog
from duration_joint_metrics import penalty
from learn_complete_joint_teachers import teacher_score
from observable_memory_joint_policy import dynamic_context
from study_io import BASE,write_json
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_shared_postpeak_memory_learning_20260920'
    assert read(stage/'receipt.json')['status'].startswith('completed')
    model=read(stage/'local_credit_model.json');history=read(stage/'state_joint/results.json')['candidates']
    indexed={c['tag']:c for c in history};zero=history[0]
    data=read(stage/'baseline.json');paths=[14,104,204,304]
    baselines=[next(r for r in data['baseline']['records'] if r['path']==p) for p in paths]
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==p) for p in paths]
    with np.load(stage/'replayed_states.npz') as z:
        observations=np.array([z['observations'][(z['path_ids']==p)&(z['decisions']==18)].squeeze(0) for p in paths])
    raw=dynamic_context(observations);mean=raw.mean(0);scale=np.maximum(raw.std(0),[5.,100.])
    normalized=np.tanh((raw-mean)/scale)
    jac=np.array(model['central_constraints']);c0=np.array(zero['measured_inequalities'])
    even=np.array(model['absolute_even_constraints']);gradients=[]
    for pair in model['pairs']:
        plus,minus=indexed[pair['plus']],indexed[pair['minus']]
        def losses(c):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r)
            for r,b,o in zip(c['records'],baselines,retained)])
        gradients.append(.5*(losses(plus)-losses(minus)))
    gradients=np.array(gradients).T
    owners=np.r_[np.repeat(np.arange(4),11),np.repeat(np.arange(4),16)]
    rows=[]
    for count in (1,2,3):
        context=np.column_stack([np.ones(4),normalized[:,:count-1]])
        # Parameter order: memory block, context basis, action.
        j=np.concatenate([jac.reshape(-1,3,1,3)*context[owners,None,k:k+1,None] for k in range(count)],axis=2).reshape(len(c0),-1)
        e=np.concatenate([even.reshape(-1,3,1,3)*context[owners,None,k:k+1,None]**2 for k in range(count)],axis=2).reshape(len(c0),-1)
        g=np.sum(gradients.reshape(4,3,1,3)*context[:,None,:,None],axis=0).reshape(-1)
        solved=linprog(g,A_ub=-j,b_ub=c0,bounds=[(-1.,1.)]*len(g),method='highs')
        rows.append(dict(context_count=count,parameters=len(g),feasible=bool(solved.success),status=str(solved.message),
            predicted_loss_change=float(solved.fun) if solved.success else None,
            proposed_scaled_parameters=solved.x.tolist() if solved.success else None,
            central_constraints=j.tolist(),absolute_even_constraints=e.tolist(),central_loss=g.tolist()))
    # Per-route controls distinguish shared-function conflict from the measured route-local model.
    independent=[]
    for i,p in enumerate(paths):
        mask=owners==i;solved=linprog(gradients[i],A_ub=-jac[mask],b_ub=c0[mask],bounds=[(-1.,1.)]*9,method='highs')
        independent.append(dict(path=p,feasible=bool(solved.success),predicted_loss_change=float(solved.fun) if solved.success else None))
    useful=[r for r in rows if r['feasible'] and r['predicted_loss_change']<-.001]
    choice=min(useful,key=lambda r:r['parameters']) if useful else None
    write_json(stage/'context_credit_audit.json',dict(paths=paths,raw_context=raw.tolist(),context_mean=mean.tolist(),
        context_scale=scale.tolist(),normalized_context=normalized.tolist(),models=rows,
        independent_route_models=independent,selected_context_count=choice['context_count'] if choice else None,
        actual_response_vectors=len(model['pairs'])*2,new_physical_rollouts=0,
        interpretation='First-order chain rule from the SAME full-route symmetric responses. '
        'Speed and tangential acceleration are reflection-even physical observations. '
        'A feasible model proposal requires independent physical validation; it is not '
        'proof of global capacity, generalization or an estimator property.'))
    print('POSTPEAK_CONTEXT',[(r['parameters'],r['feasible'],r['predicted_loss_change']) for r in rows],
          'independent',independent,'selected',choice['context_count'] if choice else None,flush=True)


if __name__=='__main__':main()
