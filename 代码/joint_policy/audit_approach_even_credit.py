"""Keep two-sided measured response offsets in joint credit, without relaxing guards."""
import numpy as np
from scipy.optimize import linprog
from duration_joint_metrics import guard_names
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_approach_coupled_joint_learning_20260920'
    older=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    history=read(stage/'state_joint/results.json')['candidates'];zero=history[0]
    indexed={c['tag']:c for c in history}
    old_indexed={c['tag']:c for c in read(older/'state_joint/results.json')['candidates']}
    model=read(stage/'joint_approach_credit_models.json');support=read(stage/'support_audit.json')
    old_model=read(older/'joint_phase_credit_models.json')
    paths=support['selected']['paths'];context=np.asarray(support['physical_contexts'])
    owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
    names=[f'{p}:{k}' for p in paths for k in ('incoming','outgoing','reverse','energy','peak','apex_speed','time_old','cruise_old','first_turn','corridor','lateral')]
    names += [f'{p}:{k}' for p in paths for k in guard_names()]
    c0=np.asarray(zero['measured_inequalities']);jac=np.asarray(model['base_constraints'])
    loss=np.asarray(model['base_per_path_loss_gradient']);lat=np.asarray(model['base_lateral_gradient'])
    even=np.zeros_like(jac);lat_even=np.zeros_like(lat)
    for k,pairs,index in [(None,old_model['pairs'],old_indexed),(2,model['pairs'],indexed)]:
        for pair in pairs:
            t=pair['time'] if k is None else pair['time_block'];kernel=pair['kernel'] if k is None else k;h=pair['head']
            plus,minus=index[pair['plus']],index[pair['minus']]
            even[:,t,kernel,h]=np.abs(.5*(np.asarray(plus['measured_inequalities'])+minus['measured_inequalities'])-c0)
            lat_even[:,t,kernel,h]=np.abs(.5*np.asarray([a['shape']['reverse_lateral_significant_mm']+b['shape']['reverse_lateral_significant_mm']-2*z['shape']['reverse_lateral_significant_mm'] for a,b,z in zip(plus['records'],minus['records'],zero['records'])]))
    chosen=next(m for m in model['models'] if m['context_count']==3)
    z=np.asarray(chosen['scaled_proposal']);matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(3)],axis=2).reshape(len(c0),-1)
    prediction=c0+matrix@z;actual=np.asarray(indexed['approach_joint_25']['measured_inequalities'])
    failures=[dict(name=n,predicted=float(p),actual=float(a),error=float(a-p)) for n,p,a in zip(names,prediction,actual) if a < -1e-8]
    failures.sort(key=lambda r:r['error'])
    print('APPROACH_PREDICTION_FAILURES',failures[:8],flush=True)
    results=[]
    # Each absolute-value auxiliary is the effective coefficient on one route,
    # not a separate absolute value for every context weight. This preserves cancellation.
    for count in (3,4,6):
        n=27*count;effective=np.zeros((len(paths)*27,n))
        for owner in range(len(paths)):
            for t in range(3):
                for k in range(3):
                    for h in range(3):
                        row=owner*27+t*9+k*3+h
                        for c in range(count):effective[row,t*count*9+c*9+k*3+h]=context[owner,c,k]
        matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),n)
        gradient=np.sum(np.stack([loss*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(n)
        lateral_gradient=np.sum(np.stack([lat*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(n)
        e=np.zeros((len(c0),len(paths)*27))
        for i,owner in enumerate(owners):e[i,owner*27:(owner+1)*27]=even[i].reshape(-1)
        e_lat=lat_even.reshape(-1)
        a=np.vstack([np.c_[-matrix,e],np.c_[effective,-np.eye(len(effective))],
                     np.c_[-effective,-np.eye(len(effective))],np.r_[lateral_gradient,e_lat][None]])
        b=np.r_[c0,np.zeros(2*len(effective)),-1e-5]
        scaling=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
        solved=linprog(np.r_[gradient,np.zeros(len(effective))],A_ub=a/scaling[:,None],b_ub=b/scaling,
                      bounds=[(-1.,1.)]*n+[(0.,1.)]*len(effective),method='highs',
                      options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,time_limit=20.))
        row=dict(context_count=count,success=bool(solved.success),status=str(solved.message),
                 predicted_loss_change=float(solved.fun) if solved.success else None,
                 scaled_proposal=solved.x[:n].tolist() if solved.success else None,
                 predicted_guard_minimum=float((c0+matrix@solved.x[:n]-e@solved.x[n:]).min()) if solved.success else None,
                 predicted_lateral_change=float(lateral_gradient@solved.x[:n]+e_lat@solved.x[n:]) if solved.success else None)
        results.append(row);print('EVEN_RESPONSE_CREDIT',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
    write_json(stage/'even_response_credit_audit.json',dict(source_sha256=sha(stage/'joint_approach_credit_models.json'),
        source_history_sha256=sha(stage/'state_joint/results.json'),prediction_failures=failures,models=results,
        base_absolute_even_constraints=even.tolist(),base_absolute_even_lateral=lat_even.tolist(),new_physical_runs=0,
        interpretation='Empirical two-sided response envelope, not a proven robust bound; actual joint validation remains required. '
                       'Nonnegative physical predictions and all real acceptance thresholds are unchanged.'))


if __name__=='__main__':main()
