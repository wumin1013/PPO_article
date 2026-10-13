"""Re-express a quantized duration guard without changing its feasible trajectories."""
import numpy as np
from heading_duration_credit import learning_guards
from duration_joint_metrics import guard_names
from audit_context_restoration_scaling import solve_correction
from study_io import BASE, write_json, sha
from search_complete_return import read


def main():
    stage=BASE/'canonical_gate_band_geometry_secant_recovery_learning_20260920'
    main_stage=BASE/'canonical_gate_band_conditioned_main_turn_joint_learning_20260920'
    old_stage=BASE/'canonical_gate_band_approach_conditioned_recovery_learning_20260920'
    history=read(stage/'state_joint/results.json')['candidates'];baseline=history[0]['records']
    n=len(baseline);offset=guard_names().index('shape.heading_overshoot_gt05_duration_s')
    rows=11*n+len(guard_names())*np.arange(n)+offset
    current=min(history[2:],key=lambda c:np.linalg.norm(np.minimum(c['measured_inequalities'],0.)))
    secant=read(stage/'final_secant_credit.json');jac=np.asarray(secant['jacobian']);gradient=np.asarray(secant['loss_gradient'])
    context=np.asarray(read(main_stage/'geometry_context_capacity.json')['context'])[:,:7]
    main_model=read(main_stage/'joint_main_turn_credit_models.json')
    old_model=read(old_stage/'conditioned_credit_models.json')
    old_index={c['tag']:c for c in read(old_stage/'state_joint/results.json')['candidates']}
    main_index={c['tag']:c for c in read(main_stage/'state_joint/results.json')['candidates']}
    heading_jac=np.zeros((n,3,4,3));checked=0
    def converted(c):
        nonlocal checked
        values=learning_guards(c['records'],baseline,c['measured_inequalities']);checked+=len(c['records'])
        return values
    for pairs,index,is_old in ((old_model['pairs'],old_index,True),(main_model['pairs'],main_index,False)):
        for pair in pairs:
            k=pair['kernel'] if is_old else pair['local_kernel'];t=pair['time'];h=pair['head']
            plus,minus=converted(index[pair['plus']]),converted(index[pair['minus']])
            heading_jac[:,t,k,h]=.5*(plus[rows]-minus[rows])
    heading_jac=np.stack([heading_jac*context[:,None,c,:,None] for c in range(7)],axis=2).reshape(n,-1)
    steps=np.repeat(np.asarray(main_model['steps'])[:,None],7,axis=1).reshape(-1)
    def coordinates(c):
        weights=np.asarray(c['coefficients']).reshape(3,7,5,3)
        return np.stack([weights[:,:,k] for k in (0,1,3,4)],axis=2).reshape(-1)/steps
    base=history[2];x=np.asarray([coordinates(c)-coordinates(base) for c in history[3:]])
    base_values=converted(base);y=np.asarray([converted(c)[rows]-base_values[rows] for c in history[3:]])
    correction=np.linalg.lstsq(x,y-x@heading_jac.T,rcond=1e-8)[0]
    heading_jac+=correction.T;jac[rows]=heading_jac
    converted_current=converted(current);models=[]
    for radius in (.25,1.,4.):
        proposal=solve_correction(jac,gradient,converted_current,history[1]['score']-current['score'],radius=radius)
        models.append(dict(radius=radius,**proposal))
        print('SIGNED_HEADING_CREDIT',radius,{k:v for k,v in proposal.items() if k!='scaled_parameter_step'},flush=True)
    write_json(stage/'signed_heading_credit_audit.json',dict(anchor=current['tag'],jacobian=jac.tolist(),
        loss_gradient=gradient.tolist(),models=models,verified_record_comparisons=checked,
        multi_secant_rank=int(np.linalg.matrix_rank(x)),heading_secant_fit_max_error=float(abs(x@heading_jac.T-y).max()),
        actual_count_acceptance_unchanged=True,source_sha256=sha(stage/'state_joint/results.json'),new_physical_runs=0,
        exact_equivalence='count(heading > limit) <= N iff the (N+1)-th largest heading <= limit. '
                          'Only the optimizer representation changes; final selection still uses every original count/physical guard.'))
    print('SIGNED_HEADING_EQUIVALENCE',checked,'anchor',current['tag'],'rank',np.linalg.matrix_rank(x),flush=True)


if __name__=='__main__':main()
