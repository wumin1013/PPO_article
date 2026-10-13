"""Convex local waveform shape constraints by supporting-plane separation."""
import argparse
import time
import numpy as np
import torch
from scipy.optimize import linprog
from waveform_constraint_model import waveform_facets
from solve_current_composition_margin import load_saved_problem
from solve_continuous_sensor_credit import constraint_problem,IMPROVEMENT_NAMES,vector
from constrained_parameter_solver import minimum_metric_step
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--waveforms',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--certify-ray',action='store_true')
    parser.add_argument('--error-audit')
    args=parser.parse_args();torch.set_num_threads(1)
    source=BASE/args.source;wave=BASE/args.waveforms
    old=read(source/'optimizer.json');model=BASE/old['model']
    assert old['parent_sha256']==sha(model)
    assert read(wave/'results.json')['parent_sha256']==sha(model)
    assert (wave/'receipt.json').exists()
    radius=.15;maximum_fraction=.35
    error_source=BASE/args.error_audit if args.error_audit else None
    error_data=read(error_source/'results.json') if error_source else None
    if error_data:assert error_data['parent_sha256']==sha(model)
    stage=register(args.stage,[__file__,source/'gradients.npz',source/'support.npz',
        source/'baselines.json',source/'optimizer.json',source/'receipt.json',
        wave/'results.json',wave/'receipt.json',model,BASE/'waveform_constraint_model.py',
        BASE/'solve_current_composition_margin.py',BASE/'solve_continuous_sensor_credit.py',
        BASE/'constrained_parameter_solver.py',BASE/'waveform_affine_constraint_contract_20260917/receipt.json']+
        ([error_source/'results.json',error_source/'receipt.json'] if error_source else []),
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=300,
            separation_rounds=8,maximum_common_reduction=maximum_fraction,
            certify_ray=args.certify_ray,ray_bisection_iterations=40 if args.certify_ray else 0,
            measured_error_audit=args.error_audit,
            maximum_metric_norm=radius,normalized_action_bound=1.,
            lp_primal_feasibility_tolerance=1e-9,lp_dual_feasibility_tolerance=1e-9),
        'Extend the confirmed moving-extremum operator repair to all current '
        'isolated-corner waveforms. Replace scalar finite-difference max/min and '
        'reverse-sum predictions by extrema and negative-angular-rate sums AFTER '
        'the affine waveform prediction. Supporting-plane separation enforces '
        'each full local convex quantity without fixing active times at the parent. '
        'Maintain scalar current energy, corridor-margin and time constraints; '
        'Keep the full parent moving domain. Where a tiny perturbation crosses '
        'the5mm/s diagnostic-support threshold, hold that sample\'s local waveform '
        'contribution at its measured parent value and disclose its count/value. '
        'Do not drop it and create fictitious nominal improvement. Its unmodeled '
        'response is a limitation, and actual complete-policy checks still inspect '
        'all samples under the unchanged physical metric definition. '
        'every physical evaluation still uses the unchanged complete metrics. '
        'No old-parent gradients or empirical error margins are borrowed. The0.15 '
        'norm is a parameter trust budget, not a proved error bound. After a '
        'validated20% update, allow the local objective to request up to35% at '
        'the SAME action/norm bounds; convex scaling reduces the proposal if '
        'needed. One proposal only, maximum8 separation rounds, then stop to '
        'analyze. Require>=10% predicted common improvement and explicit local '
        'model feasibility before new both-sign complete-policy checks. Neither '
        'mathematical feasibility nor synthetic facet tests prove physical success. '
        'The initial separation run failed the original-unit QP input-feasibility '
        'check despite HiGHS scaled success. Tighten LP primal/dual tolerances to '
        '1e-9 and record both scaled and original-unit residuals; keep the QP '
        'acceptance checks unchanged. This is a numerical recovery, not a new '
        'physical candidate or relaxed constraint. '
        'Optional ray certification checks the entire affine waveform at a '
        'smaller parameter step and computes the common reduction actually '
        'supported there, instead of demanding equality with the LP upper '
        'target. It retains every per-corner bound, the original norm/action '
        'budget and the10% admission gate. This is a numerical feasibility '
        'recovery using the same saved data, not a new physical experiment. '
        'If a current both-sign error audit is supplied, add its nonnegative '
        'one-sided curvature/energy error per norm times the proposed metric '
        'norm to the corresponding upper predictions. This convex allowance '
        'vanishes at zero update and is supported by measured current error, '
        'but is not a guaranteed bound on another direction. Nominal prediction '
        'files remain uninflated for honest subsequent model-error comparison. '
        'No template, physical-limit relaxation, development or sealed access.')
    start=time.perf_counter()
    gradient,ids,names,exclusions,metric,support,baselines=load_saved_problem(source,model)
    base={r['path']:vector(r,r['curvature']) for r in baselines}
    error_rates={}
    if error_data:
        assert error_data['metric_names']==names and error_data['path_ids']==ids
        for p,values in zip(ids,error_data['one_sided_error_per_norm']):
            error_rates[p]={name:float(values[names.index(name)])
                            for name in error_data['proposed_margin_metrics']}
    nominal={};wave_gradient={};valid={};turn={};support_audit=[]
    for r in baselines:
        if r['shape'] is None:continue
        path=r['path']
        with np.load(wave/f'path{path}_waveform.npz') as z:
            nominal[path]=z['nominal'].copy();wave_gradient[path]=z['gradient'].copy()
            common=z['common_valid'].copy();valid[path]=z['nominal_valid'].copy()
        wave_gradient[path][~common]=0.
        frozen=valid[path]&~common
        support_audit.append(dict(path=path,frozen_nominal_samples=frozen.sum(0).tolist(),
            fixed_nominal_reverse_deg=float(np.maximum(-nominal[path][frozen[:,5],5],0.).sum()*.001)))
        turn[path]=r['shape']['turn_deg']
    zero=np.zeros(768)
    def facets(direction,inflate=True):
        result={path:waveform_facets(nominal[path],wave_gradient[path],valid[path],turn[path],direction)
                for path in nominal}
        if error_data:
            norm=float(np.sqrt(max(0.,direction@metric@direction)))
            norm_gradient=metric@direction/norm if norm>1e-15 else np.zeros(768)
            for p,values in result.items():
                for name in error_rates[p]:
                    if name not in values:
                        row=gradient[ids.index(p),names.index(name)]
                        offset=base[p][names.index(name)]
                        values[name]=(float(offset+row@direction),row,offset)
                    if inflate:
                        value,row,offset=values[name];rate=error_rates[p][name]
                        values[name]=(value+rate*norm,row+rate*norm_gradient,offset)
        return result
    initial=facets(zero);wave_names=list(next(iter(initial.values())))
    for path,values in initial.items():
        for name,(value,_,_) in values.items():
            assert abs(value-base[path][names.index(name)])<1e-8,(path,name,'unsupported nominal objective')
    a,b,labels=constraint_problem(gradient,baselines,ids,support,exclusions,
        omit_names=[name for name in wave_names if 'energy' not in name])
    scalar_a=a[:,:768].copy();scalar_b=b.copy()
    aggregate={name:sum(base[p][names.index(name)] for p in nominal) for name in IMPROVEMENT_NAMES}
    audit=[]
    def separate(direction,fraction,force=False):
        current=facets(direction);rows=[];bounds=[];descriptions=[];violations=[]
        for path,values in current.items():
            for name,(value,row,offset) in values.items():
                bound=base[path][names.index(name)]
                excess=value-bound
                if force or excess>1e-8:
                    rows.append(np.r_[row,0.]);bounds.append(bound-offset)
                    descriptions.append(f'{path}_{name}_waveform')
                if excess>1e-8:violations.append(dict(path=path,metric=name,excess=excess))
        for name,total in aggregate.items():
            items=[current[p][name] for p in current]
            value=sum(r[0] for r in items);excess=value-(1.-fraction)*total
            if force or excess>1e-8:
                rows.append(np.r_[sum((r[1] for r in items),np.zeros(768)),total])
                bounds.append(total-sum(r[2] for r in items));descriptions.append(f'total_{name}_waveform')
            if excess>1e-8:violations.append(dict(path='total',metric=name,excess=excess))
        return rows,bounds,descriptions,violations
    extra,bound,description,_=separate(zero,0.,True)
    a=np.vstack([a,extra]);b=np.r_[b,bound];labels+=description
    result=dict(parent_sha256=sha(model),model=str(model.relative_to(BASE)),metric_names=names,path_ids=ids,
        prediction_kind='affine_waveform_convex_shape_operator',metric_radius=radius,
        empirical_error_audit=args.error_audit,empirical_error_is_guaranteed=False,
        eligible_for_small_probe=False,support_audit=support_audit,goal_complete=False)
    for iteration in range(8):
        remaining_wall=300-(time.perf_counter()-start)
        if remaining_wall<30:
            result.update(message='Stopped separation at the reserved completion budget')
            break
        scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
        lp=linprog(np.r_[np.zeros(768),-1.],A_ub=a/scale[:,None],b_ub=b/scale,
            bounds=[(None,None)]*768+[(0.,maximum_fraction)],method='highs',
            options=dict(time_limit=min(45.,remaining_wall-20.),
                primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
        if not lp.success:
            result.update(lp_success=False,message=lp.message);break
        eta=float(lp.x[-1])
        write_json(stage/'latest_lp_residual.json',dict(round=iteration,
            raw_maximum_violation=float(max(0.,(a@lp.x-b).max())),
            scaled_maximum_violation=float(max(0.,((a@lp.x-b)/scale).max()))))
        try:
            direction,projection=minimum_metric_step(metric,a[:,:768],b-a[:,-1]*eta,lp.x[:768],deadline=start+295.)
        except TimeoutError as error:
            result.update(message=str(error))
            break
        extra,bound,description,violations=separate(direction,eta)
        np.savez_compressed(stage/'last_iterate.npz',direction=direction,
            fraction=eta,constraint_matrix=a,constraint_bounds=b)
        write_json(stage/'last_violations.json',dict(round=iteration,violations=violations))
        audit.append(dict(round=iteration,fraction=eta,norm=projection['metric_norm'],
            new_supporting_facets=len(extra),maximum_violation=max((r['excess'] for r in violations),default=0.)))
        write_json(stage/'separation_progress.json',dict(rounds=audit))
        print('WAVEFORM_SEPARATION',audit[-1],flush=True)
        if args.certify_ray:
            # Cache the waveform change along this one direction. Certification
            # changes neither the fitted Jacobian nor the physical trajectory.
            response={p:np.einsum('tcd,d->tc',wave_gradient[p],direction)[:,:,None]
                      for p in nominal}
            scalar_response=scalar_a@direction
            def ray_values(scale):
                values={p:waveform_facets(nominal[p],response[p],valid[p],turn[p],np.array([scale]))
                        for p in nominal}
                if error_data:
                    for p,items in values.items():
                        for name,rate in error_rates[p].items():
                            if name in items:value=items[name][0]
                            else:value=base[p][names.index(name)]+scale*gradient[ids.index(p),names.index(name)]@direction
                            items[name]=(value+rate*scale*projection['metric_norm'],None,None)
                return values
            def ray_feasible(scale):
                if np.max(scale*scalar_response-scalar_b)>1e-9:return False
                return all(value<=base[p][names.index(name)]+1e-9
                           for p,values in ray_values(scale).items()
                           for name,(value,_,_) in values.items())
            lower=0.;upper=min(1.,radius/max(projection['metric_norm'],1e-20))
            assert ray_feasible(0.)
            if ray_feasible(upper):lower=upper
            else:
                for _ in range(40):
                    middle=(lower+upper)/2
                    if ray_feasible(middle):lower=middle
                    else:upper=middle
            actual=ray_values(lower)
            common=min(1.-sum(actual[p][name][0] for p in actual)/total
                       for name,total in aggregate.items() if total>1e-8)
            certified_fraction=max(0.,min(maximum_fraction,common-1e-10))
            audit[-1]['certified_ray']=dict(scale=lower,common_reduction=certified_fraction)
            write_json(stage/'separation_progress.json',dict(rounds=audit))
            print('WAVEFORM_RAY',audit[-1]['certified_ray'],flush=True)
            if certified_fraction>=.1:
                direction*=lower;eta=certified_fraction
                projection=dict(projection,ray_scale=lower,
                    pre_ray_metric_norm=projection['metric_norm'],
                    metric_norm=projection['metric_norm']*lower)
                _,_,_,violations=separate(direction,eta)
                assert not violations
                # Old aggregate cuts encoded a larger, unattained LP target;
                # at the certified fraction the new point satisfies them too.
                assert np.max(a@np.r_[direction,eta]-b)<1e-6
        if violations:
            a=np.vstack([a,extra]);b=np.r_[b,bound];labels+=description
            continue
        factor=min(1.,radius/max(projection['metric_norm'],1e-20))
        direction*=factor;eta*=factor
        _,_,_,remaining=separate(direction,eta)
        assert not remaining,'Convex scaling must retain local waveform feasibility'
        raw_violation=float(max(0.,(a@np.r_[direction,eta]-b).max()))
        assert raw_violation<1e-6
        projection=dict(projection,unscaled_metric_norm=projection['metric_norm'],
            metric_norm=projection['metric_norm']*factor,convex_radius_scale=factor,
            raw_constraint_violation=raw_violation)
        def predict(step):
            changes=np.einsum('pmd,d->pm',gradient,step*direction)
            for path,values in facets(step*direction,inflate=False).items():
                for name,(value,_,_) in values.items():
                    changes[ids.index(path),names.index(name)]=value-base[path][names.index(name)]
            return changes
        scales=np.asarray([-.125,.125,.5,1.])
        np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256),
            predicted_changes=predict(1.),prediction_scales=scales,
            predictions=np.asarray([predict(step) for step in scales]),
            path_id=ids,metric_names=np.asarray(names))
        result.update(lp_success=True,message='Full local waveform constraints satisfied',
            projection=projection,predicted_common_reduction=eta,
            maximum_normalized_action_change=float(abs(support@direction).max()),
            eligible_for_small_probe=bool(eta>=.1),waveform_paths=list(nominal))
        break
    result['separation_rounds']=audit
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_bounded_waveform_policy_credit_solve',
        eligible_for_small_probe=result['eligible_for_small_probe'],new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start,
        wall_budget_pass=time.perf_counter()-start<=300,goal_complete=False))
    print('WAVEFORM_POLICY_SOLVE',result,flush=True)


if __name__=='__main__':main()
