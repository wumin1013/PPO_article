"""Restore local feasibility directly from a saved separation iterate, no LP replay."""
import argparse
import time
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.optimize import minimize,LinearConstraint
from waveform_constraint_model import waveform_facets
from solve_current_composition_margin import load_saved_problem
from solve_continuous_sensor_credit import constraint_problem,IMPROVEMENT_NAMES,vector
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('source','waveforms','error-audit','resume','stage'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--fraction',type=float,default=.34)
    parser.add_argument('--energy',help='Optional current-parent signed-density energy operators')
    args=parser.parse_args();torch.set_num_threads(1)
    source=BASE/args.source;wave=BASE/args.waveforms;resume=BASE/args.resume
    error=read(BASE/args.error_audit/'results.json');old=read(resume/'optimizer.json')
    model=BASE/old['model'];assert old['parent_sha256']==error['parent_sha256']==sha(model)
    assert .1<=args.fraction<=.35
    warm_start=resume/'last_iterate.npz'
    if not warm_start.exists():warm_start=resume/'direction.npz'
    energy={};energy_sources=[]
    if args.energy:
        energy_stage=BASE/args.energy;energy_data=read(energy_stage/'results.json')
        assert energy_data['parent_sha256']==sha(model)
        energy_sources=[energy_stage/'results.json',energy_stage/'receipt.json']
        for r in energy_data['records']:
            file=energy_stage/r['file'];energy_sources.append(file)
            with np.load(file) as z:
                energy[r['path']]=(z['constant'].copy(),z['linear'].copy(),z['quadratic'].copy(),z['metric_names'].tolist())
    stage=register(args.stage,[__file__,source/'optimizer.json',source/'gradients.npz',
        source/'support.npz',source/'baselines.json',source/'receipt.json',wave/'results.json',
        wave/'receipt.json',BASE/args.error_audit/'results.json',BASE/args.error_audit/'receipt.json',
        warm_start,resume/'optimizer.json',resume/'receipt.json',
        model,BASE/'waveform_constraint_model.py',BASE/'solve_continuous_sensor_credit.py']+
        [BASE/d['file'] for d in error.get('waveform_error_descriptors',[])]+energy_sources,
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=120,
            full_constraint_rounds=3,optimizer_iterations=60,common_reduction=args.fraction,
            maximum_metric_norm=.15,relative_curvature_interior_reserve=1e-6,energy_operator_source=args.energy),
        'Start from the explicitly bound current-parent parameter iterate; '
        'do not rerun its LP sequence or recollect trajectories. The explicit '
        'fraction and warm-start source in this registration define the solve. '
        'Find the closest feasible point '
        'using exact convex local waveform values and subgradients in SLSQP. '
        'Request the registered common reduction and a '
        '1e-6 relative INTERIOR reserve on positive curvature/energy bounds. '
        'Keep the10% physical-probe gate, original action/radius bounds and all '
        'per-corner bounds. Check ALL nonlinear and omitted linear constraints '
        'after optimization; no tolerance or physical-limit relaxation. Norm '
        'error allowance (linear and optional quadratic in norm) remains '
        'empirical. For zero excursion bounds, optional pointwise allowances '
        'are applied BEFORE max/positive-part so an isotropic output margin '
        'does not create artificial infeasibility. Fresh both-sign complete '
        'checks are required. When energy_operator_source is set, BOTH existing '
        'energies use the current signed-density quadratic operators instead of '
        'scalar first-order predictions; all physical acceptance metrics remain '
        'identical. Stop after3 active-set rounds or120seconds.')
    start=time.perf_counter();deadline=start+115
    gradient,ids,names,exclusions,metric,support,baselines=load_saved_problem(source,model)
    base={r['path']:vector(r,r['curvature']) for r in baselines}
    for p,(constant,_,_,energy_names) in energy.items():
        assert max(abs(constant[c]-base[p][names.index(n)]) for c,n in enumerate(energy_names))<1e-10
    with np.load(warm_start) as z:initial=z['direction'].ravel().copy()
    nominal={};jac={};valid={};turn={}
    for r in baselines:
        p=r['path']
        if r['shape'] is None:continue
        with np.load(wave/f'path{p}_waveform.npz') as z:
            nominal[p]=z['nominal'].copy();jac[p]=z['gradient'].copy()
            valid[p]=z['nominal_valid'].copy();jac[p][~z['common_valid']]=0.
        turn[p]=r['shape']['turn_deg']
    rates={p:dict(zip(names,v)) for p,v in zip(error['path_ids'],error['one_sided_error_per_norm'])}
    quadratic={p:dict(zip(names,v)) for p,v in zip(error['path_ids'],
        error.get('quadratic_error_per_norm_squared',np.zeros((len(ids),len(names))))) }
    error_names=error['proposed_margin_metrics']
    smooth=[n for n in error_names if 'energy' in n or 'curvature' in n]
    field_errors=[]
    for d in error.get('waveform_error_descriptors',[]):
        assert sha(BASE/d['file'])==d['sha256']
        with np.load(BASE/d['file']) as z:field_errors.append((d,z['linear'].copy(),z['quadratic'].copy()))
    fraction=args.fraction;radius=.15
    def facets(x,inflate=True):
        norm=float(np.sqrt(max(0.,x@metric@x)));ng=metric@x/max(norm,1e-30)
        values={p:waveform_facets(nominal[p],jac[p],valid[p],turn[p],x) for p in nominal}
        for p,(constant,energy_linear,energy_quadratic,energy_names) in energy.items():
            qx=energy_quadratic@x;square=qx@x
            for c,name in enumerate(energy_names):
                values[p][name]=(float(constant[c]+energy_linear[c]@x+square[c]),
                    energy_linear[c]+2*qx[c],float(constant[c]-square[c]))
        if inflate:
            for d,e,q in field_errors:
                path=d['path'];channel=d['channel'];sign=d['sign'];offset=d['offset']
                value=sign*(nominal[path][:,channel]+jac[path][:,channel]@x)+offset+e*norm+q*norm**2
                ticks=np.flatnonzero(valid[path][:,channel]);tick=int(ticks[value[ticks].argmax()])
                if value[tick]<=0.:values[path][d['metric']]=(0.,np.zeros(768),0.)
                else:values[path][d['metric']]=(float(value[tick]),
                    sign*jac[path][tick,channel]+e[tick]*ng+2*q[tick]*(metric@x),
                    float(sign*nominal[path][tick,channel]+offset-q[tick]*norm**2))
        for p,items in values.items():
            for name in error_names:
                if name not in items:
                    g=gradient[ids.index(p),names.index(name)];b=base[p][names.index(name)]
                    items[name]=(float(b+g@x),g,b)
                if inflate:
                    v,g,b=items[name];e=rates[p][name];q=quadratic[p][name]
                    items[name]=(v+e*norm+q*norm**2,g+e*ng+2*q*(metric@x),b-q*norm**2)
        return values
    def nonlinear(x):
        current=facets(x);residual=[];rows=[];labels=[]
        for p,items in current.items():
            for name,(value,row,_) in items.items():
                bound=base[p][names.index(name)]
                reserve=1e-6*bound if name in smooth else 0.
                residual.append(value-bound+reserve);rows.append(row);labels.append(f'{p}_{name}')
        for name in IMPROVEMENT_NAMES:
            residual.append(sum(current[p][name][0]-(1.-fraction)*base[p][names.index(name)] for p in current))
            rows.append(sum((current[p][name][1] for p in current),np.zeros(768)))
            labels.append('total_'+name)
        residual.append(x@metric@x-radius**2);rows.append(2*metric@x);labels.append('metric_radius')
        return np.asarray(residual),np.asarray(rows),labels
    wave_names=list(waveform_facets(nominal[0],jac[0],valid[0],turn[0],initial))
    if energy:wave_names+=list(next(iter(energy.values()))[3])
    a,b,labels=constraint_problem(gradient,baselines,ids,support,exclusions,omit_names=wave_names)
    a=a[:,:768];chol=np.linalg.cholesky(metric)
    c=solve_triangular(chol,a.T,lower=True,check_finite=False).T
    scales=np.maximum(np.linalg.norm(c,axis=1),1e-12);c/=scales[:,None];d=b/scales
    y0=chol.T@initial;y=y0.copy()
    nr,nj,_=nonlinear(initial)
    ns=np.maximum(np.linalg.norm(solve_triangular(chol,nj.T,lower=True,check_finite=False).T,axis=1),1e-3)
    cache={}
    def cached(value):
        if time.perf_counter()>deadline:raise TimeoutError('Direct feasibility deadline')
        if 'y' not in cache or not np.array_equal(value,cache['y']):
            x=solve_triangular(chol.T,value,lower=False,check_finite=False)
            res,rows,descriptions=nonlinear(x)
            cache.update(y=value.copy(),x=x,res=res,rows=rows,labels=descriptions)
        return cache
    constraints=dict(type='ineq',fun=lambda y:-cached(y)['res']/ns,
        jac=lambda y:-solve_triangular(chol,cached(y)['rows'].T,lower=True,check_finite=False).T/ns[:,None])
    active=set(np.flatnonzero(d-c@y<1e-5).tolist());audit=[];passed=False;message=''
    try:
        for iteration in range(3):
            selected=np.asarray(sorted(active),int)
            optimizer_constraints=[constraints]
            if len(selected):optimizer_constraints.append(LinearConstraint(c[selected],-np.inf,d[selected]))
            fit=minimize(lambda y:.5*np.sum((y-y0)**2),y,jac=lambda y:y-y0,
                constraints=optimizer_constraints,
                method='SLSQP',options=dict(ftol=1e-12,maxiter=60))
            y=fit.x;x=solve_triangular(chol.T,y,lower=False,check_finite=False)
            res,_,description=nonlinear(x);linear=a@x-b
            passed=bool(res.max()<=1e-8 and linear.max()<=1e-6 and np.linalg.norm(y)<=radius+1e-10)
            audit.append(dict(round=iteration,success=bool(fit.success),message=str(fit.message),
                iterations=int(fit.nit),nonlinear_violation=float(max(0.,res.max())),
                linear_violation=float(max(0.,linear.max())),passed=passed))
            np.savez_compressed(stage/'last_iterate.npz',direction=x)
            write_json(stage/'progress.json',dict(rounds=audit,violations=[
                dict(constraint=label,excess=float(r)) for label,r in zip(description,res) if r>1e-8]))
            print('DIRECT_WAVEFORM_FEASIBILITY',audit[-1],flush=True)
            if passed:break
            missing=np.flatnonzero(c@y-d>1e-7)
            if not len(missing):break
            active.update(missing.tolist())
    except TimeoutError as exc:message=str(exc)
    result=dict(parent_sha256=sha(model),model=str(model.relative_to(BASE)),metric_names=names,
        path_ids=ids,metric_radius=radius,eligible_for_small_probe=passed,
        empirical_error_audit=args.error_audit,empirical_error_is_guaranteed=False,
        prediction_kind='affine_waveform_convex_shape_operator',rounds=audit,message=message,goal_complete=False)
    result['energy_operator_source']=args.energy
    if passed:
        scales=np.array([-.125,.125,.5,1.])
        def predict(step):
            changes=np.einsum('pmd,d->pm',gradient,step*x)
            for p,items in facets(step*x,inflate=False).items():
                for name,(value,_,_) in items.items():changes[ids.index(p),names.index(name)]=value-base[p][names.index(name)]
            return changes
        result.update(predicted_common_reduction=fraction,
            maximum_normalized_action_change=float(abs(support@x).max()),
            projection=dict(metric_norm=float(np.linalg.norm(y)),
                restoration_distance=float(np.linalg.norm(y-y0)),raw_constraint_violation=float(max(0.,linear.max()))))
        np.savez_compressed(stage/'direction.npz',direction=x.reshape(3,256),
            predicted_changes=predict(1.),prediction_scales=scales,
            predictions=np.asarray([predict(s) for s in scales]),path_id=ids,metric_names=np.asarray(names))
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_direct_waveform_feasibility_restoration',
        eligible_for_small_probe=passed,new_actor_decisions=0,new_physics_ticks=0,
        wall_s=time.perf_counter()-start,wall_budget_pass=time.perf_counter()-start<=120,goal_complete=False))


if __name__=='__main__':main()
