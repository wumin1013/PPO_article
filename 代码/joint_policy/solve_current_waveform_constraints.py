"""Constrain moving extrema through current saved waveform sensitivities."""
import time
import numpy as np
import torch
from scipy.optimize import linprog
from audit_current_waveform_credit import signals,CHANNELS
from solve_current_composition_margin import load_saved_problem
from solve_continuous_sensor_credit import constraint_problem
from constrained_parameter_solver import minimum_metric_step
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_current_parameter_solve_20260917'
    margin_source=BASE/'canonical_gate_measured_margin_solve_20260917'
    wave=BASE/'canonical_gate_waveform_credit_audit_20260917'
    small=BASE/'canonical_gate_margin_composition_20260917'
    large=BASE/'canonical_gate_learned_shape_update_20260917'
    old=read(margin_source/'optimizer.json');model=BASE/old['model'];radius=.15
    assert old['parent_sha256']==sha(model)
    stage=register('canonical_gate_waveform_constraint_solve_20260917',
        [__file__,source/'gradients.npz',source/'support.npz',source/'baselines.json',
         margin_source/'model_error.json',margin_source/'direction.npz',
         wave/'receipt.json',wave/'results.json',small/'results.json',large/'results.json',
         model,BASE/'solve_current_composition_margin.py',BASE/'solve_continuous_sensor_credit.py',
         BASE/'audit_current_waveform_credit.py',BASE/'constrained_parameter_solver.py']+
        [wave/f'path{path}_waveform.npz' for path in (2,3,13)],
        dict(new_physics_ticks=0,new_actor_decisions=0,wall_s=240,
            maximum_metric_norm=radius,maximum_common_reduction=.2,
            waveform_paths=[2,3,13],error_scales=[-.125,.125,.5,1.]),
        'The saved waveform audit shows scalar max/min credit misses movement of '
        'active extrema. Retain the current shared-parameter scalar constraints '
        'and small-composition curvature margins. Add every supported1ms positive '
        'and negative curvature-envelope inequality on paths2/3, plus every '
        'incoming half-plane inequality on path13. Both rawXY and v/a curvature '
        'must stay within the measured parent peaks in the local waveform model. '
        'Also account for measured pointwise prediction errors using the same '
        'parent/direction at four already completed scales and norm radius0.15. '
        'This combines the envelope operator and newly available empirical error '
        'coverage; do not attribute subsequent improvement uniquely to either. '
        'The waveform audit separately tests the operator. These margins are '
        'not guaranteed for new directions. A eligible proposal must receive '
        'fresh both-sign complete29-route checks. No physical-limit, objective, '
        'model architecture or reward change; no online trajectory rule, new '
        'suffix collection, development or sealed data.')
    start=time.perf_counter()
    gradient,ids,names,exclusions,metric,support,baselines=load_saved_problem(source,model)
    margins=np.asarray(read(margin_source/'model_error.json')['constraint_margin'])
    a,b,labels=constraint_problem(gradient,baselines,ids,support,exclusions,margins)
    with np.load(margin_source/'direction.npz') as z:measured_direction=z['direction'].ravel().copy()
    norm=float(np.sqrt(measured_direction@metric@measured_direction))
    variants=read(small/'results.json')['candidates']+read(large/'results.json')['candidates']
    extra_a=[];extra_b=[];extra_labels=[];reports=[]
    for path in (2,3,13):
        with np.load(wave/f'path{path}_waveform.npz') as z:
            ref=z['reference'].copy();nominal=z['nominal'].copy()
            valid=z['common_valid'].copy();wave_gradient=z['gradient'].copy()
        delta=np.einsum('tcd,d->tc',wave_gradient,measured_direction)
        upper_error=np.zeros(nominal.shape);lower_error=np.zeros(nominal.shape)
        for candidate in variants:
            j=next(i for i,r in enumerate(candidate['results']) if r['original_path']==path)
            assert candidate['results'][j]['success']
            with np.load(BASE/candidate['traces'][j]) as z:actual,actual_valid=signals(ref,z['rows'],len(nominal))
            valid&=actual_valid
            mismatch=(actual-nominal-candidate['step']*delta)/(abs(candidate['step'])*norm)
            upper_error=np.maximum(upper_error,mismatch)
            lower_error=np.maximum(lower_error,-mismatch)
        channels=(3,4) if path in (2,3) else (0,)
        baseline=next(r for r in baselines if r['path']==path)
        before=len(extra_b)
        for channel in channels:
            indices=np.flatnonzero(valid[:,channel])
            if channel==0:
                floor=-baseline['shape']['in_edge_outward_mm']
                for tick in indices:
                    extra_a.append(np.r_[-wave_gradient[tick,channel],0.])
                    extra_b.append(nominal[tick,channel]-floor-radius*lower_error[tick,channel])
                    extra_labels.append(f'{path}_inward_tick{tick}')
            else:
                metric_name=('full_xy_peak_curvature_mm_inv' if channel==3 else 'full_physical_peak_curvature_mm_inv')
                bound=baseline['curvature'][metric_name]
                for tick in indices:
                    extra_a.extend([np.r_[wave_gradient[tick,channel],0.],np.r_[-wave_gradient[tick,channel],0.]])
                    extra_b.extend([bound-nominal[tick,channel]-radius*upper_error[tick,channel],
                                    bound+nominal[tick,channel]-radius*lower_error[tick,channel]])
                    extra_labels.extend([f'{path}_{CHANNELS[channel]}_upper_tick{tick}',
                                         f'{path}_{CHANNELS[channel]}_lower_tick{tick}'])
        reports.append(dict(path=path,channels=[CHANNELS[c] for c in channels],
            constraints=len(extra_b)-before,supported_samples=valid.sum(axis=0).tolist()))
    a=np.vstack([a,np.asarray(extra_a)]);b=np.r_[b,extra_b];labels+=extra_labels
    scale=np.maximum(np.linalg.norm(a,axis=1),1e-12)
    lp=linprog(np.r_[np.zeros(768),-1.],A_ub=a/scale[:,None],b_ub=b/scale,
        bounds=[(None,None)]*768+[(0.,.2)],method='highs',options=dict(time_limit=90.))
    result=dict(lp_success=bool(lp.success),message=lp.message,parent_sha256=sha(model),
        model=str(model.relative_to(BASE)),metric_names=names,path_ids=ids,
        eligible_for_small_probe=False,empirical_radius=radius,
        waveform_reports=reports,goal_complete=False)
    if lp.success:
        eta=float(lp.x[-1])
        direction,projection=minimum_metric_step(metric,a[:,:768],b-a[:,-1]*eta,lp.x[:768])
        result.update(predicted_common_reduction=eta,projection=projection,
            maximum_normalized_action_change=float(abs(support@direction).max()),
            eligible_for_small_probe=bool(eta>=.1 and projection['metric_norm']<=radius))
        np.savez_compressed(stage/'direction.npz',direction=direction.reshape(3,256),
            predicted_changes=np.einsum('pmd,d->pm',gradient,direction),path_id=ids,metric_names=np.asarray(names))
        write_json(stage/'binding_constraints.json',dict(nonzero_duals=[
            dict(constraint=labels[i],unscaled_dual=float(lp.ineqlin.marginals[i]/scale[i]))
            for i in range(len(labels)) if abs(lp.ineqlin.marginals[i])>1e-10]))
    assert time.perf_counter()-start<240
    write_json(stage/'optimizer.json',result)
    receipt(stage,dict(status='completed_current_waveform_constraint_solve',
        eligible_for_small_probe=result['eligible_for_small_probe'],new_physics_ticks=0,
        new_actor_decisions=0,wall_s=time.perf_counter()-start))
    print('WAVEFORM_CONSTRAINT_SOLVE',result,flush=True)


if __name__=='__main__':main()
