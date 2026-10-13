"""Separate ordinary model errors, geometric linearization and actual brake tails."""
import argparse
import time
import numpy as np
import torch
from transition_waveform_model import chain_responses, signal_jacobians
from waveform_constraint_model import waveform_facets
from energy_waveform_credit import energy_signals
from prepare_current_waveform_credit import wave_signals
from canonical_gate_state import checkpoint_environment_factory
from event_adaptive_terminal_prediction import predict_terminal
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def metrics(ref, rows):
    value, valid = wave_signals(ref, rows)
    u,v = np.diff(ref,axis=0)
    turn=float(np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)))
    result={k:v[0] for k,v in waveform_facets(value,np.zeros((*value.shape,1)),valid,turn,np.zeros(1)).items()}
    energy,_=energy_signals(ref,rows)
    result.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),
                      np.sum(energy**2,axis=0).tolist()))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--proposal',default='canonical_gate_band_ordinary_joint_proposal_20260918')
    parser.add_argument('--feedback',default='canonical_gate_band_ordinary_joint_feedback_20260918')
    parser.add_argument('--stage',default='canonical_gate_band_ordinary_joint_error_sources_20260918')
    args=parser.parse_args()
    source=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    proposal=BASE/args.proposal
    feedback=BASE/args.feedback
    data,fit,actual=[read(p/'results.json') for p in (source,proposal,feedback)]
    parent=BASE/data['parent_model']
    assert sha(parent)==fit['parent_sha256']==actual['parent_sha256']
    stage=register(args.stage,
        [__file__,parent,source/'results.json',source/'receipt.json',proposal/'results.json',
         proposal/'direction.npz',feedback/'results.json',feedback/'receipt.json',
         BASE/'transition_waveform_model.py',BASE/'energy_waveform_credit.py',
         BASE/'event_adaptive_terminal_prediction.py'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=120,
             saved_corner_responses=16*len(actual['candidates'])),
        'Use every completed399a proposal in the explicitly bound feedback, including every regression. '
        'Split each full-path regression into actual ordinary-prefix change and '
        'terminal-tail contribution. Compare ordinary affine waveforms, geometry '
        'recomputed on affine XY/v/a, and actual ordinary physics. Compare exact '
        'hybrid tail reconstruction as a separate predictor. Fit nonnegative '
        'empirical ordinary scalar-error/radius-squared coefficients for a future '
        'changed proposal, explicitly not prospective bounds or acceptance. '
        'No new physical samples, repeats of completed routes, changed limits '
        'or retroactive adoption.')
    started=time.perf_counter();records=[];coefficients=[]
    costs=dict(new_actor_decisions=0,new_physics_ticks=0,terminal_planner_queries=0)
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    with np.load(proposal/'direction.npz') as z:direction=z['direction'].copy().ravel()
    radius=fit['metric_norm']
    def budget():assert time.perf_counter()-started<115
    for item in data['records']:
        if item['path']>=16:continue
        with np.load(source/item['file']) as z:local={k:z[k].copy() for k in z.files}
        ref,rows=local['reference'],local['rows']
        stop=int(np.flatnonzero(rows[:,12])[0]);ordinary=rows[:stop]
        response=chain_responses(local,direction[None,:])[:,:,0]
        nominal,valid,gradient,density,_,dg=signal_jacobians(ref,ordinary,response[:stop,:,None])
        u,v=np.diff(ref,axis=0);turn=float(np.degrees(np.arctan2(u[0]*v[1]-u[1]*v[0],u@v)))
        base_ordinary=metrics(ref,ordinary);base_full=metrics(ref,rows)
        rates={key:0. for key in base_ordinary}
        for candidate in actual['candidates']:
            budget();scale=candidate['step']
            j=next(j for j,r in enumerate(candidate['results']) if r['original_path']==item['path'])
            with np.load(BASE/candidate['traces'][j]) as z:truth=z['rows'].copy()
            assert candidate['results'][j]['success']
            actual_stop=int(np.flatnonzero(truth[:,12])[0])
            actual_ordinary=metrics(ref,truth[:actual_stop]);actual_full=metrics(ref,truth)
            predicted={k:v[0] for k,v in waveform_facets(nominal,gradient,valid,turn,np.array([scale])).items()}
            energy=np.sum((density+scale*dg[:,:,0])**2,axis=0)
            predicted.update(zip(('full_xy_normalized_energy','full_physical_normalized_energy'),energy.tolist()))
            affine=rows[:,:6]+scale*response
            geometry_rows=ordinary.copy();geometry_rows[:,:6]=affine[:stop]
            geometry=metrics(ref,geometry_rows)
            hybrid=predict_terminal(ref,affine,stop,factory,costs,budget)
            hybrid_metrics=metrics(ref,hybrid['rows']) if hybrid['supported'] else None
            regressions=[]
            for key in base_full:
                if actual_full[key]>base_full[key]+1e-6:
                    regressions.append(dict(metric=key,full_increase=actual_full[key]-base_full[key],
                        ordinary_increase=actual_ordinary[key]-base_ordinary[key],
                        actual_full_minus_ordinary=actual_full[key]-actual_ordinary[key],
                        parent_full_minus_ordinary=base_full[key]-base_ordinary[key],
                        ordinary_model_error=actual_ordinary[key]-predicted[key],
                        geometric_linearization_error=geometry[key]-predicted[key],
                        remaining_dynamics_or_support_error=actual_ordinary[key]-geometry[key],
                        hybrid_full_error=actual_full[key]-hybrid_metrics[key] if hybrid_metrics else None,
                        already_worse_in_ordinary_motion=actual_ordinary[key]>base_ordinary[key]+1e-6))
                rates[key]=max(rates[key],(actual_ordinary[key]-predicted[key])/(scale*radius)**2)
            r=dict(path=item['path'],scale=scale,actual_initiation_tick=actual_stop,
                parent_initiation_tick=stop,hybrid_supported=hybrid['supported'],
                hybrid_initiation_tick=hybrid.get('initiation_tick'),
                actual_braking_steps=len(truth)-actual_stop-23,hybrid_braking_steps=hybrid.get('braking_steps'),
                baseline_ordinary=base_ordinary,baseline_full=base_full,
                actual_ordinary=actual_ordinary,actual_full=actual_full,
                predicted_ordinary=predicted,nonlinear_geometry_on_affine_state=geometry,
                predicted_hybrid_full=hybrid_metrics,regressions=regressions)
            records.append(r)
        coefficients.extend(dict(path=item['path'],metric=key,rate=max(0.,value)) for key,value in rates.items())
    largest=[dict(path=r['path'],**v) for r in records if r['scale']==1. for v in r['regressions']]
    summary=dict(full_scale_regression_count=len(largest),
        full_scale_regressions_already_in_ordinary=sum(r['already_worse_in_ordinary_motion'] for r in largest),
        full_scale_regressions=largest,
        actual_completed_screens=len(actual['candidates']),retained_policy_unchanged=True,
        interpretation='Measured optimism of ordinary response must be corrected; exact terminal replanning alone cannot certify shape. Tail effects are reported separately, never blamed without evidence.')
    write_json(stage/'results.json',dict(parent_sha256=sha(parent),records=records,summary=summary,costs=costs,goal_complete=False))
    write_json(stage/'error_model.json',dict(parent_sha256=sha(parent),coefficients=coefficients,
        calibration_scales=[r['step'] for r in actual['candidates']],calibration_direction_sha256=sha(proposal/'direction.npz'),
        physical_metric=read(BASE/'canonical_gate_band_current_hold_geometry_20260918/results.json')['parent_sha256'],
        errors_apply_to='ordinary-prefix scalar facets and energies only',guaranteed_bound=False))
    receipt(stage,dict(status='completed_ordinary_joint_error_attribution',costs=costs,
        wall_s=time.perf_counter()-started,goal_complete=False))
    print('ORDINARY_ERROR_ATTRIBUTION',summary,flush=True)


if __name__=='__main__':main()
