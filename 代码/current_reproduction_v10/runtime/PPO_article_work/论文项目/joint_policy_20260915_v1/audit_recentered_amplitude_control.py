"""Matched finite-difference amplitude control against saved whole-policy responses."""
import time
import numpy as np
import torch
from waveform_constraint_model import waveform_facets
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    old=BASE/'canonical_gate_band_half_step_waveform_credit_20260917'
    new=BASE/'canonical_gate_band_half_step_path14_small_waveform_20260917'
    old_credit=BASE/'canonical_gate_band_half_step_current_credit_20260917'
    new_credit=BASE/'canonical_gate_band_half_step_path14_amplitude_control_20260917'
    source=BASE/'canonical_gate_band_quadratic_error_learned_update_20260917'
    small=BASE/'canonical_gate_band_half_step_locality_probe_20260917'
    data=read(source/'results.json');sd=read(small/'results.json')
    model=source/'candidate_0.5.pt';center=torch.load(model,map_location='cpu',weights_only=False)
    assert read(old/'results.json')['parent_sha256']==read(new/'results.json')['parent_sha256']==sha(model)
    stage=register('canonical_gate_band_half_step_path14_amplitude_audit_20260917',
        [__file__,model,old/'path14_waveform.npz',new/'path14_waveform.npz',old/'receipt.json',new/'receipt.json',
         old_credit/'results.json',new_credit/'results.json',old_credit/'queries.npz',new_credit/'queries.npz',
         source/'results.json',source/'receipt.json',small/'results.json',small/'receipt.json'],
        dict(new_actor_decisions=0,new_physics_ticks=0,wall_s=30,relative_amplitudes=[1.,.25]),
        'Compare full51-decision path14 waveform credit at two matched axial '
        'amplitudes, holding current parent, nominal observations and runtime '
        'identical. Test predictions against the same previously saved four '
        'whole-policy displacements around this center. Use actual checkpoint '
        'coefficient differences, including float32 rounding. This isolates '
        'finite-difference-amplitude bias; no new candidate or physical rollout.')
    start=time.perf_counter()
    oq=read(old_credit/'results.json')['queries'];nq=read(new_credit/'results.json')['queries']
    old_indices=[r['query'] for r in oq if r['path']==14]
    with np.load(old_credit/'queries.npz') as z:old_obs=z['observation'][old_indices].copy()
    with np.load(new_credit/'queries.npz') as z:new_obs=z['observation'].copy()
    assert np.array_equal(old_obs,new_obs) and len(old_obs)==51
    assert [r['anchor'] for r in oq if r['path']==14]==[r['anchor'] for r in nq]
    variants=[(-.5,data['baseline'],BASE/'canonical_gate_band_matched_probe_20260917/parent.pt'),
              (.5,next(r for r in data['candidates'] if r['step']==1.),source/'candidate_1.pt')]
    variants += [(r['step'],r,small/f'step{r["step"]:+g}.pt') for r in sd['candidates']]
    names=sd['metric_names'];operators=[]
    for folder in (old,new):
        with np.load(folder/'path14_waveform.npz') as z:
            nominal=z['nominal'].copy();jac=z['gradient'].copy();valid=z['nominal_valid'].copy()
            jac[~z['common_valid']]=0.
        operators.append((nominal,jac,valid))
    assert np.array_equal(operators[0][0],operators[1][0])
    records=[]
    for step,variant,file in variants:
        md=torch.load(file,map_location='cpu',weights_only=False)
        for key,value in center['policy'].items():
            if key!='residual_coefficients':assert torch.equal(value,md['policy'][key])
        direction=(md['policy']['residual_coefficients'].double()-center['policy']['residual_coefficients'].double()).numpy().ravel()
        actual=next(r['values'] for r in variant['metrics'] if r['path']==14)
        for scale,(nominal,jac,valid) in zip((1.,.25),operators):
            prediction=waveform_facets(nominal,jac,valid,-155.,direction)
            errors={n:float(v[0]-actual[names.index(n)]) for n,v in prediction.items()}
            r=dict(step=step,relative_axial_amplitude=scale,prediction_minus_actual=errors)
            records.append(r)
            print('AMPLITUDE_COMPOSITION',step,scale,{n:errors[n] for n in
                ('full_physical_reverse_deg','heading_overshoot_deg','negative_signed_heading_deg')},flush=True)
    write_json(stage/'results.json',dict(records=records,identical51_nominal_observations=True,
        matched_parameter_checkpoints=True,new_physics_ticks=0,goal_complete=False))
    receipt(stage,dict(status='compared_matched_current_amplitude_credit',new_actor_decisions=0,
        new_physics_ticks=0,wall_s=time.perf_counter()-start,goal_complete=False))


if __name__=='__main__':main()
