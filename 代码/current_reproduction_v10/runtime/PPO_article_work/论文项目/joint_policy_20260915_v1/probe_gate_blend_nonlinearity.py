"""Diagnose narrow gate-blend nonlinearity, retaining the exact physical executor."""
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
from continuous_gate_guidance import active_gate_transitions,GATE_TRANSITION_WIDTH_MM
from probe_coordinated_suffix import reconstruct
from probe_transition_state_parameter_interaction import difference
from probe_closed_loop_tangent_credit import ACTION_SCALE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    feedback=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
    layers=BASE/'canonical_gate_band_failed_transition_feedback_layers_20260918'
    data=read(credit/'results.json');parent=BASE/data['parent_model']
    item=next(r for r in data['records'] if r['path']==31)
    candidate=read(feedback/'results.json')['candidates'][0]
    j=next(j for j,r in enumerate(candidate['results']) if r['original_path']==31)
    trace=BASE/candidate['traces'][j];k=25
    stage=register('canonical_gate_band_narrow_blend_causal_probe_20260918',
        [__file__,parent,trace,credit/item['file'],layers/'results.json',layers/'receipt.json',
         layers/'nominal_action_at_shifted_state.npz',layers/'linear_actor_at_shifted_state.npz',
         BASE/'continuous_gate_guidance.py',BASE/'joint_runtime.py'],
        dict(new_physics_ticks=20,new_20ms_transitions=1,reused_transitions=2,
             saved_prefix_geometry_ticks=1200,target_queries=120,wall_s=60),
        'The dominant open-state nonlinear error occurs inside the0.1mm guide '
        'gate-blend band. Reuse the saved nominal and shifted-state/nominal-action '
        'transitions. Compute local XY derivatives of only the guide TARGET at '
        'each saved nominal1ms state. On the shifted state, substitute those '
        'affine targets for just20ms while keeping the nominal action, nonlinear '
        'bearing/filter, frozen projector and all physical limits unchanged. '
        'Measure the resulting error relative to the same open-state transition '
        'Jacobian. This isolates finite gate-target nonlinearity; the affine '
        'guide is a diagnostic intervention only and must never be deployed '
        'or reported as learned shape success.')
    start=time.perf_counter()
    with np.load(credit/item['file']) as z:local={q:z[q].copy() for q in z.files}
    with np.load(trace) as z:actual=z['rows'].copy();actions=z['actions'].copy()
    with np.load(layers/'nominal_action_at_shifted_state.npz') as z:latent=z['latent'].copy();fixed_rows=z['rows'].copy()
    with np.load(layers/'linear_actor_at_shifted_state.npz') as z:linear_latent=z['latent'].copy()
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,target_queries=0)
    old_states=[];new_states=[]
    old=reconstruct(local['reference'],local['rows'],local['actions'],20*(k+1),costs,state_snapshots=old_states,env_factory=factory)
    reconstruct(local['reference'],actual,actions,20*(k+1),costs,state_snapshots=new_states,env_factory=factory)
    old_next=old.state();nominal=old_states[k];shifted=new_states[k]
    env=factory(local['reference']);blend=[]
    for label,state in [('nominal',nominal),('shifted',shifted)]:
        env.restore(state)
        active=active_gate_transitions(env,GATE_TRANSITION_WIDTH_MM)
        assert len(active)==1
        gate,distance=active[0];fraction=.5+distance/GATE_TRANSITION_WIDTH_MM
        blend.append(dict(label=label,distance_mm=distance,fraction=fraction,
                          weight=fraction*fraction*(3.-2.*fraction)))
    blend[1]['nominal_linearized_weight']=blend[0]['weight']+6*blend[0]['fraction']*(1-blend[0]['fraction'])*(blend[1]['fraction']-blend[0]['fraction'])
    env.restore(nominal);env.held=local['actions'][k].copy()
    centers=[];targets=[];jacobians=[];epsilon=1e-6
    for tick in range(20*k,20*(k+1)):
        center=env.position.copy();centers.append(center)
        targets.append(env.guidance_target());costs['target_queries']+=1
        jac=np.zeros((2,2))
        for axis in range(2):
            outputs=[]
            for sign in (-1,1):
                env.position=center.copy();env.position[axis]+=sign*epsilon
                outputs.append(env.guidance_target());costs['target_queries']+=1
            jac[:,axis]=(outputs[1]-outputs[0])/(2*epsilon)
        jacobians.append(jac);row=local['rows'][tick]
        env.position=row[:2].copy();env.velocity=row[2:4].copy();env.acceleration=row[4:6].copy()
        env.heading=float(np.arctan2(row[9],row[8]));env.tracker.update(env.position);env.ticks+=1
    env=factory(local['reference']);env.restore(shifted);ordinary_guide=env.guidance_target
    def affine_target():
        tick=env.ticks-20*k
        if not 0<=tick<20:return ordinary_guide()
        return targets[tick]+jacobians[tick]@(env.position-centers[tick])
    env.guidance_target=affine_target
    _,_,_,info=env.step(latent);costs['new_physics_ticks']+=len(info['rows'])
    assert len(info['rows'])==20 and not env.done
    actual_delta=difference(shifted,nominal)
    open_prediction=(local['state_jacobian'][k,:,:7]@actual_delta-
        local['state_jacobian'][k,:,7:]@((linear_latent-latent)/ACTION_SCALE))
    residual=difference(env.state(),old_next)-open_prediction
    propagation=local['row_jacobian'][len(actual)-1,:,:7].copy()
    for t in range(len(actual)//20-1,k,-1):propagation=propagation@local['state_jacobian'][t,:,:7]
    final=propagation@residual
    old_error=read(layers/'results.json')['records'][0]
    summary=dict(blend=blend,local_xy_residual_mm=float(np.linalg.norm(residual[:2]*1.25)),
        local_heading_residual_deg=float(np.degrees(residual[6])),
        final_xy_error_contribution_mm=final[:2].tolist(),
        final_xy_contribution_norm_mm=float(np.linalg.norm(final[:2])),
        original_open_state_xy_contribution_norm_mm=old_error['final_xy_contribution_norm_mm'],
        physical_guard_ticks_original=int(fixed_rows[:,11].sum()),
        physical_guard_ticks_affine_guide=int(info['rows'][:,11].sum()))
    assert costs['new_physics_ticks']==20 and costs['target_queries']<=120 and time.perf_counter()-start<60
    np.savez_compressed(stage/'affine_guide_diagnostic.npz',rows=info['rows'],guide_centers=centers,
        guide_targets=targets,guide_xy_jacobians=jacobians)
    write_json(stage/'results.json',dict(summary=summary,costs=costs,diagnostic_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_narrow_gate_target_intervention',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('NARROW_GUIDE_BLEND',summary,flush=True)


if __name__=='__main__':main()
