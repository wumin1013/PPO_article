"""Finite-scale validity of the dominant open-state transition derivative."""
import copy
import time
import numpy as np
import torch
from canonical_gate_state import checkpoint_environment_factory
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
    trace=BASE/candidate['traces'][j];k=25;scales=(.001,.01,.1,.25,.5)
    stage=register('canonical_gate_band_transition_state_scale_audit_20260918',
        [__file__,parent,trace,credit/item['file'],layers/'results.json',layers/'receipt.json',
         layers/'nominal_action_at_shifted_state.npz',layers/'linear_actor_at_shifted_state.npz'],
        dict(new_physics_ticks=200,new_20ms_transitions=10,signed_state_scales=list(scales),
             actor_queries=0,saved_prefix_geometry_ticks=1200,wall_s=60),
        'Hold the exact nominal action and vary ONLY the measured seven-state '
        'shift along its signed ray at five fixed scales. Reuse zero and full '
        'positive-shift transitions. Compare centered finite slopes with the '
        'existing open-state derivative to distinguish an inaccurate derivative '
        'near zero from finite-displacement nonlinear growth. No changed '
        'physical limits, fitted error inflation, learned adoption or full-path '
        'time claim. A valid radius on this one ray is not a universal state tube.')
    start=time.perf_counter()
    with np.load(credit/item['file']) as z:local={q:z[q].copy() for q in z.files}
    with np.load(trace) as z:actual=z['rows'].copy();actions=z['actions'].copy()
    with np.load(layers/'nominal_action_at_shifted_state.npz') as z:latent=z['latent'].copy()
    with np.load(layers/'linear_actor_at_shifted_state.npz') as z:linear_latent=z['latent'].copy()
    factory=checkpoint_environment_factory(torch.load(parent,map_location='cpu',weights_only=False)['architecture'])
    costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0)
    old_states=[];new_states=[]
    old=reconstruct(local['reference'],local['rows'],local['actions'],20*(k+1),costs,state_snapshots=old_states,env_factory=factory)
    reconstruct(local['reference'],actual,actions,20*(k+1),costs,state_snapshots=new_states,env_factory=factory)
    old_next=old.state();nominal=old_states[k];shifted=new_states[k]
    delta=difference(shifted,nominal)
    prediction=(local['state_jacobian'][k,:,:7]@delta-
        local['state_jacobian'][k,:,7:]@((linear_latent-latent)/ACTION_SCALE))
    propagation=local['row_jacobian'][len(actual)-1,:,:7].copy()
    for t in range(len(actual)//20-1,k,-1):propagation=propagation@local['state_jacobian'][t,:,:7]
    env=factory(local['reference']);records=[];slopes=[]
    for scale in scales:
        pair=[]
        for sign in (-1,1):
            state=copy.deepcopy(nominal)
            for field in ('position','velocity','acceleration'):
                state[field]=nominal[field]+sign*scale*(shifted[field]-nominal[field])
            state['heading']=nominal['heading']+sign*scale*delta[6]
            state['tracker']['position']=state['position'].copy()
            env.restore(state);_,_,_,info=env.step(latent);costs['new_physics_ticks']+=len(info['rows'])
            assert len(info['rows'])==20 and not env.done
            response=difference(env.state(),old_next);pair.append(response)
            residual=response-sign*scale*prediction;final=propagation@residual
            records.append(dict(scale=sign*scale,local_xy_error_mm=float(np.linalg.norm(residual[:2]*1.25)),
                local_heading_error_deg=float(np.degrees(residual[6])),
                propagated_xy_error_mm=final[:2].tolist(),propagated_xy_error_norm_mm=float(np.linalg.norm(final[:2])),
                guard_ticks=int(info['rows'][:,11].sum())))
            np.savez_compressed(stage/f'scale_{sign*scale:g}.npz',rows=info['rows'],normalized_response=response)
        slope=(pair[1]-pair[0])/(2*scale);slopes.append(dict(scale=scale,
            local_xy_slope_difference_mm=float(np.linalg.norm((slope-prediction)[:2]*1.25)),
            propagated_xy_slope_difference_mm=float(np.linalg.norm((propagation@(slope-prediction))[:2]))))
    assert costs['new_physics_ticks']==200 and time.perf_counter()-start<60
    write_json(stage/'results.json',dict(records=records,centered_slopes=slopes,costs=costs,
        registered_scope='one_state_direction_only',goal_complete=False))
    receipt(stage,dict(status='completed_both_sign_state_scale_audit',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('STATE_SCALE_SLOPES',slopes,flush=True);print('STATE_SCALE_ERRORS',records,flush=True)


if __name__=='__main__':main()
