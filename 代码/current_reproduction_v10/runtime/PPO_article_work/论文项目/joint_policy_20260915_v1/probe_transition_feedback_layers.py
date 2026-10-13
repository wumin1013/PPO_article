"""Split a measured state-feedback error into actor and physical-transition layers."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import normalized_state,shift_state,ACTION_SCALE
from probe_transition_state_parameter_interaction import difference
from smooth_credit_policy import CenteredSmoothActor
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def row_state(row):
    return np.r_[row[:6]/np.array([1.25,1.25,100.,100.,2000.,2000.]),np.arctan2(row[9],row[8])]


def main():
    feedback=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    previous=BASE/'canonical_gate_band_failed_transition_factor_probe_20260918'
    data=read(feedback/'results.json');candidate=data['candidates'][0];models=read(credit/'results.json')
    parent=BASE/models['parent_model'];item=next(r for r in models['records'] if r['path']==31)
    j=next(j for j,r in enumerate(candidate['results']) if r['original_path']==31)
    trace=BASE/candidate['traces'][j];k=25
    stage=register('canonical_gate_band_failed_transition_feedback_layers_20260918',
        [__file__,parent,trace,credit/item['file'],previous/'results.json',previous/'receipt.json',
         previous/'shifted_state_parent_policy.npz',BASE/'smooth_credit_policy.py',BASE/'joint_runtime.py'],
        dict(path=31,decision=k,new_20ms_transitions=2,new_physics_ticks=40,
             actor_queries=20,saved_prefix_geometry_ticks=1200,wall_s=60),
        'The completed2x2 intervention isolates state-feedback nonlinearity as the '
        'dominant error source. Split it further at the SAME saved state using '
        'two missing actions: unchanged nominal action, then first-order actor '
        'action at the shifted state. Reuse the parent actor on shifted state '
        'from the prior probe. Estimate the actor state Jacobian with the same '
        'centered float64 extension and1e-4 scale as original credit. Separate '
        'open-state transition nonlinearity, action-response interaction at that '
        'state, and nonlinear actor response. Do not modify production actions '
        'or claim any short transition as a complete successful path.')
    start=time.perf_counter();torch.set_num_threads(1)
    with np.load(credit/item['file']) as z:local={q:z[q].copy() for q in z.files}
    with np.load(trace) as z:actual=z['rows'].copy();actions=z['actions'].copy()
    with np.load(previous/'shifted_state_parent_policy.npz') as z:s0_rows=z['rows'].copy()
    md=torch.load(parent,map_location='cpu',weights_only=False);factory=checkpoint_environment_factory(md['architecture'])
    policy=CanonicalGatePolicy(256);policy.load_state_dict(md['policy']);smooth=CenteredSmoothActor(policy)
    costs=dict(new_physics_ticks=0,actor_queries=0,saved_prefix_geometry_ticks=0)
    old_states=[];new_states=[]
    old_env=reconstruct(local['reference'],local['rows'],local['actions'],20*(k+1),costs,
                        state_snapshots=old_states,env_factory=factory)
    reconstruct(local['reference'],actual,actions,20*(k+1),costs,
                state_snapshots=new_states,env_factory=factory)
    old_state,new_state=old_states[k],new_states[k];old_next=old_env.state()
    env=factory(local['reference']);env.restore(old_state)
    latent=policy.act(env.observation(),True)[0];costs['actor_queries']+=1
    env.observation_dtype=np.float64;smooth.center(env.observation(),latent);costs['actor_queries']+=1
    jacobian=np.zeros((3,7));eps=1e-4
    for axis in range(7):
        pair=[]
        for sign in (-1,1):
            env.restore(shift_state(old_state,axis,sign*eps))
            pair.append(smooth.act(env.observation(),True)[0]);costs['actor_queries']+=1
        jacobian[:,axis]=(pair[1]-pair[0])/(2*eps)
    delta=difference(new_state,old_state);linear_latent=latent+jacobian@delta
    env.observation_dtype=np.float32;env.restore(new_state)
    actual_latent=policy.act(env.observation(),True)[0];costs['actor_queries']+=1
    outcomes={};cells=[]
    for name,action in [('nominal_action_at_shifted_state',latent),('linear_actor_at_shifted_state',linear_latent)]:
        env.restore(new_state);_,_,_,info=env.step(action);costs['new_physics_ticks']+=len(info['rows'])
        assert len(info['rows'])==20 and not env.done and not info['rows'][:,12].any()
        outcomes[name]=env.state();np.savez_compressed(stage/(name+'.npz'),rows=info['rows'],latent=action)
        cells.append(dict(cell=name,guard_ticks=int(info['rows'][:,11].sum()),held=info['held_action'].tolist()))
    fixed=difference(outcomes['nominal_action_at_shifted_state'],old_next)
    linear=difference(outcomes['linear_actor_at_shifted_state'],old_next)
    exact=row_state(s0_rows[-1])-normalized_state(old_next)
    exact[6]=wrap(np.arctan2(s0_rows[-1,9],s0_rows[-1,8])-old_next['heading'])
    a=local['state_jacobian'][k,:,:7];b=local['state_jacobian'][k,:,7:]
    actor_linear=b@((jacobian@delta)/ACTION_SCALE)
    parts=dict(open_state_transition_nonlinearity=fixed-(a@delta-actor_linear),
        state_action_response_interaction=linear-fixed-actor_linear,
        nonlinear_actor_response=exact-linear)
    q=exact-a@delta;identity=float(abs(sum(parts.values(),np.zeros(7))-q).max());assert identity<1e-10
    propagation=local['row_jacobian'][len(actual)-1,:,:7].copy()
    for t in range(len(actual)//20-1,k,-1):propagation=propagation@local['state_jacobian'][t,:,:7]
    result=[]
    for name,value in parts.items():
        final=propagation@value
        result.append(dict(component=name,local_xy_residual_mm=float(np.linalg.norm(value[:2]*1.25)),
            local_heading_residual_deg=float(np.degrees(value[6])),final_xy_error_contribution_mm=final[:2].tolist(),
            final_xy_contribution_norm_mm=float(np.linalg.norm(final[:2]))))
    assert costs['new_physics_ticks']==40 and costs['actor_queries']<=20
    assert costs['saved_prefix_geometry_ticks']<=1200 and time.perf_counter()-start<60
    write_json(stage/'results.json',dict(parent_sha256=sha(parent),records=result,cells=cells,costs=costs,
        nonlinear_actor_latent_error=(actual_latent-linear_latent).tolist(),identity_error=identity,
        diagnostic_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_state_feedback_layer_intervention',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('FEEDBACK_LAYER_CAUSE',result,'actor_error',actual_latent-linear_latent,flush=True)


if __name__=='__main__':main()
