"""Two missing cells of a saved state/parameter2x2 transition comparison."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import normalized_state,ACTION_SCALE
from joint_runtime import wrap
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def difference(left,right):
    result=normalized_state(left)-normalized_state(right)
    result[6]=wrap(left['heading']-right['heading'])
    return result


def main():
    feedback=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
    proposal=BASE/'canonical_gate_band_complete_completion_precision_20260918'
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    audit=BASE/'canonical_gate_band_failed_completion_transition_cause_20260918'
    data=read(feedback/'results.json');candidate=data['candidates'][0]
    models=read(credit/'results.json');parent=BASE/models['parent_model'];model=feedback/candidate['model']
    cause=read(audit/'results.json')['summary']['largest_error_contributors'][0]
    k=cause['decision'];assert k==25
    index=next(j for j,r in enumerate(candidate['results']) if r['original_path']==31)
    trace=BASE/candidate['traces'][index]
    item=next(r for r in models['records'] if r['path']==31)
    stage=register('canonical_gate_band_failed_transition_factor_probe_20260918',
        [__file__,parent,model,trace,credit/item['file'],proposal/'direction.npz',audit/'results.json',
         audit/'receipt.json',BASE/'joint_runtime.py',BASE/'probe_coordinated_suffix.py'],
        dict(path=31,decision=k,new_20ms_transitions=2,reused_20ms_transitions=2,
             new_physics_ticks=40,actor_calls=2,saved_prefix_geometry_ticks=1200,wall_s=60),
        'At the saved dominant error-injection decision compare two factors: '
        'nominal versus actual shifted state, and parent versus candidate shared '
        'parameters. Reuse the nominal/parent and actual/candidate cells exactly '
        'from saved rows. Execute only the missing actual/parent and nominal/candidate '
        '20ms cells. Decompose model error into nonlinear state feedback, nonlinear '
        'parameter response and finite state-parameter interaction, with exact '
        'addition identity. This is a bounded diagnostic intervention, not an '
        'alternative deployed trajectory, completed path, adopted policy or '
        'proof that projector guard changes alone cause the error.')
    start=time.perf_counter();torch.set_num_threads(1)
    with np.load(credit/item['file']) as z:local={q:z[q].copy() for q in z.files}
    with np.load(trace) as z:actual=z['rows'].copy();actions=z['actions'].copy()
    with np.load(proposal/'direction.npz') as z:direction=z['direction'].copy().ravel()
    md=torch.load(parent,map_location='cpu',weights_only=False);factory=checkpoint_environment_factory(md['architecture'])
    parent_policy=CanonicalGatePolicy(256);parent_policy.load_state_dict(md['policy'])
    new_policy=CanonicalGatePolicy(256);new_policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,actor_calls=0)
    old_states=[];new_states=[]
    old_env=reconstruct(local['reference'],local['rows'],local['actions'],20*(k+1),costs,
                        state_snapshots=old_states,env_factory=factory)
    new_env=reconstruct(local['reference'],actual,actions,20*(k+1),costs,
                        state_snapshots=new_states,env_factory=factory)
    old_next=old_env.state();new_next=new_env.state();old_state=old_states[k];new_state=new_states[k]
    env=factory(local['reference']);outcomes={};diagnostics=[]
    for name,state,policy in [('shifted_state_parent_policy',new_state,parent_policy),
                              ('nominal_state_candidate_policy',old_state,new_policy)]:
        env.restore(state);latent=policy.act(env.observation(),True)[0];costs['actor_calls']+=1
        _,_,_,info=env.step(latent);costs['new_physics_ticks']+=len(info['rows'])
        assert len(info['rows'])==20 and not env.done and not info['rows'][:,12].any()
        outcomes[name]=env.state()
        np.savez_compressed(stage/(name+'.npz'),rows=info['rows'],latent=latent,held=info['held_action'])
        diagnostics.append(dict(cell=name,guard_ticks=int(info['rows'][:,11].sum()),
                                held=info['held_action'].tolist()))
    a,b=local['state_jacobian'][k,:,:7],local['state_jacobian'][k,:,7:]
    direct=np.einsum('hc,hc->h',local['features'][k],direction.reshape(3,256))/ACTION_SCALE
    ds=difference(outcomes['shifted_state_parent_policy'],old_next)
    dp=difference(outcomes['nominal_state_candidate_policy'],old_next)
    full=difference(new_next,old_next)
    pieces=dict(state_nonlinearity=ds-a@difference(new_state,old_state),
        parameter_nonlinearity=dp-b@direct,state_parameter_interaction=full-ds-dp)
    error=full-a@difference(new_state,old_state)-b@direct
    identity=float(abs(sum(pieces.values(),np.zeros(7))-error).max());assert identity<1e-10
    propagation=local['row_jacobian'][len(actual)-1,:,:7].copy()
    for t in range(len(actual)//20-1,k,-1):propagation=propagation@local['state_jacobian'][t,:,:7]
    result=[]
    for name,q in pieces.items():
        contribution=propagation@q
        result.append(dict(component=name,normalized_state_residual=q.tolist(),
            local_xy_residual_mm=float(np.linalg.norm(q[:2]*1.25)),
            local_heading_residual_deg=float(np.degrees(q[6])),
            final_xy_error_contribution_mm=contribution[:2].tolist(),
            final_xy_contribution_norm_mm=float(np.linalg.norm(contribution[:2]))))
    assert costs['new_physics_ticks']==40 and costs['actor_calls']==2
    assert costs['saved_prefix_geometry_ticks']<=1200 and time.perf_counter()-start<60
    write_json(stage/'results.json',dict(parent_sha256=sha(parent),candidate_sha256=sha(model),path=31,
        decision=k,records=result,cells=diagnostics,identity_error=identity,costs=costs,diagnostic_only=True,goal_complete=False))
    receipt(stage,dict(status='completed_state_parameter_factor_probe',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False))
    print('TRANSITION_FACTOR_CAUSE',result,flush=True)


if __name__=='__main__':main()
