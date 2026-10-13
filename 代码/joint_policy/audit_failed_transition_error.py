"""Locate model-error injections on a saved failed prefix; no successful time claim."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_closed_loop_tangent_credit import normalized_state,STATE_SCALE,ACTION_SCALE
from transition_waveform_model import chain_responses
from joint_runtime import wrap
from audit_function_constraint_interference import distances_to_segments
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    source=BASE/'canonical_gate_band_complete_completion_feedback_20260918'
    proposal=BASE/'canonical_gate_band_complete_completion_precision_20260918'
    credit=BASE/'canonical_gate_band_shape_priority_current_credit_20260918'
    data=read(source/'results.json');candidate=data['candidates'][0]
    records=read(credit/'results.json');parent=BASE/records['parent_model']
    assert sha(parent)==data['parent_sha256']==records['parent_sha256']
    j=next(j for j,r in enumerate(candidate['results']) if not r['success'])
    assert candidate['results'][j]['original_path']==31
    trace=BASE/candidate['traces'][j]
    item=next(r for r in records['records'] if r['path']==31)
    stage=register('canonical_gate_band_failed_completion_transition_cause_20260918',
        [__file__,source/'results.json',source/'receipt.json',trace,proposal/'direction.npz',
         parent,credit/item['file'],credit/'results.json',BASE/'transition_waveform_model.py',
         BASE/'probe_coordinated_suffix.py'],
        dict(paths=[31],new_physics_ticks=0,saved_prefix_geometry_ticks=3000,
             actor_feature_evaluations=100,wall_s=60),
        'Use the saved corridor-failure prefix, never its time as completion. '
        'At each20ms boundary compare the actual normalized seven-state change '
        'with the current closed-loop Jacobian applied to the ACTUAL previous '
        'state change and nominal parameter injection. Propagate each local '
        'residual through the same Jacobians to decompose the final affine XY '
        'prediction error exactly. Record gate-frontier mismatch, projector-guard '
        'switch counts and parameter-feature interaction as explanatory evidence, '
        'not causation by correlation. No new trajectory or repeated physics.')
    started=time.perf_counter();torch.set_num_threads(1)
    with np.load(credit/item['file']) as z:local={k:z[k].copy() for k in z.files}
    with np.load(trace) as z:actual=z['rows'].copy();actions=z['actions'].copy()
    with np.load(proposal/'direction.npz') as z:direction=z['direction'].copy().ravel()
    md=torch.load(parent,map_location='cpu',weights_only=False)
    factory=checkpoint_environment_factory(md['architecture']);policy=CanonicalGatePolicy(256)
    policy.load_state_dict(md['policy'])
    full=len(actual)//20;frame=full*20;costs=dict(saved_prefix_geometry_ticks=0)
    baseline_states=[];actual_states=[];actual_obs=[]
    old_env=reconstruct(local['reference'],local['rows'],local['actions'],frame,costs,
                        state_snapshots=baseline_states,env_factory=factory)
    new_env=reconstruct(local['reference'],actual,actions,frame,costs,
                        observations=actual_obs,state_snapshots=actual_states,env_factory=factory)
    baseline_states.append(old_env.state());actual_states.append(new_env.state())
    actual_obs.append(new_env.observation())
    delta=np.stack([normalized_state(a)-normalized_state(b) for a,b in zip(actual_states,baseline_states)])
    delta[:,6]=[wrap(a['heading']-b['heading']) for a,b in zip(actual_states,baseline_states)]
    injections=np.einsum('khc,hc->kh',local['features'][:full+1],direction.reshape(3,256))/ACTION_SCALE
    residual=[];detail=[]
    for k in range(full):
        sj=local['state_jacobian'][k]
        q=delta[k+1]-sj[:,:7]@delta[k]-sj[:,7:]@injections[k]
        residual.append(q)
        with torch.no_grad():features=policy.residual_features(torch.tensor(actual_obs[k])).numpy()*policy.residual_scale.numpy()[:,None]
        interaction=np.einsum('hc,hc->h',features-local['features'][k],direction.reshape(3,256))
        detail.append(dict(decision=k,tick=20*k,
            parent_frontier=int(baseline_states[k]['tracker']['frontier']),
            actual_frontier=int(actual_states[k]['tracker']['frontier']),
            local_xy_residual_mm=float(np.linalg.norm(q[:2]*STATE_SCALE[:2])),
            local_velocity_residual_mm_s=float(np.linalg.norm(q[2:4]*STATE_SCALE[2:4])),
            local_heading_residual_deg=float(np.degrees(q[6])),
            exact_feature_interaction_latent=interaction.tolist(),
            parent_guard_ticks=int(local['rows'][20*k:20*(k+1),11].sum()),
            actual_guard_ticks=int(actual[20*k:20*(k+1),11].sum())))
    response=chain_responses(local,direction[None,:])[:,:,0]
    predicted=local['rows'][:len(actual),:6]+response[:len(actual)]
    error=actual[:,:6]-predicted
    final=len(actual)-1;rj=local['row_jacobian'][final]
    last_residual=actual[-1,:6]-local['rows'][final,:6]-rj[:,:7]@delta[full]-rj[:,7:]@injections[full]
    propagation=rj[:,:7].copy();contributions=[]
    for k in range(full-1,-1,-1):
        contribution=propagation@residual[k]
        detail[k]['final_xy_error_contribution_mm']=contribution[:2].tolist()
        detail[k]['final_xy_contribution_norm_mm']=float(np.linalg.norm(contribution[:2]))
        contributions.append(contribution)
        propagation=propagation@local['state_jacobian'][k,:,:7]
    reconstructed=sum(contributions,np.zeros(6))+last_residual
    identity=float(abs(reconstructed-error[-1]).max())
    assert identity<1e-7
    ref=local['reference'];distance=distances_to_segments(ref,actual[:,:2]).min(axis=1)
    forecast_distance=distances_to_segments(ref,predicted[:,:2]).min(axis=1)
    by_error=sorted(detail,key=lambda r:r['final_xy_contribution_norm_mm'],reverse=True)
    first={}
    for threshold in (.001,.01):
        hits=np.flatnonzero(np.linalg.norm(error[:,:2],axis=1)>threshold)
        first[str(threshold)]=int(hits[0]+1) if len(hits) else None
    summary=dict(failure_tick=len(actual),actual_final_point_distance_mm=float(distance[-1]),
        predicted_final_point_distance_mm=float(forecast_distance[-1]),
        final_xy_error_mm=error[-1,:2].tolist(),final_partial_transition_error_mm=last_residual[:2].tolist(),
        first_xy_error_exceedance_tick=first,exact_error_decomposition_max_residual=identity,
        mismatched_frontier_decisions=[r['decision'] for r in detail if r['parent_frontier']!=r['actual_frontier']],
        largest_error_contributors=by_error[:6])
    assert costs['saved_prefix_geometry_ticks']<=3000 and time.perf_counter()-started<60
    np.savez_compressed(stage/'prediction_error.npz',actual=actual[:,:6],predicted=predicted,
        state_residual=np.asarray(residual),final_partial_residual=last_residual)
    write_json(stage/'results.json',dict(parent_sha256=sha(parent),path=31,records=detail,summary=summary,
        costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_saved_transition_error_decomposition',new_physics_ticks=0,
        costs=costs,wall_s=time.perf_counter()-started,goal_complete=False))
    print('FAILED_TRANSITION_CAUSE',summary,flush=True)


if __name__=='__main__':main()
