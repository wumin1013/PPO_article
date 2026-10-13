"""Bounded initialization of continuous gate-state representation from saved training data."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy,canonical_observation,environment_factory
from continuous_gate_guidance import environment_factory as old_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from diagnose_local_basis_fit import feature_batches
from learn_continuous_sensor_transfer import means
from probe_coordinated_suffix import reconstruct
from learn_single_decision_credit import review,adoption_checks
from probe_direct_constraint_composition import metric_rows
from solve_continuous_sensor_credit import NAMES
from train_pilot import evaluate,save_torch
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    guide=BASE/'continuous_gate_guidance_mechanism_probe_20260917'
    selected=read(guide/'results.json')['candidates'][0]
    model=BASE/selected['source_model'];ids=[r['original_path'] for r in selected['results']]
    stage=register('canonical_gate_state_initialization_20260917',
        [__file__,BASE/'canonical_gate_state.py',BASE/'continuous_gate_guidance.py',
         BASE/'continuous_sensor_runtime.py',BASE/'probe_coordinated_suffix.py',
         BASE/'current_gate_observation_continuity_audit_20260917/results.json',
         guide/'results.json',model,BASE/'diagnose_local_basis_fit.py'],
        dict(saved_training_paths=29,saved_geometry_ticks=65000,
            state_transition_width_mm=2.,guide_transition_width_mm=.1,
            analytic_fits=1,learned_coefficients=768,ridge=.001,
            evaluation_actor_calls=6000,evaluation_physics_ticks=140000,wall_s=180),
        'The current actor has same-physical-state gate feed jumps up to42.3mm/s '
        'on the155-degree example despite the corrected guide target. Encode '
        'neighboring gate observations in the SAME heading frame before blending '
        'them over a global2mm neighborhood, derived from100mm/s*20ms. Also '
        'replace binary endpoint-sensor validity by its2mm interval fraction. '
        'The actor still learns steering/feed/preview; this defines observation '
        'geometry, not a turn template or output correction. The72 slots are '
        'retained for an interpretable matched initialization, not a minimality '
        'claim. Validate pure coordinate conversion as an exact parent identity '
        'and record unsupported overlapping gate neighborhoods. One analytic '
        'ridge fit transfers existing learned feedback targets on the29 saved '
        'current-guide training paths; frozen hidden network and768 shared residual '
        'coefficients. Compare unadapted and fitted encodings on those29 full '
        'paths. Report complete shape/curvature/time/cruise; no development or '
        'promotion in this initialization. Stop after this one fit, analyze any '
        'closed-loop failure; do not repeat width/ridge/epoch sweeps. Coverage '
        'tracker is always restored after virtual observation queries.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,evaluation_actor_calls=0,
        evaluation_physics_ticks=0,analytic_fits=0)
    saved=torch.load(model,map_location='cpu',weights_only=False)['policy']
    teacher=KernelLocalFeedbackPolicy(256);teacher.load_state_dict(saved)
    policy=CanonicalGatePolicy(256);policy.load_state_dict(saved)
    observations=[];targets=[];weights=[];path_ids=[];identity_errors=[];unsupported=[]
    for r,trace in zip(selected['results'],selected['traces']):
        path=r['original_path']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,state_snapshots=states,env_factory=old_factory)
        states.append(env.state())
        for anchor,state in enumerate(states):
            env.restore(state);raw=env.observation();old=teacher.act(raw,True)[0]
            assert np.array_equal(env.decode_action(old),actions[anchor])
            pure=canonical_observation(env,blend_gate=False,smooth_valid=False)
            converted=policy.act(pure,True)[0]
            identity_errors.append(float(abs(converted-old).max()))
            assert np.array_equal(converted,old)
            try:x=canonical_observation(env)
            except ValueError as error:
                unsupported.append(dict(path=path,anchor=anchor,error=str(error)));continue
            assert env.tracker.frontier==state['tracker']['frontier']
            observations.append(x);targets.append(old);weights.append(1/(len(ids)*len(states)));path_ids.append(path)
    if unsupported:
        write_json(stage/'unsupported.json',dict(states=unsupported,goal_complete=False))
        receipt(stage,dict(status='state_encoding_requires_overlap_resolution',unsupported_states=len(unsupported),
            costs=costs,wall_s=time.perf_counter()-start,goal_complete=False));return
    x=np.asarray(observations);target=np.asarray(targets);w=np.asarray(weights)
    assert abs(w.sum()-1)<1e-10
    np.savez_compressed(stage/'training_pairs.npz',observation=x,target=target,weight=w,path_id=path_ids)
    before=means(policy,x);scale=np.array([.005,.1,.1]);rhs=(target-before)/scale
    f=feature_batches(policy,x)*policy.residual_scale.detach().numpy()[None,:,None]/scale[None,:,None]
    delta=np.stack([np.linalg.solve(f[:,h].T@(w[:,None]*f[:,h])+.001*np.eye(256),
        f[:,h].T@(w*rhs[:,h])) for h in range(3)])
    with torch.no_grad():policy.residual_coefficients.add_(torch.tensor(delta,dtype=torch.float32))
    after=means(policy,x);costs['analytic_fits']=1
    fit=dict(identity_max_error=max(identity_errors),saved_states=len(x),
        normalized_rms_before=np.sqrt((w[:,None]*rhs**2).sum(0)).tolist(),
        normalized_rms_after=np.sqrt((w[:,None]*((after-target)/scale)**2).sum(0)).tolist())
    for name,state in (('unadapted',saved),('fitted',policy.state_dict())):
        save_torch(stage/f'{name}.pt',dict(policy=state,architecture='canonical_gate_kernel',centers=256,
            observation_encoding='heading_frame_blended_gate_and_fractional_endpoint',
            guidance_encoding='continuous_gate_target_0.1mm',parent_sha256=sha(model),
            learning_method='shared_residual_function_transfer' if name=='fitted' else 'unadapted_representation'))
    write_json(stage/'fit.json',fit);print('CANONICAL_STATE_FIT',fit,flush=True)
    training=read(BASE/'learning_preparation_v4/paths.json')['training'];candidates=[]
    for tag in ('unadapted','fitted'):
        policy.load_state_dict(torch.load(stage/f'{tag}.pt',map_location='cpu',weights_only=False)['policy'])
        rs=evaluate(policy,[np.asarray(training[i]) for i in ids],stage,tag,None,costs,start+180,environment_factory)
        rs=[dict(r,original_path=i) for r,i in zip(rs,ids)];traces=[stage/tag/f'{i}.npz' for i in range(len(ids))]
        summary=review(rs,traces);metrics=metric_rows(rs,traces,summary,NAMES)
        candidates.append(dict(tag=tag,model=f'{tag}.pt',summary=summary,metrics=metrics,results=rs,
            traces=[str(t.relative_to(BASE)) for t in traces],checks=adoption_checks(summary,selected['summary'])))
        write_json(stage/'results.json',dict(candidates=candidates,fit=fit,costs=costs,metric_names=NAMES,
            goal_complete=False,promoted=False))
        print('CANONICAL_STATE_SCREEN',tag,{k:summary[k] for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s')},flush=True)
    assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=140000
    assert costs['saved_prefix_geometry_ticks']<=65000
    receipt(stage,dict(status='completed_canonical_gate_state_initialization',costs=costs,
        wall_s=time.perf_counter()-start,goal_complete=False,promoted=False))


if __name__=='__main__':main()
