"""Matched latent/physical proposal metrics and exact saved-action bounds."""
import time
import argparse
import numpy as np
import torch
from scipy.linalg import block_diag
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from probe_one_decision_credit import AMPLITUDE
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',default='canonical_gate_band_bend_interpolated_learning_20260917')
    parser.add_argument('--stage',default='canonical_gate_band_current_parameter_geometries_20260917')
    parser.add_argument('--cruise-support',choices=('pure_straights','segment_interiors'),default='pure_straights')
    args=parser.parse_args()
    parent=BASE/args.parent
    data=read(parent/'results.json');selected=data['full_candidate'];model=parent/data['chosen']['model']
    assert selected['checks']['eligible'] and selected['summary']['successes']==64
    evidence=BASE/'canonical_gate_band_current_cruise_request_probe_20260917'
    support_evidence=BASE/'canonical_gate_band_segment_cruise_probe_20260918'
    extra_sources=([support_evidence/'results.json',support_evidence/'receipt.json']
                   if args.cruise_support=='segment_interiors' else [])
    stage=register(args.stage,
        [__file__,model,parent/'results.json',parent/'receipt.json',evidence/'results.json',
         evidence/'receipt.json',BASE/'canonical_gate_state.py',BASE/'probe_coordinated_suffix.py',
         BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'terminal_candidate_runtime_v1.py',BASE/'joint_runtime_v5.py']+extra_sources,
        dict(new_physics_ticks=0,saved_geometry_ticks=100000,replayed_actor_calls=6000,
            wall_s=120,physical_units=[float(np.deg2rad(1.)),5.,.5],normalized_action_bound=.5,
            metric_radius=.1,ridge=.001,training_paths=64,cruise_support=args.cruise_support,
            pure_straight_margin_mm=6.2,curved_segment_margin_mm=12.4 if args.cruise_support=='segment_interiors' else None),
        'Build two matched UPDATE-GEOMETRY arms from the identical64 saved '
        'training trajectories and equal path weights. Latent arm uses the '
        'existing [.005,.1,.1] scales. Physical arm uses [1degree,5mm/s,.5mm] '
        'units for the metric and exact saved-state action-change bounds '
        'of half those units. Invert the unchanged tanh action map to linear '
        'coefficient halfspaces using centers/ranges obtained from the actual checkpoint '
        'runtime decoder, and compare every reconstructed nominal action and sampled '
        'perturbed action directly with that decoder. Do not use the older base-runtime '
        'preview range. Invert to linear '
        'coefficient halfspaces, including genuinely unbounded latent sides '
        'when the physical request limit is already stricter. This changes '
        'the learning geometry (metric plus support), not the policy, action '
        'map, guide, runtime or physical limits. Both arms keep radius0.1 '
        'and coefficient ridge0.001, and use the same forthcoming explicit '
        '99.5mm/s request-deficit objective on existing6.2mm endpoint-excluded '
        'straight TRAINING states. When explicitly selecting segment_interiors, '
        'add otherwise omitted straight segments of curved training paths at '
        '12.4mm clearance from BOTH segment endpoints, using ordered progress. '
        'All physical metric/support arrays are unchanged between support arms; '
        'only target-state membership and equal-target-path loss weights change. '
        'Four finite feed-only probes establish some reachable cruise and also '
        'show shape regressions, requiring joint learning and full acceptance. '
        'Those labels are an offline maximum-speed '
        'learning goal, not a deployed path/angle rule. The actual adoption '
        'must check real cruise, time and complete shape; request fit alone '
        'is insufficient. Reconstruct saved geometry only; do not sample '
        'new physics or inspect development/sealed paths.')
    start=time.perf_counter();torch.set_num_threads(1)
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture'])
    costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,replayed_actor_calls=0)
    observations=[];latents=[];decoded_actions=[];weights=[];cruise_indices=[];cruise_weights=[];state_paths=[];cruise_groups=[]
    for result,trace in zip(selected['results'],selected['traces']):
        path=result['original_path']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
            state_snapshots=states,env_factory=factory);states.append(env.state());local_cruise=[]
        for k,state in enumerate(states):
            env.restore(state);o=env.observation();a=p.act(o,True)[0];costs['replayed_actor_calls']+=1
            assert np.array_equal(env.decode_action(a),actions[k])
            decoded_actions.append(actions[k])
            index=len(observations);observations.append(o);latents.append(a);weights.append(1./(64*len(states)))
            state_paths.append(path)
            if len(ref)==2:
                length=np.linalg.norm(ref[-1]-ref[0]);unit=(ref[-1]-ref[0])/length
                along=float((state['position']-ref[0])@unit)
                if 6.2<=along<=length-6.2:local_cruise.append(index)
            elif args.cruise_support=='segment_interiors':
                progress,segment=env.progress()
                along=progress-env.corridor.arcs[segment]
                remaining=env.corridor.arcs[segment+1]-progress
                if min(along,remaining)>=12.4 and k<len(states)-1:local_cruise.append(index)
        if local_cruise:
            cruise_groups.append(local_cruise)
        assert time.perf_counter()-start<115 and costs['replayed_actor_calls']<=6000 and costs['saved_prefix_geometry_ticks']<=100000
    for group in cruise_groups:
        cruise_indices.extend(group);cruise_weights.extend([1./(len(cruise_groups)*len(group))]*len(group))
    observations=np.asarray(observations);latent=np.asarray(latents,dtype=float);weights=np.asarray(weights)
    with torch.no_grad():features=(p.residual_features(torch.tensor(observations))*p.residual_scale[None,:,None]).numpy().astype(float)
    decoder=env.decode_action
    centers=decoder(np.zeros(3));ranges=np.empty(3)
    for h in range(3):
        perturb=np.zeros(3);perturb[h]=np.arctanh(.5)
        change=decoder(perturb)-centers
        assert np.all(abs(np.delete(change,h))<1e-12)
        ranges[h]=change[h]/.5
    units=np.array([np.deg2rad(1.),5.,.5])
    nominal_decoder_error=float(abs(centers+ranges*np.tanh(latent)-np.asarray(decoded_actions)).max())
    assert nominal_decoder_error<1e-12
    derivative=ranges[None,:]*(1.-np.tanh(latent)**2)
    latent_features=features/AMPLITUDE[None,:,None]
    physical_features=features*derivative[:,:,None]/units[None,:,None]
    metrics={}
    for name,f in [('latent',latent_features),('physical',physical_features)]:
        metrics[name]=block_diag(*[f[:,h].T@(weights[:,None]*f[:,h])/3.+.001*np.eye(256) for h in range(3)])
        np.linalg.cholesky(metrics[name])
    support=np.zeros((len(features),3,768))
    for h in range(3):support[:,h,h*256:(h+1)*256]=features[:,h]
    physical=centers+ranges*np.tanh(latent)
    low_u=(physical-.5*units-centers)/ranges;high_u=(physical+.5*units-centers)/ranges
    low=np.full_like(latent,-np.inf);high=np.full_like(latent,np.inf)
    mask=low_u>-1.;low[mask]=np.arctanh(low_u[mask])-latent[mask]
    mask=high_u<1.;high[mask]=np.arctanh(high_u[mask])-latent[mask]
    assert abs(weights.sum()-1.)<1e-12 and abs(sum(cruise_weights)-1.)<1e-12
    # Exact equivalence of the saved-state halfspaces and original action map.
    rng=np.random.default_rng(20260917)
    errors=[]
    decoder_errors=[]
    for scale in (.001,.01,.1,.3):
        delta=support@rng.normal(size=768)*scale
        mapped=centers+ranges*np.tanh(latent+delta)
        executed=np.array([decoder(row) for row in latent+delta])
        decoder_errors.append(float(abs(mapped-executed).max()))
        assert decoder_errors[-1]<1e-12
        algebra=(delta>=low-1e-10)&(delta<=high+1e-10)
        direct=abs(executed-np.asarray(decoded_actions))<=.5*units+1e-10
        assert np.array_equal(algebra,direct)
        finite=np.isfinite(high)
        errors.append(float(abs((centers+ranges*np.tanh(latent+np.where(finite,high,0)))-physical-.5*units)[finite].max()))
    assert max(errors)<1e-10
    cruise_indices=np.asarray(cruise_indices)
    np.savez_compressed(stage/'geometry.npz',metric_latent=metrics['latent'],metric_physical=metrics['physical'],
        observation=observations,
        support=support.reshape(-1,768),lower_latent=np.tile(-.5*AMPLITUDE,(len(features),1)).ravel(),
        upper_latent=np.tile(.5*AMPLITUDE,(len(features),1)).ravel(),lower_physical=low.ravel(),upper_physical=high.ravel(),
        cruise_features=features[cruise_indices,1],cruise_latent=latent[cruise_indices,1],cruise_weight=np.asarray(cruise_weights),
        state_path=np.array(state_paths),cruise_state_indices=cruise_indices,physical_units=units,
        action_centers=centers,action_ranges=ranges,latent=latent,decoded_actions=np.asarray(decoded_actions),state_weight=weights)
    write_json(stage/'results.json',dict(parent_sha256=sha(model),states=len(features),cruise_states=len(cruise_indices),
        arms=['latent','physical'],physical_units=units.tolist(),exact_bound_mapping_max_error=max(errors),
        cruise_support=args.cruise_support,cruise_target_paths=len(cruise_groups),
        current_policy_changed=False,common_objective_target_mm_s=99.5,costs=costs,goal_complete=False,
        actual_decoder=f'{decoder.__module__}.{decoder.__qualname__}',action_centers=centers.tolist(),action_ranges=ranges.tolist(),
        nominal_runtime_decoder_max_error=nominal_decoder_error,perturbed_runtime_decoder_max_error=max(decoder_errors)))
    receipt(stage,dict(status='prepared_matched_physical_and_latent_update_geometries',states=len(features),
        cruise_states=len(cruise_indices),costs=costs,wall_s=time.perf_counter()-start))
    print('PARAMETER_GEOMETRIES',len(features),'cruise',len(cruise_indices),'mapping_error',max(errors),flush=True)


if __name__=='__main__':main()
