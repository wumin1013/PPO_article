"""Saved-state feed reach and terminal-time sensitivity of the current learner."""
import time
import numpy as np
import torch
from scipy.linalg import solve_triangular
from canonical_gate_state import CanonicalGatePolicy,checkpoint_environment_factory
from probe_coordinated_suffix import reconstruct
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    parent=BASE/'canonical_gate_band_bend_interpolated_learning_20260917'
    reference=BASE/'canonical_gate_band_refined_energy_current_reference_20260917'
    data=read(parent/'results.json');candidate=data['full_candidate'];model=parent/data['chosen']['model']
    assert candidate['checks']['eligible'] and candidate['summary']['successes']==64
    old_duration=BASE/'canonical_feed_credit_duration_probe_20260917'
    stage=register('canonical_gate_band_current_cruise_learning_support_20260917',
        [__file__,model,parent/'results.json',parent/'receipt.json',reference/'basis.npz',
         reference/'results.json',reference/'receipt.json',BASE/'solve_transition_shape_step.py',
         BASE/'joint_runtime.py',BASE/'probe_coordinated_suffix.py',
         old_duration/'results.json',old_duration/'receipt.json'],
        dict(new_physics_ticks=0,saved_geometry_ticks=30000,replayed_actor_calls=2000,wall_s=60),
        'Diagnose why the accepted shape/energy update still lacks sufficient '
        'cruise. Reconstruct only the16 saved pure-straight training paths. '
        'At actual20ms decisions in the existing6.2mm endpoint-excluded '
        'diagnostic window, measure actor demand and invert its exact tanh '
        'feed mapping to99 and99.5mm/s. Compute the largest possible feed '
        'latent change at each saved state under the CURRENT metric radius0.1 '
        'by Cauchy-Schwarz, ignoring all other constraints, so it is an '
        'optimistic necessary reachability bound in this local shared-residual '
        'proposal. Do not claim that a fixed-state bound limits all future '
        'closed-loop trajectories. Separately document the held-action bound '
        '0.5*feed scale0.1=0.05 at the29 credit-path saved states only; do not '
        'incorrectly apply that extra support bound to unseen straight states. '
        'Compare saved parent/current brake initiation and complete times '
        'on all64 paths; reuse the prior20 duration probes, including nonzero '
        'time changes. The current objective contains shape/energy gain and '
        'only a time nondegradation screen, not a cruise/time descent target. '
        'This audit selects a mechanism change, not a deployed feed schedule '
        'or evidence of time improvement from failed prefixes.')
    start=time.perf_counter();torch.set_num_threads(1)
    metadata=torch.load(model,map_location='cpu',weights_only=False)
    p=CanonicalGatePolicy(256);p.load_state_dict(metadata['policy'])
    factory=checkpoint_environment_factory(metadata['architecture'])
    with np.load(reference/'basis.npz') as z:chol=np.linalg.cholesky(z['metric'])
    costs=dict(saved_prefix_geometry_ticks=0,new_physics_ticks=0,replayed_actor_calls=0)
    records=[];state_rows=[];time_rows=[]
    old={r['original_path']:(r,t) for r,t in zip(data['full_parent']['results'],data['full_parent']['traces'])}
    for result,trace in zip(candidate['results'],candidate['traces']):
        path=result['original_path']
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        old_result,old_trace=old[path]
        with np.load(BASE/old_trace) as z:old_rows=z['rows'].copy()
        first=lambda r:int(np.flatnonzero(r[:,12])[0])
        time_rows.append(dict(path=path,parent_time_s=old_result['completion_time_s'],
            current_time_s=result['completion_time_s'],parent_brake_tick=first(old_rows),current_brake_tick=first(rows)))
        if len(ref)!=2:continue
        states=[];env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,
            state_snapshots=states,env_factory=factory);states.append(env.state())
        unit=(ref[-1]-ref[0])/np.linalg.norm(ref[-1]-ref[0]);length=float(np.linalg.norm(ref[-1]-ref[0]))
        observations=[];decisions=[];latent=[]
        for k,state in enumerate(states):
            along=float((state['position']-ref[0])@unit)
            if not 6.2<=along<=length-6.2:continue
            env.restore(state);observation=env.observation();a=p.act(observation,True)[0]
            costs['replayed_actor_calls']+=1
            assert np.array_equal(env.decode_action(a),actions[k])
            observations.append(observation);decisions.append(k);latent.append(float(a[1]))
        with torch.no_grad():features=(p.residual_features(torch.tensor(np.array(observations)))[:,1]*p.residual_scale[1]).numpy()
        # The metric is block diagonal; whitening the full row preserves the exact bound.
        support=np.zeros((len(features),768));support[:,256:512]=features
        dual_norm=np.linalg.norm(solve_triangular(chol,support.T,lower=True),axis=0)
        latent=np.array(latent);feed=50*(1+np.tanh(latent));upper=50*(1+np.tanh(latent+.1*dual_norm))
        required99=np.maximum(0.,np.arctanh(.98)-latent)
        required995=np.maximum(0.,np.arctanh(.99)-latent)
        needed99=required99/np.maximum(dual_norm,1e-30)
        needed995=required995/np.maximum(dual_norm,1e-30)
        for k,z,f,n,u,r99,r995 in zip(decisions,latent,feed,dual_norm,upper,needed99,needed995):
            state_rows.append([path,k,z,f,n,u,r99,r995])
        records.append(dict(path=path,decisions=len(decisions),demand_median_mm_s=float(np.median(feed)),
            optimistic_single_step_demand_upper_median_mm_s=float(np.median(upper)),
            fraction_saved_states_cannot_request99_in_radius=float(np.mean(upper<99.)),
            fraction_saved_states_cannot_request99p5_in_radius=float(np.mean(upper<99.5)),
            necessary_metric_norm_for99_quantiles=np.quantile(needed99,[0,.5,1]).tolist(),
            necessary_metric_norm_for99p5_quantiles=np.quantile(needed995,[0,.5,1]).tolist()))
        assert time.perf_counter()-start<55 and costs['saved_prefix_geometry_ticks']<=30000 and costs['replayed_actor_calls']<=2000
    old_probe=read(old_duration/'results.json')
    nonzero=[dict(path=r['path'],sign=r['sign'],duration=r['duration'],change_s=r['time_change_s'])
        for r in old_probe['records'] if r['success'] and abs(r['time_change_s'])>1e-9]
    np.savez_compressed(stage/'saved_straight_state_bounds.npz',rows=np.asarray(state_rows),
        columns=np.array(['path','decision','feed_latent','feed_mm_s','metric_dual_norm','optimistic_upper_feed_mm_s','needed_metric_norm_99','needed_metric_norm_99p5']))
    write_json(stage/'results.json',dict(parent_sha256=sha(model),straight_records=records,
        timing=time_rows,changed_timing=[r for r in time_rows if r['current_time_s']!=r['parent_time_s']],
        historical_duration_probe_nonzero_changes=nonzero,
        current_objective_has_explicit_time_or_cruise_descent=False,
        saved_credit_state_maximum_feed_latent_change=.05,
        saved_state_upper_bounds_are_not_closed_loop_guarantees=True,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_saved_cruise_learning_support_audit',straight_paths=len(records),
        costs=costs,wall_s=time.perf_counter()-start,goal_complete=False))
    print('CRUISE_SUPPORT',records,flush=True)
    print('CHANGED_TIMES',[r for r in time_rows if r['current_time_s']!=r['parent_time_s']],flush=True)


if __name__=='__main__':main()
