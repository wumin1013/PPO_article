"""Finite-width continuous sensing preparation from existing full training traces."""
import time
import numpy as np
import torch
from continuous_sensor_runtime import continuous_sensor_observation,environment_factory
from kernel_local_feedback_policy import KernelLocalFeedbackPolicy
from probe_coordinated_suffix import reconstruct
from learn_single_decision_credit import review
from joint_runtime import SENSOR_DISTANCES
from search_complete_return import read
from train_pilot import evaluate
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'recentered_reverse_shape_credit_learning_20260917'
    selection=read(parent/'training_selection.json');model=parent/selection['model']
    stage=register('continuous_sensor_preparation_20260917',
        [__file__,BASE/'continuous_sensor_runtime.py',BASE/'joint_runtime_v5.py',
         BASE/'joint_runtime.py',BASE/'kernel_local_feedback_policy.py',
         BASE/'probe_coordinated_suffix.py',model,parent/'training_selection.json',
         BASE/'large_step_single_sensor_causal_probe_20260917/receipt.json'],
        dict(saved_training_states=4167,physical_screen_paths=[3,8,13,15],
            sensor_full_width_mm=2.,saved_prefix_geometry_ticks=90000,
            evaluation_actor_calls=1000,evaluation_physics_ticks=25000,wall_s=180,
            training_updates=0),
        'The current bidirectional intervention identifies point-tangent switching as '
        'a cause of large-step shape regressions. Replace point directions by exact '
        'mean reference tangent over a2mm centered arc interval (maximum displacement '
        'within one existing20ms decision). Width is global and derived from the '
        'common speed/time limits, not angle or path tuned. Clip the interval to '
        'finite path endpoints and divide by its actual arc width; do not normalize '
        'the chord. Other observations and all physical execution remain unchanged. '
        'This differs from the old unequal9-interval geometry distillation: sensors '
        'retain their original centers and local support. Reconstruct64 saved '
        'training traces, verify original means/actions, save paired encodings for '
        'a later bounded learned adaptation. Check continuity and other-input '
        'identity. Four unchanged-weight closed-loop rollouts diagnose initialization '
        'only; they are not learned corrections or promotion. No development/sealed '
        'access, teacher trajectories, control overrides, or new credit collection.')
    start=time.perf_counter();costs=dict(saved_prefix_geometry_ticks=0,saved_actor_calls=0,
        evaluation_actor_calls=0,evaluation_physics_ticks=0)
    policy=KernelLocalFeedbackPolicy(256)
    policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    old_all=[];new_all=[];targets=[];path_ids=[];weights=[];limits=[]
    other=np.r_[np.arange(42),np.arange(62,72)]
    for path,trace in enumerate(selection['traces']):
        with np.load(BASE/trace) as z:ref,rows,actions=[z[k].copy() for k in ('reference','rows','actions')]
        obs=[];states=[]
        env=reconstruct(ref,rows,actions,(len(actions)-1)*20,costs,obs,states)
        obs.append(env.observation());states.append(env.state())
        transformed=[];means=[]
        for x,a,state in zip(obs,actions,states):
            env.restore(state);new=continuous_sensor_observation(env,x)
            assert np.array_equal(x[other],new[other])
            assert np.linalg.norm(new[42:62].reshape(10,2),axis=1).max()<=1+1e-6
            mean=policy.act(x,True)[0];costs['saved_actor_calls']+=1
            assert np.array_equal(env.decode_action(mean),a)
            transformed.append(new);means.append(mean)
        if path in (3,8,13,15):
            incoming=[j for j,s in enumerate(states) if s['tracker']['frontier']==0]
            u=(ref[1]-ref[0])/np.linalg.norm(ref[1]-ref[0]);length=np.linalg.norm(ref[1]-ref[0])
            position=np.asarray([(states[j]['position']-ref[0])@u for j in incoming])
            for distance in SENSOR_DISTANCES[1:]:
                location=length-distance
                if not position.min()<=location<=position.max():continue
                anchor=incoming[int(np.argmin(abs(position-location)))];pair=[]
                for sign in (-1,1):
                    env.restore(states[anchor]);env.position+=(location+sign*1e-5-(env.position-ref[0])@u)*u
                    pair.append(continuous_sensor_observation(env))
                with torch.no_grad():value=policy.distribution(torch.tensor(np.asarray(pair))).mean.numpy()
                limits.append(dict(path=path,sensor_distance_mm=float(distance),
                    maximum_feature_change=float(abs(pair[1][42:62]-pair[0][42:62]).max()),
                    latent_jump=(value[1]-value[0]).tolist()))
        old_all.extend(obs);new_all.extend(transformed);targets.extend(means)
        path_ids.extend([path]*len(obs));weights.extend([1/(64*len(obs))]*len(obs))
    assert len(old_all)==4167 and costs['saved_prefix_geometry_ticks']<=90000
    np.savez_compressed(stage/'training_pairs.npz',old_observation=np.asarray(old_all),
        observation=np.asarray(new_all),target=np.asarray(targets),
        path_id=np.asarray(path_ids),weight=np.asarray(weights))
    write_json(stage/'continuity.json',dict(sensor_half_width_mm=1.,limits=limits,
        maximum_feature_jump=max(r['maximum_feature_change'] for r in limits),
        maximum_latent_jump=max(max(abs(np.asarray(r['latent_jump']))) for r in limits),
        all_other_inputs_identical=True,parent_sha256=sha(model),
        gate_progress_discontinuity_removed=False))
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];ids=[3,8,13,15]
    results=evaluate(policy,[np.asarray(paths[i]) for i in ids],stage,'unchanged_weights',None,
        costs,start+180,environment_factory)
    results=[dict(r,original_path=i) for r,i in zip(results,ids)]
    traces=[stage/'unchanged_weights'/f'{j}.npz' for j in range(len(ids))]
    value=review(results,traces)
    baseline=review([selection['results'][i] for i in ids],[BASE/selection['traces'][i] for i in ids])
    write_json(stage/'results.json',dict(baseline=baseline,unchanged_weights=value,costs=costs,
        learning_performed=False,promoted=False,goal_complete=False))
    assert costs['evaluation_actor_calls']<=1000 and costs['evaluation_physics_ticks']<=25000
    receipt(stage,dict(status='completed_continuous_sensor_learning_preparation',costs=costs,wall_s=time.perf_counter()-start))
    for k in ('successes','entry_sum_mm','exit_sum_mm','full_reverse_sum_deg','full_time_s'):
        print('CONTINUOUS_SENSOR',k,baseline[k],value[k],flush=True)


if __name__=='__main__':main()
