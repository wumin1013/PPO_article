"""Isolate normal-feedforward magnitude/direction sources with frozen policy."""
import time
import numpy as np
import torch
from joint_runtime import tracking
from joint_runtime_v5 import environment_factory
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from train_pilot import evaluate
from learn_exit_local_credit import summary
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read

ORIGINAL=tracking.compute_tracking_demand


def demand_with_sources(velocity,acceleration,desired,guidance_rate,*,actual_magnitude,actual_direction):
    if not actual_magnitude and not actual_direction:
        return ORIGINAL(velocity,acceleration,desired,guidance_rate)
    velocity,acceleration,desired=[np.asarray(x,float) for x in (velocity,acceleration,desired)]
    actual_norm=float(np.linalg.norm(velocity));desired_norm=float(np.linalg.norm(desired))
    if actual_magnitude and actual_direction:source=velocity
    elif actual_magnitude:source=desired*(actual_norm/desired_norm) if desired_norm>1e-12 else np.zeros(2)
    else:source=velocity*(desired_norm/actual_norm) if actual_norm>1e-12 else desired
    requested=float(guidance_rate)*np.array([-source[1],source[0]])
    applied=tracking.ball(requested,np.zeros(2),1998.)
    jerk=1600.*(desired-velocity)-80.*(acceleration-applied)
    return velocity,acceleration,jerk,dict(normal_ff_requested_mm_s2=float(np.linalg.norm(requested)),
        normal_ff_applied_mm_s2=float(np.linalg.norm(applied)),filtered_guidance_rate_rad_s=float(guidance_rate),
        tracking_desired_jerk_mm_s3=float(np.linalg.norm(jerk)))


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';model=parent/'candidate_1.pt'
    indices=[0,1,6,7,12,13,14,15,47,50]
    stage=register('normal_feedforward_source_probe_20260916',
        [__file__,model,parent/'receipt.json',BASE/'joint_runtime.py',BASE/'joint_runtime_v5.py',
         BASE/'exit_local_credit_full_xy_review_20260916/feedforward_saved_data_audit.json',
         BASE/'local_geometry_feedback_policy.py',BASE/'learn_exit_local_credit.py']+
         [parent/'training_1'/f'{i}.npz' for i in indices],
        dict(new_variants=3,training_paths=indices,actor_calls=6000,physics_ticks=150000,wall_s=120),
        'Fixed learned-policy2x2 executor mechanism probe: normal feedforward magnitude from desired '
        'versus actual velocity, and direction from desired versus actual velocity. Reuse completed '
        'desired/desired baseline; run only three new variants. Keep1600/80 tracking gains,1998 '
        'feedforward regularization, frozen planar V/A/J projector, LOS, policy weights, actions, '
        '20ms decisions,1ms execution, ordered coverage and full terminal stop unchanged. Startup '
        'actual direction uses desired direction only when actual velocity is zero. This runtime '
        'patch exists only inside this diagnostic process and is always restored; no frozen source '
        'is edited. Evaluate both turn directions, shallow/mid/sharp corners, long route and straight. '
        'Require complete paths and actual shape benefit before considering retraining under a new '
        'common executor; no claim of newly learned skill or fair algorithmic superiority.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True);policy.load_state_dict(torch.load(model,map_location='cpu',weights_only=False)['policy'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];training=[np.asarray(paths[i]) for i in indices]
    original=read(parent/'training_1/summary.json')['results']
    base_results=[dict(original[i],original_path=i) for i in indices]
    baseline=summary(parent/'training_1',base_results,[parent/'training_1'/f'{i}.npz' for i in indices])
    records=[dict(bits=[0,0],reused=True,**baseline)]
    # The00 branch delegates exactly; verify representative saved physical states.
    for i in indices:
        with np.load(parent/'training_1'/f'{i}.npz') as z:rows=z['rows']
        for row in rows[::31]:
            args=(row[2:4],row[4:6],row[8:10],row[10])
            expected=ORIGINAL(*args);actual=demand_with_sources(*args,actual_magnitude=False,actual_direction=False)
            assert all(np.array_equal(expected[k],actual[k]) for k in range(3))
    try:
        for mag,direction in [(1,0),(0,1),(1,1)]:
            def replacement(v,a,d,r):return demand_with_sources(v,a,d,r,actual_magnitude=mag,actual_direction=direction)
            tracking.compute_tracking_demand=replacement
            tag=f'magnitude{mag}_direction{direction}'
            result=evaluate(policy,training,stage,tag,None,costs,started+120,environment_factory)
            for r,i in zip(result,indices):r['original_path']=i
            row=summary(stage/tag,result);row.update(bits=[mag,direction],reused=False,paths=result)
            records.append(row);write_json(stage/'results.json',dict(records=records,costs=costs,goal_complete=False))
            print('FF_SOURCE',mag,direction,'success',row['successes'],'time',row['full_time_s'],
                  'exit',row['median_exit_mm'],'reverse',row['full_reverse_sum_deg'],flush=True)
            assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=150000
    finally:
        tracking.compute_tracking_demand=ORIGINAL
    receipt(stage,dict(status='completed_normal_feedforward_source_probe',costs=costs,
        frozen_tracking_function_restored=tracking.compute_tracking_demand is ORIGINAL,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
