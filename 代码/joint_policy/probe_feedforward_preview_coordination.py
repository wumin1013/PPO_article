"""Finite preview-bias compensation for an altered feedforward direction."""
import time
import numpy as np
import torch
from joint_runtime import tracking
from joint_runtime_v5 import environment_factory
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from probe_feedforward_velocity_source import demand_with_sources
from train_pilot import evaluate,save_torch
from learn_exit_local_credit import summary
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    parent=BASE/'exit_local_credit_parameter_learning_20260916';initial=parent/'candidate_1.pt'
    source=BASE/'normal_feedforward_source_probe_20260916'
    indices=[0,1,6,7,12,13,14,15,47,50]
    stage=register('feedforward_preview_coordination_probe_20260916',
        [__file__,initial,parent/'receipt.json',source/'receipt.json',source/'results.json',
         BASE/'probe_feedforward_velocity_source.py',BASE/'local_geometry_feedback_policy.py',BASE/'learn_exit_local_credit.py'],
        dict(new_variants=2,training_paths=indices,preview_latent_biases=[.15,.3],actor_calls=4000,
             physics_ticks=100000,wall_s=90),
        'Limited coordination probe after real-velocity-direction feedforward broadened sharp '
        'corners but increased exit crossing. Under that diagnostic executor only, add0.15 or0.3 '
        'to the learned preview latent head bias globally; retain the state-dependent learned '
        'steering/feed functions and all constraints. Zero-bias10-path physics is reused. '
        'This tests whether anticipation compensates the changed tracking response without '
        'angle-specific timing or geometry templates. It is a finite parameter intervention, '
        'not retraining, deployment or final learned performance. Reject absent complete success '
        'and real inward/tangent shape; do not treat a failure prefix as a broad successful turn.')
    original=tracking.compute_tracking_demand;started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=LocalGeometryFeedbackPolicy(mask_endpoint=True)
    paths=read(BASE/'learning_preparation_v4/paths.json')['training'];training=[np.asarray(paths[i]) for i in indices]
    baseline=next(r for r in read(source/'results.json')['records'] if r['bits']==[0,1]);records=[]
    try:
        tracking.compute_tracking_demand=lambda v,a,d,r:demand_with_sources(v,a,d,r,actual_magnitude=False,actual_direction=True)
        for bias in (.15,.3):
            policy.load_state_dict(saved['policy'])
            with torch.no_grad():policy.actor[-1].bias[2].add_(bias)
            name=f'probe_bias_{bias:g}.pt';save_torch(stage/name,dict(policy=policy.state_dict(),
                architecture='local_geometry_feedback',mask_endpoint=True,preview_latent_bias=bias,
                diagnostic_executor='actual_velocity_direction_feedforward',parent_sha256=sha(initial),deployment_accepted=False))
            tag=f'bias_{bias:g}';results=evaluate(policy,training,stage,tag,None,costs,started+90,environment_factory)
            for row,i in zip(results,indices):row['original_path']=i
            row=summary(stage/tag,results);row.update(bias=bias,model=name,paths=results);records.append(row)
            write_json(stage/'results.json',dict(baseline_reused=baseline,records=records,costs=costs,goal_complete=False))
            print('FF_PREVIEW',bias,row['successes'],row['full_time_s'],[(s['path'],s['success'],round(s['out_edge_outward_mm'],3),round(s['full_physical_reverse_deg'],2),round(s['full_middle80_turn_arc_mm'],2)) for s in row['shapes']],flush=True)
    finally:tracking.compute_tracking_demand=original
    assert costs['evaluation_actor_calls']<=4000 and costs['evaluation_physics_ticks']<=100000
    receipt(stage,dict(status='completed_feedforward_preview_compensation_probe',costs=costs,
                       frozen_executor_restored=tracking.compute_tracking_demand is original,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
