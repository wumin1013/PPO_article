"""Separate historical origin from the live retained exit-update tradeoff."""
import itertools
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from learn_constrained_entry_credit import complete_summary
from complete_corner_metrics import complete_corner_metrics
from train_pilot import evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json


def main():
    torch.set_num_threads(1)
    old_model=BASE/'local_geometry_continuous_reward_ppo_20260916/policy_18645.pt'
    new_model=BASE/'exit_local_credit_parameter_learning_20260916/candidate_1.pt'
    old_pool=BASE/'local_geometry_full_training_audit_20260916/training_pool'
    new_pool=BASE/'exit_local_credit_parameter_learning_20260916/training_1'
    paths=list(range(6,14))
    stage=register('retained_exit_update_entry_causality_20260916',
        [__file__,old_model,new_model,old_pool/'summary.json',new_pool/'summary.json',
         BASE/'exit_local_credit_parameter_learning_20260916/receipt.json',
         BASE/'preswing_mechanism_chain_20260916/receipt.json',BASE/'local_geometry_feedback_policy.py',
         BASE/'complete_corner_metrics.py',BASE/'joint_runtime_v5.py'],
        dict(new_full_paths=48,reused_full_paths=16,evaluation_actor_calls=5000,
             evaluation_physics_ticks=120000,wall_s=180,new_training_decisions=0),
        'Live-model causal audit: exact2x2x2 last-output-head composition between the '
        'PPO18645 parent and its retained learned exit-credit update. Reuse both000/111 '
        'complete trajectories on8 existing training corners. New six head compositions '
        'share every other tensor, original runtime and constraints. Measure true entry '
        'excursion/duration, heading/reversal and exit shape. No selection or adoption; '
        'this is distinct from the earlier rejected-MSE-model head ablation. Determine '
        'whether the currently retained exit improvement itself changes entry behavior, '
        'and which head or interaction mediates that physical effect. Do not confuse '
        'this update-specific causal effect with the entire historical origin.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0)
    old=torch.load(old_model,map_location='cpu',weights_only=False)['policy']
    new=torch.load(new_model,map_location='cpu',weights_only=False)['policy']
    for name in old:
        if name not in ('actor.4.weight','actor.4.bias'):assert torch.equal(old[name],new[name])
    geometry=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['training']]
    records=[]
    for bits in itertools.product((0,1),repeat=3):
        tag=''.join(map(str,bits))
        if tag in ('000','111'):
            pool=old_pool if tag=='000' else new_pool
            saved=read(pool/'summary.json')['results']
            result=[dict(saved[i],original_path=i) for i in paths]
            traces=[pool/f'{i}.npz' for i in paths]
            reused=True
        else:
            p=LocalGeometryFeedbackPolicy(mask_endpoint=True);p.load_state_dict(old)
            with torch.no_grad():
                for h,enabled in enumerate(bits):
                    if enabled:
                        p.actor[-1].weight[h].copy_(new['actor.4.weight'][h])
                        p.actor[-1].bias[h].copy_(new['actor.4.bias'][h])
            result=evaluate(p,[geometry[i] for i in paths],stage,tag,None,costs,started+180,environment_factory)
            result=[dict(r,original_path=i) for r,i in zip(result,paths)]
            traces=[stage/tag/f'{i}.npz' for i in range(len(paths))]
            reused=False
        row=dict(bits=list(bits),reused=reused,**complete_summary(result,traces))
        records.append(row)
        print('RETAINED_CAUSE',tag,row['successes'],
            'entries',[round(s['in_edge_outward_mm'],5) for s in row['shapes']],flush=True)
        write_json(stage/'results.json',dict(records=records,costs=costs,candidate_selected=False))
        assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=120000
    receipt(stage,dict(status='completed_retained_update_head_causality',costs=costs,
        reused_full_paths=16,new_full_paths=48,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
