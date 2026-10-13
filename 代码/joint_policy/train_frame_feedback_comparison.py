"""Matched offline coordinate-frame experiment; no teacher schedule at runtime."""
import time
import argparse
import traceback
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy,heading_frame,PAIRS
from reflection_policy import reflect_observation
from train_feedback_from_optimized_rollouts import decoded
from train_pilot import save_torch,evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from study_io import BASE,register,receipt,sha,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frame',required=True,choices=['reference','heading']);parser.add_argument('--mask-history',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(916853)
    source=BASE/'interior_teacher_feedback_learning_20260916'
    stage=register(('history_independent_' if args.mask_history else '')+f'{args.frame}_frame_feedback_learning_20260916',[__file__,BASE/'frame_feedback_policy.py',
        BASE/'reflection_policy.py',BASE/'joint_policy.py',source/'training_data.npz',source/'training_sources.json',source/'receipt.json',
        BASE/'gate_observation_effect_audit_20260916/receipt.json',BASE/'joint_runtime.py'],
        dict(supervised_updates=2000,learning_rate=.001,seed=916853,evaluation_epochs=[500,1000,2000],
             training_decisions=0,evaluation_decisions=6000,physics_ticks=180000,wall_s=360,frame=args.frame,history_independent=args.mask_history),
        ('Single-factor history-input ablation within the fresh heading-frame recipe: identical initial arrays, data, weights, loss and2000 updates as heading_frame_feedback_learning. Replace previous held commands8:11 with reset constants. This is distinct from the earlier200-update warm-start history test; gains0.89 steering and0.91 preview in the new trained actor motivate this controlled test. Do not claim that masking alone proves the cause of initial reverse instructions. ' if args.mask_history else '')+
        'Matched fresh-network coordinate representation comparison. Both arms use identical96x96 tanh actor, exact reflection averaging, initial parameter arrays, existing interior teacher dataset/path weights, physical-action loss and2000 bounded fitting updates. Only vector-coordinate encoding differs between the original two arms; heading arm rotates to current filtered heading while preserving angle scalars and every input dimension. Compared with prior warm starts, fresh architecture/training recipe also changes, so only between the original two arms is a coordinate-only comparison justified. Runtime has network weights and coordinate transform only. No path timing coefficients, angle tables, relaxed constraints or sealed tests. Stop at budget, retain only actual completion/shape/cruise gains.')
    started=time.perf_counter();p=FrameFeedbackPolicy(72,frame=args.frame,history_independent=args.mask_history)
    with np.load(source/'training_data.npz') as z:x=torch.tensor(z['observation']);y=torch.tensor(z['actions']);w=torch.tensor(z['sample_weight'])
    encoded=heading_frame(x)
    assert torch.max(abs(heading_frame(reflect_observation(x))-reflect_observation(encoded)))<1e-6
    inverse=encoded.clone();inverse[:,7]*=-1;recovered=heading_frame(inverse)
    assert max(float(torch.max(abs(recovered[:,i:i+2]-x[:,i:i+2]))) for i in PAIRS)<1e-6
    with torch.no_grad():
        a=p.distribution(x).mean;b=p.distribution(reflect_observation(x)).mean
        assert torch.equal(a*x.new_tensor([-1.,1.,1.]),b)
    params=list(p.actor.parameters());opt=torch.optim.Adam(params,lr=.001);scale=x.new_tensor([np.deg2rad(3),10.,1.])
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    validation=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    checkpoints=[];history=[]
    try:
        for epoch in range(1,2001):
            assert time.perf_counter()-started<360
            pred=decoded(p.distribution(x).mean);errors=(pred-y)/scale
            loss=(torch.nn.functional.smooth_l1_loss(errors,torch.zeros_like(errors),reduction='none').mean(1)*w).sum()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();costs['supervised_updates']+=1
            if epoch%100==0:history.append(dict(epoch=epoch,loss=float(loss.detach()),physical_rmse=torch.sqrt(((pred-y)**2*w[:,None]).sum(0)).detach().tolist()))
            if epoch not in (500,1000,2000):continue
            model=stage/f'epoch_{epoch}.pt';save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='frame_feedback',frame=args.frame,
                history_independent=args.mask_history,learning_method='supervised_feedback_coordinate_comparison',epoch=epoch))
            results=evaluate(p,validation,stage,f'validation_{epoch}',None,costs,started+360,environment_factory)
            assert costs['evaluation_actor_calls']<=6000 and costs['evaluation_physics_ticks']<=180000
            shapes=[];cruise=[]
            for i in range(12):
                with np.load(stage/f'validation_{epoch}/{i}.npz') as z:
                    speed=np.linalg.norm(z['rows'][:,2:4],axis=1);cruise.append(dict(path=i,peak=float(speed.max()),ge95_seconds=float((speed>=95).sum()*.001)))
                    if i<4:
                        s=corner_shape(z['reference'],z['rows'],z['actions']);s['spatial']=geometric_turn_metrics(z['reference'],z['rows']);shapes.append(s)
            row=dict(epoch=epoch,successes=sum(r['success'] for r in results),time_s=sum(r['completion_time_s'] for r in results) if all(r['success'] for r in results) else None,
                shapes=shapes,cruise=cruise,model_sha256=sha(model),final_shape_acceptance=False)
            checkpoints.append(row);write_json(stage/'progress.json',dict(checkpoints=checkpoints,fit_history=history,costs=costs))
            print('FRAME',args.frame,'EPOCH',epoch,row['successes'],row['time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
        write_json(stage/'results.json',dict(checkpoints=checkpoints,fit_history=history,costs=costs,goal_complete=False,
            transform_information_preserved=True,reflection_commutation_verified=True))
        receipt(stage,dict(status='completed_coordinate_frame_comparison_arm',frame=args.frame,costs=costs,wall_s=time.perf_counter()-started,training_data_sha256=sha(source/'training_data.npz')))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
