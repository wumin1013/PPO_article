"""One-factor replay-data contrast: prohibit redundant held-action copying."""
import time
import traceback
import numpy as np
import torch
from history_independent_policy import HistoryIndependentPolicy
from train_feedback_from_optimized_rollouts import decoded
from train_pilot import save_torch,evaluate
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from study_io import BASE,register,receipt,sha,write_json


def main():
    torch.set_num_threads(1);torch.manual_seed(916509)
    source=BASE/'interior_teacher_feedback_learning_20260916'
    initial=BASE/'preview_state_residual_learning_20260916/best_generation_5.pt'
    stage=register('history_independent_feedback_learning_20260916',[__file__,BASE/'history_independent_policy.py',
        BASE/'train_feedback_from_optimized_rollouts.py',source/'training_data.npz',source/'receipt.json',initial,
        BASE/'feedback_credit_history_audit_20260916/receipt.json',BASE/'joint_runtime.py',BASE/'dynamic_residual_policy.py'],
        dict(supervised_updates=200,learning_rate=.0003,seed=916509,evaluation_epochs=[50,100,200],
             training_decisions=0,evaluation_decisions=6000,physics_ticks=180000,wall_s=240),
        'Single-factor representation contrast to interior_teacher_feedback_learning: replace previous held-action input8:11 with reset constants, since step replaces held before computing dynamics. Same exact32 trajectory dataset, sample weights, initial parameter arrays, optimizer,200 updates and validation checkpoints. This tests shortcut sensitivity. Reject if full completion, actual shape and cruising do not materially improve; do not extend epochs. No sealed test or changed constraints.')
    started=time.perf_counter();saved=torch.load(initial,map_location='cpu',weights_only=False)
    p=HistoryIndependentPolicy(72);p.load_state_dict(saved['policy'])
    with np.load(source/'training_data.npz') as z:
        x=torch.tensor(z['observation']);y=torch.tensor(z['actions']);w=torch.tensor(z['sample_weight'])
    with torch.no_grad():
        changed=x.clone();changed[:,8:11]=torch.randn_like(changed[:,8:11])
        assert torch.equal(p.distribution(x).mean,p.distribution(changed).mean)
    params=list(p.actor.parameters())+list(p.dynamic_head.parameters())+list(p.preview_head.parameters())
    opt=torch.optim.Adam(params,lr=.0003);scale=x.new_tensor([np.deg2rad(3),10.,1.])
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    validation=[np.asarray(r) for r in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    checkpoints=[];history=[]
    try:
        for epoch in range(1,201):
            assert time.perf_counter()-started<240
            pred=decoded(p.distribution(x).mean);errors=(pred-y)/scale
            loss=(torch.nn.functional.smooth_l1_loss(errors,torch.zeros_like(errors),reduction='none').mean(1)*w).sum()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();costs['supervised_updates']+=1
            if epoch%10==0:history.append(dict(epoch=epoch,loss=float(loss.detach()),physical_rmse=torch.sqrt(((pred-y)**2*w[:,None]).sum(0)).detach().tolist()))
            if epoch not in (50,100,200):continue
            model=stage/f'epoch_{epoch}.pt';save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='history_independent_preview_residual',
                learning_method='supervised_feedback_history_input_ablation',epoch=epoch,parent_sha256=sha(initial)))
            results=evaluate(p,validation,stage,f'validation_{epoch}',None,costs,started+240,environment_factory)
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
            print('EPOCH',epoch,row['successes'],row['time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
        write_json(stage/'results.json',dict(checkpoints=checkpoints,fit_history=history,costs=costs,goal_complete=False,history_invariance_exact=True))
        receipt(stage,dict(status='completed_history_input_ablation',costs=costs,wall_s=time.perf_counter()-started,training_data_sha256=sha(source/'training_data.npz')))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
