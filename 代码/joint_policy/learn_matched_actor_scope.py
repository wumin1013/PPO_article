"""Matched output-layer/full-actor adaptation on the same coordinated data."""
import copy
import time
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from learn_exit_local_credit import summary
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'exit_local_credit_parameter_learning_20260916';initial=source/'candidate_1.pt'
    fitted=BASE/'matched_entry_interaction_feedback_20260916'
    indices=list(range(16))+[16,28,31,40,41]
    stage=register('matched_actor_scope_feedback_20260916',
        [__file__,initial,source/'receipt.json',fitted/'window_training_data.npz',fitted/'receipt.json',
         BASE/'updated_head_causality_probe_20260916/receipt.json',BASE/'local_geometry_feedback_policy.py',
         BASE/'frame_feedback_policy.py',BASE/'learn_exit_local_credit.py',BASE/'learning_preparation_v4/paths.json'],
        dict(seed=9161271,scopes=['output','full_actor'],actor_lr=.00003,full_batch_updates=300,
             checkpoints=[30,100,300],gradient_clip=1.,ridge=.001,screen_indices=indices,
             evaluation_decisions=20000,physics_ticks=450000,wall_s=480,full_pool_candidates_at_most=2,
             development_candidates_at_most=2),
        'Test learned function selectivity after exact head ablation showed isolated steering/preview '
        'updates cause unrelated-path failures. Both arms share parent, coordinated branch data, weighted '
        'normalized latent MSE, parameter penalty, Adam settings and30/100/300 checkpoint opportunities. '
        'Only trainable actor scope differs: final output layer or all actor layers. The critic is frozen. '
        'This matched pair is not a one-factor comparison to the previous closed-form ridge fit. '
        'Screen every isolated training corner plus all mixed paths failing in the prior fit. Select '
        'at most one checkpoint per arm only with21/21 success,entry<=0.05mm,median exit<=80% parent, '
        'no increased maximum exit/full reversal sum,and lower complete time+two-edge cost. Evaluate '
        'only remaining43 training paths, reusing the21 screening paths, before any development run. '
        'Only64/64 physical/shape qualifying candidates enter development. Report regression losses as '
        'optimization evidence only, not policy success; stop at budget without unchanged extension.')
    started=time.perf_counter();costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False);parent=saved['policy']
    teacher=LocalGeometryFeedbackPolicy(mask_endpoint=True);teacher.load_state_dict(parent)
    with np.load(fitted/'window_training_data.npz') as z:
        x=torch.tensor(z['observation']);y=torch.tensor(z['target_residual'],dtype=torch.float32);w=torch.tensor(z['weight'],dtype=torch.float32)
    scale=torch.tensor([.025,.3,.3])
    with torch.no_grad():target=teacher.distribution(x).mean+y*scale
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(p) for p in paths['training']]
    base_results=read(source/'training_1/summary.json')['results']
    baseline=summary(source/'training_1',[dict(base_results[i],original_path=i) for i in indices],
                     [source/'training_1'/f'{i}.npz' for i in indices])
    base_full=summary(source/'training_1',base_results)
    def qualify(row,base,count):
        return bool(row['successes']==count and row['max_entry_mm']<=.05 and
            row['median_exit_mm']<=.8*base['median_exit_mm'] and row['max_exit_mm']<=base['max_exit_mm'] and
            row['full_reverse_sum_deg']<=base['full_reverse_sum_deg'] and row['time_plus_shape_s']<base['time_plus_shape_s'])
    screen_records=[];selected=[];logs=[]
    for scope in ('output','full_actor'):
        torch.manual_seed(9161271);model=LocalGeometryFeedbackPolicy(mask_endpoint=True);model.load_state_dict(parent)
        params=list(model.actor[-1].parameters()) if scope=='output' else list(model.actor.parameters())
        allowed={id(p) for p in params}
        for p in model.parameters():p.requires_grad_(id(p) in allowed)
        optimizer=torch.optim.Adam(params,lr=3e-5)
        originals={name:p.detach().clone() for name,p in model.actor.named_parameters()}
        records=[]
        for epoch in range(1,301):
            prediction=model.distribution(x).mean
            data_loss=(((prediction-target)/scale).square().mean(-1)*w).sum()
            penalty=prediction.new_zeros(())
            for name,p in model.actor.named_parameters():
                delta=p-originals[name]
                if name=='4.weight':delta=delta/scale[:,None]
                elif name=='4.bias':delta=delta/scale
                penalty=penalty+delta.square().sum()/3
            loss=data_loss+.001*penalty
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
            costs['supervised_updates']+=1
            if epoch not in (30,100,300):continue
            name=f'{scope}_{epoch}.pt';save_torch(stage/name,dict(policy=model.state_dict(),observation_dim=72,
                architecture='local_geometry_feedback',mask_endpoint=True,frame='heading',history_independent=True,
                learning_method='matched_supervised_actor_scope',scope=scope,epoch=epoch,parent_sha256=sha(initial)))
            tag=f'{scope}_screen_{epoch}'
            result=evaluate(model,[training[i] for i in indices],stage,tag,None,costs,started+480,environment_factory)
            for r,i in zip(result,indices):r['original_path']=i
            row=summary(stage/tag,result);row.update(scope=scope,epoch=epoch,model=name,tag=tag,
                data_loss=float(data_loss.detach()),regularizer=float(penalty.detach()),eligible=qualify(summary(stage/tag,result),baseline,len(indices)),
                paths=result)
            records.append(row);screen_records.append(row);logs.append(dict(scope=scope,epoch=epoch,data_loss=float(data_loss.detach()),loss=float(loss.detach())))
            write_json(stage/'screening_results.json',dict(baseline=baseline,candidates=screen_records,costs=costs,logs=logs))
            print('ACTOR_SCOPE',scope,epoch,row['successes'],'loss',row['data_loss'],'medianexit',row['median_exit_mm'],'eligible',row['eligible'],flush=True)
            assert costs['evaluation_actor_calls']<=20000
        eligible=[r for r in records if r['eligible']]
        if eligible:selected.append(min(eligible,key=lambda r:r['time_plus_shape_s']))
    write_json(stage/'screening_selection.json',dict(selected=[dict(scope=r['scope'],model=r['model']) for r in selected],training_only=True))
    full=[];development=[]
    for chosen in selected:
        model=LocalGeometryFeedbackPolicy(mask_endpoint=True);model.load_state_dict(torch.load(stage/chosen['model'],map_location='cpu',weights_only=False)['policy'])
        remaining=[i for i in range(64) if i not in indices];tag=chosen['scope']+'_remaining'
        results=evaluate(model,[training[i] for i in remaining],stage,tag,None,costs,started+480,environment_factory)
        for r,i in zip(results,remaining):r['original_path']=i
        mapping={i:(r,stage/chosen['tag']/f'{j}.npz') for j,(i,r) in enumerate(zip(indices,chosen['paths']))}
        mapping.update({i:(r,stage/tag/f'{j}.npz') for j,(i,r) in enumerate(zip(remaining,results))})
        reviewed=summary(stage,[mapping[i][0] for i in range(64)],[mapping[i][1] for i in range(64)])
        reviewed.update(scope=chosen['scope'],model=chosen['model'],eligible=qualify(reviewed,base_full,64),
                        trace_paths=[mapping[i][1].relative_to(BASE).as_posix() for i in range(64)])
        full.append(reviewed);write_json(stage/'full_training_results.json',dict(candidates=full,baseline=base_full,costs=costs))
    write_json(stage/'selection.json',dict(selected=[r for r in full if r['eligible']],training_only=True,goal_complete=False))
    for row in full:
        if not row['eligible']:continue
        model=LocalGeometryFeedbackPolicy(mask_endpoint=True);model.load_state_dict(torch.load(stage/row['model'],map_location='cpu',weights_only=False)['policy'])
        tag=row['scope']+'_development';result=evaluate(model,[np.asarray(p) for p in paths['validation']],stage,tag,None,costs,started+480,environment_factory)
        development.append(dict(scope=row['scope'],model=row['model'],**summary(stage/tag,result)))
    assert costs['evaluation_actor_calls']<=20000 and costs['evaluation_physics_ticks']<=450000
    write_json(stage/'results.json',dict(full_training=full,development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_matched_actor_scope_fit',costs=costs,
                       selected_models=[r['model'] for r in full if r['eligible']],wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
