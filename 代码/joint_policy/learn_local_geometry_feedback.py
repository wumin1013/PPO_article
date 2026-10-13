"""Matched representation test on the same local counterfactual training data."""
import time
import copy
import numpy as np
import torch
from local_geometry_feedback_policy import LocalGeometryFeedbackPolicy
from frame_feedback_policy import FrameFeedbackPolicy
from reflection_policy import reflect_observation
from probe_constrained_parameter_step import summarize
from train_constrained_frame_search import rank
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def main():
    torch.set_num_threads(1)
    source=BASE/'constrained_parameter_step_probe_20260916';initial=source/'candidate_1.pt'
    fitted=BASE/'local_counterfactual_feedback_learning_20260916'
    stage=register('matched_endpoint_representation_learning_20260916',
        [__file__,BASE/'local_geometry_feedback_policy.py',BASE/'frame_feedback_policy.py',BASE/'reflection_policy.py',
         initial,source/'training_results.json',fitted/'training_data.npz',fitted/'fit.json',fitted/'receipt.json'],
        dict(training_evaluation_decisions=6000,development_decisions=2200,physics_ticks=180000,wall_s=300,
             seed=9161023,epochs=300,checkpoints=[50,150,300],actor_lr=.0003,gradient_clip=1.,
             arms=['full_features','mask_endpoint_bearing'],loss='identical weighted normalized latent MSE on existing local branch dataset'),
        'Audit reset-state endpoint-bearing sensitivity without physics, then matched nonlinear feedback fits. '
        'Only arm difference is masking observation13:15(endpoint vector); remaining distance, ordered gate, '
        'local preview coordinates/tangents/validity and physical states remain. Same initial actor, training '
        'targets, weighted loss, learning rate,300 full-batch updates and checkpoint opportunities. '
        'This checks a measured distant-bearing shortcut; removal is not by dimension-count preference. '
        'Labels are previously completed training-only local counterfactual improvements. No new probe data, '
        'angle tables or runtime schedules. Select independently by16-path training physical rank relative '
        'to preserved initialization; at most one selected checkpoint per arm enters development. '
        'No claim that input ablation alone proves Markov sufficiency or the original learning cause.')
    started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    teacher=FrameFeedbackPolicy(frame='heading',history_independent=True);teacher.load_state_dict(saved['policy'])
    with np.load(fitted/'training_data.npz') as z:x=torch.tensor(z['observation']);residual=torch.tensor(z['target_residual'],dtype=torch.float32);weight=torch.tensor(z['weight'],dtype=torch.float32)
    scale=torch.tensor([.02,.3,.3])
    with torch.no_grad():target=teacher.distribution(x).mean+residual*scale
    paths=read(BASE/'learning_preparation_v4/paths.json');indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    audit=[]
    for i,ref in enumerate(paths['training']):
        env=environment_factory(np.asarray(ref));obs=env.observation();masked=obs.copy();masked[13:15]=0
        original=env.decode_action(teacher.act(obs,True)[0]);changed=env.decode_action(teacher.act(masked,True)[0])
        audit.append(dict(path=i,local_turn_visible=bool(np.max(abs(obs[42:62].reshape(10,2)[:,1]))>1e-5),
            original_steering_deg=float(np.rad2deg(original[0])),masked_steering_deg=float(np.rad2deg(changed[0])),
            original_action=original.tolist(),masked_action=changed.tolist()))
    write_json(stage/'reset_endpoint_audit.json',dict(records=audit,physics_ticks=0,
        caveat='Identifies command dependence at identical physical reset states; removing a feature from an already fitted network also shifts feed and D. It does not establish historical learning causation.'))
    training=[np.asarray(paths['training'][i]) for i in indices]
    baseline=copy.deepcopy(next(r for r in read(source/'training_results.json')['candidates'] if r['alpha']==.5));baseline['rank']=rank(baseline)
    arms=[]
    for masked in [False,True]:
        torch.manual_seed(9161023);model=LocalGeometryFeedbackPolicy(mask_endpoint=masked);model.load_state_dict(saved['policy'])
        optimizer=torch.optim.Adam(model.actor.parameters(),lr=.0003)
        arm='mask_endpoint_bearing' if masked else 'full_features';records=[]
        for epoch in range(1,301):
            predicted=model.distribution(x).mean
            loss=(((predicted-target)/scale).square().mean(-1)*weight).sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.);optimizer.step();costs['supervised_updates']+=1
            if epoch not in [50,150,300]:continue
            name=f'{arm}_{epoch}.pt';save_torch(stage/name,dict(policy=model.state_dict(),observation_dim=72,
                architecture='local_geometry_feedback',mask_endpoint=masked,frame='heading',history_independent=True,
                learning_method='matched_local_counterfactual_feedback_fit',epoch=epoch,parent_sha256=sha(initial)))
            before_calls=costs['evaluation_actor_calls'];before_ticks=costs['evaluation_physics_ticks']
            tag=f'{arm}_training_{epoch}';evaluated=evaluate(model,training,stage,tag,None,costs,started+300,environment_factory)
            costs['training_actor_calls']+=costs['evaluation_actor_calls']-before_calls;costs['evaluation_actor_calls']=before_calls
            costs['training_physics_ticks']+=costs['evaluation_physics_ticks']-before_ticks;costs['evaluation_physics_ticks']=before_ticks
            result=summarize(stage,tag,evaluated,indices);result.update(epoch=epoch,model=name,loss=float(loss.detach()));result['rank']=rank(result);records.append(result)
            write_json(stage/f'{arm}_training_results.json',dict(baseline=baseline,candidates=records,costs=costs))
            print('REPRESENTATION',arm,epoch,result['rank'],flush=True)
            assert costs['training_actor_calls']<=6000
        best=max(records,key=lambda r:r['rank']);selected=best if best['successes']==16 and best['rank']>baseline['rank'] else None
        arms.append(dict(arm=arm,selected=selected,mask_endpoint=masked))
    write_json(stage/'selection.json',dict(arms=arms,baseline_rank=baseline['rank'],selection='training-only; written before any development evaluation'))
    development=[]
    for arm in arms:
        if arm['selected'] is None:continue
        saved_model=torch.load(stage/arm['selected']['model'],map_location='cpu',weights_only=False)
        model=LocalGeometryFeedbackPolicy(mask_endpoint=arm['mask_endpoint']);model.load_state_dict(saved_model['policy'])
        tag=arm['arm']+'_validation';evaluated=evaluate(model,[np.asarray(p) for p in paths['validation']],stage,tag,None,costs,started+300,environment_factory)
        result=summarize(stage,tag,evaluated,list(range(12)));result.update(arm=arm['arm'],model=arm['selected']['model']);development.append(result)
        print('REPRESENTATION_DEVELOPMENT',arm['arm'],result['successes'],result['full_time_s'],flush=True)
    assert costs['evaluation_actor_calls']<=2200 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=180000
    write_json(stage/'results.json',dict(arms=arms,development=development,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_matched_endpoint_representation_fit',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
