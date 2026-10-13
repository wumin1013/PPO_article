"""Training-only parameter backtracking along an already learned PPO update."""
import time
import copy
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy
from train_pilot import evaluate,save_torch
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def summarize(stage,tag,results,path_ids):
    rows=[]
    for i,result in enumerate(results):
        with np.load(stage/tag/f'{i}.npz') as z:
            speed=np.linalg.norm(z['rows'][:,2:4],axis=1)
            shape=corner_shape(z['reference'],z['rows'],z['actions']) if len(z['reference'])==3 else None
            if shape:shape['spatial']=geometric_turn_metrics(z['reference'],z['rows'])
        rows.append(dict(path=path_ids[i],success=result['success'],complete_time_s=result['completion_time_s'],
            elapsed_s=result['elapsed_s'],shape=shape,peak_speed=float(speed.max()),cruise_ge95_s=float((speed>=95).sum()*.001)))
    successes=sum(r['success'] for r in rows)
    corners=[r['shape'] for r in rows if r['shape'] is not None and r['success']]
    incoming=sum((s['first_physical_turn']!='inside')+max(s['outward_excursion_mm']-.05,0)/1.25 for s in corners)
    reverse=sum(max(s['reverse_turning_total_deg']-5,0)/abs(s['turn_deg']) for s in corners)
    time_sum=sum(r['elapsed_s'] for r in rows)
    energy=sum(1.25*s['spatial']['normalized_bending_mm_inv'] for s in corners)
    return dict(successes=successes,full_time_s=time_sum if successes==len(rows) else None,
        rank=[successes,-incoming,-reverse,-time_sum-energy],incoming_violation=incoming,reverse_violation=reverse,paths=rows)


def main():
    torch.set_num_threads(1)
    initial=BASE/'history_independent_heading_frame_feedback_learning_20260916/epoch_500.pt'
    updated=BASE/'history_independent_heading_return_ppo_20260916/policy_10049.pt'
    alphas=[0.,.1,.25,.5,.75,1.]
    stage=register('constrained_parameter_step_probe_20260916',[__file__,initial,updated,BASE/'frame_feedback_policy.py',
        initial.parent/'receipt.json',updated.parent/'receipt.json'],
        dict(training_decisions=9000,evaluation_decisions=3000,physics_ticks=280000,wall_s=360,step_fractions=alphas),
        'Check whether a smaller parameter step along the existing PPO update can retain inward approach while repairing complete success. Evaluate fixed6 parameter fractions on original16 training paths only; rank lexicographically by full successes, incoming violation(first inward,outward<=0.05), normalized excess reversal above5deg, then time+normalized energy. Evaluate best2 training-ranked fractions once on development. This is bounded learned-parameter selection, not online action blending or a corner template. Failure-prefix shape cannot count as completion. No extension if no feasible step.')
    a=torch.load(initial,map_location='cpu',weights_only=False);b=torch.load(updated,map_location='cpu',weights_only=False)
    p=FrameFeedbackPolicy(frame='heading',history_independent=True)
    paths=read(BASE/'learning_preparation_v4/paths.json');indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    training=[np.asarray(paths['training'][i]) for i in indices];validation=[np.asarray(r) for r in paths['validation']]
    started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    candidates=[]
    def load(alpha):
        state=copy.deepcopy(a['policy'])
        for key in state:
            if key.startswith('actor.') and torch.is_floating_point(state[key]):state[key]=a['policy'][key]+alpha*(b['policy'][key]-a['policy'][key])
        p.load_state_dict(state)
    for i,alpha in enumerate(alphas):
        load(alpha);before=costs['evaluation_actor_calls'];ticks=costs['evaluation_physics_ticks']
        results=evaluate(p,training,stage,f'training_{i}',None,costs,started+360,environment_factory)
        costs['training_actor_calls']+=costs['evaluation_actor_calls']-before;costs['evaluation_actor_calls']=before
        costs['training_physics_ticks']+=costs['evaluation_physics_ticks']-ticks;costs['evaluation_physics_ticks']=ticks
        row=summarize(stage,f'training_{i}',results,indices);row.update(alpha=alpha,tag=f'training_{i}')
        candidates.append(row);write_json(stage/'training_results.json',dict(candidates=candidates,costs=costs))
        assert costs['training_actor_calls']<=9000
        print('STEP',alpha,row['rank'],flush=True)
    chosen=sorted(candidates,key=lambda r:r['rank'],reverse=True)[:2]
    write_json(stage/'selection.json',dict(selected=[dict(alpha=r['alpha'],rank=r['rank']) for r in chosen],basis='training only, written before development runs'))
    evaluations=[]
    for i,row in enumerate(chosen):
        load(row['alpha']);model=stage/f'candidate_{i}.pt';save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='frame_feedback',frame='heading',history_independent=True,
            learning_method='training_constrained_PPO_parameter_backtracking',alpha=row['alpha'],parent_sha256=[sha(initial),sha(updated)]))
        results=evaluate(p,validation,stage,f'validation_{i}',None,costs,started+360,environment_factory)
        result=summarize(stage,f'validation_{i}',results,list(range(12)));result.update(alpha=row['alpha'],model_sha256=sha(model),final_acceptance=False);evaluations.append(result)
        print('VALIDATION',row['alpha'],result['rank'],result['full_time_s'],flush=True)
    assert costs['evaluation_actor_calls']<=3000 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=280000
    write_json(stage/'results.json',dict(selected=evaluations,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_training_constrained_parameter_step_probe',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
