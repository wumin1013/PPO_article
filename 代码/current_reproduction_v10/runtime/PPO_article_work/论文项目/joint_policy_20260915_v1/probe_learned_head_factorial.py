"""Six new head combinations; both completed endpoints are reused exactly."""
import itertools
import time
import copy
import numpy as np
import torch
from learned_head_mixture_policy import LearnedHeadMixturePolicy
from probe_constrained_parameter_step import summarize
from train_constrained_frame_search import rank
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,sha,write_json


def main():
    torch.set_num_threads(1)
    a_path=BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt'
    b_path=BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt'
    stage=register('learned_head_factorial_probe_20260916',[__file__,BASE/'learned_head_mixture_policy.py',BASE/'frame_feedback_policy.py',a_path,b_path,
        a_path.parent/'receipt.json',b_path.parent/'receipt.json',BASE/'frame_feedback_training_reproduction_20260916/receipt.json',BASE/'constrained_frame_parameter_learning_20260916/analysis.json'],
        dict(new_training_combinations=6,training_decisions=9000,evaluation_decisions=3000,physics_ticks=280000,wall_s=360,coefficients=[0.,1.]),
        'Full2^3 factorial over steering/feed/preview learned feedback heads from the broad-turn step0.5 network and higher-cruise heading network. Reuse completed endpoints on16 training and12 development paths; evaluate only six new combinations. Same states/runtime/constraints; interactions are disclosed, not assigned to one head in isolation. Select best2 new combinations by training-only complete-success/incoming/reversal/time+energy rank before development. Outputs remain state-dependent learned functions; no corner templates or hand trajectories. This is a head-interaction mechanism pilot and parameter/model selection, not new PPO updates or final success.')
    a=torch.load(a_path,map_location='cpu',weights_only=False);b=torch.load(b_path,map_location='cpu',weights_only=False)
    p=LearnedHeadMixturePolicy(a['policy'],b['policy'])
    with np.load(BASE/'interior_teacher_feedback_learning_20260916/training_data.npz') as z:x=torch.tensor(z['observation'])
    with torch.no_grad():
        p.set_coefficients((0,0,0));assert torch.equal(p.mean(x),p.shape_policy.distribution(x).mean)
        p.set_coefficients((1,1,1));endpoint_error=float((p.mean(x)-p.cruise_policy.distribution(x).mean).abs().max())
        assert endpoint_error<1e-6
    paths=read(BASE/'learning_preparation_v4/paths.json');indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    training=[np.asarray(paths['training'][i]) for i in indices];validation=[np.asarray(r) for r in paths['validation']]
    source_a=read(a_path.parent/'training_results.json')['candidates'];base=copy.deepcopy(next(r for r in source_a if r['alpha']==.5));base.update(coefficients=[0,0,0],reused=True)
    base['rank']=rank(base)
    source_b=BASE/'frame_feedback_training_reproduction_20260916'
    b_result=next(r for r in read(source_b/'results.json')['records'] if 'heading_frame' in r['model'])
    endpoint=copy.deepcopy(b_result)
    for i,r in enumerate(endpoint['paths']):
        original=read(source_b/'heading'/f'{i}.json');r['elapsed_s']=original['elapsed_s'];r['complete_time_s']=original['completion_time_s']
    endpoint.update(coefficients=[1,1,1],reused=True);endpoint['rank']=rank(endpoint)
    candidates=[base,endpoint];started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    for coefficients in itertools.product((0.,1.),repeat=3):
        if coefficients in ((0.,0.,0.),(1.,1.,1.)):continue
        p.set_coefficients(coefficients);tag='training_'+''.join(str(int(c)) for c in coefficients)
        calls=costs['evaluation_actor_calls'];ticks=costs['evaluation_physics_ticks']
        results=evaluate(p,training,stage,tag,None,costs,started+360,environment_factory)
        costs['training_actor_calls']+=costs['evaluation_actor_calls']-calls;costs['evaluation_actor_calls']=calls
        costs['training_physics_ticks']+=costs['evaluation_physics_ticks']-ticks;costs['evaluation_physics_ticks']=ticks
        row=summarize(stage,tag,results,indices);row.update(coefficients=list(coefficients),reused=False);row['rank']=rank(row)
        candidates.append(row);write_json(stage/'training_results.json',dict(candidates=candidates,costs=costs))
        assert costs['training_actor_calls']<=9000
        print('HEADS',coefficients,row['rank'],flush=True)
    selected=sorted([r for r in candidates if not r['reused']],key=lambda r:r['rank'],reverse=True)[:2]
    write_json(stage/'selection.json',dict(selected=[dict(coefficients=r['coefficients'],rank=r['rank']) for r in selected],training_only=True))
    development=[]
    for i,selected_row in enumerate(selected):
        p.set_coefficients(selected_row['coefficients']);model=stage/f'candidate_{i}.pt'
        save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='learned_head_mixture',
            learning_method='training_only_three_head_factorial_selection',coefficients=selected_row['coefficients'],parent_sha256=[sha(a_path),sha(b_path)]))
        tag=f'validation_{i}';results=evaluate(p,validation,stage,tag,None,costs,started+360,environment_factory)
        row=summarize(stage,tag,results,list(range(12)));row.update(coefficients=selected_row['coefficients'],model_sha256=sha(model),final_acceptance=False);row['rank']=rank(row)
        development.append(row);print('DEVELOPMENT',row['coefficients'],row['rank'],row['full_time_s'],flush=True)
    assert costs['evaluation_actor_calls']<=3000 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=280000
    write_json(stage/'results.json',dict(development=development,costs=costs,goal_complete=False,endpoint_actor_error=endpoint_error))
    receipt(stage,dict(status='completed_learned_head_factorial_probe',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
