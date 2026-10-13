"""Finite frozen architecture contrast, with new mirrored development cases."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from reflection_policy import ReflectionPolicy, reflect_observation
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from train_pilot import evaluate, save_torch
from study_io import BASE, register, receipt, sha, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume-after-label-collision',action='store_true')
    args=parser.parse_args()
    models = [('source', BASE/'ar_conditional_joint_20260916/final.pt'),
              ('shape_search_g3', BASE/'shape_constrained_search_20260916/best_generation_3.pt')]
    if args.resume_after_label_collision:
        stage=BASE/'reflection_policy_probe_20260916'
        assert (stage/'source.pt').exists() and not (stage/'source/summary.json').exists()
        assert not (stage/'resume_after_label_fix.json').exists()
        write_json(stage/'failure_before_evaluation.json',dict(reason='source evaluation folder collided with registration source folder',
            stack_location='evaluate directory.mkdir before any evaluation',parity_physics_ticks=120,
            parity_checks_passed=3,formal_evaluation_actor_calls=0,model_loads=1))
        write_json(stage/'resume_after_label_fix.json',dict(changed_only='evaluation folder gets eval_ prefix',
            source_sha256=sha(__file__),skip_completed_parity_checks=True,total_model_load_budget=3,
            formal_evaluation_budget_unchanged=True))
        (stage/'source_after_label_fix.py').write_bytes(Path(__file__).read_bytes())
    else:
        stage = register('reflection_policy_probe_20260916',
        [__file__, BASE/'reflection_policy.py', BASE/'joint_policy.py', BASE/'isolated_corner_shape.py',
         BASE/'learning_preparation_v4/paths.json']+[p for _,p in models],
        dict(physics_ticks=300000, evaluation_decisions=10000, model_loads=2,
             validations_per_model=16, training_updates=0, wall_s=600),
            'Frozen exact-reflection architecture contrast on original 12 development paths plus four mirrored isolated cases; shape-search model is learned, group averaging is a disclosed inductive bias. Not a trained architecture comparison or independent test.')
    torch.set_num_threads(1); started=time.perf_counter()
    paths=[np.array(r) for r in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    mirror=np.diag([1.,-1.]); paths += [p@mirror for p in paths[:4]]
    # Validate state parity and the physical reflection at finite test commands.
    left=environment_factory(paths[1]); right=environment_factory(paths[13]); checks=0
    for action in ([] if args.resume_after_label_collision else ([.02, .6, -.4], [.03,.6,-.3], [.01,.5,-.2])):
        expected=reflect_observation(torch.tensor(left.observation())).numpy()
        assert np.max(np.abs(expected-right.observation()))<1e-6
        _,_,_,li=left.step(np.array(action)); _,_,_,ri=right.step(np.array(action)*[-1,1,1])
        transformed=li['rows'].copy()
        for offset in (0,2,4,6,8): transformed[:,offset:offset+2] = transformed[:,offset:offset+2]@mirror
        transformed[:,10]*=-1
        assert np.max(np.abs(transformed-ri['rows']))<1e-5
        checks+=1
    costs=dict(evaluation_actor_calls=0,evaluation_physics_ticks=0,parity_physics_ticks=120)
    summaries=[]
    for label,path in models:
        incoming=torch.load(path,map_location='cpu',weights_only=False)
        policy=ReflectionPolicy(incoming['observation_dim']);policy.load_state_dict(incoming['policy'])
        observation=torch.tensor(left.observation())
        with torch.no_grad():
            original=policy.distribution(observation).mean
            reflected=policy.distribution(reflect_observation(observation)).mean*torch.tensor([-1.,1.,1.])
        assert torch.equal(original,reflected)
        save_torch(stage/f'{label}.pt',dict(policy=policy.state_dict(),observation_dim=72,
            architecture='reflection_group_average',source_model_sha256=sha(path),mode='joint'))
        evaluation_label='eval_'+label
        results=evaluate(policy,paths,stage,evaluation_label,None,costs,started+600,environment_factory)
        shapes=[]
        for i in list(range(4))+list(range(12,16)):
            with np.load(stage/evaluation_label/f'{i}.npz') as d:
                shapes.append(dict(path=i,success=results[i]['success'],**corner_shape(d['reference'],d['rows'],d['actions'])))
        summaries.append(dict(model=label,original_successes=sum(r['success'] for r in results[:12]),
            original_time_s=sum(r['completion_time_s'] for r in results[:12]) if all(r['success'] for r in results[:12]) else None,
            mirrored_successes=sum(r['success'] for r in results[12:]),shapes=shapes))
        write_json(stage/'results.json',dict(results=summaries,costs=costs,goal_complete=False))
        print(label,summaries[-1]['original_successes'],summaries[-1]['original_time_s'],
              [(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],1),s['first_physical_turn']) for s in shapes],flush=True)
    assert costs['evaluation_actor_calls']<=10000 and costs['evaluation_physics_ticks']+120<=300000
    receipt(stage,dict(status='completed_frozen_reflection_architecture_probe',costs=costs,
        finite_physics_parity_checks=3 if args.resume_after_label_collision else checks,
        actor_forward_calls=2*costs['evaluation_actor_calls'],
        model_loads=3 if args.resume_after_label_collision else 2,completed_checks_reused=args.resume_after_label_collision,
        training_updates=0,wall_s=time.perf_counter()-started,goal_complete=False))


if __name__=='__main__':main()
