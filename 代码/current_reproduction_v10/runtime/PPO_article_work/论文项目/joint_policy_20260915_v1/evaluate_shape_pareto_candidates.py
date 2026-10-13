"""Post-hoc development evaluation selected from existing training evidence.

No new optimization or rollout repeats. Both candidate tags are selected using
training outcomes only, before looking at their new development trajectories.
"""
import time
import json
import numpy as np
import torch
from reflection_policy import ReflectionPolicy
from dynamic_residual_policy import DynamicResidualPolicy
from search_complete_return import read
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from joint_runtime_v5 import environment_factory
from train_pilot import save_torch,evaluate
from study_io import BASE,register,receipt,sha,write_json


def key(candidate):
    shapes=[r['shape'] for r in candidate['results'] if r['shape'] is not None]
    incoming=sum(max(s['outward_excursion_mm']-.05,0)/1.25+.1*(s['first_physical_turn']!='inside') for s in shapes)
    reverse=sum(max(s['reverse_turning_total_deg']-5.,0)/abs(s['turn_deg']) for s in shapes)
    energy=sum(1.25*r['spatial']['normalized_bending_mm_inv'] for r in candidate['results'] if r['shape'] is not None)
    return (-candidate['successes'],incoming,reverse,-candidate['return_sum']+energy)


def main():
    torch.set_num_threads(1)
    folders=['spatial_bending_learning_20260916','dynamic_state_residual_learning_20260916']
    initial=BASE/'reflection_shape_search_20260916/best_generation_5.pt'
    stage=register('shape_pareto_candidate_review_20260916',[__file__,initial,BASE/'learning_preparation_v4/paths.json',
        BASE/'dynamic_residual_policy.py',BASE/'reflection_policy.py',BASE/'geometric_turn_metrics.py']+
        [BASE/f/name for f in folders for name in ('receipt.json','training_results.json','basis.npz')],
        dict(training_decisions=0,training_updates=0,maximum_candidates=2,evaluation_decisions=5000,physics_ticks=150000,wall_s=120),
        'Select one previously trained candidate per subspace by complete-success, incoming-shape, normalized excess reverse-turn, then full return minus spatial energy. Post-hoc development selection only; not a final comparison or independent test.')
    selected=[]
    for folder in folders:
        rows=read(BASE/folder/'training_results.json')['candidates']
        winner=min([r for r in rows if r.get('candidate_complete',True)],key=key)
        selected.append(dict(folder=folder,tag=winner['tag'],training_key=key(winner),coefficients=winner['coefficients']))
    write_json(stage/'selected_before_evaluation.json',dict(selected=selected))
    started=time.perf_counter();costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0)
    validation=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    outputs=[]
    for index,chosen in enumerate(selected):
        incoming=torch.load(initial,map_location='cpu',weights_only=False)
        dynamic=chosen['folder'].startswith('dynamic')
        policy=DynamicResidualPolicy(incoming['observation_dim']) if dynamic else ReflectionPolicy(incoming['observation_dim'])
        missing=policy.load_state_dict(incoming['policy'],strict=not dynamic)
        if dynamic:assert set(missing.missing_keys)=={'dynamic_head.weight','dynamic_head.bias'} and not missing.unexpected_keys
        head=policy.dynamic_head if dynamic else policy.actor[-1]
        basis=np.load(BASE/chosen['folder']/'basis.npz')['basis']
        delta=np.asarray(chosen['coefficients'])@basis
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(head.weight.numpy()+delta[:,:head.in_features],dtype=torch.float32))
            head.bias.copy_(torch.as_tensor(head.bias.numpy()+delta[:,-1],dtype=torch.float32))
        model_path=stage/f'candidate_{index}.pt'
        save_torch(model_path,dict(policy=policy.state_dict(),observation_dim=incoming['observation_dim'],mode='joint',
            architecture='reflection_dynamic_residual' if dynamic else 'reflection_group_average',
            source=chosen['folder'],source_tag=chosen['tag'],optimizer_method='complete_return_CEM'))
        # Verify first recorded training action against reconstructed candidate.
        source_rows=read(BASE/chosen['folder']/'training_results.json')['candidates']
        candidate=next(r for r in source_rows if r['tag']==chosen['tag'])
        for record in candidate['results']:
            with np.load(BASE/chosen['folder']/'training_traces'/record['trace']) as trace:
                env=environment_factory(trace['reference'])
                action=env.decode_action(policy.act(env.observation(),deterministic=True)[0])
                assert np.array_equal(action,trace['actions'][0])
        results=evaluate(policy,validation,stage,f'validation_{index}',None,costs,started+120,environment_factory)
        assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=150000
        shapes=[]
        for i in range(4):
            with np.load(stage/f'validation_{index}'/f'{i}.npz') as data:
                shape=corner_shape(data['reference'],data['rows'],data['actions'])
                shape['spatial']=geometric_turn_metrics(data['reference'],data['rows'])
                shapes.append(shape)
        row=dict(selection=chosen,model_sha256=sha(model_path),successes=sum(r['success'] for r in results),
            full_time_s=sum(r['completion_time_s'] for r in results) if all(r['success'] for r in results) else None,
            shapes=shapes,final_shape_acceptance=False)
        outputs.append(row)
        print(chosen['folder'],chosen['tag'],row['successes'],row['full_time_s'],
            [(round(s['turn_deg']),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
        write_json(stage/'partial_results.json',dict(results=outputs,costs=costs))
    write_json(stage/'results.json',dict(results=outputs,costs=costs,goal_complete=False,boundary='Post-hoc development review; neither independent seeds nor sealed tests.'))
    receipt(stage,dict(status='completed',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
