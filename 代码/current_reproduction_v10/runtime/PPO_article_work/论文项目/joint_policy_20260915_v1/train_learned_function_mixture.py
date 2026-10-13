"""Complete-return learning of three global state-dependent function weights."""
import time
import math
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from learned_head_mixture_policy import LearnedHeadMixturePolicy
from train_constrained_frame_search import init_worker,run_candidate,rank
from probe_constrained_parameter_step import summarize
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from search_complete_return import read
from study_io import BASE,register,receipt,sha,write_json


def main():
    torch.set_num_threads(1);rng=np.random.default_rng(916977)
    a_path=BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt';b_path=BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt'
    source=BASE/'learned_head_factorial_probe_20260916'
    stage=register('learned_function_mixture_optimization_20260916',[__file__,BASE/'learned_head_mixture_policy.py',BASE/'train_constrained_frame_search.py',
        BASE/'frame_feedback_policy.py',BASE/'reflection_policy.py',a_path,b_path,source/'receipt.json',source/'training_results.json',source/'results.json',
        BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',BASE/'terminal_candidate_runtime_v1.py',
        BASE/'geometry_spec.py',BASE/'geometric_turn_metrics.py',BASE/'isolated_corner_shape.py',BASE/'learning_preparation_v4/paths.json'],
        dict(training_decisions=60000,evaluation_decisions=5000,physics_ticks=1500000,wall_s=600,generations=6,population=8,elites=3,
             coefficient_bounds=[-.5,1.5],starting_standard_deviations=[.35,.5,.35],seed=916977,workers=4,no_gain_stop_generations=3),
        'Jointly learn three global coefficients over two already learned state-feedback networks, motivated by all6 mixed factorial corners preserving16/16 original training completion while32-feature random parameter perturbations did not. This observed coverage is not a pure dimensionality or optimizer superiority claim. Coefficients may interpolate or modestly extrapolate learned functions; original tanh action bounds, LOS guider,1.25mm ordered corridor,V/A/J,decision/execution periods and stopping remain unchanged. Rank complete success, first-inward/outward above0.01mm, excess reversal above5deg, then full time+normalized spatial energy. Reuse every exact coefficient already evaluated; evaluate new selected models on development only at2/4/6 and final. Runtime contains two networks and3 constants, no angle table, path schedule, manual corner trajectory or online search. This is complete-outcome parameter learning, not PPO gradient training. No unchanged extension after3 generations without improvement; no final acceptance without actual XY, full completion and cruising evidence.')
    a=torch.load(a_path,map_location='cpu',weights_only=False);b=torch.load(b_path,map_location='cpu',weights_only=False)
    p=LearnedHeadMixturePolicy(a['policy'],b['policy'])
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(paths['training'][i]) for i in indices];validation=[np.asarray(r) for r in paths['validation']]
    (stage/'training_traces').mkdir();started=time.perf_counter();deadline=time.time()+600
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,parameter_updates=0)
    cache={}
    for i,record in enumerate(read(source/'training_results.json')['candidates']):
        record=dict(record,tag=f'reused_factorial_{i}',reused_from=str(source/'training_results.json'),training_actor_calls=0,training_physics_ticks=0)
        record['rank']=rank(record);cache[tuple(np.asarray(record['coefficients'],dtype=np.float32))]=record
    incumbent=max(cache.values(),key=lambda r:r['rank']);all_candidates=list(cache.values());mean=np.array(incumbent['coefficients'],float);cov=np.diag(np.array([.35,.5,.35])**2)
    evaluated={tuple(np.asarray(r['coefficients'],dtype=np.float32)) for r in read(source/'results.json')['development']}|{(0.,0.,0.),(1.,1.,1.)}
    checkpoints=[];generations=[];no_gain=0
    reserve_decisions=8*sum(math.ceil(environment_factory(r).max_ticks/20) for r in training)
    reserve_ticks=8*sum(environment_factory(r).max_ticks for r in training)
    def checkpoint(generation):
        key=tuple(np.asarray(incumbent['coefficients'],dtype=np.float32))
        if key in evaluated:return
        p.set_coefficients(incumbent['coefficients']);model=stage/f'best_generation_{generation}.pt'
        save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='learned_head_mixture',learning_method='complete_outcome_joint_coefficient_CEM',
            coefficients=incumbent['coefficients'],generation=generation,parent_sha256=[sha(a_path),sha(b_path)]))
        tag=f'validation_{generation}';result=evaluate(p,validation,stage,tag,None,costs,time.perf_counter()+max(5,deadline-time.time()),environment_factory)
        row=summarize(stage,tag,result,list(range(12)));row.update(coefficients=incumbent['coefficients'],generation=generation,model_sha256=sha(model),final_acceptance=False)
        checkpoints.append(row);evaluated.add(key);write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
        print('CHECKPOINT',generation,row['coefficients'],row['successes'],row['full_time_s'],flush=True)
    try:
        with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(stage),a,None,indices,training,deadline-45,b)) as pool:
            generation=0
            for generation in range(1,7):
                if costs['training_actor_calls']+reserve_decisions>60000 or costs['training_physics_ticks']+costs['evaluation_physics_ticks']+reserve_ticks>1500000 or time.time()>deadline-120:break
                factor=np.linalg.cholesky(cov);noise=rng.normal(size=(4,3));population=[np.clip(mean+factor@n,-.5,1.5).astype(np.float32) for n in np.vstack([noise,-noise])]
                tasks=[];chosen=[]
                for i,coeff in enumerate(population):
                    key=tuple(coeff)
                    if key in cache:chosen.append(cache[key])
                    elif key not in [tuple(t[1]) for t in tasks]:tasks.append((f'g{generation}_c{i}',coeff))
                new=list(pool.map(run_candidate,tasks))
                for record in new:
                    record['rank']=rank(record);cache[tuple(np.asarray(record['coefficients'],dtype=np.float32))]=record
                    costs['training_actor_calls']+=record['training_actor_calls'];costs['training_physics_ticks']+=record['training_physics_ticks']
                chosen+=new;all_candidates+=new;old=incumbent
                ordered=sorted([incumbent]+chosen,key=lambda r:r['rank'],reverse=True);incumbent=ordered[0]
                elite=np.array([r['coefficients'] for r in ordered[:3]]);mean=.5*mean+.5*elite.mean(0);delta=elite-elite.mean(0);cov=.8*cov+.2*delta.T@delta/len(elite)
                costs['parameter_updates']+=1;no_gain=no_gain+1 if incumbent['tag']==old['tag'] else 0
                generations.append(dict(generation=generation,best_tag=incumbent['tag'],coefficients=incumbent['coefficients'],rank=incumbent['rank'],mean=mean.tolist(),covariance=cov.tolist(),no_gain=no_gain))
                write_json(stage/'training_results.json',dict(candidates=all_candidates,costs=costs));write_json(stage/'generations.json',dict(generations=generations,costs=costs))
                print('GENERATION',generation,incumbent['coefficients'],incumbent['rank'],'decisions',costs['training_actor_calls'],flush=True)
                if generation in (2,4,6):checkpoint(generation)
                if no_gain>=3:break
            checkpoint(generation)
        assert costs['training_actor_calls']<=60000 and costs['evaluation_actor_calls']<=5000 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=1500000
        write_json(stage/'results.json',dict(checkpoints=checkpoints,selected_training_tag=incumbent['tag'],coefficients=incumbent['coefficients'],costs=costs,goal_complete=False))
        receipt(stage,dict(status='completed_bounded_learned_function_coefficient_optimization',costs=costs,wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
