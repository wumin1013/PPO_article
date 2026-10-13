"""Offline per-training-path trajectory optimization, never a deployed policy.

A generic five-knot temporal perturbation of all three feedback actions is
optimized by physical complete-episode outcomes. No angle speed/preview table
or reference trajectory fit. This script only produces teacher diagnostics;
successful trajectories do not establish a learned reusable feedback policy.
"""
import time
import argparse
import math
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from joint_runtime_v5 import environment_factory
from probe_coordinated_suffix import encode
from geometric_turn_metrics import geometric_turn_metrics
from isolated_corner_shape import corner_shape
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,write_json,sha


def rank(row):
    if row['shape'] is None:return row['success'],0.,-row['elapsed_s']
    shape=row['shape'];incoming=max(shape['outward_excursion_mm']-.05,0)/1.25+.1*(shape['first_physical_turn']!='inside')
    reverse=max(shape['reverse_turning_total_deg']-5,0)/abs(shape['turn_deg'])
    quality=row['elapsed_s']+1.25*row['spatial']['normalized_bending_mm_inv']+2*reverse
    return row['success'],-incoming,-quality


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--straight',action='store_true');parser.add_argument('--remaining',action='store_true');args=parser.parse_args()
    assert not (args.straight and args.remaining)
    train_indices=(4,5,6,8,9,11,12,13,18,22,26,34) if args.remaining else (50,53) if args.straight else (14,15)
    decision_budget=60000 if args.remaining else 30000
    tick_budget=1400000 if args.remaining else 700000
    wall_budget=900 if args.remaining else 300
    source=BASE/'preview_state_residual_learning_20260916';model=source/'best_generation_5.pt'
    selected=next(g for g in read(source/'generations.json')['generations'] if g['generation']==5)
    assert selected['model_sha256']==sha(model)
    parent=next(r for r in read(source/'training_results.json')['candidates'] if r['tag']==selected['best_tag'])
    stage=register('training_remaining_teacher_optimization_20260916' if args.remaining else 'training_straight_teacher_optimization_20260916' if args.straight else 'training_corner_teacher_optimization_20260916',[__file__,model,source/'receipt.json',
        source/'training_results.json',source/'generations.json',BASE/'learning_preparation_v4/paths.json',
        BASE/'dynamic_residual_policy.py',BASE/'geometric_turn_metrics.py',
        BASE/'correlated_exploration_support_probe_20260916/receipt.json'],
        dict(training_paths=list(train_indices),policy_gradient_updates=0,teacher_generations=6,population=12,elites=3,
             decisions=decision_budget,physics_ticks=tick_budget,wall_s=wall_budget,temporal_knots=5,
             initial_physical_rms=[math.radians(3),15.,3.],seed=916417),
        'Offline local trajectory optimization on registered existing training paths only. Optimize generic smooth temporal perturbations to all three feedback controls, retaining complete-success and inward-approach priority, then full time+normalized bending+2*excess-reversal; non-isolated paths use complete time, with shape still subject to actual-XY review. Remaining-path expansion addresses inconsistent anchor objectives found in the first feedback fit. No already optimized path is repeated. No deployed corner templates or learned-student claim. Stop a path after three generations without gain; analyze before student training.')
    torch.set_num_threads(1);saved=torch.load(model,map_location='cpu',weights_only=False)
    policy=PreviewResidualPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    paths=read(BASE/'learning_preparation_v4/paths.json')['training']
    started=time.perf_counter();costs=dict(teacher_decisions=0,physics_ticks=0,policy_gradient_updates=0,teacher_parameter_updates=0)
    rng=np.random.default_rng(916417);all_results=[];summaries=[]
    def evaluate(coeff,path_index,generation,candidate,horizon):
        ref=np.asarray(paths[path_index]);env=environment_factory(ref);chunks=[];actions=[];observations=[]
        knot_time=np.linspace(0,horizon,7);values=np.c_[np.zeros(3),coeff,np.zeros(3)]
        while not env.done:
            assert costs['teacher_decisions']<decision_budget and costs['physics_ticks']<tick_budget-2200 and time.perf_counter()-started<wall_budget
            obs=env.observation();latent=policy.act(obs,True)[0]
            delta=np.array([np.interp(env.ticks*.001,knot_time,values[i]) for i in range(3)])
            if np.any(delta):
                desired=env.decode_action(latent)+delta
                desired=np.clip(desired,[-math.pi/2+1e-6,1e-5,.800001],[math.pi/2-1e-6,99.99999,11.999999])
                latent=encode(desired)
            _,_,_,info=env.step(latent);chunks.append(info['rows']);actions.append(info['held_action']);observations.append(obs)
            costs['teacher_decisions']+=1;costs['physics_ticks']+=len(info['rows'])
        rows=np.vstack(chunks);held=np.asarray(actions);audit=difference_audit(ref[0],rows)
        success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
        tag=f'p{path_index}_g{generation}_c{candidate}'
        np.savez_compressed(stage/f'{tag}.npz',reference=ref,rows=rows,actions=held,observation=np.asarray(observations),
            temporal_coefficients=coeff,knot_time=knot_time)
        result=dict(tag=tag,path=path_index,coefficients=coeff.tolist(),success=success,reason=env.reason,
            elapsed_s=env.ticks*.001,complete_time_s=env.ticks*.001 if success else None,
            shape=corner_shape(ref,rows,held) if len(ref)==3 else None,spatial=geometric_turn_metrics(ref,rows) if len(ref)==3 else None,
            audit=audit,geometry=env.tracker.snapshot(1.25),trace=f'{tag}.npz',deployed=False)
        all_results.append(result)
        write_json(stage/'training_results.json',dict(candidates=all_results,costs=costs))
        return result
    for path_index in train_indices:
        original=next(r for r in parent['results'] if r['path']==path_index)
        incumbent=dict(original,tag='reused_baseline',coefficients=np.zeros((3,5)).tolist(),deployed=False,
            elapsed_s=original['elapsed_s'],reused_from=str(source/'training_results.json'),physical_source_tag=parent['tag'])
        incumbent.setdefault('spatial',None)
        horizon=original['complete_time_s'];mean=np.zeros((3,5))
        sd=np.repeat(np.array([math.radians(3),15.,3.]),5)
        covariance=np.diag(sd**2)
        for k in range(5):covariance[5+k,10+k]=covariance[10+k,5+k]=-.5*15*3
        no_gain=0;history=[]
        for generation in range(1,7):
            # Reserve enough work for a worst-case complete generation before launch.
            max_ticks=environment_factory(np.asarray(paths[path_index])).max_ticks
            if costs['teacher_decisions']+12*math.ceil(max_ticks/20)>decision_budget or costs['physics_ticks']+12*max_ticks>tick_budget or time.perf_counter()-started>wall_budget-60:break
            factor=np.linalg.cholesky(covariance);noise=rng.normal(size=(6,15))
            population=[mean+(factor@e).reshape(3,5) for e in np.vstack([noise,-noise])]
            candidates=[incumbent]+[evaluate(c,path_index,generation,i,horizon) for i,c in enumerate(population)]
            ordered=sorted(candidates,key=rank,reverse=True);previous=incumbent;incumbent=ordered[0]
            elites=np.array([r['coefficients'] for r in ordered[:3]])
            mean=.5*mean+.5*elites.mean(0)
            centered=elites.reshape(3,15)-elites.reshape(3,15).mean(0)
            covariance=.8*covariance+.2*centered.T@centered/3
            costs['teacher_parameter_updates']+=1
            no_gain=no_gain+1 if incumbent['tag']==previous['tag'] else 0
            history.append(dict(generation=generation,best_tag=incumbent['tag'],rank=rank(incumbent),mean=mean.tolist(),covariance=covariance.tolist()))
            write_json(stage/f'progress_{path_index}.json',dict(best=incumbent,history=history,costs=costs))
            s=incumbent['shape'];e=incumbent['spatial']
            print(path_index,generation,incumbent['tag'],incumbent['success'],incumbent['complete_time_s'],
                'E',round(e['bending_energy_mm_inv'],2) if e else None,'width',round(e['middle80_turn_arc_length_mm'],2) if e else None,'reverse',round(s['reverse_turning_total_deg'],2) if s else None,flush=True)
            if no_gain>=3:break
        summaries.append(dict(path=path_index,best=incumbent,history=history,not_deployable=True))
    write_json(stage/'results.json',dict(paths=summaries,costs=costs,goal_complete=False,
        boundary='Path-specific offline optimized time coefficients are teacher diagnostics, never the final feedback policy. No validation or sealed test was used.'))
    receipt(stage,dict(status='completed_bounded_offline_teacher_optimization',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
