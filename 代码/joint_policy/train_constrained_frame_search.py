"""Joint network-parameter learning with training-only physical shape ranking.

Workers are ordinary simulator processes, not AI agents. Deployment remains
one fixed learned network; no trajectory coefficients or search run online.
"""
import time
import math
import copy
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from frame_feedback_policy import FrameFeedbackPolicy,heading_frame
from reflection_policy import reflect_observation
from audit_search_feature_coverage import saved_observations
from train_feedback_from_optimized_rollouts import decoded
from train_pilot import evaluate,save_torch
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from validate_actions import difference_audit
from search_complete_return import read
from study_io import BASE,register,receipt,sha,write_json


_WORKER=None


def parameter_state(saved,bases,coefficients):
    state=copy.deepcopy(saved['policy']);offset=0
    for head,basis in enumerate(bases):
        n=len(basis);delta=np.asarray(coefficients[offset:offset+n])@basis;offset+=n
        state['actor.4.weight'][head]+=torch.tensor(delta[:-1],dtype=torch.float32)
        state['actor.4.bias'][head]+=float(delta[-1])
    assert offset==len(coefficients)
    return state


def init_worker(stage,initial,bases,indices,paths,deadline,mixture_state=None):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER=(Path(stage),initial,bases,indices,paths,deadline,mixture_state)


def run_candidate(task):
    stage,saved,bases,indices,paths,deadline,mixture_state=_WORKER
    tag,coefficients=task
    if mixture_state is None:
        p=FrameFeedbackPolicy(frame='heading',history_independent=True)
        p.load_state_dict(parameter_state(saved,bases,coefficients))
    else:
        from learned_head_mixture_policy import LearnedHeadMixturePolicy
        p=LearnedHeadMixturePolicy(saved['policy'],mixture_state['policy'],coefficients)
        p.set_coefficients(coefficients)
    records=[];calls=0;ticks=0
    with torch.no_grad():
        for path_index,reference in zip(indices,paths):
            env=environment_factory(reference);chunks=[];held=[]
            while not env.done:
                if time.time()>deadline:raise TimeoutError('Registered CEM wall budget')
                obs=torch.tensor(env.observation())
                latent=(p.mean(obs) if mixture_state is not None else p.distribution(obs).mean).numpy()
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);calls+=1;ticks+=len(info['rows'])
            rows=np.vstack(chunks);actions=np.asarray(held);audit=difference_audit(reference[0],rows)
            success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            shape=corner_shape(reference,rows,actions) if len(reference)==3 else None
            if shape:shape['spatial']=geometric_turn_metrics(reference,rows)
            filename=f'{tag}_p{path_index}.npz'
            np.savez_compressed(stage/'training_traces'/filename,reference=reference,rows=rows,actions=actions)
            speed=np.linalg.norm(rows[:,2:4],axis=1)
            records.append(dict(path=path_index,success=success,reason=env.reason,elapsed_s=env.ticks*.001,
                complete_time_s=env.ticks*.001 if success else None,shape=shape,audit=audit,geometry=env.tracker.snapshot(1.25),
                trace=filename,peak_speed=float(speed.max()),cruise_ge95_s=float((speed>=95).sum()*.001)))
    return dict(tag=tag,coefficients=np.asarray(coefficients).tolist(),paths=records,training_actor_calls=calls,training_physics_ticks=ticks)


def rank(candidate):
    rows=candidate['paths'];good=[r for r in rows if r['success']]
    corners=[r['shape'] for r in good if r['shape'] is not None]
    incoming=sum((s['first_physical_turn']!='inside')+max(s['outward_excursion_mm']-.01,0)/1.25 for s in corners)
    reverse=sum(max(s['reverse_turning_total_deg']-5,0)/abs(s['turn_deg']) for s in corners)
    objective=sum(r['elapsed_s'] for r in rows)+sum(1.25*s['spatial']['normalized_bending_mm_inv'] for s in corners)
    return [len(good),-incoming,-reverse,-objective]


def main():
    torch.set_num_threads(1);torch.manual_seed(916913);rng=np.random.default_rng(916913)
    source=BASE/'constrained_parameter_step_probe_20260916';initial=source/'candidate_1.pt'
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    assert saved['alpha']==.5 and saved['history_independent']
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    stage=register('constrained_frame_parameter_learning_20260916',[__file__,initial,source/'receipt.json',source/'training_results.json',
        BASE/'frame_feedback_policy.py',BASE/'reflection_policy.py',BASE/'joint_policy.py',BASE/'audit_search_feature_coverage.py',
        BASE/'isolated_corner_shape.py',BASE/'geometric_turn_metrics.py',BASE/'joint_runtime.py',BASE/'joint_runtime_v2.py',
        BASE/'joint_runtime_v3.py',BASE/'joint_runtime_v5.py',BASE/'terminal_candidate_runtime_v1.py',BASE/'geometry_spec.py',BASE/'learning_preparation_v4/paths.json'],
        dict(training_decisions=80000,evaluation_decisions=6000,physics_ticks=2000000,wall_s=900,generations=6,population=12,elites=3,
             seed=916913,train_indices=indices,workers=4,odd_even_feature_variance_target=.99,latent_rms=[.025,.18,.20],
             validation_generations=[2,4,6],no_improvement_stop_generations=3),
        'Starting from the learned0.5 PPO parameter step with16/16 training and12/12 development completion, optimize all three final-layer outputs jointly by complete physical training episodes. Use separate reflected odd/even learned-feature SVD bases retaining99% observed variance; variance does not prove action relevance. Rank full success first, then first-inward and excess outward above0.01mm, then normalized excess reversal above5deg, then complete time+1.25E/angle^2. These are disclosed training criteria, not online rules or final shape acceptance. Preserve original guider and all physical constraints. Retain a candidate only if actual developmental shape is inspected and cruise improves; reject narrow-turn or completion regressions. No independent-seed, fair-control or sealed-test claim. Stop after3 generations without training-rank improvement or budget.')
    started=time.perf_counter();deadline=time.time()+900
    (stage/'training_traces').mkdir()
    p=FrameFeedbackPolicy(frame='heading',history_independent=True);p.load_state_dict(saved['policy'])
    observations=[];baseline_paths=[];baseline=next(r for r in read(source/'training_results.json')['candidates'] if r['alpha']==.5)
    errors=[]
    for i,index in enumerate(indices):
        trace=source/'training_3'/f'{i}.npz'
        with np.load(trace) as z:
            obs=saved_observations(z['reference'],z['rows'],z['actions'])
            with torch.no_grad():actions=decoded(p.distribution(torch.tensor(obs)).mean).numpy()
            errors.append(float(np.max(abs(actions-z['actions'][:len(obs)]))))
            observations.append(obs)
        row=copy.deepcopy(baseline['paths'][i]);row['reused_trace']=str(trace);row['reused_trace_sha256']=sha(trace);baseline_paths.append(row)
    assert max(errors)<1e-4
    x=torch.tensor(np.vstack(observations));x[:,8:11]=x.new_tensor([0.,.3,.25])
    with torch.no_grad():
        h=p.actor[:4](heading_frame(x)).numpy().astype(float)
        hm=p.actor[:4](heading_frame(reflect_observation(x))).numpy().astype(float)
    bases=[];metadata=[]
    for j in range(3):
        features=(h-hm)/2 if j==0 else (h+hm)/2
        if j==0:features=np.vstack([features,-features])
        center=np.zeros(features.shape[1]) if j==0 else features.mean(0)
        _,singular,axes=np.linalg.svd(features-center,full_matrices=False)
        fraction=np.cumsum(singular**2)/sum(singular**2)
        count=int(np.searchsorted(fraction,.99))+1
        transform=axes[:count]/np.maximum(singular[:count]/np.sqrt(len(features)-1),.05)[:,None]
        basis=np.zeros((count+(j>0),97));basis[:count,:96]=transform
        if j>0:
            basis[:count,96]=-transform@center;basis[-1,96]=1.
        bases.append(basis);metadata.append(dict(head=j,rank=count,parameters=len(basis),retained_variance=float(fraction[count-1]),singular=singular.tolist()))
    dimension=sum(len(b) for b in bases)
    np.savez_compressed(stage/'basis.npz',steering=bases[0],feed=bases[1],preview=bases[2])
    write_json(stage/'basis_contract.json',dict(heads=metadata,dimensions=dimension,source_observations=len(x),max_action_reconstruction_error=max(errors),physics_ticks=0))
    print('BASIS',[(r['head'],r['rank'],round(r['retained_variance'],4)) for r in metadata],flush=True)
    all_candidates=[];generations=[];checkpoints=[]
    incumbent=dict(tag='reused_baseline',coefficients=np.zeros(dimension).tolist(),paths=baseline_paths,training_actor_calls=0,training_physics_ticks=0)
    incumbent['rank']=rank(incumbent);all_candidates.append(incumbent)
    mean=np.zeros(dimension);sd=np.concatenate([np.full(len(b),scale/np.sqrt(len(b))) for b,scale in zip(bases,[.025,.18,.20])]);cov=np.diag(sd**2)
    paths=read(BASE/'learning_preparation_v4/paths.json');training=[np.asarray(paths['training'][i]) for i in indices];validation=[np.asarray(r) for r in paths['validation']]
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,parameter_updates=0)
    reserve_decisions=12*sum(math.ceil(environment_factory(r).max_ticks/20) for r in training)
    reserve_ticks=12*sum(environment_factory(r).max_ticks for r in training)
    no_gain=0;last_eval='reused_baseline'
    def checkpoint(generation):
        nonlocal last_eval
        if incumbent['tag']==last_eval:return
        p.load_state_dict(parameter_state(saved,bases,incumbent['coefficients']))
        model=stage/f'best_generation_{generation}.pt';save_torch(model,dict(policy=p.state_dict(),observation_dim=72,mode='joint',architecture='frame_feedback',frame='heading',history_independent=True,
            learning_method='constrained_joint_parameter_CEM',generation=generation,parent_sha256=sha(initial)))
        label=f'validation_{generation}';results=evaluate(p,validation,stage,label,None,costs,time.perf_counter()+max(5,deadline-time.time()),environment_factory)
        shapes=[];cruise=[]
        for i in range(12):
            with np.load(stage/label/f'{i}.npz') as z:
                speed=np.linalg.norm(z['rows'][:,2:4],axis=1);cruise.append(dict(path=i,peak=float(speed.max()),ge95_seconds=float((speed>=95).sum()*.001)))
                if i<4:
                    s=corner_shape(z['reference'],z['rows'],z['actions']);s['spatial']=geometric_turn_metrics(z['reference'],z['rows']);shapes.append(s)
        row=dict(generation=generation,tag=incumbent['tag'],successes=sum(r['success'] for r in results),
            full_time_s=sum(r['completion_time_s'] for r in results) if all(r['success'] for r in results) else None,
            shapes=shapes,cruise=cruise,model_sha256=sha(model),final_acceptance=False)
        checkpoints.append(row);last_eval=incumbent['tag'];write_json(stage/'checkpoints.json',dict(checkpoints=checkpoints,costs=costs))
        print('CHECKPOINT',generation,row['successes'],row['full_time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
    try:
        with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(stage),saved,bases,indices,training,deadline-45)) as pool:
            generation=0
            for generation in range(1,7):
                if costs['training_actor_calls']+reserve_decisions>80000 or costs['training_physics_ticks']+costs['evaluation_physics_ticks']+reserve_ticks>2000000 or time.time()>deadline-150:break
                factor=np.linalg.cholesky(cov);noise=rng.normal(size=(6,dimension));population=[mean+factor@n for n in np.vstack([noise,-noise])]
                candidates=list(pool.map(run_candidate,[(f'g{generation}_c{i}',c) for i,c in enumerate(population)]))
                for c in candidates:
                    c['rank']=rank(c);costs['training_actor_calls']+=c['training_actor_calls'];costs['training_physics_ticks']+=c['training_physics_ticks']
                all_candidates.extend(candidates);old=incumbent;ordered=sorted([incumbent]+candidates,key=lambda r:r['rank'],reverse=True);incumbent=ordered[0]
                elite=np.array([r['coefficients'] for r in ordered[:3]]);mean=.5*mean+.5*elite.mean(0)
                centered=elite-elite.mean(0);cov=.8*cov+.2*centered.T@centered/3;costs['parameter_updates']+=1
                no_gain=no_gain+1 if incumbent['tag']==old['tag'] else 0
                generations.append(dict(generation=generation,best_tag=incumbent['tag'],rank=incumbent['rank'],mean=mean.tolist(),covariance=cov.tolist(),no_gain=no_gain))
                write_json(stage/'training_results.json',dict(candidates=all_candidates,costs=costs));write_json(stage/'generations.json',dict(generations=generations,costs=costs))
                print('GENERATION',generation,incumbent['tag'],incumbent['rank'],'decisions',costs['training_actor_calls'],flush=True)
                if generation in (2,4,6):checkpoint(generation)
                if no_gain>=3:break
            checkpoint(generation)
        assert costs['training_actor_calls']<=80000 and costs['evaluation_actor_calls']<=6000 and costs['training_physics_ticks']+costs['evaluation_physics_ticks']<=2000000
        write_json(stage/'results.json',dict(checkpoints=checkpoints,costs=costs,selected_training_tag=incumbent['tag'],goal_complete=False))
        receipt(stage,dict(status='completed_bounded_constrained_frame_parameter_learning',costs=costs,wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json',dict(traceback=traceback.format_exc(),costs=costs));raise


if __name__=='__main__':main()
