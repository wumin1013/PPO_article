"""Fit a single three-action feedback network from training-only episode data.

The deployed artifact contains network weights only. Offline per-path time
coefficients are used to select source training trajectories, never at run time.
This is supervised policy improvement from physically optimized episodes,
not a PPO gradient update and not proof that the resulting policy is safe.
"""
import time
import argparse
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from audit_search_feature_coverage import saved_observations,policy_held
from search_complete_return import read
from train_pilot import save_torch,evaluate
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from study_io import BASE,register,receipt,write_json,sha


def decoded(mean):
    z=torch.tanh(mean)
    return torch.stack([np.pi/2*z[:,0],50*(1+z[:,1]),6.4+5.6*z[:,2]],dim=1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--consistent-teachers',action='store_true');parser.add_argument('--interior-margin',action='store_true');args=parser.parse_args()
    if args.interior_margin and not args.consistent_teachers:raise ValueError('Margin-only contrast uses the consistent training pool')
    torch.set_num_threads(1);torch.manual_seed(916509)
    source=BASE/'preview_state_residual_learning_20260916';initial=source/'best_generation_5.pt'
    teachers=[BASE/'training_corner_teacher_optimization_20260916',BASE/'training_straight_teacher_optimization_20260916']
    if args.consistent_teachers:teachers.append(BASE/'training_remaining_teacher_optimization_20260916')
    prior_pools=[BASE/f for f in ('spatial_bending_learning_20260916','dynamic_state_residual_learning_20260916','preview_state_residual_learning_20260916')] if args.consistent_teachers else []
    stage=register('interior_teacher_feedback_learning_20260916' if args.interior_margin else 'consistent_teacher_feedback_learning_20260916' if args.consistent_teachers else 'optimized_rollout_feedback_learning_20260916',[__file__,initial,source/'receipt.json',source/'training_results.json',source/'generations.json',
        BASE/'dynamic_residual_policy.py',BASE/'audit_search_feature_coverage.py',BASE/'learning_preparation_v4/paths.json']+
        [f/n for f in teachers+prior_pools for n in ('receipt.json','training_results.json')],
        dict(supervised_updates=200,learning_rate=.0003,seed=916509,evaluation_epochs=[50,100,200],
             teacher_eligibility_euclidean_distance_mm=1.05 if args.interior_margin else 1.25,
             unchanged_environment_and_acceptance_tolerance_mm=1.25,
             physical_training_decisions=0,evaluation_decisions=5000,physics_ticks=150000,wall_s=240,
             teacher_paths=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53] if args.consistent_teachers else [14,15,50,53],
             parent_anchor_paths=[] if args.consistent_teachers else [4,5,6,8,9,11,12,13,18,22,26,34]),
        ('Margin-only source-selection contrast to consistent_teacher_feedback_learning: keep the same pool, source weights, seed, optimizer,200 updates and checkpoint schedule; require all selected recorded motion segments to remain inside a1.05mm Euclidean corridor. Original simulator and external acceptance stay at1.25mm. This conservative data criterion tests approximation robustness, not a relaxed constraint or an online trajectory rule. ' if args.interior_margin else
         'Training-source contrast: same initial weights, seed, update count, optimizer, equal path weights and evaluation schedule as optimized_rollout_feedback_learning; include optimized outcomes on all sixteen training paths, retaining eligible parent and previously computed complete training trajectories in candidate selection. Select with the same per-path objective, not a mixture of optimized labels and obligatory old-policy anchors. No additional epochs. ' if args.consistent_teachers else '')+
        'Fit all actor and existing residual weights jointly from complete successful training episodes selected before development evaluation. Isolated turns require inward first/outward<=0.05mm/reversal<=5deg, then rank full time+normalized energy; other paths use full time. Equal total weight per path. No timing coefficients or path lookup in the learned runtime. Supervised mechanism pilot, not new PPO updates or final validation.')
    started=time.perf_counter();saved=torch.load(initial,map_location='cpu',weights_only=False)
    policy=PreviewResidualPolicy(saved['observation_dim']);policy.load_state_dict(saved['policy'])
    parent_tag=next(g['best_tag'] for g in read(source/'generations.json')['generations'] if g['generation']==5)
    parent=next(c for c in read(source/'training_results.json')['candidates'] if c['tag']==parent_tag)
    extras={};catalogs={}
    for folder in prior_pools:
        rows=read(folder/'training_results.json')['candidates']
        catalogs[str(folder/'training_results.json')]={r['tag']:r for r in rows}
        for candidate in rows:
            for record in candidate['results']:
                if not record['success']:continue
                origin=folder;row=candidate
                while not (origin/'training_traces'/record['trace']).exists():
                    from pathlib import Path
                    source_file=Path(row['reused_from'])
                    if str(source_file) not in catalogs:catalogs[str(source_file)]={r['tag']:r for r in read(source_file)['candidates']}
                    row=catalogs[str(source_file)][row.get('physical_source_tag',row['tag'])];origin=source_file.parent
                extras.setdefault(record['path'],[]).append((dict(record,tag=folder.name+'/'+candidate['tag']),origin/'training_traces'/record['trace']))
    selection={}
    margin_cache={};margin_checks=[]
    def maximum_distance(trace):
        key=str(trace)
        if key not in margin_cache:
            with np.load(trace) as z:
                ref=z['reference'];xy=z['rows'][:,:2];vectors=np.diff(ref,axis=0)
                delta=xy[:,None,:]-ref[:-1]
                t=np.clip(np.sum(delta*vectors,axis=2)/np.sum(vectors*vectors,axis=1),0.,1.)
                distance=np.linalg.norm(delta-t[...,None]*vectors,axis=2).min(axis=1)
                margin_cache[key]=float(distance.max())
        return margin_cache[key]
    for folder in teachers:
        all_rows=read(folder/'training_results.json')['candidates']
        for path_index in sorted({r['path'] for r in all_rows}):
            good=[]
            candidates=[(r,folder/r['trace']) for r in all_rows]
            if args.consistent_teachers:
                original=next(r for r in parent['results'] if r['path']==path_index)
                candidates.append((dict(original,tag='eligible_parent_baseline'),source/'training_traces'/original['trace']))
                candidates.extend(extras.get(path_index,[]))
            for r,trace in candidates:
                if r['path']!=path_index or not r['success']:continue
                s=r['shape']
                if s is not None and (s['first_physical_turn']!='inside' or s['outward_excursion_mm']>.05 or s['reverse_turning_total_deg']>5):continue
                if args.interior_margin and maximum_distance(trace)>1.05:continue
                good.append((r,trace))
            assert good,('No acceptable training source',path_index)
            good.sort(key=lambda entry:entry[0]['complete_time_s']+(1.25*entry[0]['spatial']['normalized_bending_mm_inv'] if entry[0].get('spatial') else 0))
            distinct=[];seen=set()
            for r,trace in good:
                digest=sha(trace)
                if digest in seen:continue
                if args.interior_margin:
                    from geometry_spec import Corridor
                    with np.load(trace) as z:
                        corridor=Corridor(z['reference'],epsilon=1.05);points=np.vstack([z['reference'][0],z['rows'][:,:2]])
                        inside=all(corridor.segment_inside(a,b) for a,b in zip(points[:-1],points[1:]))
                    margin_checks.append(dict(path=path_index,trace=str(trace),sampled_max_distance=maximum_distance(trace),all_segments_inside_1_05=inside))
                    if not inside:continue
                seen.add(digest);distinct.append((trace,r['tag']))
                if len(distinct)==2:break
            assert distinct,('No complete interior training source',path_index)
            selection[path_index]=distinct
    if args.interior_margin:write_json(stage/'margin_selection_audit.json',dict(checked_sources=len(margin_cache),selected_checks=margin_checks,physics_ticks=0,external_tolerance_unchanged=1.25))
    data=[];targets=[];weights=[];manifest=[];path_ids=[]
    for record in parent['results']:
        path_index=record['path']
        files=selection.get(path_index,[(source/'training_traces'/record['trace'],parent_tag)])
        for file,tag in files:
            with np.load(file) as d:
                observations=d['observation'].copy() if 'observation' in d else saved_observations(d['reference'],d['rows'],d['actions'])
                held=d['actions'].copy()
                count=min(len(observations),len(held));ticks=np.arange(count)*20
                used=d['rows'][ticks,12]<.5
                observations=observations[:count][used];held=held[:count][used]
                if path_index not in selection:
                    with torch.no_grad():pred=decoded(policy.distribution(torch.tensor(observations)).mean).numpy()
                    assert np.max(np.abs(pred-held))<1e-4
            data.append(observations);targets.append(held);weights.extend([1/(len(files)*len(held))]*len(held));path_ids.extend([path_index]*len(held))
            manifest.append(dict(path=path_index,trace=str(file),sha256=sha(file),optimized_source=file.parent!=source/'training_traces',source_tag=tag,rows=len(held)))
    x=torch.tensor(np.vstack(data),dtype=torch.float32);y=torch.tensor(np.vstack(targets),dtype=torch.float32)
    w=torch.tensor(weights,dtype=torch.float32);w/=w.sum()
    np.savez_compressed(stage/'training_data.npz',observation=x.numpy(),actions=y.numpy(),sample_weight=w.numpy(),path=np.array(path_ids))
    write_json(stage/'training_sources.json',dict(manifest=manifest,rows=len(x),selected_before_evaluation=True))
    parameters=list(policy.actor.parameters())+list(policy.dynamic_head.parameters())+list(policy.preview_head.parameters())
    optimizer=torch.optim.Adam(parameters,lr=.0003);scale=x.new_tensor([np.deg2rad(3),10.,1.])
    costs=dict(training_actor_calls=0,training_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    history=[];checkpoints=[];validation=[np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['validation']]
    for epoch in range(1,201):
        assert time.perf_counter()-started<240
        prediction=decoded(policy.distribution(x).mean)
        errors=(prediction-y)/scale
        loss=(torch.nn.functional.smooth_l1_loss(errors,torch.zeros_like(errors),reduction='none').mean(1)*w).sum()
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();costs['supervised_updates']+=1
        if epoch%10==0:
            history.append(dict(epoch=epoch,loss=float(loss.detach()),physical_rmse=torch.sqrt(((prediction-y)**2*w[:,None]).sum(0)).detach().tolist()))
        if epoch not in (50,100,200):continue
        path=stage/f'epoch_{epoch}.pt'
        save_torch(path,dict(policy=policy.state_dict(),observation_dim=saved['observation_dim'],mode='joint',architecture='reflection_preview_residual',
            learning_method='supervised_feedback_learning_from_optimized_training_episodes',epoch=epoch,parent_sha256=sha(initial)))
        result=evaluate(policy,validation,stage,f'validation_{epoch}',None,costs,started+240,environment_factory)
        assert costs['evaluation_actor_calls']<=5000 and costs['evaluation_physics_ticks']<=150000
        shapes=[];cruise=[]
        for i in range(12):
            with np.load(stage/f'validation_{epoch}'/f'{i}.npz') as d:
                speed=np.linalg.norm(d['rows'][:,2:4],axis=1)
                cruise.append(dict(path=i,peak=float(speed.max()),ge95_seconds=float((speed>=95).sum()*.001)))
                if i<4:
                    s=corner_shape(d['reference'],d['rows'],d['actions']);s['spatial']=geometric_turn_metrics(d['reference'],d['rows']);shapes.append(s)
        row=dict(epoch=epoch,successes=sum(r['success'] for r in result),time_s=sum(r['completion_time_s'] for r in result) if all(r['success'] for r in result) else None,
            shapes=shapes,cruise=cruise,model_sha256=sha(path),final_shape_acceptance=False)
        checkpoints.append(row);write_json(stage/'progress.json',dict(checkpoints=checkpoints,fit_history=history,costs=costs))
        print('EPOCH',epoch,row['successes'],row['time_s'],[(round(s['turn_deg']),round(s['outward_excursion_mm'],3),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
    write_json(stage/'results.json',dict(checkpoints=checkpoints,fit_history=history,costs=costs,goal_complete=False,
        boundary='Single feedback network, no timing schedule or per-path parameters at evaluation. Supervised fitting is not proof of matching optimized teacher trajectories or generalization.'))
    receipt(stage,dict(status='completed_bounded_feedback_learning',costs=costs,wall_s=time.perf_counter()-started,training_data_sha256=sha(stage/'training_data.npz')))


if __name__=='__main__':main()
