"""Matched regression continuation with verified executed-teacher data.

Offline mixtures are collection controllers only. Labels are actual teacher
actions from fully successful mixed trajectories, not unexecuted proposals.
Both evaluation arms run the neural feedback policy alone.
"""
import time
import math
import numpy as np
import torch
from dynamic_residual_policy import PreviewResidualPolicy
from train_feedback_from_optimized_rollouts import decoded
from search_complete_return import read
from probe_coordinated_suffix import encode
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from train_pilot import evaluate,save_torch
from validate_actions import difference_audit
from study_io import BASE,register,receipt,write_json,sha


def load(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=PreviewResidualPolicy(saved['observation_dim']);model.load_state_dict(saved['policy'])
    return model


def main():
    torch.set_num_threads(1)
    source=BASE/'consistent_teacher_feedback_learning_20260916';initial=source/'epoch_50.pt'
    parent=BASE/'preview_state_residual_learning_20260916'
    teacher_folders=[BASE/f for f in ('training_corner_teacher_optimization_20260916','training_straight_teacher_optimization_20260916','training_remaining_teacher_optimization_20260916')]
    indices=[4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53]
    stage=register('feedback_dataset_aggregation_20260916',[__file__,initial,source/'receipt.json',source/'training_data.npz',
        parent/'best_generation_5.pt',BASE/'dynamic_residual_policy.py',BASE/'train_feedback_from_optimized_rollouts.py',
        BASE/'feedback_training_reproduction_audit_20260916/receipt.json',BASE/'learning_preparation_v4/paths.json']+
        [folder/'training_results.json' for folder in teacher_folders],
        dict(collection_episodes=32,collection_paths=indices,executed_teacher_probability=.7,seed=916611,
             collection_decisions=5000,evaluation_decisions=5000,physics_ticks=220000,wall_s=300,
             arms=['static_data','aggregated_data'],updates_per_arm=100,evaluation_updates=[50,100],learning_rate=.0003),
        'From the same epoch50 network compare100 new full-actor regression steps with unchanged data versus added states visited by learner/teacher mixtures. Same optimizer reset, learning rate, path weighting and validation schedule. Teacher action labels are included only when actually executed in a fully successful mixed episode, excluding terminal-only actions. Teachers remain offline and path-specific; neither evaluation arm has a lookup, schedule or override. Collect first; do not fit if either previously failing mixed training path18/26 has no successful labeled coverage.')
    learner=load(initial);teacher=load(parent/'best_generation_5.pt')
    settings={}
    for folder in teacher_folders:
        rows=read(folder/'training_results.json')['candidates']
        for path_index in sorted({r['path'] for r in rows}):
            good=[]
            for r in rows:
                if r['path']!=path_index or not r['success']:continue
                s=r['shape']
                if s and (s['first_physical_turn']!='inside' or s['outward_excursion_mm']>.05 or s['reverse_turning_total_deg']>5):continue
                good.append(r)
            if good:
                chosen=min(good,key=lambda r:r['complete_time_s']+(1.25*r['spatial']['normalized_bending_mm_inv'] if r['spatial'] else 0))
                with np.load(folder/chosen['trace']) as z:settings[path_index]=(z['knot_time'].copy(),np.c_[np.zeros(3),z['temporal_coefficients'],np.zeros(3)])
            else:settings[path_index]=(np.array([0.,100.]),np.zeros((3,2)))
    write_json(stage/'collection_teachers.json',{str(k):dict(knots=v[0].tolist(),values=v[1].tolist(),deployment=False) for k,v in settings.items()})
    paths=read(BASE/'learning_preparation_v4/paths.json');started=time.perf_counter()
    costs=dict(collection_decisions=0,collection_physics_ticks=0,evaluation_actor_calls=0,evaluation_physics_ticks=0,supervised_updates=0)
    rng=np.random.default_rng(916611);collection=[];new_x=[];new_y=[];new_ids=[]
    for repeat in range(2):
        for path_index in indices:
            env=environment_factory(np.asarray(paths['training'][path_index]));knots,values=settings[path_index]
            chunks=[];held=[];obs_rows=[];teacher_mask=[]
            while not env.done:
                assert costs['collection_decisions']<5000 and time.perf_counter()-started<300
                assert costs['collection_physics_ticks']+costs['evaluation_physics_ticks']<217800
                obs=env.observation();use_teacher=bool(rng.uniform()<.7)
                latent=(teacher if use_teacher else learner).act(obs,True)[0]
                if use_teacher:
                    delta=np.array([np.interp(env.ticks*.001,knots,values[i]) for i in range(3)])
                    if np.any(delta):
                        target=np.clip(env.decode_action(latent)+delta,[-math.pi/2+1e-6,1e-5,.800001],[math.pi/2-1e-6,99.99999,11.999999])
                        latent=encode(target)
                _,_,_,info=env.step(latent);chunks.append(info['rows']);held.append(info['held_action']);obs_rows.append(obs)
                teacher_mask.append(use_teacher and not bool(info['rows'][0,12]))
                costs['collection_decisions']+=1;costs['collection_physics_ticks']+=len(info['rows'])
            rows=np.vstack(chunks);actions=np.asarray(held);observations=np.asarray(obs_rows);mask=np.asarray(teacher_mask)
            audit=difference_audit(env.corridor.reference[0],rows);success=bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
            tag=f'mixture_r{repeat}_p{path_index}'
            np.savez_compressed(stage/f'{tag}.npz',reference=env.corridor.reference,rows=rows,actions=actions,observation=observations,executed_teacher=mask)
            if success:
                new_x.append(observations[mask]);new_y.append(actions[mask]);new_ids.extend([path_index]*int(mask.sum()))
            collection.append(dict(path=path_index,repeat=repeat,success=success,reason=env.reason,trace=f'{tag}.npz',
                full_time_s=env.ticks*.001 if success else None,teacher_labels_retained=int(mask.sum()) if success else 0,
                audit=audit,geometry=env.tracker.snapshot(1.25)))
            write_json(stage/'collection_results.json',dict(records=collection,costs=costs))
            print('COLLECT',repeat,path_index,success,'labels',collection[-1]['teacher_labels_retained'],flush=True)
    if not {18,26}.issubset(new_ids):
        receipt(stage,dict(status='collection_insufficient_for_registered_fit',costs=costs,wall_s=time.perf_counter()-started,
            reason='No successful labeled coverage of both failing mixed training paths; no training performed.',goal_complete=False))
        return
    new_x=np.vstack(new_x);new_y=np.vstack(new_y);new_ids=np.asarray(new_ids)
    np.savez_compressed(stage/'new_data.npz',observation=new_x,actions=new_y,path=new_ids)
    with np.load(source/'training_data.npz') as z:old_x=z['observation'];old_y=z['actions'];old_w=z['sample_weight'];old_ids=z['path']
    histories=[];checkpoints=[];validation=[np.asarray(p) for p in paths['validation']]
    for arm in ('static_data','aggregated_data'):
        torch.manual_seed(916611);model=load(initial)
        x=old_x;y=old_y;w=old_w.copy()
        if arm=='aggregated_data':
            weights=np.zeros(len(new_ids))
            for path_index in indices:
                mask=new_ids==path_index
                if not mask.any():continue
                w[old_ids==path_index]*=.5;weights[mask]=.5/(len(indices)*mask.sum())
            x=np.vstack([old_x,new_x]);y=np.vstack([old_y,new_y]);w=np.r_[w,weights]
        x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.float32);w=torch.tensor(w,dtype=torch.float32);w/=w.sum()
        params=list(model.actor.parameters())+list(model.dynamic_head.parameters())+list(model.preview_head.parameters())
        optimizer=torch.optim.Adam(params,lr=.0003);scale=x.new_tensor([np.deg2rad(3),10.,1.])
        for update in range(1,101):
            assert time.perf_counter()-started<300
            prediction=decoded(model.distribution(x).mean);error=(prediction-y)/scale
            loss=(torch.nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').mean(1)*w).sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();costs['supervised_updates']+=1
            if update%10==0:histories.append(dict(arm=arm,update=update,loss=float(loss.detach())))
            if update not in (50,100):continue
            model_path=stage/f'{arm}_{update}.pt'
            save_torch(model_path,dict(policy=model.state_dict(),observation_dim=72,mode='joint',architecture='reflection_preview_residual',
                learning_method='matched_feedback_dataset_aggregation',arm=arm,updates=update,initial_sha256=sha(initial)))
            name=f'{arm}_{update}_validation'
            result=evaluate(model,validation,stage,name,None,costs,started+300,environment_factory)
            assert costs['evaluation_actor_calls']<=5000 and costs['collection_physics_ticks']+costs['evaluation_physics_ticks']<=220000
            shapes=[];cruise=[]
            for i in range(12):
                with np.load(stage/name/f'{i}.npz') as d:
                    speed=np.linalg.norm(d['rows'][:,2:4],axis=1);cruise.append(float((speed>=95).sum()*.001))
                    if i<4:
                        s=corner_shape(d['reference'],d['rows'],d['actions']);s['spatial']=geometric_turn_metrics(d['reference'],d['rows']);shapes.append(s)
            row=dict(arm=arm,update=update,successes=sum(r['success'] for r in result),
                full_time_s=sum(r['completion_time_s'] for r in result) if all(r['success'] for r in result) else None,
                shapes=shapes,cruise95_seconds=cruise,model_sha256=sha(model_path),final_shape_acceptance=False)
            checkpoints.append(row);write_json(stage/'progress.json',dict(checkpoints=checkpoints,fit_history=histories,costs=costs))
            print('EVAL',arm,update,row['successes'],row['full_time_s'],
                [(round(s['turn_deg']),round(s['reverse_turning_total_deg'],2),round(s['spatial']['bending_energy_mm_inv'],2)) for s in shapes],flush=True)
    write_json(stage/'results.json',dict(collection=collection,checkpoints=checkpoints,fit_history=histories,costs=costs,goal_complete=False))
    receipt(stage,dict(status='completed_matched_dataset_aggregation',costs=costs,wall_s=time.perf_counter()-started))


if __name__=='__main__':main()
