"""Adaptive current-response minimax learning with measured low-dimensional steps."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from audit_worst_search_boundary import setup,STAGE as SOURCE
from completion_event_response import event_values,fit
from worst_guard_response import solve,worst_merit
from joint_response_models import fit_local_response
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


NAME='canonical_gate_band_recentered_minimax_joint_learning_20260920'


def exploration_directions(anchor,transform,model,emodel,buffer,normalizer,specs):
    local=dict(anchor,subspace_coordinates=[0.]*18);base=np.array(anchor['coefficients'])
    directions=[]
    for radius in (.25,.5,1.):
        result=solve(local,transform,model,emodel,buffer,radius,normalizer,base,specs)
        if result is not None and np.linalg.norm(result[0])>1e-10:
            directions.append(result[0]/np.linalg.norm(result[0]))
    # The current worst guard gradients supplement nearly collinear LP steps.
    order=np.argsort(anchor['augmented_learning_inequalities'])[:6]
    for index in order:
        row=model[:,index];norm=np.linalg.norm(row)
        if norm>1e-10:directions.append(row/norm)
    _,singular,vh=np.linalg.svd(np.array(directions),full_matrices=False)
    k=min(3,int(sum(singular>1e-8)))
    assert k>0
    return vh[:k].T,singular


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+580.
    history,initial,prior,global_transform,global_base,model,emodel,buffer,normalizer,specs=setup()
    assert read(SOURCE/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    assert read(SOURCE/'search_boundary_audit.json')['active_measurement_box_coordinates']==17
    md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    data=read(SOURCE/'baseline.json');support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in data['baseline']['records'] if r['path'] in paths]
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'shadow_training_policy.pt',
        SOURCE/'shadow_raw_identity_audit.json',SOURCE/'state_joint/results.json',SOURCE/'search_boundary_audit.json',
        SOURCE/'baseline.json',SOURCE/'support_audit.json',SOURCE/'completion_specs.json']+
        [BASE/f for f in ('audit_worst_search_boundary.py','worst_guard_response.py','completion_event_response.py',
            'joint_response_models.py','joint_training_evaluator.py','learn_coupled_early_late.py')],
        dict(wall_s=600,workers=4,maximum_rounds=6,maximum_new_policies=50,maximum_round_proposals=2,
            actor_parameters=54,global_search_dimensions=18,maximum_local_dimensions=3,
            normalized_direction_amplitude=.5,local_radius=1.,ppo_updates=0),
        'Worst-guard credit repaired actual worst violation3.839->2.572 with good response '
        'agreement,then17/18 SEARCH coordinates reached the old measurement-center box; '
        'zero actor coefficient bounds were active. Offline recentering predicts further '
        'descent but is not validation. Integrate current-response recentering: at each '
        'accepted actor choose at most3 joint directions from current minimax proposals '
        'and worst-guard gradients in the existing18-space; collect symmetric COMPLETE '
        'returns at that actor BEFORE updating. Historical full-space fits select only '
        'exploration directions and can never directly authorize an actor update. Fit the '
        'actually measured local response; require full local rank and physical success '
        'plus actual/predicted worst-merit improvement>=0.1. At most2 updates per fresh '
        'center,then remeasure. First round also compares prior-projected/current response '
        'at identical local space and radius. Keep all raw adoption guards and endpoint '
        'event checks; unsupported event branches stop for audit.6 rounds/50 new policies/ '
        '600s,all15 affected full routes per policy.54 actor weights and ordinary deployment '
        'unchanged;this is closed-loop parameter optimization,not PPO or a speed rule. '
        'If current measured directions fail twice consecutively,stop this mechanism for '
        'a representation/response audit,not more unmodified iterations.')
    folder=stage/'state_joint';folder.mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support);write_json(stage/'completion_specs.json',specs)
    write_json(stage/'subspace.json',dict(base=global_base.tolist(),transform=global_transform.tolist()))
    write_json(stage/'selection_hold.json',dict(reason='Adaptive measured minimax training; original guards unchanged.'))
    torch.save(md,stage/'initial_policy.pt');entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    new=[];blocks=[];rounds=[];pool=None;anchor=initial;minimum_buffer=buffer.copy();failed_rounds=0
    event_prior=emodel.copy();amplitude=.5;cache={tuple(c['coefficients']):c for c in history}
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(global_coordinates,tag,local_coordinates):
        if time.perf_counter()>deadline-15 or len(new)>=50:raise TimeoutError('Recentered minimax budget')
        u=global_base+global_transform@global_coordinates
        if tuple(u) in cache:
            c=copy.deepcopy(cache[tuple(u)]);c['local_coordinates']=local_coordinates.tolist();return c
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag)
        c.update(subspace_coordinates=global_coordinates.tolist(),local_coordinates=local_coordinates.tolist())
        try:event_values(c,specs);c['event_support_valid']=True
        except (ValueError,AssertionError) as exc:c.update(event_support_valid=False,event_support_error=str(exc))
        history.append(c);new.append(c);cache[tuple(u)]=c
        write_json(folder/'results.json',dict(candidates=history,best=None))
        return c
    def local_view(c):return dict(c,subspace_coordinates=c['local_coordinates'])
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,initial['records'],folder,deadline))
        for round_index in range(6):
            if time.perf_counter()>deadline-80 or failed_rounds>=2:break
            supported=[c for c in history if c['physical_pass'] and c['event_support_valid']]
            proxy=[dict(c,coefficients=c['subspace_coordinates']) for c in supported]
            prior_model,information=fit_local_response(proxy,dict(anchor,coefficients=anchor['subspace_coordinates']),
                np.ones(18),prior)
            center=np.array(anchor['subspace_coordinates']);values=np.array(anchor['completion_event_margins_mm'])
            near=[c for c in supported if c['tag']!=anchor['tag'] and np.max(abs(np.array(c['subspace_coordinates'])-center))<=4.]
            x=np.array([np.array(c['subspace_coordinates'])-center for c in near])
            y=np.array([np.array(c['completion_event_margins_mm'])-values for c in near])
            weight=1./np.maximum(np.linalg.norm(x,axis=1),.25)
            prior_event=np.linalg.solve((x*weight[:,None]).T@x+.01*np.eye(18),(x*weight[:,None]).T@y+.01*event_prior)
            basis,singular=exploration_directions(anchor,global_transform,prior_model,prior_event,
                minimum_buffer,normalizer,specs)
            local_transform=amplitude*global_transform@basis;k=basis.shape[1];local_base=np.array(anchor['coefficients'])
            local_initial=copy.deepcopy(anchor);local_initial['local_coordinates']=[0.]*k;local_history=[local_initial]
            pairs=[];unsupported=False
            for j in range(k):
                d=np.eye(k)[j]
                pair=(evaluate(center+amplitude*basis@d,f'round{round_index}_response{j}_plus',d),
                      evaluate(center-amplitude*basis@d,f'round{round_index}_response{j}_minus',-d))
                pairs.append(pair);local_history.extend(pair)
                unsupported |= any(not c['event_support_valid'] or not c['physical_pass'] for c in pair)
            if unsupported:
                rounds.append(dict(round=round_index,stop='Current response encountered unsupported event or physical failure'))
                break
            fresh=np.array([np.r_[(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2,
                (a['training_score']-b['training_score'])/2] for a,b in pairs])
            event=np.array([(np.array(a['completion_event_margins_mm'])-b['completion_event_margins_mm'])/2 for a,b in pairs])
            residual=(np.array([c['completion_event_margins_mm'] for c in local_history])-values-
                np.array([c['local_coordinates'] for c in local_history])@event)
            local_buffer=np.maximum(minimum_buffer,np.max(abs(residual),axis=0)+.001)
            np.savez_compressed(stage/f'round{round_index}_current_response.npz',basis=basis,transform=local_transform,
                center=center,base=local_base,fresh=fresh,event=event,buffer=local_buffer)
            round_record=dict(round=round_index,anchor=anchor['tag'],local_dimensions=k,
                prior_global_information=information,new_current_responses=2*k,singular_values=singular.tolist())
            print('RECENTERED_CURRENT_RESPONSE',round_record,flush=True)
            local_anchor=local_initial;radius=1.;accepted_in_round=0
            arms=['prior_control','current','current_update'] if round_index==0 else ['current','current_update']
            for arm in arms:
                if time.perf_counter()>deadline-15:break
                if arm=='prior_control':
                    fitted=amplitude*basis.T@prior_model;ejac=amplitude*basis.T@prior_event;chosen_buffer=local_buffer
                elif arm=='current':fitted,ejac,chosen_buffer=fresh,event,local_buffer
                else:
                    fitted,ejac,estimated_buffer=fit([local_view(c) for c in local_history],local_view(local_anchor),fresh)
                    chosen_buffer=np.maximum(local_buffer,estimated_buffer)
                solution=solve(local_view(local_anchor),local_transform,fitted,ejac,chosen_buffer,
                    radius,normalizer,local_base,specs)
                if solution is None:
                    blocks.append(dict(round=round_index,arm=arm,stop='No event-feasible local step'));continue
                delta,predicted,events,info=solution
                if predicted<=1e-9:
                    blocks.append(dict(round=round_index,arm=arm,stop='No predicted worst-merit descent'));continue
                a=np.array(local_anchor['local_coordinates'])+delta
                c=evaluate(center+amplitude*basis@a,f'round{round_index}_{arm}',a);local_history.append(c)
                actual=worst_merit(local_anchor,normalizer)-worst_merit(c,normalizer);ratio=actual/predicted
                accepted=c['physical_pass'] and actual>0 and ratio>=.1
                block=dict(round=round_index,arm=arm,anchor=local_anchor['tag'],measured=c['tag'],radius=radius,
                    predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,
                    worst_merit=worst_merit(c,normalizer),original_merit=merit(c),
                    original_guard_min=min(c['augmented_measured_inequalities']),event_margins_mm=c.get('completion_event_margins_mm'),
                    predicted_event_margins_mm=events.tolist(),**info)
                blocks.append(block);print('RECENTERED_MINIMAX_UPDATE',block,flush=True)
                if arm!='prior_control':
                    if accepted:local_anchor=c;anchor=c;accepted_in_round+=1
                    if ratio<.25 or not c['physical_pass']:radius*=.25
                    elif ratio>.75 and accepted:radius=min(1.,radius*2.)
                write_json(stage/'learning_updates.json',blocks)
                if not c['event_support_valid'] or not c['physical_pass'] or acceptable(c):unsupported=True;break
            round_record.update(accepted_current_updates=accepted_in_round,best=anchor['tag'])
            rounds.append(round_record);write_json(stage/'rounds.json',rounds)
            failed_rounds=failed_rounds+1 if accepted_in_round==0 else 0
            if accepted_in_round==0:amplitude*=.5
            if unsupported:break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,new));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);u=np.array(chosen['coefficients'])
        actor.update(coefficients=u[:18].reshape(-1,3),accepted=False,training_branch_only=True)
        actor['parent']['coefficients']=u[18:].reshape(-1,3);torch.save(actor,stage/'shadow_training_policy.pt')
        by_path={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by_path.get(r['path'],r)) for r in read(SOURCE/'shadow_training_result.json')['records']]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),worst_merit=worst_merit(chosen,normalizer),
            original_all_guard_pass=bool(passing),adopted=False))
        write_json(stage/'learning_updates.json',blocks);write_json(stage/'rounds.json',rounds)
        receipt(stage,dict(status='completed_recentered_minimax_joint_learning',new_policies=len(new),rounds=len(rounds),
            training_best=chosen['tag'],accepted_current_updates=sum(b.get('accepted',False) and b['arm']!='prior_control' for b in blocks),
            original_guard_passing_new_policies=len(passing),research_acceptable=False,pending_candidate_audit=bool(passing),
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),ppo_updates=0,wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_recentered_minimax_joint_learning',new_policies=len(new),error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
