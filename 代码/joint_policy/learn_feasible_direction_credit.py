"""Measure an unprojected minimax direction before trusting its predicted credit."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from completion_event_response import event_values,fit
from worst_guard_response import solve,worst_merit
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_recentered_minimax_joint_learning_20260920'
NAME='canonical_gate_band_feasible_direction_current_credit_20260920'


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+135.
    all_history=read(SOURCE/'state_joint/results.json')['candidates']
    initial=copy.deepcopy(next(c for c in all_history if c['tag']==read(SOURCE/'receipt.json')['training_best']))
    audit=read(SOURCE/'subspace_retention_audit.json')['records'][-1]
    assert initial['tag']==audit['anchor']
    assert read(SOURCE/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    subspace=read(SOURCE/'subspace.json');global_t=np.array(subspace['transform'])
    direction=np.array(audit['full_direction']);transform=(global_t@direction)[:,None]
    base=np.array(initial['coefficients']);md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    data=read(SOURCE/'baseline.json');support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in data['baseline']['records'] if r['path'] in paths]
    specs=read(SOURCE/'completion_specs.json');event_values(initial,specs)
    normalizer=read(BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920/response_error_channels.json')['normalizer']
    with np.load(SOURCE/'round5_current_response.npz') as z:buffer=z['buffer']
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'shadow_training_policy.pt',
        SOURCE/'shadow_raw_identity_audit.json',SOURCE/'state_joint/results.json',SOURCE/'subspace_retention_audit.json',
        SOURCE/'subspace.json',SOURCE/'round5_current_response.npz',SOURCE/'baseline.json',SOURCE/'support_audit.json',
        SOURCE/'completion_specs.json']+[BASE/f for f in ('worst_guard_response.py','completion_event_response.py',
            'joint_training_evaluator.py','learn_coupled_early_late.py')],
        dict(wall_s=150,workers=4,maximum_new_policies=6,current_direction_responses=[-.5,.5],
            full_model_direction_control=1.,maximum_followup_updates=3,actor_parameters=54,ppo_updates=0),
        'The last refreshed3-space has no predicted descent,while the historical/current '
        '18-space model predicts.191 worst-merit improvement for an unprojected direction. '
        'Its projection loses14.7percent of vector norm;this does not prove missing causal '
        'capacity because full-model prediction may be wrong. At the SAME final audited actor '
        'measure +/-.5 of that exact joint54-weight direction,then the full alpha1 proposal '
        'as an explicit model check. Learn at most3 subsequent one-dimensional coefficients '
        'from measured current full-return responses. Exact cached alpha points are never '
        'rerun. All15 affected routes,full physics and original final raw guards remain. '
        '150s/6 new policies; no PPO,no deployed phase/angle/velocity rule. Accept only actual '
        'worst-merit decrease,physical success and model agreement for learned updates. '
        'If observed direction disagrees,do not infer infeasibility or extend this line search.')
    folder=stage/'state_joint';folder.mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support);write_json(stage/'completion_specs.json',specs)
    write_json(stage/'direction.json',dict(full18_direction=direction.tolist(),actor_direction=transform[:,0].tolist()))
    write_json(stage/'selection_hold.json',dict(reason='Unprojected direction/current response test; final guards unchanged.'))
    torch.save(md,stage/'initial_policy.pt');entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    initial['global_subspace_coordinates']=initial['subspace_coordinates'];initial['subspace_coordinates']=[0.]
    initial['tag']='reused_current_actor';initial['event_support_valid']=True
    history=[initial];cache={0.:initial};new=[];blocks=[];anchor=initial;pool=None
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(alpha,tag):
        alpha=float(alpha)
        if alpha in cache:return cache[alpha]
        if time.perf_counter()>deadline-15 or len(new)>=6:raise TimeoutError('Current feasible direction budget')
        u=base+transform[:,0]*alpha
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag);c['subspace_coordinates']=[alpha]
        c['global_subspace_coordinates']=(np.array(initial['global_subspace_coordinates'])+direction*alpha).tolist()
        try:event_values(c,specs);c['event_support_valid']=True
        except (ValueError,AssertionError) as exc:c.update(event_support_valid=False,event_support_error=str(exc))
        history.append(c);new.append(c);cache[alpha]=c
        write_json(folder/'results.json',dict(candidates=history,best=None))
        print('FEASIBLE_DIRECTION_ACTUAL',tag,alpha,c['physical_pass'],worst_merit(c,normalizer),flush=True)
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,initial['records'],folder,deadline))
        plus,minus=evaluate(.5,'response_plus'),evaluate(-.5,'response_minus')
        control=evaluate(1.,'full18_model_proposal')
        actual=worst_merit(initial,normalizer)-worst_merit(control,normalizer)
        control_good=control['physical_pass'] and actual>0 and actual/audit['full18_predicted_drop']>=.1
        blocks.append(dict(arm='full18_model_control',measured=control['tag'],predicted_drop=audit['full18_predicted_drop'],
            actual_drop=actual,ratio=actual/audit['full18_predicted_drop'],accepted=control_good))
        print('FEASIBLE_DIRECTION_CONTROL',blocks[-1],flush=True)
        if control_good:anchor=control
        supported=all(c['physical_pass'] and c['event_support_valid'] for c in history)
        if supported:
            fresh=np.r_[np.array(plus['augmented_learning_inequalities'])-minus['augmented_learning_inequalities'],
                plus['training_score']-minus['training_score']][None,:]
            radius=.5
            for iteration in range(3):
                model,emodel,residual_buffer=fit(history,anchor,fresh)
                solution=solve(anchor,transform,model,emodel,np.maximum(buffer,residual_buffer),radius,normalizer,base,specs)
                if solution is None or solution[1]<=1e-9:break
                delta,predicted,events,info=solution
                c=evaluate(anchor['subspace_coordinates'][0]+delta[0],f'current_direction_{iteration}')
                actual=worst_merit(anchor,normalizer)-worst_merit(c,normalizer);ratio=actual/predicted
                accepted=c['physical_pass'] and actual>0 and ratio>=.1
                block=dict(arm='current_direction',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],
                    predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,
                    worst_merit=worst_merit(c,normalizer),original_merit=merit(c),**info)
                blocks.append(block);print('FEASIBLE_DIRECTION_UPDATE',block,flush=True)
                if accepted:anchor=c
                if ratio<.25:radius*=.25
                write_json(stage/'learning_updates.json',blocks)
                if not c['event_support_valid'] or not c['physical_pass'] or acceptable(c) or radius<.03125:break
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
        write_json(stage/'learning_updates.json',blocks)
        receipt(stage,dict(status='completed_feasible_direction_current_credit',new_policies=len(new),training_best=chosen['tag'],
            accepted_updates=sum(b.get('accepted',False) for b in blocks),original_guard_passing_new_policies=len(passing),
            research_acceptable=False,pending_candidate_audit=bool(passing),shadow_sha256=sha(stage/'shadow_training_policy.pt'),
            ppo_updates=0,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_feasible_direction_current_credit',new_policies=len(new),error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
