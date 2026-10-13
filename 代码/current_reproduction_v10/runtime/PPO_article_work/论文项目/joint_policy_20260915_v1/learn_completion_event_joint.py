"""Compare staircase credit with observed completion-event credit in joint learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from completion_event_response import event_values,fit,propose
from learn_information_refreshed_joint import propose_subspace
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'
NAME='canonical_gate_band_completion_event_joint_learning_20260920'


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+185.
    assert read(SOURCE/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    assert sha(SOURCE/'shadow_training_policy.pt')==read(SOURCE/'receipt.json')['shadow_sha256']
    history=copy.deepcopy(read(SOURCE/'state_joint/results.json')['candidates'])
    anchor=next(c for c in history if c['tag']=='refreshed_trust_2')
    audit=read(SOURCE/'completion_event_fit_audit.json');specs=audit['specs']
    for c in history:event_values(c,specs)
    with np.load(SOURCE/'current_joint_response.npz') as z:
        fresh,transform,base=z['fresh'],z['transform'],z['base']
    normalizer=read(SOURCE/'response_error_channels.json')['normalizer']
    data=read(SOURCE/'baseline.json');support=read(SOURCE/'support_audit.json')
    paths=support['evaluation_paths'];selected=[r for r in data['baseline']['records'] if r['path'] in paths]
    md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'shadow_training_policy.pt',
        SOURCE/'shadow_raw_identity_audit.json',SOURCE/'state_joint/results.json',SOURCE/'baseline.json',
        SOURCE/'support_audit.json',SOURCE/'current_joint_response.npz',SOURCE/'response_error_channels.json',
        SOURCE/'terminal_event_error_audit.json',SOURCE/'completion_event_fit_audit.json']+
        [BASE/f for f in ('completion_event_response.py','audit_refreshed_terminal_events.py',
            'joint_training_evaluator.py','learn_coupled_early_late.py')],
        dict(wall_s=200,maximum_new_policies=10,workers=4,actor_parameters=54,search_dimensions=12,
            matched_radius=1.,maximum_radius=1.,maximum_updates=8,ppo_updates=0),
        'Saved exact replay attributes the two smallest rejected steps primarily to path304 '
        'completion time jumping20ms. At980ms the exact unchanged brake planner endpoint '
        'margin crosses zero while gates,remaining distance and97ms duration remain fixed. '
        'All33 saved candidates validate this predicate equivalence for304/305. Fit continuous '
        'endpoint margins from their complete observed returns. At the SAME actor,space, '
        'objective and radius1 compare the ordinary staircase-response proposal with a '
        'proposal requiring learned positive endpoint margins and substituting the implied '
        'complete-time branch in prediction. Use residual90th-percentile/max(.001mm) '
        'buffers as heuristics,not confidence bounds. Reuse all old policies without reruns. '
        'Accept only measured improvement of the SAME original merit with physical success; '
        'final all-guard adoption unchanged. At most8 further event-aware updates,with '
        'actual agreement shrinkage and no radius growth beyond1. If observed gate,duration '
        'or earlier-brake branch changes,stop this local response mechanism and audit support; '
        'do not deploy a trigger rule or freeze runtime. All15 affected routes including '
        'multicorner31 must fully execute per new policy.200s/10 new policies,not PPO, '
        'not a final comparison or generalization claim.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'selection_hold.json',dict(reason='Completion-event training mechanism; all final guards unchanged.'))
    write_json(stage/'completion_specs.json',specs)
    torch.save(md,stage/'initial_policy.pt')
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    initial=anchor;blocks=[];pool=None;new_count=0;new_candidates=[]
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(a,tag):
        nonlocal new_count
        if time.perf_counter()>deadline-15:raise TimeoutError('Completion-event joint budget')
        u=base+transform@a
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag)
        c['subspace_coordinates']=np.asarray(a).tolist()
        try:event_values(c,specs);c['event_support_valid']=True
        except (ValueError,AssertionError) as exc:c.update(event_support_valid=False,event_support_error=str(exc))
        history.append(c);new_candidates.append(c);new_count+=1
        write_json(folder/'results.json',dict(candidates=history,best=None))
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,initial['records'],folder,deadline))
        model,emodel,buffer=fit(history,initial,fresh)
        plans=[('staircase',propose_subspace(initial,transform,model,1.,normalizer,base)),
               ('event',propose(initial,transform,model,emodel,buffer,1.,normalizer,base,specs))]
        stop=False;radius=1.
        for arm,solution in plans:
            if solution is None:continue
            delta,predicted=solution[:2];c=evaluate(np.array(initial['subspace_coordinates'])+delta,arm+'_matched')
            actual=merit(initial)-merit(c);ratio=actual/predicted if predicted>1e-12 else None
            accepted=c['physical_pass'] and actual>0 and ratio is not None and ratio>=.1
            block=dict(arm=arm,anchor=initial['tag'],measured=c['tag'],radius=1.,predicted_drop=predicted,
                actual_drop=actual,ratio=ratio,accepted=accepted,merit=merit(c),
                event_support_valid=c['event_support_valid'],event_margins_mm=c.get('completion_event_margins_mm'),
                predicted_event_margins_mm=solution[2].tolist() if len(solution)>2 else None,
                original_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('COMPLETION_EVENT_MATCHED',block,flush=True)
            if arm=='event':
                if accepted:anchor=c
                if ratio is None or ratio<.25:radius*=.25
                if not c['event_support_valid']:stop=True
            write_json(stage/'learning_updates.json',blocks)
        for iteration in range(8):
            if stop or time.perf_counter()>deadline-15 or radius<.03125:break
            supported=[c for c in history if c.get('event_support_valid',True)]
            model,emodel,buffer=fit(supported,anchor,fresh)
            solution=propose(anchor,transform,model,emodel,buffer,radius,normalizer,base,specs)
            if solution is None:
                blocks.append(dict(arm='event',iteration=iteration,stop='No predicted event-feasible step at current trust radius'))
                break
            delta,predicted,events=solution
            if predicted<=1e-9:
                blocks.append(dict(arm='event',iteration=iteration,stop='No predicted original-merit descent',predicted_drop=predicted))
                break
            c=evaluate(np.array(anchor['subspace_coordinates'])+delta,f'event_trust_{iteration}')
            actual=merit(anchor)-merit(c);ratio=actual/predicted
            accepted=c['physical_pass'] and actual>0. and ratio>=.1
            block=dict(arm='event',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],radius=radius,
                predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,merit=merit(c),
                event_support_valid=c['event_support_valid'],event_margins_mm=c.get('completion_event_margins_mm'),
                predicted_event_margins_mm=events.tolist(),buffer_mm=buffer.tolist(),
                original_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('COMPLETION_EVENT_UPDATE',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(1.,radius*2.)
            write_json(stage/'learning_updates.json',blocks)
            if not c['event_support_valid'] or acceptable(c):break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,new_candidates));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);u=np.array(chosen['coefficients'])
        actor.update(coefficients=u[:18].reshape(-1,3),accepted=False,training_branch_only=True)
        actor['parent']['coefficients']=u[18:].reshape(-1,3)
        torch.save(actor,stage/'shadow_training_policy.pt')
        by_path={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by_path.get(r['path'],r)) for r in read(SOURCE/'shadow_training_result.json')['records']]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),original_all_guard_pass=bool(passing),adopted=False))
        write_json(stage/'learning_updates.json',blocks)
        receipt(stage,dict(status='completed_completion_event_joint_learning',new_policies=new_count,
            training_best=chosen['tag'],accepted_updates=sum(b.get('accepted',False) for b in blocks),
            original_guard_passing_new_policies=len(passing),research_acceptable=False,pending_candidate_audit=bool(passing),
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            ppo_updates=0,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_completion_event_joint_learning',new_policies=new_count,error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
