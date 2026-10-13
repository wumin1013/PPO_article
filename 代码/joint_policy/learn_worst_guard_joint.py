"""Matched summed-violation versus worst-violation complete-loop parameter learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from audit_worst_guard_credit import setup,STAGE as SOURCE
from worst_guard_response import solve,worst_merit
from completion_event_response import fit,propose,event_values
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


NAME='canonical_gate_band_worst_guard_joint_learning_20260920'


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+165.
    history,initial,fresh,transform,base,model,emodel,buffer,normalizer,specs=setup()
    assert read(SOURCE/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    data=read(SOURCE/'baseline.json');support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in data['baseline']['records'] if r['path'] in paths]
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'shadow_training_policy.pt',
        SOURCE/'shadow_raw_identity_audit.json',SOURCE/'state_joint/results.json',SOURCE/'current_joint_response.npz',
        SOURCE/'learning_updates.json',SOURCE/'worst_guard_credit_audit.json',SOURCE/'current_geometry_audit.json',
        SOURCE/'baseline.json',SOURCE/'support_audit.json',SOURCE/'completion_specs.json']+
        [BASE/f for f in ('worst_guard_response.py','audit_worst_guard_credit.py','completion_event_response.py',
            'joint_training_evaluator.py','learn_coupled_early_late.py')],
        dict(wall_s=180,maximum_new_policies=8,workers=4,parameters=54,search_dimensions=18,
             initial_radius=.5,maximum_radius=1.,l1_per_radius=12.,maximum_followup_updates=6,ppo_updates=0),
        'The measured expanded18 learner improved summed merit while the worst path204 '
        'exit-duration violation rose from3.789 to3.839 normalized units. Test scalarization '
        'masking persistent route failures. Use the SAME saved full returns,current18-space '
        'response,anchor,event margin buffers,radius0.5 andL1 budget6 for summed-violation '
        'and worst-violation proposals. Worst-first LP minimizes maximum existing LEARNING '
        'guard violation,then improves original summed merit while retaining90percent of '
        'that predicted worst-guard decrease. Actual acceptance uses max violation plus '
        '.001 times the original merit,with actual/predicted agreement>=0.1 and complete '
        'physical success. This is an explicit training objective change,not removal or '
        'rescaling of any final raw guard. Event buffers incorporate actual positive '
        'prediction errors from preceding independent proposals plus.001mm,as a heuristic. '
        'At most6 subsequent joint steps/180s/8 new complete policies; if no measured worst '
        'descent or unsupported terminal branch,stop this mechanism and inspect raw '
        'geometry/credit.15 affected routes per policy;all88 full outcome assembled only '
        'with previously proven zero support on the others. No PPO or final superiority claim.')
    folder=stage/'state_joint';folder.mkdir();write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',support);write_json(stage/'completion_specs.json',specs)
    write_json(stage/'subspace.json',dict(base=base.tolist(),transform=transform.tolist()))
    write_json(stage/'selection_hold.json',dict(reason='Worst-guard training objective comparison; original adoption unchanged.'))
    torch.save(md,stage/'initial_policy.pt');entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    new=[];blocks=[];pool=None;anchor=initial;radius=.5;minimum_buffer=buffer.copy()
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(a,tag):
        if time.perf_counter()>deadline-15:raise TimeoutError('Worst-guard joint budget')
        u=base+transform@a
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag);c['subspace_coordinates']=np.asarray(a).tolist()
        try:event_values(c,specs);c['event_support_valid']=True
        except (ValueError,AssertionError) as exc:c.update(event_support_valid=False,event_support_error=str(exc))
        history.append(c);new.append(c);write_json(folder/'results.json',dict(candidates=history,best=None))
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,initial['records'],folder,deadline))
        summed=propose(initial,transform,model,emodel,buffer,.5,normalizer,base,specs,l1_budget=6.)
        worst=solve(initial,transform,model,emodel,buffer,.5,normalizer,base,specs)
        stop=False
        for arm,solution in (('summed',summed),('worst',worst)):
            if solution is None:
                blocks.append(dict(arm=arm,feasible=False));continue
            delta,predicted,events=solution[:3];c=evaluate(np.array(initial['subspace_coordinates'])+delta,arm+'_matched')
            actual=(merit(initial)-merit(c)) if arm=='summed' else (worst_merit(initial,normalizer)-worst_merit(c,normalizer))
            ratio=actual/predicted if predicted>1e-12 else None
            accepted=c['physical_pass'] and actual>0 and ratio is not None and ratio>=.1
            block=dict(arm=arm,anchor=initial['tag'],measured=c['tag'],predicted_drop=predicted,actual_drop=actual,
                ratio=ratio,accepted=accepted,original_merit=merit(c),worst_merit=worst_merit(c,normalizer),
                worst_learning_violation=max(0.,-min(c['augmented_learning_inequalities'])),
                predicted_event_margins_mm=events.tolist(),event_margins_mm=c.get('completion_event_margins_mm'),
                original_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('WORST_GUARD_MATCHED',block,flush=True)
            if arm=='worst':
                if accepted:anchor=c
                if ratio is None or ratio<.25:radius*=.25
                elif ratio>.75 and accepted:radius=1.
                if not c['event_support_valid']:stop=True
            write_json(stage/'learning_updates.json',blocks)
        for iteration in range(6):
            if stop or time.perf_counter()>deadline-15 or radius<.03125:break
            model,emodel,buffer=fit([c for c in history if c['event_support_valid']],anchor,fresh)
            buffer=np.maximum(buffer,minimum_buffer)
            solution=solve(anchor,transform,model,emodel,buffer,radius,normalizer,base,specs)
            if solution is None:
                blocks.append(dict(arm='worst',iteration=iteration,stop='No event-feasible step'));break
            delta,predicted,events,info=solution
            if predicted<=1e-9:
                blocks.append(dict(arm='worst',iteration=iteration,stop='No predicted worst-merit descent'));break
            c=evaluate(np.array(anchor['subspace_coordinates'])+delta,f'worst_trust_{iteration}')
            actual=worst_merit(anchor,normalizer)-worst_merit(c,normalizer);ratio=actual/predicted
            accepted=c['physical_pass'] and actual>0 and ratio>=.1
            block=dict(arm='worst',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],radius=radius,
                predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,original_merit=merit(c),
                worst_merit=worst_merit(c,normalizer),worst_learning_violation=max(0.,-min(c['augmented_learning_inequalities'])),
                original_guard_min=min(c['augmented_measured_inequalities']),
                predicted_event_margins_mm=events.tolist(),event_margins_mm=c.get('completion_event_margins_mm'),**info)
            blocks.append(block);print('WORST_GUARD_UPDATE',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(1.,radius*2.)
            write_json(stage/'learning_updates.json',blocks)
            if not c['event_support_valid'] or acceptable(c):break
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
        receipt(stage,dict(status='completed_worst_guard_joint_learning',new_policies=len(new),training_best=chosen['tag'],
            accepted_updates=sum(b.get('accepted',False) for b in blocks),original_guard_passing_new_policies=len(passing),
            research_acceptable=False,pending_candidate_audit=bool(passing),shadow_sha256=sha(stage/'shadow_training_policy.pt'),
            ppo_updates=0,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_worst_guard_joint_learning',new_policies=len(new),error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
