"""Measure missing joint directions and compare current restricted/expanded credit."""
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
from audit_coupled_trust_model import labels
from learn_coupled_early_late import coupled_worker,worker_initialize
from learn_verified_trust_joint import positive_sum
from joint_training_evaluator import build_evaluator
from learn_full_pool_joint_credit import aggregate
from study_io import BASE,register,write_json,receipt,sha
from search_complete_return import read


SOURCE=BASE/'canonical_gate_band_completion_event_joint_learning_20260920'
FRESH=BASE/'canonical_gate_band_information_refreshed_joint_learning_20260920'
OLDER=BASE/'canonical_gate_band_coupled_early_late_joint_completion_20260920'
NAME='canonical_gate_band_complementary_joint_direction_learning_20260920'


def construct_basis(initial,specs):
    credit=read(OLDER/'measured_coupled_credit.json');steps=np.array(credit['steps'])
    old_transform=np.array(read(FRESH/'subspace.json')['transform'])
    basis=old_transform/(.5*steps[:,None])
    assert np.max(abs(basis.T@basis-np.eye(12)))<1e-10
    names=labels(initial['records']);jac=np.array(credit['jacobian'])
    channels=['path105.reverse_deg','path105.energy','path105.spatial_curvature',
        'path105.intrinsic_energy','path105.motion.after_apex_5mm_speed_mm_s',
        'path304.negative_lateral','path305.negative_lateral','path204.exit_outward',
        'path204.duration_audit.exit_time_gt005_s','path14.reverse_deg']
    rows=[jac[names.index(n)] for n in channels]
    # Historical full-dimensional responses select exploration directions only.
    # Fresh complete measurements below are required before any update.
    history=read(OLDER/'state_joint/results.json')['candidates'];supported=[];excluded=[]
    for c in history:
        try:event_values(c,specs);supported.append(c)
        except (ValueError,AssertionError) as exc:excluded.append(dict(tag=c['tag'],reason=str(exc)))
    center=next(c for c in history if c['tag']==credit['anchor'])
    x=np.array([(np.array(c['coefficients'])-center['coefficients'])/steps for c in supported])
    y=np.array([c['completion_event_margins_mm'] for c in supported]);y-=np.mean(y,axis=0)
    x-=np.mean(x,axis=0)
    event_gradient=np.linalg.lstsq(x,y,rcond=1e-8)[0]
    rows.extend(event_gradient.T)
    rows=np.array(rows);projected=rows-(rows@basis)@basis.T
    norms=np.linalg.norm(projected,axis=1);projected=projected[norms>1e-10]/norms[norms>1e-10,None]
    _,singular,vh=np.linalg.svd(projected,full_matrices=False)
    assert sum(singular>1e-8)>=6
    extra=vh[:6].T;combined=np.column_stack([basis,extra])
    assert np.max(abs(combined.T@combined-np.eye(18)))<1e-8
    return steps,combined,dict(channels=channels,historical_event_supported=len(supported),
        historical_event_excluded=excluded,complement_singular_values=singular.tolist(),
        orthogonality_error=float(np.max(abs(combined.T@combined-np.eye(18)))))


def main():
    torch.set_num_threads(1);started=time.perf_counter();deadline=started+580.
    initial=copy.deepcopy(next(c for c in read(SOURCE/'state_joint/results.json')['candidates']
        if c['tag']==read(SOURCE/'receipt.json')['training_best']))
    specs=read(SOURCE/'completion_specs.json');steps,basis,design=construct_basis(initial,specs)
    base=np.array(initial['coefficients']);transform=.5*steps[:,None]*basis;k=18
    md=torch.load(SOURCE/'shadow_training_policy.pt',map_location='cpu',weights_only=False)
    assert sha(SOURCE/'shadow_training_policy.pt')==sha(FRESH/'shadow_training_policy.pt')
    assert read(FRESH/'shadow_raw_identity_audit.json')['actual_actor_and_saved_decoded_actions_bitwise']
    data=read(SOURCE/'baseline.json');support=read(SOURCE/'support_audit.json');paths=support['evaluation_paths']
    selected=[r for r in data['baseline']['records'] if r['path'] in paths]
    normalizer=read(FRESH/'response_error_channels.json')['normalizer']
    stage=register(NAME,[__file__,SOURCE/'receipt.json',SOURCE/'outcome_audit.json',SOURCE/'state_joint/results.json',
        SOURCE/'shadow_training_policy.pt',SOURCE/'completion_specs.json',SOURCE/'baseline.json',SOURCE/'support_audit.json',
        FRESH/'shadow_raw_identity_audit.json',FRESH/'subspace.json',OLDER/'measured_coupled_credit.json',
        OLDER/'state_joint/results.json']+[BASE/f for f in ('completion_event_response.py','joint_training_evaluator.py',
            'learn_coupled_early_late.py','observable_memory_joint_policy.py')],
        dict(wall_s=600,workers=4,actor_parameters=54,original_search_dimensions=12,new_orthogonal_dimensions=6,
            current_symmetric_responses=36,matched_radius=1.,matched_l1_budget=12.,maximum_new_policies=42,
            maximum_followup_updates=4,ppo_updates=0),
        'Completion-event learning repaired304/305 time predicates but worsened105 geometry. '
        'Test a SEARCH-space restriction before declaring actor capacity or a universal credit '
        'failure. Preserve the actual54-parameter three-head early/late actor. Add6 directions '
        'orthogonal to the former12 in normalized parameter coordinates,oriented by historical '
        'full-dimensional gradients of the observed conflict channels and completion margins. '
        'Historical responses select exploration only; remeasure ALL36 +/-18 directions at '
        'the latest audited current actor. Compare restricted12 versus expanded18 proposals '
        'at this SAME center,current measured responses,original merit,event constraints, '
        'per-coordinate radius1 and total coordinateL1 budget12. No deployment basis,angle '
        'table or template. Retain exact full-loop execution on all15 affected paths. '
        'At most4 further expanded updates use measured/predicted agreement; reset center '
        'within2 units,never enlarge radius beyond1. Any new unsupported completion branch '
        'is saved and halts this event-conditioned learner rather than being counted as '
        'predicted feasible.600s/42 new policies. Keep physical limits and every original '
        'adoption guard. If no measured feasible improvement, investigate current trajectory '
        'responses or policy representation; do not run the same optimizer for longer.')
    folder=stage/'state_joint';folder.mkdir()
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    write_json(stage/'completion_specs.json',specs);write_json(stage/'direction_design.json',design)
    write_json(stage/'subspace.json',dict(base=base.tolist(),transform=transform.tolist(),
        heads_squared_norm=[float(np.sum(transform[h::3]**2)) for h in range(3)]))
    write_json(stage/'selection_hold.json',dict(reason='Joint search-space mechanism comparison; final guards unchanged.'))
    torch.save(md,stage/'initial_policy.pt')
    entry=build_evaluator(data,selected,stage,md['parent'],deadline)
    initial['tag']='reused_audited_actor';initial['subspace_coordinates']=[0.]*k;event_values(initial,specs)
    initial['event_support_valid']=True;history=[initial];anchor=initial;new_count=0;pool=None;blocks=[]
    def merit(c):return c['training_score']/normalizer+.1*positive_sum(c['augmented_learning_inequalities'])
    def acceptable(c):return c['physical_pass'] and min(c['augmented_measured_inequalities'])>=-1e-8 and c['training_score']<normalizer-1e-6
    def evaluate(a,tag):
        nonlocal new_count
        if time.perf_counter()>deadline-15:raise TimeoutError('Complementary joint budget')
        u=base+transform@a
        records=pool.map(coupled_worker,[(p,u,tag) for p in paths]);by_path={r['path']:r for r in records}
        c=entry([by_path[r['path']] for r in selected],u,tag);c['subspace_coordinates']=np.asarray(a).tolist()
        try:event_values(c,specs);c['event_support_valid']=True
        except (ValueError,AssertionError) as exc:c.update(event_support_valid=False,event_support_error=str(exc))
        history.append(c);new_count+=1
        write_json(folder/'results.json',dict(candidates=history,best=None))
        if new_count%6==0:print('COMPLEMENTARY_RESPONSE',new_count,tag,c['physical_pass'],c['event_support_valid'],flush=True)
        return c
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,initial['records'],folder,deadline))
        pairs=[]
        for j in range(k):
            d=np.eye(k)[j];pairs.append((evaluate(d,f'response_{j}_plus'),evaluate(-d,f'response_{j}_minus')))
        if not all(c['event_support_valid'] for c in history):
            raise ValueError('Saved a new event branch; audit required before fitting')
        model=np.array([np.r_[(np.array(a['augmented_learning_inequalities'])-b['augmented_learning_inequalities'])/2,
            (a['training_score']-b['training_score'])/2] for a,b in pairs])
        emodel=np.array([(np.array(a['completion_event_margins_mm'])-b['completion_event_margins_mm'])/2 for a,b in pairs])
        buffer=np.maximum(.001,np.quantile(np.abs(np.array([c['completion_event_margins_mm'] for c in history])-
            initial['completion_event_margins_mm']-np.array([c['subspace_coordinates'] for c in history])@emodel),.9,axis=0))
        np.savez_compressed(stage/'current_joint_response.npz',fresh=model,event=emodel,transform=transform,base=base)
        radius=1.;stop=False
        for arm,size in (('restricted',12),('expanded',18)):
            local=dict(initial,subspace_coordinates=[0.]*size)
            solution=propose(local,transform[:,:size],model[:size],emodel[:size],buffer,1.,normalizer,base,specs,l1_budget=12.)
            if solution is None:
                blocks.append(dict(arm=arm,feasible=False));continue
            delta,predicted,events=solution;a=np.r_[delta,np.zeros(k-size)];c=evaluate(a,arm+'_matched')
            actual=merit(initial)-merit(c);ratio=actual/predicted if predicted>1e-12 else None
            accepted=c['physical_pass'] and actual>0 and ratio is not None and ratio>=.1
            block=dict(arm=arm,anchor=initial['tag'],measured=c['tag'],radius=1.,l1_budget=12.,
                step_l1=float(abs(a).sum()),step_l2=float(np.linalg.norm(a)),predicted_drop=predicted,
                actual_drop=actual,ratio=ratio,accepted=accepted,merit=merit(c),
                event_margins_mm=c.get('completion_event_margins_mm'),predicted_event_margins_mm=events.tolist(),
                original_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('COMPLEMENTARY_MATCHED',block,flush=True)
            if arm=='expanded':
                if accepted:anchor=c
                if ratio is None or ratio<.25:radius*=.25
                if not c['event_support_valid']:stop=True
            write_json(stage/'learning_updates.json',blocks)
        for iteration in range(4):
            if stop or time.perf_counter()>deadline-15 or radius<.03125:break
            fitted,ejac,buffer=fit([c for c in history if c['event_support_valid']],anchor,model)
            solution=propose(anchor,transform,fitted,ejac,buffer,radius,normalizer,base,specs,l1_budget=12.*radius)
            if solution is None:
                blocks.append(dict(arm='expanded',iteration=iteration,stop='No event-feasible step'));break
            delta,predicted,events=solution
            if predicted<=1e-9:
                blocks.append(dict(arm='expanded',iteration=iteration,stop='No predicted original-merit descent'));break
            c=evaluate(np.array(anchor['subspace_coordinates'])+delta,f'expanded_trust_{iteration}')
            actual=merit(anchor)-merit(c);ratio=actual/predicted
            accepted=c['physical_pass'] and actual>0 and ratio>=.1
            block=dict(arm='expanded',iteration=iteration,anchor=anchor['tag'],measured=c['tag'],radius=radius,
                predicted_drop=predicted,actual_drop=actual,ratio=ratio,accepted=accepted,merit=merit(c),
                event_margins_mm=c.get('completion_event_margins_mm'),predicted_event_margins_mm=events.tolist(),
                original_guard_min=min(c['augmented_measured_inequalities']))
            blocks.append(block);print('COMPLEMENTARY_UPDATE',block,flush=True)
            if accepted:anchor=c
            if ratio<.25 or not c['physical_pass']:radius*=.25
            elif ratio>.75 and accepted:radius=min(1.,radius*2.)
            write_json(stage/'learning_updates.json',blocks)
            if not c['event_support_valid'] or acceptable(c):break
        pool.close();pool.join();pool=None
        passing=list(filter(acceptable,history[1:]));chosen=min(passing,key=lambda c:c['training_score']) if passing else anchor
        actor=copy.deepcopy(md);u=np.array(chosen['coefficients'])
        actor.update(coefficients=u[:18].reshape(-1,3),accepted=False,training_branch_only=True)
        actor['parent']['coefficients']=u[18:].reshape(-1,3);torch.save(actor,stage/'shadow_training_policy.pt')
        by_path={r['path']:r for r in chosen['records']}
        full=[copy.deepcopy(by_path.get(r['path'],r)) for r in read(SOURCE/'shadow_training_result.json')['records']]
        write_json(stage/'shadow_training_result.json',dict(tag=chosen['tag'],records=full,summary=aggregate(full),
            training_score=chosen['training_score'],merit=merit(chosen),original_all_guard_pass=bool(passing),adopted=False))
        write_json(stage/'learning_updates.json',blocks)
        receipt(stage,dict(status='completed_complementary_joint_direction_learning',new_policies=new_count,
            training_best=chosen['tag'],accepted_updates=sum(b.get('accepted',False) for b in blocks),
            original_guard_passing_new_policies=len(passing),research_acceptable=False,pending_candidate_audit=bool(passing),
            shadow_sha256=sha(stage/'shadow_training_policy.pt'),wall_s=time.perf_counter()-started,
            ppo_updates=0,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_complementary_joint_direction_learning',new_policies=new_count,error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
