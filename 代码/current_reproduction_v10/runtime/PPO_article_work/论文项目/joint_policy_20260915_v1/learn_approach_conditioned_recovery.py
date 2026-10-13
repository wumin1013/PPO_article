"""Learn later three-action compensation conditional on a measured earlier gain."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog
from consolidate_joint_teacher_policy import collect_data
from joint_teacher_state_policy import features
from observable_memory_joint_policy import dynamic_context
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from search_complete_return import read
from study_io import BASE, register, receipt, write_json


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_approach_coupled_joint_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    zero=history[0];anchor=next(c for c in history if c['tag']=='approach_joint_18')
    early=np.asarray(anchor['coefficients']);md['reused_candidates']=len(history)
    # Select a measured gain before optimizing any later correction.
    early_keys=('reverse_lateral_total_mm','reverse_lateral_significant_mm','reverse_lateral_significant_duration_s')
    for r,b in zip(anchor['records'],zero['records']):
        assert all(r['shape'][k]<=b['shape'][k]+1e-12 for k in early_keys)
        assert r['target_motion']['first_turn_inside']
        assert r['shape']['in_edge_outward_mm']<=b['shape']['in_edge_outward_mm']+1e-12
    stage=register('canonical_gate_band_approach_conditioned_recovery_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',parent_file,source/'receipt.json',source/'state_joint/results.json',
         source/'state_joint/candidate.pt',source/'joint_approach_credit_models.json',source/'even_response_credit_audit.json'],
        dict(wall_s=600,workers=4,maximum_later_axis_vectors=36,maximum_scale_replacements=36,
             maximum_joint_validations=6,context_controls=[1,3,4,6],coefficient_bounds=[-1.,1.]),
        'The central zero-centered proposal mispredicts early lateral improvement. '
        'Two-sided envelopes expose omitted offsets and have no feasible approach '
        'proposal. However an ALREADY MEASURED early memory-steering vector improves '
        'significant lateral excursion on all seven affected routes and reduces '
        'duration on14/204 without violating incoming/first-turn/lateral guards. '
        'Its later curvature/speed/exit costs remain unacceptable. Freeze that '
        'learned early vector, replay its actual changed states, and collect '
        'matched later three-action responses there, not at the old zero parent. '
        'Use the actual nonzero constraint intercept; fit shared constant/linear/'
        'bilinear/quadratic motion-context corrections and validate at most6 full '
        'joint vectors. The deployed representation, if accepted, remains a single '
        'state/history function; there is no timing or path switch. Every old guard '
        'stays referenced to the original protected parent. Verify that later '
        'support cannot alter the already improved prefix. No additional optimizer '
        'budget, physical relaxation, final acceptance or PPO-bias claim.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:
        np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.asarray([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    for c in history:
        assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
    cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[]
    try:
        checked=copy.deepcopy(md);checked['coefficients']=early.reshape(-1,3)
        training=collect_data(checked,md['parent'],anchor['records'])
        np.savez_compressed(stage/'conditioned_states.npz',**training)
        f=features(md,training['observations'],training['innovations']);context=np.zeros((len(paths),6,2));context[:,0]=1.
        support_paths=[];prefix={};support_rows=[]
        for k in range(2):
            active=np.any(f[:,k]!=0.,axis=1);support_paths.append(np.unique(training['path_ids'][active]).tolist())
        for i,p in enumerate(paths):
            ticks=[]
            for k in range(2):
                ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
                assert len(ids)<=1
                if len(ids):
                    ticks.append(int(training['decisions'][ids[0]])*20)
                    v,a=np.tanh((dynamic_context(training['observations'][ids[0]])-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                    context[i,1:,k]=[v,a,v*a,v*v,a*a]
            assert ticks and min(ticks)>=280
            prefix[p]=min(ticks)
            support_rows.append(dict(path=p,first_later_kernel_ms=prefix[p]))
        write_json(stage/'conditioned_support_audit.json',dict(records=support_rows,context=context.tolist(),
            phase_support_paths=support_paths,decoded_action_identity_decisions=len(training['decisions']),
            parent_constraints_unchanged=True,approach_coefficients_fixed=early.tolist()))
        print('CONDITIONED_RECOVERY_START',anchor['tag'],'guard',min(anchor['measured_inequalities']),
              'replay',len(training['decisions']),flush=True)
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                assert np.array_equal(u.reshape(3,6,3,3)[:,:,2],early.reshape(3,6,3,3)[:,:,2])
                offset=(u-early).reshape(3,6,3,3);active=np.any(offset[:,:,:2]!=0.,axis=(0,1,3))
                affected=sorted(set(p for k in range(2) if active[k] for p in support_paths[k]))
                tag=f'conditioned_{len(history)}';measured=pool.map(worker_evaluate,[(p,u,tag) for p in affected])
                by_path={r['path']:r for r in measured};records=[copy.deepcopy(by_path.get(r['path'],r)) for r in anchor['records']]
                for r,b in zip(records,anchor['records']):
                    if r['path'] in affected:
                        with np.load(r['trace']) as z:actual=z['rows'][:prefix[r['path']]].copy()
                        with np.load(b['trace']) as z:reference=z['rows'][:prefix[r['path']]].copy()
                        assert np.array_equal(actual,reference),('early_prefix_changed',tag,r['path'])
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=sorted(set(affected+support['phase_support_paths'][2])),
                    evaluated_paths_this_vector=affected,early_prefix_bitwise=True,
                    exact_zero_support_paths=[p for p in paths if p not in affected and p not in support['phase_support_paths'][2]])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('CONDITIONED_RECOVERY',tag,c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        previous=read(source/'joint_approach_credit_models.json');steps=np.asarray(previous['steps'])[:,:2].copy()
        c0=np.asarray(anchor['measured_inequalities']);jac=np.zeros((len(c0),3,2,3));g=np.zeros((len(paths),3,2,3));pairs=[]
        for t in range(3):
            for k in range(2):
                for h in range(3):
                    step=steps[t,k,h];delta=np.zeros((3,6,3,3));delta[t,0,k,h]=step
                    plus=evaluate(early+delta.reshape(-1));minus=evaluate(early-delta.reshape(-1))
                    size=max(np.max(abs(np.asarray(c['measured_inequalities'])-c0)) for c in (plus,minus))
                    if size>.1:
                        factor=max(.001,.1/size);step*=factor;steps[t,k,h]=step;delta*=factor
                        plus=evaluate(early+delta.reshape(-1));minus=evaluate(early-delta.reshape(-1))
                    jac[:,t,k,h]=.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities'])
                    g[:,t,k,h]=.5*(losses(plus['records'])-losses(minus['records']))
                    pairs.append(dict(time=t,kernel=k,head=h,step=step,plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
                    write_json(stage/'matched_conditioned_responses.json',dict(pairs=pairs,steps=steps.tolist()))
        owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)];models=[]
        for count in (1,3,4,6):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            scaling=np.maximum(np.maximum(abs(matrix).max(1),abs(c0)),1e-12)
            solved=linprog(gradient,A_ub=-matrix/scaling[:,None],b_ub=c0/scaling,bounds=[(-1.,1.)]*(18*count),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(context_count=count,parameters=18*count,success=bool(solved.success),status=str(solved.message),
                predicted_loss=float(anchor['score']+solved.fun) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None,
                predicted_guard_minimum=float((c0+matrix@solved.x).min()) if solved.success else None)
            models.append(row);print('CONDITIONED_CREDIT',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
        write_json(stage/'conditioned_credit_models.json',dict(models=models,steps=steps.tolist(),pairs=pairs,
            base_constraints=jac.tolist(),base_loss_gradient=g.tolist(),constraint_intercept=c0.tolist()))
        useful=[m for m in models if m['success'] and m['predicted_loss']<zero['score']-.001]
        for model in useful[:2]:
            count=model['context_count'];direction=np.zeros((3,6,3,3))
            direction[:,:count,:2]=np.asarray(model['scaled_proposal']).reshape(3,count,2,3)*steps[:,None]
            for scale in (1.,.5,.25):
                c=evaluate(early+scale*direction.reshape(-1))
                blocks.append(dict(method='measured_early_conditioned_joint_learning',context_count=count,scale=scale,
                    tag=c['tag'],score=c['score'],guard_minimum=min(c['measured_inequalities']),early_prefix_bitwise=True))
                if best()['score']<zero['score']-.001:break
            if best()['score']<zero['score']-.001:break
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_approach_conditioned_recovery',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
