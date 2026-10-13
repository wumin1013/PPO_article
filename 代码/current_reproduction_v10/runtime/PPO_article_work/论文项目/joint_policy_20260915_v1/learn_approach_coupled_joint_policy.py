"""Learn approach, pre-jump and recovery actions as one causal state function."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import linprog, minimize
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
    source=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    previous=BASE/'canonical_gate_band_bilinear_motion_context_joint_learning_20260920'
    assert not read(previous/'receipt.json')['research_acceptable']
    approach=read(source/'approach_joint_support.json')
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    old_md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    md=copy.deepcopy(old_md)
    md.update(centers=np.concatenate([old_md['centers'],np.asarray(approach['center'])[None]]),
        radii=np.r_[old_md['radii'],approach['radius']],
        per_center_mean=np.vstack([old_md['per_center_mean'],approach['mean']]),
        per_center_std=np.vstack([old_md['per_center_std'],approach['std']]),
        dynamic_context_basis='quadratic',coefficients=np.zeros((54,3)),
        learning_method='approach_coupled_complete_three_action_credit')
    stage=register('canonical_gate_band_approach_coupled_joint_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         BASE/'consolidate_joint_teacher_policy.py',parent_file,source/'receipt.json',
         source/'joint_phase_credit_models.json',source/'shape_control_timing.json',
         source/'approach_joint_support.json',source/'state_joint/candidate.pt',
         previous/'receipt.json',previous/'nonlinear_zero_extension_identity.json'],
        dict(wall_s=600,workers=4,maximum_new_axis_vectors=36,maximum_joint_validations=4,
             maximum_amplitude_optimizer_calls=42,context_controls=[1,3,4,6],
             kernels=3,coefficient_bounds=[-1.,1.],minimum_predicted_lateral_improvement_mm=1e-5),
        'Latest adjustable kernels activate at280/360ms, after wrong-way lateral '
        'motion on seven difficult training paths at140-266ms. Their causal support '
        'cannot fix that existing defect. Add an observable state center immediately '
        'before measured onset, with audited seven-route support and no runtime '
        'time/path switch. Jointly learn its instantaneous and causal-memory THREE '
        'actions with the existing pre-jump/recovery function. Reuse the measured '
        'late-kernel response model; collect only18 calibrated +/- approach axes '
        '(up to18 smaller-scale replacements if needed). Compare constant,linear, '
        'bilinear,quadratic physical-context models from those identical responses. '
        'Require predicted improvement of significant lateral excursion as well as '
        'all original guards; a late-only optimum is not counted as an approach '
        'solution. Validate joint,half and separated approach/late components in '
        'full13-path feedback. Only an actually fully guarded improvement starts '
        '42 amplitude-optimization calls. If infeasible, retain source evidence '
        'and change mechanism. Keep physics,original64,all duration/speed/cruise '
        'protections and final acceptance unchanged. Parameter learning, not PPO.')
    started=time.perf_counter();deadline=started+580;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');paths=support['selected']['paths']
    assert set(approach['paths']).issubset(paths)
    selected=[r for r in baseline if r['path'] in paths]
    with np.load(source/'replayed_states.npz') as z:training={k:z[k].copy() for k in z.files}
    f=features(md,training['observations'],training['innovations'])
    assert np.array_equal(f[:,:2],features(old_md,training['observations'],training['innovations']))
    active=np.any(f!=0.,axis=(1,2));assert np.unique(training['path_ids'][active]).tolist()==paths
    approach_indices=[np.flatnonzero((training['path_ids']==p)&np.any(f[:,2]!=0.,axis=1)).item() for p in approach['paths']]
    raw=dynamic_context(training['observations'][approach_indices]);mean=raw.mean(0);scale=np.maximum(raw.std(0),[5.,100.])
    md.update(per_center_context_mean=np.vstack([old_md['per_center_context_mean'],mean]),
              per_center_context_scale=np.vstack([old_md['per_center_context_scale'],scale]))
    context=np.zeros((len(paths),6,3));context[:,0]=1.
    for i,p in enumerate(paths):
        for k in range(3):
            ids=np.flatnonzero((training['path_ids']==p)&np.any(f[:,k]!=0.,axis=1))
            assert len(ids)<=1
            if len(ids):
                v,a=np.tanh((dynamic_context(training['observations'][ids[0]])-md['per_center_context_mean'][k])/md['per_center_context_scale'][k])
                context[i,1:,k]=[v,a,v*a,v*v,a*a]
    choice=dict(paths=paths,decisions=int(active.sum()),radii=md['radii'].tolist())
    phase_paths=[np.unique(training['path_ids'][np.any(f[:,k]!=0.,axis=1)]).tolist() for k in range(3)]
    write_json(stage/'baseline.json',data)
    write_json(stage/'support_audit.json',dict(selected=choice,phase_support_paths=phase_paths,
        physical_contexts=context.tolist(),training_only_approach_center_ms=approach['training_only_center_ms'],
        original_kernels_bitwise_preserved_states=len(active),unaffected_paths=[r['path'] for r in baseline if r['path'] not in paths]))
    np.savez_compressed(stage/'replayed_states.npz',**training)
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.asarray([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def lateral(records):return float(sum(r['shape']['reverse_lateral_significant_mm'] for r in records))
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=[0.]*162,records=selected,score=score(selected),
        measured_inequalities=constraints(selected).tolist())
    history=[zero];md['reused_candidates']=1;cache={tuple(zero['coefficients']):zero}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[]
    try:
        check=collect_data(md,md['parent'],[next(r for r in baseline if r['path']==304)])
        assert not np.any(check['labels'])
        write_json(stage/'zero_extension_identity.json',dict(decoded_action_replay_decisions=len(check['decisions']),
            checked_path=304,all_saved_kernel_states_preserved=len(active),zero_residual_preserves_original_dtype=True,
            prior_all88_runtime_identity=str(previous/'nonlinear_zero_extension_identity.json'),new_physical_runs=0))
        print('APPROACH_COUPLED_START',choice,'lateral',lateral(selected),flush=True)
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                assert parameters_in_bounds(u,-1.,1.)
                touched=np.any(u.reshape(3,6,3,3)!=0.,axis=(0,1,3))
                affected=sorted(set(p for k in range(3) if touched[k] for p in phase_paths[k]))
                tag=f'approach_joint_{len(history)}';measured=pool.map(worker_evaluate,[(p,u,tag) for p in affected])
                by_path={r['path']:r for r in measured};records=[copy.deepcopy(by_path.get(r['path'],r)) for r in selected]
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=affected,
                    exact_zero_support_paths=[p for p in paths if p not in affected])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('APPROACH_COUPLED',tag,c['score'],min(c['measured_inequalities']),lateral(records),flush=True)
            return cache[key]
        old=read(source/'joint_phase_credit_models.json');jac=np.zeros((len(zero['measured_inequalities']),3,3,3));g=np.zeros((len(paths),3,3,3));lat_g=np.zeros((len(paths),3,3,3))
        jac[:,:,:2]=old['base_constraints'];g[:,:,:2]=old['base_per_path_loss_gradient']
        steps=np.zeros((3,3,3));steps[:,:2]=old['steps'];steps[:,2]=[.003,.01,.01]
        pairs=[]
        for t in range(3):
            for h in range(3):
                step=steps[t,2,h]
                delta=np.zeros((3,6,3,3));delta[t,0,2,h]=step
                plus=evaluate(delta);minus=evaluate(-delta)
                size=max(np.max(abs(np.asarray(c['measured_inequalities'])-zero['measured_inequalities'])) for c in (plus,minus))
                if size>.1:
                    factor=max(.001,.1/size);step*=factor;steps[t,2,h]=step
                    delta*=factor;plus=evaluate(delta);minus=evaluate(-delta)
                jac[:,t,2,h]=.5*(np.asarray(plus['measured_inequalities'])-minus['measured_inequalities'])
                g[:,t,2,h]=.5*(losses(plus['records'])-losses(minus['records']))
                lat_g[:,t,2,h]=.5*np.asarray([a['shape']['reverse_lateral_significant_mm']-b['shape']['reverse_lateral_significant_mm'] for a,b in zip(plus['records'],minus['records'])])
                pairs.append(dict(time_block=t,head=h,step=step,plus=plus['tag'],minus=minus['tag'],initial_response=float(size)))
                write_json(stage/'matched_approach_responses.json',dict(pairs=pairs,steps=steps.tolist()))
        c0=np.asarray(zero['measured_inequalities']);owners=np.r_[np.repeat(np.arange(len(paths)),11),np.repeat(np.arange(len(paths)),16)]
        models=[]
        for count in (1,3,4,6):
            matrix=np.stack([jac*context[owners,None,c,:,None] for c in range(count)],axis=2).reshape(len(c0),-1)
            gradient=np.sum(np.stack([g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            lateral_gradient=np.sum(np.stack([lat_g*context[:,None,c,:,None] for c in range(count)],axis=2),axis=0).reshape(-1)
            a=np.vstack([-matrix,lateral_gradient]);b=np.r_[c0,-1e-5]
            scaling=np.maximum(np.maximum(abs(a).max(1),abs(b)),1e-12)
            solved=linprog(gradient,A_ub=a/scaling[:,None],b_ub=b/scaling,bounds=[(-1.,1.)]*(27*count),method='highs',
                options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            row=dict(context_count=count,parameters=27*count,success=bool(solved.success),status=str(solved.message),
                predicted_loss_change=float(solved.fun) if solved.success else None,
                predicted_lateral_change=float(lateral_gradient@solved.x) if solved.success else None,
                scaled_proposal=solved.x.tolist() if solved.success else None)
            models.append(row);print('APPROACH_CREDIT',{k:v for k,v in row.items() if k!='scaled_proposal'},flush=True)
        write_json(stage/'joint_approach_credit_models.json',dict(pairs=pairs,steps=steps.tolist(),models=models,
            base_constraints=jac.tolist(),base_per_path_loss_gradient=g.tolist(),base_lateral_gradient=lat_g.tolist()))
        useful=[r for r in models if r['success'] and r['predicted_loss_change']<-.001]
        if useful:
            chosen=min(useful,key=lambda r:r['parameters']);count=chosen['context_count']
            joint=np.zeros((3,6,3,3));joint[:,:count]=np.asarray(chosen['scaled_proposal']).reshape(3,count,3,3)*steps[:,None]
            pre=joint.copy();pre[:,:,:2]=0.;late=joint.copy();late[:,:,2]=0.
            arms={}
            for label,u in [('joint',joint),('joint_half',joint*.5),('approach_only',pre),('late_only',late)]:
                c=evaluate(u);arms[label]=c
                blocks.append(dict(method='matched_approach_joint_control',arm=label,tag=c['tag'],actual_change=c['score']-zero['score'],
                    lateral_change=lateral(c['records'])-lateral(selected),guard_minimum=min(c['measured_inequalities'])))
            write_json(stage/'physical_approach_interaction.json',dict(
                interaction=arms['joint']['score']-arms['approach_only']['score']-arms['late_only']['score']+zero['score'],
                components_deployed=False,arms=blocks))
            if best()['score']<zero['score']-.001:
                anchor=np.asarray(best()['coefficients']);vectors=[]
                for k in range(3):
                    for h in range(3):
                        v=np.zeros_like(joint);v[:,:,k,h]=joint[:,:,k,h];vectors.append(v.reshape(-1))
                basis=np.asarray(vectors).T;basis=basis[:,np.any(basis!=0.,axis=0)]
                solution=minimize(lambda z:evaluate(anchor+basis@z)['score'],np.zeros(basis.shape[1]),method='COBYLA',
                    bounds=[(-2.,2.)]*basis.shape[1],constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(anchor+basis@z)['measured_inequalities']))],
                    options=dict(maxiter=42,rhobeg=.5,tol=1e-6,catol=1e-8))
                blocks.append(dict(method='learned_three_phase_action_amplitudes',solver=str(solution.message)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_approach_coupled_joint_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
