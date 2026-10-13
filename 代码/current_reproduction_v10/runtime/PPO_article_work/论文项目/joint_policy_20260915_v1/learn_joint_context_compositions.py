"""Validate genuine cross-time three-action compositions, then learn their shared amplitudes."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from duration_joint_metrics import guards,penalty
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize,worker_evaluate,metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    assert not read(source/'receipt.json')['research_acceptable']
    audit=read(source/'matched_phase_credit_audit.json')
    assert all(c['prejump_coefficient_norm']==0. for c in audit['compositions'])
    md=torch.load(source/'state_joint/candidate.pt',map_location='cpu',weights_only=False)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    model=read(source/'joint_phase_credit_models.json');combined=next(m for m in model['models'] if m['context_count']==3)
    control=next(m for m in audit['matched_one_phase_controls'] if m['context_count']==3 and m['phase']=='postpeak')
    assert combined['predicted_loss_change']<control['predicted_change']-.001
    steps=np.repeat(np.asarray(model['steps'])[:,None,:,:],3,axis=1)
    joint=steps*np.asarray(combined['scaled_proposal']).reshape(3,3,2,3)
    early=joint.copy();early[:,:,1,:]=0.
    late=joint.copy();late[:,:,0,:]=0.
    post_control=np.zeros_like(joint);post_control[:,:,1,:]=np.asarray(control['scaled_parameters']).reshape(3,3,3)*steps[:,:,1,:]
    stage=register('canonical_gate_band_coupled_context_composition_learning_20260920',
        [__file__,BASE/'observable_memory_joint_policy.py',BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py',BASE/'learn_early_memory_joint_policy.py',
         BASE/'learn_shared_temporal_joint_policy.py',BASE/'learn_direct_observable_joint_policy.py',
         parent_file,source/'receipt.json',source/'matched_phase_credit_audit.json',
         source/'joint_phase_credit_models.json',source/'state_joint/results.json',source/'state_joint/candidate.pt'],
        dict(wall_s=420,workers=4,maximum_new_composition_vectors=6,maximum_optimizer_calls=48,
             optimizer_dimensions=6,policy_parameters=54,coefficient_bounds=[-1.,1.]),
        'Matched same-parent controls show that the tested18-parameter optimum uses '
        'zero pre-jump coefficients, so it is not evidence of a temporal coordination '
        'benefit. The54-parameter physical-context model predicts a better joint '
        'direction than its matched post-only optimum. Keep this exact function and '
        'all guards. Reuse all44 vectors, then execute the learned joint direction, '
        'its isolated pre/post components, the independently optimized post-only '
        'direction, and half/quarter joint scales. Isolated components are attribution '
        'controls, not deployable schedules. Quantify real full-return/shape interaction '
        'on the same parent. Only after a measured fully protected improvement, learn '
        'six phase-by-action amplitudes on the measured joint direction for48 full-loop '
        'calls, avoiding a54-dimensional simplex initialization. Final policy remains '
        'one state/history function with learned three-action coefficients. This is '
        'parameter learning, not PPO. No physical/shape bound or test split changes.')
    started=time.perf_counter();deadline=started+400;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records'];support=read(source/'support_audit.json');choice=support['selected']
    paths=choice['paths'];selected=[r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def losses(records):return np.array([teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)])
    def score(records):return float(losses(records).sum())
    def constraints(records):return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    history=copy.deepcopy(read(source/'state_joint/results.json')['candidates'])
    for c in history:
        assert abs(c['score']-score(c['records']))<1e-10
        assert np.array_equal(c['measured_inequalities'],constraints(c['records']))
        c['tag']='reused_'+c['tag']
    zero=history[0];md['reused_candidates']=len(history);cache={tuple(np.round(c['coefficients'],12)):c for c in history}
    def feasible(c):return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    blocks=[];pool=None;arms={}
    print('JOINT_CONTEXT_COMPOSITION_START',len(history),'pre_norm',float(np.linalg.norm(early)),
          'post_norm',float(np.linalg.norm(late)),flush=True)
    try:
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'context_joint_{len(history)}'
                active=paths if np.any(u.reshape(3,3,2,3)[:,:,0,:]) else [14,104,204,304]
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in active]);by_path={r['path']:r for r in measured}
                records=[copy.deepcopy(by_path.get(r['path'],r)) for r in selected]
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=active,
                    exact_zero_support_paths=[p for p in paths if p not in active])
                history.append(c);cache[key]=c;write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('JOINT_CONTEXT',tag,'paths',len(active),'score',c['score'],'guard',min(c['measured_inequalities']),flush=True)
            return cache[key]
        for label,u in [('pre_component',early),('post_component',late),('joint',joint),
                        ('post_optimum',post_control),('joint_half',joint*.5),('joint_quarter',joint*.25)]:
            c=evaluate(u);arms[label]=c
            blocks.append(dict(method='matched_joint_component_control',arm=label,tag=c['tag'],
                actual_change=c['score']-zero['score'],guard_minimum=min(c['measured_inequalities'])))
        interaction=arms['joint']['score']-arms['pre_component']['score']-arms['post_component']['score']+zero['score']
        write_json(stage/'physical_phase_interaction.json',dict(full_loss_interaction=interaction,
            per_path=[dict(path=p,interaction=float(v)) for p,v in zip(paths,
                losses(arms['joint']['records'])-losses(arms['pre_component']['records'])-
                losses(arms['post_component']['records'])+losses(zero['records']))],
            arms=[dict(arm=k,tag=v['tag'],score=v['score'],guard_minimum=min(v['measured_inequalities']),
                       complete_paths=sum(r['success'] for r in v['records'])) for k,v in arms.items()],
            same_parent_state_function=True,component_controls_deployed=False,
            interpretation='Actual complete-feedback differences, conditional on these coefficients '
            'and training paths. Not a universal causal explanation or a PPO-estimator statement.'))
        if best()['score']<zero['score']-.001:
            anchor=np.asarray(best()['coefficients']);basis=[]
            for kernel in (0,1):
                for head in range(3):
                    vector=np.zeros_like(joint);vector[:,:,kernel,head]=joint[:,:,kernel,head];basis.append(vector.reshape(-1))
            basis=np.asarray(basis).T;basis=basis[:,np.any(basis!=0.,axis=0)]
            write_json(stage/'learned_amplitude_basis.json',dict(anchor=best()['tag'],basis=basis.tolist(),
                construction='Six disjoint phase-by-action groups of the physically validated joint credit direction; no runtime switches.'))
            solution=minimize(lambda z:evaluate(anchor+basis@z)['score'],np.zeros(basis.shape[1]),method='COBYLA',
                bounds=[(-2.,2.)]*basis.shape[1],constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(anchor+basis@z)['measured_inequalities']))],
                options=dict(maxiter=48,rhobeg=.5,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='learned_shared_phase_action_amplitudes',solver=str(solution.message)))
        pool.close();pool.join();pool=None;finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_joint_context_composition_learning',error=str(exc),wall_s=time.perf_counter()-started,
            adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
