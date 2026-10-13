"""Learn nonlinear physical-context joint feedback from complete-route responses."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import copy
import time
import multiprocessing as mp
import numpy as np
import torch
from scipy.optimize import minimize
from consolidate_joint_teacher_policy import collect_data
from duration_joint_metrics import guards, penalty
from learn_complete_joint_teachers import teacher_score, parameters_in_bounds
from learn_early_memory_joint_policy import worker_initialize, worker_evaluate, metric_functions
from learn_direct_observable_joint_policy import finish
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--basis', choices=['bilinear','quadratic'], default='bilinear')
    args=parser.parse_args(); torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_coupled_phase_context_memory_learning_20260920'
    previous=BASE/'canonical_gate_band_coupled_context_composition_learning_20260920'
    recovery=BASE/'canonical_gate_band_measured_joint_amplitude_restoration_20260920'
    assert not read(recovery/'receipt.json')['research_acceptable']
    capacity=read(source/'nonlinear_motion_context_capacity.json')
    count={'bilinear':4,'quadratic':6}[args.basis]
    model=next(r for r in capacity['models'] if r['context_count']==count)
    parent_file=BASE/'canonical_gate_band_postpeak_memory_joint_learning_20260920/research_candidate.pt'
    md=torch.load(source/'state_joint/candidate.pt', map_location='cpu', weights_only=False)
    md.update(dynamic_context_basis=args.basis, coefficients=np.zeros((6*count,3)),
              learning_method='nonlinear_motion_context_complete_joint_response_learning')
    steps=np.repeat(np.asarray(read(source/'joint_phase_credit_models.json')['steps'])[:,None],count,axis=1)
    joint=steps*np.asarray(model['scaled_proposal']).reshape(3,count,2,3)
    early=joint.copy(); early[:,:,1]=0.
    late=joint.copy(); late[:,:,0]=0.
    stage=register(f'canonical_gate_band_{args.basis}_motion_context_joint_learning_20260920',
        [__file__, BASE/'observable_memory_joint_policy.py', BASE/'joint_teacher_state_policy.py',
         BASE/'duration_joint_metrics.py', BASE/'consolidate_joint_teacher_policy.py',
         BASE/'learn_early_memory_joint_policy.py', BASE/'learn_shared_temporal_joint_policy.py',
         BASE/'learn_direct_observable_joint_policy.py', parent_file, source/'receipt.json',
         source/'state_joint/candidate.pt', source/'joint_phase_credit_models.json',
         source/'nonlinear_motion_context_capacity.json', previous/'state_joint/results.json',
         recovery/'receipt.json', recovery/'optimization_blocks.json'],
        dict(wall_s=420,workers=4,nonlinear_context_basis=args.basis,parameters=18*count,
             maximum_compositions=5,maximum_amplitude_optimizer_calls=42,
             optimizer_dimensions=6,coefficient_bounds=[-1.,1.]),
        'Measured six-amplitude correction at the actual linear-context joint point '
        'has no feasible local direction. Change the shared motion representation: '
        'use products of bounded observed speed and tangential acceleration, then '
        'store them in the existing causal kernel memories. Fit all three action '
        'coefficients jointly from the SAME full-route responses with unchanged '
        'guards. Verify zero-function identity on all88 parent paths and nonzero '
        'linear-function embedding on the13 known affected paths. Execute full, '
        'half,quarter joint directions and isolated phase components as offline '
        'controls. Only an actual protected gain starts six-amplitude optimization '
        'for at most42 calls. Otherwise stop this basis and inspect actual outcome '
        'errors. Runtime receives no time/path ID or angle table; no rule/template '
        'is learned as a label. Physics, completion, duration and shape protections '
        'are unchanged. This is complete closed-loop parameter learning, not PPO.')
    started=time.perf_counter();deadline=started+400;folder=stage/'state_joint';folder.mkdir()
    data=read(source/'baseline.json');baseline=data['baseline']['records']
    support=read(source/'support_audit.json');choice=support['selected'];paths=choice['paths']
    selected=[r for r in baseline if r['path'] in paths]
    write_json(stage/'baseline.json',data);write_json(stage/'support_audit.json',support)
    with np.load(source/'replayed_states.npz') as z:
        np.savez_compressed(stage/'replayed_states.npz',**{k:z[k] for k in z.files})
    _,old_guards=metric_functions(data,selected,stage,md['parent'],deadline)
    retained=[next(r['baseline_shape'] for r in data['retained'] if r['path']==b['path']) for b in selected]
    def score(records):
        return float(sum(teacher_score(r,b,o,barrier=False)+penalty(r) for r,b,o in zip(records,selected,retained)))
    def constraints(records):
        return np.concatenate([old_guards(records)]+[guards(r,b) for r,b in zip(records,selected)])
    zero=dict(tag='reused_identity',coefficients=np.zeros(joint.size).tolist(),records=selected,
              score=score(selected),measured_inequalities=constraints(selected).tolist())
    history=[zero];md['reused_candidates']=1;cache={tuple(zero['coefficients']):zero}
    def feasible(c):
        return parameters_in_bounds(c['coefficients'],-1.,1.) and min(c['measured_inequalities'])>=-1e-8
    def best():return min(filter(feasible,history),key=lambda c:c['score'])
    pool=None;blocks=[];arms={}
    try:
        identity=collect_data(md,md['parent'],baseline)
        assert not np.any(identity['labels'])
        old=next(c for c in read(previous/'state_joint/results.json')['candidates'] if c['tag']=='context_joint_46')
        check=copy.deepcopy(md);extended=np.zeros((3,count,2,3))
        extended[:,:3]=np.asarray(old['coefficients']).reshape(3,3,2,3)
        check['coefficients']=extended.reshape(-1,3)
        check_data=collect_data(check,md['parent'],old['records'])
        old_embedded=copy.deepcopy(old);old_embedded.update(tag='verified_linear_joint',coefficients=extended.reshape(-1).tolist())
        history.append(old_embedded);cache[tuple(np.round(extended.reshape(-1),12))]=old_embedded;md['reused_candidates']=2
        write_json(stage/'nonlinear_zero_extension_identity.json',dict(parent_paths=len(baseline),
            parent_decisions=len(identity['decisions']),nonzero_paths=len(old['records']),
            nonzero_decisions=len(check_data['decisions']),decoded_actions_bitwise=True,new_physical_runs=0))
        print('NONLINEAR_CONTEXT_START',args.basis,count,'identity',len(identity['decisions']),
              'linear',len(check_data['decisions']),'predicted',model['predicted_loss_change'],flush=True)
        pool=mp.get_context('spawn').Pool(4,worker_initialize,(md,selected,folder,deadline))
        def evaluate(u):
            u=np.asarray(u).reshape(-1);key=tuple(np.round(u,12))
            if key not in cache:
                tag=f'nonlinear_context_{len(history)}'
                active=paths if np.any(u.reshape(3,count,2,3)[:,:,0]) else support['phase_support_paths'][1]
                measured=pool.map(worker_evaluate,[(p,u,tag) for p in active]);by_path={r['path']:r for r in measured}
                records=[copy.deepcopy(by_path.get(r['path'],r)) for r in selected]
                c=dict(tag=tag,coefficients=u.tolist(),records=records,score=score(records),
                    measured_inequalities=constraints(records).tolist(),actual_rerun_paths=active,
                    exact_zero_support_paths=[p for p in paths if p not in active])
                history.append(c);cache[key]=c
                write_json(folder/'results.json',dict(candidates=history,best=best()['tag']))
                print('NONLINEAR_CONTEXT',tag,c['score'],min(c['measured_inequalities']),flush=True)
            return cache[key]
        for label,u in [('joint',joint),('joint_half',joint*.5),('joint_quarter',joint*.25),
                        ('pre_component',early),('post_component',late)]:
            c=evaluate(u);arms[label]=c
            blocks.append(dict(method='nonlinear_joint_context_validation',arm=label,tag=c['tag'],
                actual_change=c['score']-zero['score'],guard_minimum=min(c['measured_inequalities'])))
        write_json(stage/'physical_phase_interaction.json',dict(
            interaction=arms['joint']['score']-arms['pre_component']['score']-arms['post_component']['score']+zero['score'],
            arms=[dict(arm=k,tag=v['tag'],score=v['score'],minimum=min(v['measured_inequalities'])) for k,v in arms.items()],
            same_parent=True,components_deployed=False))
        if best()['score']<zero['score']-.001:
            anchor=np.asarray(best()['coefficients']);vectors=[]
            for kernel in (0,1):
                for head in range(3):
                    v=np.zeros_like(joint);v[:,:,kernel,head]=joint[:,:,kernel,head];vectors.append(v.reshape(-1))
            basis=np.asarray(vectors).T
            solution=minimize(lambda z:evaluate(anchor+basis@z)['score'],np.zeros(6),method='COBYLA',
                bounds=[(-2.,2.)]*6,constraints=[dict(type='ineq',fun=lambda z:np.asarray(evaluate(anchor+basis@z)['measured_inequalities']))],
                options=dict(maxiter=42,rhobeg=.5,tol=1e-6,catol=1e-8))
            blocks.append(dict(method='learned_nonlinear_context_amplitudes',solver=str(solution.message)))
        pool.close();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started)
    except TimeoutError:
        if pool is not None:pool.terminate();pool.join();pool=None
        finish(stage,md,baseline,data,best(),zero,history,blocks,choice,parent_file,started,budget_exhausted=True)
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_nonlinear_context_joint_learning',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
