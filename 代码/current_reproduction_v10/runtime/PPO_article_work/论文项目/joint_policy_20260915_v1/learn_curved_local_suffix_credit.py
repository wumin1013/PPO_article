"""Matched representation/credit controls using existing complete suffix outcomes."""
import copy
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from deep_joint_credit_policy import DeepJointCreditPolicy, credit_normalization
from local_suffix_credit_policy import LocalSuffixCreditPolicy, local_basis
from reflection_policy import reflect_observation
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def main():
    os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_fast_absolute_shape_joint_credit_20260919'
    previous=BASE/'canonical_gate_band_stratified_suffix_joint_credit_20260919'
    data=read(source/'results.json')
    baseline=next(c for c in data['candidates'] if c['tag']==data['best'])
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    queries=sorted(read(previous/'query_results.json')['queries'],key=lambda q:q['query'])
    stage=register('canonical_gate_band_curved_local_suffix_credit_20260919',
        [__file__,BASE/'local_suffix_credit_policy.py',BASE/'deep_joint_credit_policy.py',
         BASE/'learn_stratified_suffix_joint_credit.py',source/'research_candidate.pt',source/'results.json',
         previous/'query_results.json',previous/'action_credit.npz',previous/'receipt.json',
         BASE/'canonical_gate_band_shared_network_credit_calibration_20260919/results.json',
         BASE/'canonical_gate_band_shared_credit_shape_localization_20260919/results.json'],
        dict(wall_s=300,workers=4,reused_queries=120,reused_branches=720,new_candidates_maximum=5,
             paths=20,optimizer_steps=250,learning_rate=.003,first_scale=.125),
        'Shared linear action-credit training caused broad closed-loop violations. Complete '
        'head-amplitude calibration preserved primary conditions but did not reduce reversal. '
        'Saved actual turning shows reversal in early/late turning, nearly none in the main '
        'turn or terminal brake. Change learning locally: retain the same complete-consequence '
        'first differences and add nonnegative diagonal second differences of the measured '
        'FULL scalar outcome, plus the same saved-state action trust projection and anchor '
        'regularizer. Compare MLP quadratic credit, local Wendland state features with linear '
        'credit, and identical local features with quadratic credit at scale0.125. Local '
        'centers are the120 observed query states; support radii use state-neighbor distances, '
        'never path IDs, angle tables, time masks or desired actions. If local quadratic '
        'at0.125 completes and preserves primary conditions with actual loss improvement, '
        'test0.5/1; otherwise stop expansion. These are deterministic parameter optimizations '
        'on counterfactual data, NOT PPO or speed-label imitation. Second differences are '
        'a local regularizer, not an exact Hessian or unbiased credit proof. No adoption '
        'without all original shape, physical, cruise,64-pool/fair/generalization checks.')
    started=time.perf_counter()
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next','复用完整后缀证据，对照局部状态表示与曲率信用的完整闭环结果。'],
        check=True,stdout=subprocess.DEVNULL)
    support=np.vstack([np.load(source/r['trace'])['observations'] for r in baseline['records']])
    qobs=np.array([q['observation'] for q in queries],np.float32)
    mean,std=credit_normalization(support)
    centers,radius=local_basis(qobs,mean,std)
    gradient=np.array([q['gradient'] for q in queries],np.float32)
    curvature=np.zeros_like(gradient)
    for index,q in enumerate(queries):
        for h in range(3):
            lo,hi=q['records'][2*h:2*h+2]
            if lo['success'] and hi['success']:
                curvature[index,h]=max(0.,(hi['absolute_loss']+lo['absolute_loss']-2*q['baseline_absolute_loss'])/suffix.AMPLITUDE[h]**2)
    weights=torch.tensor([q['stratum_size'] for q in queries],dtype=torch.float32)[:,None]
    obs,anchors=torch.tensor(qobs),torch.tensor(support)
    g,h=torch.tensor(gradient),torch.tensor(curvature)
    limits=torch.tensor(suffix.LIMITS,dtype=torch.float32)
    normalization=float((abs(g)*weights*limits).sum())
    jobs,training=[],{}
    pool=None;outcomes=[]
    try:
        for arm in ('mlp_quadratic','local_linear','local_quadratic'):
            torch.manual_seed(20260919)
            p=(DeepJointCreditPolicy(parent,mean,std) if arm.startswith('mlp') else
               LocalSuffixCreditPolicy(parent,mean,std,centers,radius))
            params=list(p.residual.parameters()) if arm.startswith('mlp') else [p.coefficients]
            optimizer=torch.optim.Adam(params,lr=.003)
            history=[]
            # Cache fixed local features: this is algebra, not new physical sampling.
            if not arm.startswith('mlp'):
                fq,fa=p.features(obs).detach(),p.features(anchors).detach()
            def corrections():
                if arm.startswith('mlp'): return p.correction(obs),p.correction(anchors)
                return (fq*p.coefficients).sum(-2)*p.output_scale,(fa*p.coefficients).sum(-2)*p.output_scale
            for index in range(250):
                delta,anchor=corrections()
                linear=(weights*g*delta).sum()
                quadratic=.5*(weights*h*delta.square()).sum() if arm.endswith('quadratic') else delta.sum()*0
                loss=(linear+quadratic)/normalization+.02*(anchor/limits).square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():
                    maximum=float(abs(corrections()[1]/limits).max())
                    if maximum>1:
                        if arm.startswith('mlp'):
                            p.residual[-1].weight.div_(maximum);p.residual[-1].bias.div_(maximum)
                        else:p.coefficients.div_(maximum)
                if index in (0,249):history.append(dict(step=index,linear=float(linear.detach()),quadratic=float(quadratic.detach())))
            with torch.no_grad():
                delta,anchor=corrections()
                parity=float(abs(p.correction(reflect_observation(anchors))-anchor*torch.tensor([-1.,1.,1.])).max())
                assert parity<1e-7 and abs(anchor/limits).max()<=1.00001
                training[arm]=dict(history=history,predicted_linear_change=float((weights*g*delta).sum()),
                    predicted_quadratic_change=float(.5*(weights*h*delta.square()).sum()),reflection_error=parity,
                    support_fraction=float((abs(anchor).max(-1).values>1e-10).float().mean()))
            common=dict(parent=parent,mean=mean,std=std,step_scale=.125,
                environment_architecture='gate_band_canonical_kernel',learning_method=arm+'_complete_suffix_parameter_optimization')
            if arm.startswith('mlp'):
                md=dict(common,architecture='deep_joint_credit',residual=copy.deepcopy(p.residual.state_dict()),output_mask=np.ones(3,np.float32))
            else:
                md=dict(common,architecture='local_suffix_credit',centers=centers,radius=radius,coefficients=p.coefficients.detach().numpy().copy())
            torch.save(md,stage/f'{arm}_0.125.pt');jobs.append((arm+'_0.125',md))
        write_json(stage/'actor_optimization.json',dict(arms=training,normalization=normalization,
            nonzero_curvatures=int((curvature>0).sum()),new_physics_ticks_during_training=0))
        pool=mp.get_context('spawn').Pool(4,suffix.initialize,(stage,parent,baseline,retained,source,started+290))
        def evaluate(jobs):
            for result in pool.imap_unordered(suffix.evaluate_network,jobs):
                outcomes.append(result)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('LOCAL_SUFFIX_ACTUAL',result['tag'],result['absolute_objective'],result['summary'],result['promotion'],flush=True)
        evaluate(jobs)
        first=next(c for c in outcomes if c['tag']=='local_quadratic_0.125')
        if first['primary_preserved'] and first['absolute_objective'] < queries[0]['baseline_absolute_loss']-1e-5:
            larger=[]
            for scale in (.5,1.):
                md=copy.deepcopy(jobs[-1][1]);md['step_scale']=scale;tag=f'local_quadratic_{scale}'
                torch.save(md,stage/(tag+'.pt'));larger.append((tag,md))
            jobs+=larger;evaluate(larger)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['primary_preserved'] and c['absolute_objective'] < queries[0]['baseline_absolute_loss']-1e-5]
        best=min(good,key=lambda c:c['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_curved_local_suffix_credit_comparison',new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,best=None if best is None else best['tag'],adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_curved_local_suffix_credit_comparison',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next','检查局部表示及曲率信用对照；按真实结果修正联合学习，不采用未验收候选。'],
            check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
