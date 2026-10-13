"""Use validated local-amplitude credit in bounded complete-policy learning."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
import learn_stratified_suffix_joint_credit as suffix
from guarded_trace_optimizer import solve_guarded
from dense_trace_credit_model import coordinates,coefficients,values,objective
from learn_retained_joint_growth import growth_deficits
from learn_error_buffered_trace_credit import actual_guard_audit
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    fine=BASE/'canonical_gate_band_local_amplitude_trace_credit_20260919'
    assert read(fine/'model_validation.json')['improved'],'Quarter-amplitude model did not improve; change mechanism'
    source=BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    previous=BASE/'canonical_gate_band_per_corner_dense_credit_20260919'
    latest=BASE/'canonical_gate_band_dense_signed_trace_completion_20260919'
    model=torch.load(fine/'response_model.pt',map_location='cpu',weights_only=False)
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    metadata=torch.load(latest/'signed_trace_0.0625.pt',map_location='cpu',weights_only=False)
    retained=model['retained'];baseline=model['baseline']
    assert sha(source/'research_candidate.pt')==read(fine/'query_plan.json')['parent_sha256']
    stage=register('canonical_gate_band_local_amplitude_guarded_learning_20260919',
        [__file__,BASE/'guarded_trace_optimizer.py',BASE/'dense_trace_credit_model.py',
         BASE/'learn_error_buffered_trace_credit.py',BASE/'learn_stratified_suffix_joint_credit.py',
         fine/'response_model.pt',fine/'model_validation.json',fine/'receipt.json',
         source/'research_candidate.pt',latest/'signed_trace_0.0625.pt',latest/'results.json',previous/'results.json'],
        dict(wall_s=420,maximum_new_candidates=2,paths=20,workers=2,solver_per_arm_s=125,
             trust_scale=.125,new_response_queries=0),
        'Only proceed after the matched quarter-amplitude response improves prediction '
        'on three previously measured policies. Reuse the same parent,108-center '
        'actor and dense time credit. Compare unbuffered and empirical-error-buffered '
        '0.125 joint updates with identical objective, start and solver budget. '
        'The prior epigraph solve spent its budget without actual candidates; use '
        'direct metric constraints with generated trust-support cuts. Check EVERY '
        'saved support row, full trace guard and margin before retaining an iterate. '
        'Persist feasible iterates during optimization and preserve them on solver '
        'timeout; label nonconvergence explicitly. No constraint is accepted from '
        'a subset alone. Actual full20-path validation decides shape/physics, never '
        'the response model. No action-label fit, PPO claim, new runtime rule or '
        'final acceptance without the original64/fairness/generalization requirements.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[];audits=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '用已验证的小幅响应进行有无训练误差余量的联合参数学习，并完成真实20路径。'],check=True,stdout=subprocess.DEVNULL)
    try:
        error=np.zeros_like(model['base']);known=[];controls=[]
        for folder,tags in [(previous,['per_corner_constrained_0.0625','per_corner_constrained_0.125']),
                            (latest,['signed_trace_0.0625'])]:
            for c in read(folder/'results.json')['candidates']:
                if c['tag'] not in tags:continue
                md=torch.load(folder/(c['tag']+'.pt'),map_location='cpu',weights_only=False)
                z=coordinates(model,md);known.append(z)
                radius=float(abs(model['anchors']@z).max())
                residual=growth_deficits(c,retained)-values(model,z)[0]
                error=np.maximum(error,np.maximum(residual,0.)/radius**2)
                controls.append(dict(tag=c['tag'],radius=radius,residual=residual.tolist()))
        write_json(stage/'error_calibration.json',dict(controls=controls,envelope=error.tolist(),factor=1.25))
        initial=coordinates(model,metadata)
        for tag,buffer in [('fine_unbuffered',np.zeros_like(error)),('fine_error_buffered',1.25*.125**2*error)]:
            z,audit=solve_guarded(model,initial,.125,stage/(tag+'_solver'),min(started+290,time.perf_counter()+125),buffer)
            audit.update(tag=tag,buffer=buffer.tolist());audits.append(audit)
            write_json(stage/'optimizer.json',dict(candidates=audits))
            if z is None:continue
            if any(abs(model['anchors']@(z-other)).max()<1e-7 for other in known):
                audit['skipped_duplicate']=True;continue
            candidate=copy.deepcopy(metadata)
            candidate.update(coefficients=coefficients(model,z),step_scale=1.,
                learning_method='locally_validated_finite_amplitude_joint_parameter_optimization')
            with torch.no_grad():model['policy'].coefficients.copy_(torch.tensor(candidate['coefficients']))
            prefix=max(float(abs(model['policy'].act(x,True)[0]-model['policy'].base.act(x,True)[0]).max()) for x in model['protected'])
            assert prefix<=1e-12,prefix
            audit['maximum_protected_latent_difference']=prefix
            torch.save(candidate,stage/(tag+'.pt'));jobs.append((tag,candidate));known.append(z)
        write_json(stage/'optimizer.json',dict(candidates=audits))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,suffix.initialize,(stage,parent,baseline,retained,source,started+410))
            for result in pool.imap_unordered(suffix.evaluate_network,jobs):
                result['local_guard_audit']=actual_guard_audit(model,result) if result['summary']['successes']==20 else None
                outcomes.append(result)
                write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('LOCAL_AMPLITUDE_ACTUAL',result['tag'],result['summary'],result['local_guard_audit'],flush=True)
            pool.close();pool.join();pool=None
        good=[r for r in outcomes if r['primary_preserved'] and r['local_guard_audit']['guard_pass']
              and r['absolute_objective']<objective(model,np.zeros(model['metric_jac'].shape[1]))[0]]
        best=min(good,key=lambda r:r['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_local_amplitude_guarded_joint_learning',new_candidates=len(outcomes),
            best=None if best is None else best['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_local_amplitude_guarded_joint_learning',error=str(exc),
            new_candidates=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '核实三动作新候选实际形状和巡航，按可信闭环收益继续学习；正式保留模型不变。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
