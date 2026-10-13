"""Validate measured full-policy secants before larger bounded joint updates."""
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
from guarded_trace_optimizer import solve_guarded,guard_indices
from dense_trace_credit_model import coordinates,coefficients,values,objective
from learn_trace_resolved_joint_credit import signed_channels
from learn_retained_joint_growth import growth_deficits
from learn_error_buffered_trace_credit import actual_guard_audit
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def calibrated_model(original,controls):
    model=copy.deepcopy(original)
    z=np.asarray([coordinates(original,md) for _,_,md in controls])
    a=original['anchors'];g=a.T@a/len(a)
    gram=z@g@z.T
    ridge=.05*float(np.linalg.eigvalsh(gram).max())
    correction_map=np.linalg.solve(gram+ridge*np.eye(len(controls)),z@g)
    residual=np.asarray([growth_deficits(c,original['retained'])-original['base']-
                         original['metric_jac']@q for (_,c,_),q in zip(controls,z)])
    model['metric_jac']+=residual.T@correction_map
    for path,m in model['models'].items():
        original_path=original['models'][path]
        residuals={k:[] for k in m['jac']}
        for (folder,c,_),q in zip(controls,z):
            r=next(r for r in c['records'] if r['path']==path)
            assert r['success']
            with np.load(folder/r['trace']) as data:ref,rows=data['reference'].copy(),data['rows'].copy()
            actual=signed_channels(ref,rows,m['base']['grid'],np.arange(len(m['base']['incoming'])))
            for k in residuals:
                residuals[k].append(actual[k]-original_path['base'][k]-original_path['jac'][k]@q)
        for k in residuals:m['jac'][k]+=np.asarray(residuals[k]).T@correction_map
    return model,dict(ridge=ridge,gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
        calibration_controls=[c['tag'] for _,c,_ in controls],margin_response_unchanged=True)


def main():
    torch.set_num_threads(1)
    fine=BASE/'canonical_gate_band_local_amplitude_trace_credit_20260919'
    source=BASE/'canonical_gate_band_dense_margin_suffix_learning_20260919'
    latest=BASE/'canonical_gate_band_local_amplitude_guarded_learning_20260919'
    model=torch.load(fine/'response_model.pt',map_location='cpu',weights_only=False)
    controls=[]
    for name,tags in [('canonical_gate_band_per_corner_dense_credit_20260919',['per_corner_constrained_0.0625','per_corner_constrained_0.125']),
                      ('canonical_gate_band_dense_signed_trace_completion_20260919',['signed_trace_0.0625']),
                      (latest.name,['fine_unbuffered'])]:
        folder=BASE/name
        for tag in tags:
            c=next(c for c in read(folder/'results.json')['candidates'] if c['tag']==tag)
            md=torch.load(folder/(tag+'.pt'),map_location='cpu',weights_only=False)
            controls.append((folder,c,md))
    parent=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    metadata=controls[-1][2];baseline=model['baseline'];retained=model['retained']
    stage=register('canonical_gate_band_secant_calibrated_joint_learning_20260919',
        [__file__,BASE/'guarded_trace_optimizer.py',BASE/'dense_trace_credit_model.py',
         BASE/'learn_trace_resolved_joint_credit.py',BASE/'learn_error_buffered_trace_credit.py',
         fine/'response_model.pt',fine/'model_validation.json',latest/'receipt.json']+
        [folder/(c['tag']+'.pt') for folder,c,_ in controls]+[folder/'results.json' for folder,_,_ in controls],
        dict(wall_s=420,maximum_new_candidates=2,paths=20,workers=2,
             trust_scales=[.25,.5],solver_per_arm_s=120,new_queries=0),
        'Fine-amplitude joint learning completed20/20 and reduced reversal to18.8837deg '
        'with3.697s curved99 cruise, but small actual local guard errors remain. Before '
        'larger joint updates, fit a damped low-rank full-policy secant correction to '
        'signed traces and scalar responses using THREE already measured actors; '
        'predict the FOURTH fine_unbuffered actor without fitting it. Accept this '
        'model correction only if normalized shape error decreases at least20percent '
        'and the155deg peak error does not increase. Otherwise use the validated '
        'fine-amplitude model unchanged. If accepted, refit on all four controls. '
        'The complete-decision corridor model remains unchanged. Compare two bounded '
        'joint policy radii0.25/0.5; both require complete physical20-path trials, '
        'prefix identity, original physical constraints and full metric reporting. '
        'These explore the validated local model outside prior radii; no reliability '
        'claim until actual trials. Research selection may retain a primary-feasible '
        'shape-improving branch while explicitly recording every remaining per-corner '
        'defect; it is not adoption. Retain86cd, S2, all final64/fair/generalization '
        'requirements, and do not open sealed tests or call this PPO.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[];audits=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '先用未参与拟合的已测联合候选核实割线校准，再验证两个较大联合更新。'],check=True,stdout=subprocess.DEVNULL)
    try:
        fitted,fit_audit=calibrated_model(model,controls[:-1])
        held=controls[-1];z=coordinates(model,held[2]);actual=growth_deficits(held[1],retained)
        before=values(model,z)[0];after=values(fitted,z)[0];idx=guard_indices(model)
        old_error=float(np.square(before[idx]-actual[idx]).sum())
        new_error=float(np.square(after[idx]-actual[idx]).sum())
        peak_index=9*14+5
        accepted=bool(new_error<.8*old_error and abs(after[peak_index]-actual[peak_index])<=abs(before[peak_index]-actual[peak_index]))
        validation=dict(fit=fit_audit,held_out_actor=held[1]['tag'],before_error=old_error,
            after_error=new_error,before_peak_error=float(before[peak_index]-actual[peak_index]),
            after_peak_error=float(after[peak_index]-actual[peak_index]),accepted=accepted)
        write_json(stage/'model_validation.json',validation)
        print('SECANT_VALIDATION',validation,flush=True)
        if accepted:
            model,fit_audit=calibrated_model(model,controls)
            write_json(stage/'final_model_fit.json',fit_audit)
        torch.save(model,stage/'response_model.pt')
        initial=coordinates(model,metadata);known=[coordinates(model,md) for _,_,md in controls]
        for scale in (.25,.5):
            tag=f'joint_radius_{scale}'
            z,audit=solve_guarded(model,initial,scale,stage/(tag+'_solver'),min(started+295,time.perf_counter()+120))
            audit.update(tag=tag);audits.append(audit)
            write_json(stage/'optimizer.json',dict(candidates=audits))
            if z is None:continue
            if any(abs(model['anchors']@(z-q)).max()<1e-7 for q in known):
                audit['skipped_duplicate']=True;continue
            candidate=copy.deepcopy(metadata)
            candidate.update(coefficients=coefficients(model,z),step_scale=1.,
                learning_method='validated_secant_joint_parameter_optimization' if accepted else 'fine_amplitude_joint_parameter_optimization')
            with torch.no_grad():model['policy'].coefficients.copy_(torch.tensor(candidate['coefficients']))
            error=max(float(abs(model['policy'].act(x,True)[0]-model['policy'].base.act(x,True)[0]).max()) for x in model['protected'])
            assert error<=1e-12,error
            audit['maximum_protected_latent_difference']=error
            torch.save(candidate,stage/(tag+'.pt'));jobs.append((tag,candidate));known.append(z)
        write_json(stage/'optimizer.json',dict(candidates=audits))
        if jobs:
            pool=mp.get_context('spawn').Pool(2,suffix.initialize,(stage,parent,baseline,retained,source,started+410))
            for result in pool.imap_unordered(suffix.evaluate_network,jobs):
                result['local_guard_audit']=actual_guard_audit(model,result) if result['summary']['successes']==20 else None
                outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
                print('SECANT_ACTUAL',result['tag'],result['summary'],result['local_guard_audit'],flush=True)
            pool.close();pool.join();pool=None
        comparison=controls[-1][1]
        good=[r for r in outcomes if r['primary_preserved'] and r['absolute_objective']<comparison['absolute_objective']
              and r['summary']['reverse_sum_deg']<comparison['summary']['reverse_sum_deg']
              and r['summary']['exit_sum_mm']<=comparison['summary']['exit_sum_mm']
              and r['summary']['energy_sum']<=baseline['summary']['energy_sum']]
        best=min(good,key=lambda r:r['absolute_objective']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(next(md for tag,md in jobs if tag==best['tag']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_secant_calibrated_joint_learning',new_candidates=len(outcomes),
            best=None if best is None else best['tag'],secant_correction_used=accepted,
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_secant_calibrated_joint_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '按真实形状和完整路径结果选择研究分支，继续更新当前闭环响应及三动作参数。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
