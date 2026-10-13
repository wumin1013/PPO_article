"""Train on physical motion toward the corridor, not only prefix position."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from learn_common_horizon_constraint_credit import distances
import learn_training_pool_constraint_repair as repair
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_training_pool_constraint_learning_20260919'
    position=BASE/'canonical_gate_band_common_horizon_constraint_learning_20260919'
    full=BASE/'canonical_gate_band_recentered_training_pool_check_20260919'
    spec=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:scale,trust=z['scale'].copy(),z['trust'].copy()
    with np.load(full/'path40.npz') as z:ref,baseline=z['reference'].copy(),z['rows'].copy()
    controls=read(source/'results.json')['candidates']
    count=min([len(baseline)]+[c['records'][0]['recorded_ticks'] for c in controls])
    base=baseline[:count,:6];columns=[]
    for i in range(18):
        pair=[]
        for sign in (-1,1):
            c=next(c for c in controls if c['tag']==f'axis{i}_{sign:+d}')
            with np.load(source/c['records'][0]['trace']) as z:pair.append(z['rows'][:count,:6].copy())
        columns.append((pair[1]-pair[0])/2.)
    jac=np.stack(columns,axis=-1)
    horizons=np.array([0.,.005,.010,.020])
    tail=slice(count-80,count)
    forecast=np.concatenate([base[tail,:2]+t*base[tail,2:4]+.5*t*t*base[tail,4:6] for t in horizons])
    fj=np.concatenate([jac[tail,:2]+t*jac[tail,2:4]+.5*t*t*jac[tail,4:6] for t in horizons])
    # Validate the kinematic approximation ONLY where future ground truth exists.
    calibration=[]
    for ticks in (5,10,20):
        t=ticks*.001
        indices=np.arange(250,count-ticks)
        prediction=baseline[indices,:2]+t*baseline[indices,2:4]+.5*t*t*baseline[indices,4:6]
        error=np.linalg.norm(prediction-baseline[indices+ticks,:2],axis=1)
        calibration.append(dict(horizon_ms=ticks,median_xy_error_mm=float(np.median(error)),
                                maximum_xy_error_mm=float(error.max())))
    stage=register('canonical_gate_band_motion_aware_constraint_learning_20260919',
        [__file__,BASE/'learn_common_horizon_constraint_credit.py',BASE/'learn_training_pool_constraint_repair.py',
         source/'zero_extension.pt',source/'parameter_space.npz',source/'results.json',source/'receipt.json',
         position/'terminal_motion_audit.json',position/'receipt.json',full/'path40.npz'],
        dict(wall_s=180,workers=2,reused_parameter_trials=38,new_path_trials=2,
             observed_prefix_ms=count,training_forecast_horizons_ms=[0,5,10,20],trust_radius=8.),
        'Position-only common-horizon learning correctly predicted about1.15mm '
        'at451ms, but measured distance velocity remained23.3mm/s toward the '
        'boundary and both candidates failed at455ms. Reuse the same full-policy '
        'parameter response traces and114-center actor; include observed velocity '
        'and acceleration responses in a training-only short kinematic forecast '
        'over5/10/20ms. Report forecast error where future recorded ground truth '
        'exists; beyond the observed prefix it is an approximate constraint '
        'surrogate, not a simulated/certified trajectory. No failed physical '
        'prefix is extended as actual data and no efficiency is assigned to '
        'one. Minimize predicted reference-corridor exposure at the same1.15mm '
        'training target, compare half/full learned parameter updates, then '
        'run each actor through genuine complete path40 or failure. Execution '
        'remains1.25mm,20/1ms,V/A/J and full braking/rest. No forecast, barrier '
        'controller, speed limit or time switch is deployed. Full64 and all '
        'shape/fair/generalization checks still decide acceptance.')
    started=time.perf_counter();pool=None;outcomes=[]
    write_json(stage/'forecast_calibration.json',dict(within_observed_trajectory=calibration,
        unobserved_forecast_is_not_actual_data=True))
    print('MOTION_FORECAST_CALIBRATION',calibration,flush=True)
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '联合学习位置、速度、加速度对未来约束的影响，并用实际完整路径检验。'],check=True,stdout=subprocess.DEVNULL)
    try:
        def fun(u):
            xy=forecast+np.einsum('tcn,n->tc',fj,u)
            distance,dxy=distances(ref,xy)
            excess=np.maximum((distance-1.15)/.1,0.)
            derivative=np.einsum('tc,tcn->tn',dxy,fj)/.1
            return float(np.mean(excess*excess)+1e-6*(u@u)),2*derivative.T@excess/len(excess)+2e-6*u
        solution=minimize(fun,np.zeros(18),jac=True,method='SLSQP',bounds=[(-8.,8.)]*18,
            constraints=[dict(type='ineq',fun=lambda u:np.r_[8.-trust@u,8.+trust@u],
                              jac=lambda u:np.vstack([-trust,trust]))],options=dict(maxiter=300,ftol=1e-10))
        assert abs(trust@solution.x).max()<=8.+1e-6
        write_json(stage/'optimizer.json',dict(converged=bool(solution.success),message=solution.message,
            initial_objective=fun(np.zeros(18))[0],predicted_objective=fun(solution.x)[0],
            parameters=solution.x.tolist(),maximum_training_function_change=float(abs(trust@solution.x).max())))
        print('MOTION_AWARE_PROPOSAL',fun(np.zeros(18))[0],fun(solution.x)[0],solution.message,flush=True)
        jobs=[(f'motion_aware_{s}',s*solution.x) for s in (.5,1.)]
        for tag,u in jobs:torch.save(repair.candidate_metadata(spec,scale,u),stage/(tag+'.pt'))
        pool=mp.get_context('spawn').Pool(2,repair.initialize,(stage,spec,scale,ref,started+170))
        for result in pool.imap_unordered(repair.evaluate,jobs):
            outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            r=result['records'][0]
            print('MOTION_AWARE_ACTUAL',result['tag'],r['success'],r['recorded_ticks'],r['geometry'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['records'][0]['success']]
        best=min(good,key=lambda c:c['feasibility_loss']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(repair.candidate_metadata(spec,scale,best['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_motion_aware_constraint_learning',new_candidates=len(outcomes),
            complete_path40_candidates=len(good),best=None if best is None else best['tag'],
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_motion_aware_constraint_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '按真实完整路径40结果继续修复并复核64训练池，保留全部失败证据与原模型。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
