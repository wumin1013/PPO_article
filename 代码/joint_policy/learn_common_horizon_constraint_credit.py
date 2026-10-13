"""Correct stopped-prefix event aliasing using jointly predicted physical XY."""
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
import learn_training_pool_constraint_repair as repair
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def distances(reference,xy):
    start=reference[:-1];direction=np.diff(reference,axis=0)
    length2=np.sum(direction*direction,axis=1)
    rel=xy[:,None,:]-start[None,:,:]
    fraction=np.clip(np.sum(rel*direction[None,:,:],axis=2)/length2,0.,1.)
    residual=rel-fraction[:,:,None]*direction[None,:,:]
    norm=np.linalg.norm(residual,axis=2);index=norm.argmin(1)
    selected=norm[np.arange(len(xy)),index]
    gradient=residual[np.arange(len(xy)),index]/np.maximum(selected[:,None],1e-12)
    return selected,gradient


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_training_pool_constraint_learning_20260919'
    full=BASE/'canonical_gate_band_recentered_training_pool_check_20260919'
    controls=read(source/'results.json')['candidates']
    spec=torch.load(source/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(source/'parameter_space.npz') as z:scale,trust=z['scale'].copy(),z['trust'].copy()
    with np.load(full/'path40.npz') as z:ref,baseline=z['reference'].copy(),z['rows'].copy()
    count=min([len(baseline)]+[r['records'][0]['recorded_ticks'] for r in controls])
    assert count>=400
    base=baseline[:count,:2]
    jac=[];sign_audit=[]
    baseline_distance=distances(ref,base)[0]
    for i in range(18):
        low=next(c for c in controls if c['tag']==f'axis{i}_-1')
        high=next(c for c in controls if c['tag']==f'axis{i}_+1')
        with np.load(source/low['records'][0]['trace']) as z:lo=z['rows'][:count,:2].copy()
        with np.load(source/high['records'][0]['trace']) as z:hi=z['rows'][:count,:2].copy()
        jac.append((hi-lo)/2.)
        fixed_slope=(distances(ref,hi)[0][-1]-distances(ref,lo)[0][-1])/2.
        stopped_slope=(high['feasibility_loss']-low['feasibility_loss'])/2.
        sign_audit.append(dict(axis=i,fixed_horizon_endpoint_distance_slope=float(fixed_slope),
            stopped_event_loss_slope=float(stopped_slope),opposite_sign=bool(fixed_slope*stopped_slope<0)))
    jac=np.stack(jac,axis=-1)
    stage=register('canonical_gate_band_common_horizon_constraint_learning_20260919',
        [__file__,BASE/'learn_training_pool_constraint_repair.py',source/'zero_extension.pt',
         source/'parameter_space.npz',source/'results.json',source/'receipt.json',full/'path40.npz'],
        dict(wall_s=180,workers=2,reused_full_parameter_trials=38,new_path40_trials=2,
             common_observed_prefix_ms=count,trust_radii=[2.,8.]),
        'Previous38 whole-policy trials still failed at gate3. First-crossing '
        'overshoot depends on whether execution stops at451 or452ms; it is not '
        'a consistent local distance objective. Reuse all raw observed traces '
        'without extending ANY failed prefix. Sum central full-policy parameter '
        'responses of physicalXY on their shared observed time horizon, then '
        'compute Euclidean distance to the true reference segments. Quantify '
        'sign disagreement with the stopped-event loss. Learn two joint '
        'coefficient updates minimizing common-horizon corridor exposure with '
        'a1.15mm training target, while retaining the actual1.25mm execution '
        'limit. The larger training function trust radii2/8 are explicit '
        'exploration, not safety guarantees, and every candidate is executed '
        'to true full-path success or failure. Prefix geometry is only a '
        'constraint learning signal, never full time, cruise or success. '
        'No time-indexed action is deployed; only the114-center state policy '
        'coefficients change. Full64 validation remains mandatory before any '
        'claim of recovered training-pool feasibility;86cd remains retained.')
    started=time.perf_counter();pool=None;outcomes=[];jobs=[]
    write_json(stage/'event_alias_audit.json',dict(common_horizon_ms=count,axes=sign_audit,
        opposite_sign_axes=sum(r['opposite_sign'] for r in sign_audit),
        baseline_common_endpoint_distance_mm=float(baseline_distance[-1])))
    np.savez_compressed(stage/'physical_prefix_response.npz',base_xy=base,xy_jacobian=jac)
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '用实际共同时间窗的XY响应消除停止事件评分混淆，学习并执行两个完整闭环修复候选。'],check=True,stdout=subprocess.DEVNULL)
    try:
        def fun(u):
            xy=base+np.einsum('tcn,n->tc',jac,u)
            distance,dxy=distances(ref,xy)
            excess=np.maximum((distance-1.15)/.1,0.)
            dparameter=np.einsum('tc,tcn->tn',dxy,jac)/.1
            weights=np.ones(count)/count;weights[-1]+=1.
            return float(weights@(excess*excess)+1e-6*(u@u)),2*dparameter.T@(weights*excess)+2e-6*u
        audits=[]
        for radius in (2.,8.):
            solution=minimize(fun,np.zeros(18),jac=True,method='SLSQP',bounds=[(-radius,radius)]*18,
                constraints=[dict(type='ineq',fun=lambda u:np.r_[radius-trust@u,radius+trust@u],
                                  jac=lambda u:np.vstack([-trust,trust]))],
                options=dict(maxiter=240,ftol=1e-10))
            assert abs(trust@solution.x).max()<=radius+1e-6
            tag=f'common_horizon_{radius}'
            md=repair.candidate_metadata(spec,scale,solution.x)
            torch.save(md,stage/(tag+'.pt'));jobs.append((tag,solution.x.copy()))
            prediction=distances(ref,base+np.einsum('tcn,n->tc',jac,solution.x))[0]
            audits.append(dict(tag=tag,parameters=solution.x.tolist(),converged=bool(solution.success),
                message=solution.message,initial_constraint_objective=fun(np.zeros(18))[0],
                predicted_constraint_objective=fun(solution.x)[0],
                predicted_prefix_max_distance_mm=float(prediction.max()),
                maximum_training_function_change=float(abs(trust@solution.x).max())))
            print('COMMON_HORIZON_PROPOSAL',audits[-1],flush=True)
        write_json(stage/'optimizer.json',dict(candidates=audits))
        pool=mp.get_context('spawn').Pool(2,repair.initialize,(stage,spec,scale,ref,started+170))
        for result in pool.imap_unordered(repair.evaluate,jobs):
            outcomes.append(result)
            write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            r=result['records'][0]
            print('COMMON_HORIZON_ACTUAL',result['tag'],r['success'],r['recorded_ticks'],r['geometry'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['records'][0]['success']]
        best=min(good,key=lambda c:c['feasibility_loss']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];torch.save(repair.candidate_metadata(spec,scale,best['parameters']),stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_common_horizon_constraint_learning',new_candidates=len(outcomes),
            complete_path40_candidates=len(good),best=None if best is None else best['tag'],
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_common_horizon_constraint_learning',error=str(exc),new_candidates=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '按完整路径40和后续64训练池的真实结果继续学习修复，不采用失败前缀或探针。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
