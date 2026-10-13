"""Sequential measured full-policy constraint learning on an unfinished route."""
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


def ordered_progress(reference,rows,record):
    length=np.linalg.norm(np.diff(reference,axis=0),axis=1)
    edge=min(record['geometry']['gates_passed'],len(length)-1)
    direction=reference[edge+1]-reference[edge]
    fraction=np.clip((rows[-1,:2]-reference[edge])@direction/(length[edge]**2),0.,1.)
    return float(length[:edge].sum()+fraction*length[edge])


def main():
    torch.set_num_threads(1)
    basis=BASE/'canonical_gate_band_training_pool_constraint_learning_20260919'
    source=BASE/'canonical_gate_band_motion_aware_constraint_learning_20260919'
    spec=torch.load(basis/'zero_extension.pt',map_location='cpu',weights_only=False)
    with np.load(basis/'parameter_space.npz') as z:scale,trust=z['scale'].copy(),z['trust'].copy()
    current=next(c for c in read(source/'results.json')['candidates'] if c['tag']=='motion_aware_1.0')
    with np.load(source/current['records'][0]['trace']) as z:ref,base_rows=z['reference'].copy(),z['rows'].copy()
    u0=np.array(current['parameters'])
    stage=register('canonical_gate_band_iterative_motion_constraint_learning_20260919',
        [__file__,BASE/'learn_common_horizon_constraint_credit.py',BASE/'learn_training_pool_constraint_repair.py',
         basis/'zero_extension.pt',basis/'parameter_space.npz',source/'results.json',source/'receipt.json',
         source/'motion_aware_1.0.pt'],
        dict(wall_s=600,workers=4,maximum_refresh_cycles=3,parameter_axes=18,axis_step=.5,
             proposals_per_cycle=2,maximum_new_path40_trials=114,maximum_total_training_function_change=8.),
        'Motion-aware learning passed the original gate3 failure but failed at554ms '
        'after gate4. Recenter whole-policy parameter responses on each actual '
        'improving actor, retaining the SAME114 centers and original63-path '
        'function trust. Each cycle measures +/-0.5 on18 parameter axes, models '
        'observedXY/V/A through the CURRENT observed failure time, and uses '
        'both sides only if both are physically recorded through that time; '
        'otherwise use one recorded side, or freeze an unsupported coordinate. '
        'Never extend a failed trace. Apply training-only5/10/20ms kinematic '
        'forecast constraints throughout the observed prefix, retaining earlier '
        'corner protection as the frontier advances. Evaluate half/full joint '
        'updates to real full-route completion or failure. Select a next research '
        'center by full physical success first, then ordered reference progress '
        'for feasibility learning ONLY; failure has no complete-time efficiency '
        'credit. Stop this branch on no progress or three cycles, preserving '
        'all source/response/candidate traces. No forecast/barrier/pulse runtime '
        'controller, path schedule or relaxed constraints. A path40 success '
        'still needs original full64/shape/cruise/fairness/generalization gates.')
    started=time.perf_counter();pool=None;outcomes=[];iterations=[];best=None
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '在路径40真实推进后的状态上刷新完整参数信用，保留已通过区间并争取完整停稳。'],check=True,stdout=subprocess.DEVNULL)
    try:
        pool=mp.get_context('spawn').Pool(4,repair.initialize,(stage,spec,scale,ref,started+580))
        for iteration in range(3):
            if time.perf_counter()>started+450:break
            center_record=current['records'][0]
            center_progress=ordered_progress(ref,base_rows,center_record)
            jobs=[(f'iter{iteration}_axis{i}_{sign:+d}',u0+sign*.5*np.eye(18)[i]) for i in range(18) for sign in (-1,1)]
            batch=[]
            for result in pool.imap_unordered(repair.evaluate,jobs):
                batch.append(result);outcomes.append(result)
                if len(batch)%12==0:
                    write_json(stage/'results.json',dict(candidates=outcomes,iterations=iterations,adopted=False,goal_complete=False))
                    print('ITERATIVE_CONSTRAINT_RESPONSES',iteration,len(batch),'complete',sum(c['records'][0]['success'] for c in batch),flush=True)
            count=len(base_rows);base=base_rows[:,:6];columns=[];modes=[];bounds=[]
            for i in range(18):
                sides=[]
                for sign in (-1,1):
                    c=next(c for c in batch if c['tag']==f'iter{iteration}_axis{i}_{sign:+d}')
                    with np.load(stage/c['records'][0]['trace']) as z:rows=z['rows'].copy()
                    sides.append(rows[:count,:6] if len(rows)>=count else None)
                lo,hi=sides
                if lo is not None and hi is not None:
                    columns.append(hi-lo);modes.append('central_observed');bounds.append((-2.,2.))
                elif hi is not None:
                    columns.append((hi-base)/.5);modes.append('positive_observed');bounds.append((0.,2.))
                elif lo is not None:
                    columns.append((base-lo)/.5);modes.append('negative_observed');bounds.append((-2.,0.))
                else:
                    columns.append(np.zeros_like(base));modes.append('unobserved_frozen');bounds.append((0.,0.))
            jac=np.stack(columns,axis=-1)
            horizons=(0.,.005,.010,.020)
            forecast=np.concatenate([base[:,:2]+t*base[:,2:4]+.5*t*t*base[:,4:6] for t in horizons])
            fj=np.concatenate([jac[:,:2]+t*jac[:,2:4]+.5*t*t*jac[:,4:6] for t in horizons])
            def fun(delta):
                distance,normal=distances(ref,forecast+np.einsum('tcn,n->tc',fj,delta))
                excess=np.maximum((distance-1.15)/.1,0.)
                derivative=np.einsum('tc,tcn->tn',normal,fj)/.1
                return float(np.mean(excess*excess)+1e-6*(delta@delta)),2*derivative.T@excess/len(excess)+2e-6*delta
            solution=minimize(fun,np.zeros(18),jac=True,method='SLSQP',bounds=bounds,
                constraints=[dict(type='ineq',fun=lambda d:np.r_[8.-trust@(u0+d),8.+trust@(u0+d)],
                                  jac=lambda d:np.vstack([-trust,trust]))],options=dict(maxiter=240,ftol=1e-10))
            audit=dict(iteration=iteration,center_parameters=u0.tolist(),center_progress_mm=center_progress,
                center_observed_ms=count,derivative_modes=modes,converged=bool(solution.success),message=solution.message,
                initial_objective=fun(np.zeros(18))[0],predicted_objective=fun(solution.x)[0],delta=solution.x.tolist())
            write_json(stage/f'iter{iteration}_optimizer.json',audit)
            proposal_jobs=[]
            if abs(trust@(u0+solution.x)).max()<=8.+1e-6:
                proposal_jobs=[(f'iter{iteration}_joint_{s}',u0+s*solution.x) for s in (.5,1.)]
            for result in pool.imap_unordered(repair.evaluate,proposal_jobs):
                batch.append(result);outcomes.append(result)
                print('ITERATIVE_CONSTRAINT_JOINT',iteration,result['tag'],result['records'][0]['success'],
                      result['records'][0]['geometry']['gates_passed'],result['records'][0]['recorded_ticks'],flush=True)
            complete=[c for c in batch if c['records'][0]['success'] and abs(trust@np.array(c['parameters'])).max()<=8.+1e-6]
            if complete:
                best=min(complete,key=lambda c:c['feasibility_loss'])
                audit['selected']=best['tag'];audit['completed_path40']=True;iterations.append(audit)
                break
            eligible=[]
            for c in batch:
                if abs(trust@np.array(c['parameters'])).max()>8.+1e-6:continue
                with np.load(stage/c['records'][0]['trace']) as z:rows=z['rows'].copy()
                progress=ordered_progress(ref,rows,c['records'][0])
                if progress>center_progress+.01:eligible.append((progress,c,rows))
            if not eligible:
                audit['selected']=None;audit['reason']='no_observed_ordered_progress';iterations.append(audit);break
            progress,current,base_rows=max(eligible,key=lambda x:x[0])
            u0=np.asarray(current['parameters']);audit.update(selected=current['tag'],selected_progress_mm=progress,
                selected_gates=current['records'][0]['geometry']['gates_passed'],completed_path40=False)
            iterations.append(audit)
            torch.save(repair.candidate_metadata(spec,scale,u0),stage/f'iter{iteration}_research_center.pt')
            write_json(stage/'results.json',dict(candidates=outcomes,iterations=iterations,adopted=False,goal_complete=False))
            print('ITERATIVE_CONSTRAINT_CENTER',iteration,audit['selected'],progress,audit['selected_gates'],flush=True)
        pool.close();pool.join();pool=None
        result=dict(candidates=outcomes,iterations=iterations,adopted=False,goal_complete=False)
        if best:
            result['best']=best['tag'];md=repair.candidate_metadata(spec,scale,best['parameters'])
            torch.save(md,stage/(best['tag']+'.pt'));torch.save(md,stage/'research_candidate.pt')
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_iterative_motion_constraint_learning',cycles=len(iterations),
            new_path_trials=len(outcomes),complete_path40=best is not None,best=None if best is None else best['tag'],
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        write_json(stage/'results.json',dict(candidates=outcomes,iterations=iterations,adopted=False,goal_complete=False))
        receipt(stage,dict(status='failed_iterative_motion_constraint_learning',error=str(exc),new_path_trials=len(outcomes),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '完整路径40修复后复核64训练池；若仍失败，依据最新真实运动状态继续修正学习。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
