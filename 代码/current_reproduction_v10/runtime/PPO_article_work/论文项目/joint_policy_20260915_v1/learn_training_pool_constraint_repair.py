"""Learn a continuous-state policy correction from full-path parameter responses."""
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
from scipy.optimize import minimize
from deep_joint_credit_policy import load_credit_policy
from local_suffix_credit_policy import LocalSuffixCreditPolicy
from raw_state_joint_policy import LOCAL_INDICES
from reflection_policy import reflect_observation
from learn_prefix_preserving_suffix_credit import prefix_indices
from canonical_gate_state import checkpoint_environment_factory
from learn_nonlinear_joint_feedback import rollout
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}


def feasibility_loss(record):
    if record['success']:
        return record['complete_time_s']+.01/(max(record['corridor']['margin_lower_mm'],0.)+.01)
    # Failure gets no time/efficiency credit. Covered gates are only a feasibility
    # learning signal, always worse than a full physical/rest success here.
    geometry=record['geometry']
    remaining=1.-geometry['gates_passed']/max(geometry['gate_count'],1)
    return 100.+10.*remaining+100.*max(-record['corridor']['margin_lower_mm'],0.)


def initialize(stage,spec,scale,reference,deadline):
    torch.set_num_threads(1)
    WORK.update(stage=stage,spec=spec,scale=scale,reference=reference,
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'),deadline=deadline)


def candidate_metadata(spec,scale,u):
    candidate=copy.deepcopy(spec)
    candidate['coefficients'][-len(scale):]+=np.asarray(u).reshape(scale.shape)*scale
    candidate['learning_method']='complete_path_constraint_joint_parameter_optimization'
    return candidate


def evaluate(job):
    tag,u=job
    candidate=candidate_metadata(WORK['spec'],WORK['scale'],u)
    policy=load_credit_policy(candidate);costs=dict(actor_calls=0,physics_ticks=0)
    r,rows,actions,observations=rollout(policy,WORK['factory'],WORK['reference'],costs,WORK['deadline'])
    file=WORK['stage']/(tag+'.npz')
    np.savez_compressed(file,reference=WORK['reference'],rows=rows,actions=actions,observations=observations)
    r.update(path=40,trace=file.name,costs=costs,recorded_ticks=len(rows))
    result=dict(tag=tag,parameters=np.asarray(u).tolist(),records=[r],feasibility_loss=feasibility_loss(r))
    write_json(WORK['stage']/(tag+'.json'),result)
    return result


def build_extension(metadata,failed_observations,records):
    support=[];protected=[]
    for r in records:
        if not r['success']:continue
        with np.load(BASE/r['trace']) as z:ref,rows,obs=[z[k].copy() for k in ('reference','rows','observations')]
        support.append(obs)
        if not r['curved_path']:protected.append(obs)
        elif 'shape' in r:protected.append(obs[prefix_indices(ref,rows,obs)])
    support=np.vstack(support);protected=np.vstack(protected)
    mean,std=metadata['mean'],metadata['std']
    def normalized(x):return (x[:,LOCAL_INDICES]-mean)/std
    x=normalized(failed_observations)
    mirrored=normalized(reflect_observation(torch.tensor(failed_observations)).numpy())
    p=np.vstack([normalized(protected),normalized(reflect_observation(torch.tensor(protected)).numpy())])
    clearance=np.sqrt(((x[:,None]-p[None,:])**2).mean(-1)).min(1)
    viable=np.flatnonzero(clearance>1e-4)
    assert len(viable)>=6
    selected=[int(viable[np.argmax(clearance[viable])])]
    distance=np.full(len(x),np.inf)
    for _ in range(5):
        center=selected[-1]
        distance=np.minimum(distance,np.minimum(((x-x[center])**2).mean(-1),((mirrored-x[center])**2).mean(-1)))
        score=distance.copy();score[clearance<=1e-4]=-1.;score[selected]=-1.
        selected.append(int(np.argmax(score)))
    centers=x[selected]
    pair=np.sqrt(np.minimum(((centers[:,None]-x[None,:])**2).mean(-1),
                            ((centers[:,None]-mirrored[None,:])**2).mean(-1)))
    neighbor=np.sort(pair,axis=1)[:,min(4,len(x)-1)]
    radius=np.minimum(np.maximum(1.5*neighbor,.03),.9*clearance[selected])
    spec=copy.deepcopy(metadata)
    spec.update(centers=np.vstack([metadata['centers'],centers]),
                radius=np.r_[metadata['radius'],radius],
                coefficients=np.vstack([metadata['coefficients'],np.zeros((6,3))]),step_scale=1.)
    policy=load_credit_policy(spec);current=load_credit_policy(metadata)
    with torch.no_grad():
        a=policy.features(torch.tensor(support)).numpy()[:,-6:,:]*policy.output_scale.numpy()
        f=policy.features(torch.tensor(failed_observations)).numpy()[:,-6:,:]*policy.output_scale.numpy()
        px=policy.features(torch.tensor(protected)).numpy()[:,-6:,:]
    assert abs(px).max()==0.,abs(px).max()
    error=max(float(abs(policy.act(o,True)[0]-current.act(o,True)[0]).max()) for o in np.vstack([support,failed_observations]))
    assert error<=1e-12,error
    scale=np.minimum(np.array([.015,.10,.10])/np.maximum(abs(f).max(0),1e-12),
                     np.array([.0005,.005,.005])/np.maximum(abs(a).max(0),1e-12))
    scale=np.minimum(scale,10.)
    # Each parameter owns one center/head; all bounds are training-only.
    def matrix(features,limits):
        result=np.zeros((len(features),3,18))
        for center in range(6):
            for head in range(3):result[:,head,3*center+head]=features[:,center,head]*scale[center,head]/limits[head]
        return result.reshape(-1,18)
    trust=np.vstack([matrix(a,np.array([.001,.01,.01])),matrix(f,np.array([.03,.2,.2]))])
    return spec,scale,trust,dict(selected_training_observations=selected,radii=radius.tolist(),
        protected_state_count=len(protected),support_state_count=len(support),
        zero_update_actor_error=error,protected_new_features_maximum=float(abs(px).max()),
        old_centers=len(metadata['centers']),new_centers=len(spec['centers']),network_layers_added=0)


def main():
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_recentered_fine_trace_learning_20260919'
    full=BASE/'canonical_gate_band_recentered_training_pool_check_20260919'
    probe=BASE/'canonical_gate_band_training_pool_failure_credit_20260919'
    metadata=torch.load(source/'research_candidate.pt',map_location='cpu',weights_only=False)
    full_records=read(full/'results.json')['records']
    failed=next(r for r in full_records if r['path']==40)
    with np.load(BASE/failed['trace']) as z:reference,observations=z['reference'].copy(),z['observations'].copy()
    stage=register('canonical_gate_band_training_pool_constraint_learning_20260919',
        [__file__,BASE/'local_suffix_credit_policy.py',BASE/'deep_joint_credit_policy.py',
         BASE/'learn_nonlinear_joint_feedback.py',source/'research_candidate.pt',full/'results.json',
         full/'receipt.json',probe/'results.json',probe/'receipt.json'],
        dict(wall_s=300,workers=4,added_state_centers=6,added_network_layers=0,
             full_parameter_response_paths=36,joint_proposals=2,path=40),
        'All30 isolated20ms interventions still failed near gate3, and both earlier '
        'fast research actors also failed there. Move from isolated pulses to full '
        'feedback-policy parameter credit. Add six reflection-equivariant compact '
        'state centers selected by state-space coverage on this newly included '
        'training path, retaining the same input coordinates and existing actor '
        'layer. Set radii from state neighbors and protected-prefix clearance, '
        'never path ID, angle bins or time schedules. Zero coefficients must '
        'reproduce all saved states; new features are exactly zero on16 isolated '
        'entry prefixes and all observed straight paths. Bound action changes '
        'on every successful63-path observation and failed-path observation '
        'during learning. Measure +/-18 whole-policy parameter directions on '
        'path40, then learn two simultaneous three-head parameter proposals '
        'from those complete physical outcomes. Failures have feasibility/gate '
        'credit only, never complete-time efficiency. A selected repair still '
        'requires actual64-path validation, all shape/cruise/fair/generalization '
        'gates and does not replace retained86cd. No pulse deployment, hand '
        'speed rule, angle/trajectory template or PPO claim.')
    started=time.perf_counter();pool=None;outcomes=[]
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--active',stage.name,
        '--kind','complete_closed_loop_parameter_optimization','--next',
        '扩充训练分布的连续状态函数，测量完整参数响应并联合学习路径40约束修复。'],check=True,stdout=subprocess.DEVNULL)
    try:
        spec,scale,trust,basis_audit=build_extension(metadata,observations,full_records)
        torch.save(spec,stage/'zero_extension.pt');np.savez_compressed(stage/'parameter_space.npz',scale=scale,trust=trust)
        write_json(stage/'basis_audit.json',basis_audit)
        print('CONSTRAINT_REPAIR_BASIS',basis_audit,flush=True)
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,spec,scale,reference,started+285))
        jobs=[(f'axis{i}_{sign:+d}',sign*np.eye(18)[i]) for i in range(18) for sign in (-1,1)]
        for result in pool.imap_unordered(evaluate,jobs):
            outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            r=result['records'][0]
            print('CONSTRAINT_REPAIR_AXIS',result['tag'],r['success'],r['geometry']['gates_passed'],result['feasibility_loss'],flush=True)
        base=feasibility_loss(failed)
        gradient=np.zeros(18);curvature=np.zeros(18)
        for i in range(18):
            lo=next(c['feasibility_loss'] for c in outcomes if c['tag']==f'axis{i}_-1')
            hi=next(c['feasibility_loss'] for c in outcomes if c['tag']==f'axis{i}_+1')
            gradient[i]=(hi-lo)/2.;curvature[i]=max(hi+lo-2*base,.1)
        best_axis=min(outcomes,key=lambda c:c['feasibility_loss'])
        def fun(u):return float(gradient@u+.5*np.sum(curvature*u*u)+.01*(u@u)),gradient+curvature*u+.02*u
        solution=minimize(fun,np.asarray(best_axis['parameters']),jac=True,method='SLSQP',bounds=[(-1.,1.)]*18,
            constraints=[dict(type='ineq',fun=lambda u:np.r_[1-trust@u,1+trust@u],jac=lambda u:np.vstack([-trust,trust]))],
            options=dict(maxiter=200,ftol=1e-9))
        assert abs(trust@solution.x).max()<=1.+1e-7
        write_json(stage/'optimizer.json',dict(converged=bool(solution.success),message=solution.message,
            gradient=gradient.tolist(),diagonal_curvature=curvature.tolist(),parameters=solution.x.tolist(),
            full_support_maximum=float(abs(trust@solution.x).max()),surrogate_is_approximate=True))
        candidates=[(f'joint_{s}',s*solution.x) for s in (.5,1.)]
        for result in pool.imap_unordered(evaluate,candidates):
            outcomes.append(result);write_json(stage/'results.json',dict(candidates=outcomes,adopted=False,goal_complete=False))
            r=result['records'][0]
            print('CONSTRAINT_REPAIR_JOINT',result['tag'],r['success'],r['geometry']['gates_passed'],result['feasibility_loss'],flush=True)
        pool.close();pool.join();pool=None
        good=[c for c in outcomes if c['records'][0]['success']]
        best=min(good,key=lambda c:c['feasibility_loss']) if good else None
        result=dict(candidates=outcomes,adopted=False,goal_complete=False)
        if best:
            selected=candidate_metadata(spec,scale,best['parameters'])
            torch.save(selected,stage/(best['tag']+'.pt'));torch.save(selected,stage/'research_candidate.pt')
            result['best']=best['tag']
        write_json(stage/'results.json',result)
        receipt(stage,dict(status='completed_training_pool_constraint_learning',new_path_trials=len(outcomes),
            complete_path40_candidates=len(good),best=None if best is None else best['tag'],
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_training_pool_constraint_learning',error=str(exc),
            new_path_trials=len(outcomes),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),'--kind',
            'complete_closed_loop_parameter_optimization_finished','--next',
            '若学得修复已完整停稳则验证全部64训练路径；否则依完整参数后果继续修正表示和学习。'],check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
