"""Offline joint-action teachers optimized against actual complete trajectories."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from deep_joint_credit_policy import load_credit_policy
from canonical_gate_state import checkpoint_environment_factory
from analyze_nonlinear_joint_response import measure
from probe_coordinated_suffix import reconstruct
from decision_innovation_policy import innovation_features
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read
from learn_full_pool_compensation import progress


SCALE=np.array([.12,.25,.25])
WORK={}


def parameters_in_bounds(parameters,lower,upper,tolerance=1e-8):
    values=np.asarray(parameters,dtype=float)
    return bool(np.all(np.isfinite(values)) and np.all(values>=lower-tolerance)
                and np.all(values<=upper+tolerance))


def teacher_score(r,baseline,retained,*,barrier=True):
    if not r['success']:return 1e6
    s,b,o=r['shape'],baseline['shape'],retained
    if barrier and (not r['target_motion']['first_turn_inside'] or s['in_edge_outward_mm']>1e-6):return 1e5
    # Shape goals are evaluated on actual full XY, without prescribing a curve.
    loss=(s['full_physical_reverse_deg']/.5)**2+(s['out_edge_outward_mm']/.01)**2
    loss+=(s['full_normalized_energy']/.1)**2
    if 'reverse_lateral_total_mm' in b:
        loss+=(s['reverse_lateral_total_mm']/.005)**2
        loss+=(s['reverse_lateral_significant_duration_s']/.05)**2
    loss+=max((s['full_spatial_peak_curvature_mm_inv']-o['full_spatial_peak_curvature_mm_inv'])/.2,0)**2
    loss+=max((o['apex_speed_mm_s']-s['apex_speed_mm_s'])/2,0)**2
    regret=[max((s[k]-max(b[k],o[k]))/scale,0.) for k,scale in (
        ('out_edge_outward_mm',.005),('full_physical_reverse_deg',.5),
        ('full_normalized_energy',.02),('full_spatial_peak_curvature_mm_inv',.2))]
    regret += [max((min(b['apex_speed_mm_s'],o['apex_speed_mm_s'])-s['apex_speed_mm_s'])/2,0.),
               max((r['complete_time_s']-baseline['complete_time_s']-.01)/.02,0.),
               max((baseline['motion']['cruise_ge99_s']-.02-r['motion']['cruise_ge99_s'])/.05,0.)]
    return float(loss+10*np.square(regret).sum())


def initialize(stage,parent,deadline,method='cem',temporal_scope='apex',reference_envelope=False,protect_energy=False,
               expert_state=None,protect_exit=False):
    torch.set_num_threads(1)
    WORK.update(stage=stage,policy=load_credit_policy(parent),deadline=deadline,method=method,
                temporal_scope=temporal_scope,
                reference_envelope=reference_envelope,
                protect_energy=protect_energy,
                expert=None,protect_exit=protect_exit,
                factory=checkpoint_environment_factory('gate_band_canonical_kernel'))
    if expert_state is not None:
        from canonical_gate_state import CanonicalGatePolicy
        WORK['expert']=CanonicalGatePolicy(256)
        WORK['expert'].load_state_dict(expert_state)


def optimize_path(job):
    path,baseline,retained=job[:3]
    rng=np.random.default_rng(202609194+path)
    with np.load(baseline['trace']) as z:reference,nominal,nominal_actions=[z[k].copy() for k in ('reference','rows','actions')]
    speed=np.linalg.norm(nominal[:,2:4],axis=1)
    apex=int(np.linalg.norm(nominal[:,:2]-reference[1],axis=1).argmin())
    start=max(0,(apex//20-6)*20)
    if WORK['temporal_scope']=='full_window':start=0
    stop=min((len(nominal_actions)-1)*20,(apex//20+12)*20)
    if WORK['temporal_scope'] in ('observed_approach','visible_approach','event_resolved','causal_blocks'):
        from joint_anticipation_features import anticipation_features
        with np.load(baseline['trace']) as z:observed=z['observations'].copy()
        visible=np.flatnonzero([max(anticipation_features(x,np.zeros(3))[:2])>1e-6 for x in observed])
        if len(visible):start=max(0,(int(visible[0])-(WORK['temporal_scope']=='observed_approach'))*20)
    if WORK['temporal_scope']=='target_jump_window':
        from localize_joint_shape_residual import characterize
        jump=characterize(baseline['trace'])['target_jumps'][0]['time_ms']
        start=max(0,(jump//20-2)*20)
        stop=min((len(nominal_actions)-1)*20,start+80)
    if WORK['temporal_scope']=='causal_blocks':
        from localize_joint_shape_residual import characterize
        localized=characterize(baseline['trace'])
        jump=localized['target_jumps'][0]['time_ms']//20*20
        recovery=max((r for r in localized['negative_turn_intervals'] if r['start_ms']>apex),
                     key=lambda r:r['reverse_deg'])
        post_start=recovery['start_ms']//20*20
        block_frames=[list(range(post_start,post_start+80,20)),
                      list(range(jump-40,jump+40,20)),list(range(start,start+80,20))]
        active_frames=[f for block in block_frames for f in block]
        assert len(set(active_frames))==12 and min(active_frames)>=0
        assert max(active_frames)<len(nominal_actions)*20
        start,stop=min(active_frames),max(active_frames)
    costs=dict(saved_prefix_geometry_ticks=0,actor_calls=0,physics_ticks=0)
    restored=reconstruct(reference,nominal,nominal_actions,start,costs,env_factory=WORK['factory'])
    initial=restored.state();policy=WORK['policy']
    if hasattr(policy,'bind'):
        policy.bind(restored)
        action=policy.act(restored.observation(),True)[0]
        assert np.array_equal(restored.decode_action(action),nominal_actions[start//20])
        _,_,_,check=restored.step(action)
        assert np.array_equal(check['rows'],nominal[start:start+20])
        costs['actor_calls']+=1;costs['physics_ticks']+=len(check['rows'])
    count=(stop-start)//20+1
    knots=np.linspace(0,1,6)
    # Four learned internal temporal coefficients per head; fixed zero endpoints.
    basis=CubicSpline(knots,np.vstack([np.zeros((1,4)),np.eye(4),np.zeros((1,4))]),bc_type='clamped')(np.linspace(0,1,count))
    if WORK['temporal_scope'] in ('full_window','target_jump_window','observed_approach','visible_approach'):
        # Same twelve parameters; include startup credit and learn the first action.
        basis=CubicSpline(np.linspace(0,1,5),np.vstack([np.eye(4),np.zeros((1,4))]),
                          bc_type='clamped')(np.linspace(0,1,count))
    if WORK['temporal_scope']=='event_resolved':
        from localize_joint_shape_residual import characterize
        localized=characterize(baseline['trace'])
        jump=localized['target_jumps'][0]['time_ms']//20*20
        recovery=max((r for r in localized['negative_turn_intervals'] if r['start_ms']>apex),
                     key=lambda r:r['reverse_deg'])
        knots=np.array([start,jump-20,jump,(apex//20+1)*20,
                        (recovery['stop_ms']//20+1)*20,stop],dtype=float)
        assert np.all(np.diff(knots)>0),knots
        basis=CubicSpline(knots,np.vstack([np.zeros((1,4)),np.eye(4),np.zeros((1,4))]),
                          bc_type='clamped')(np.arange(start,stop+1,20))
    if WORK['temporal_scope']=='causal_blocks':
        basis=np.zeros((count,12))
        for column,frame in enumerate(active_frames):basis[(frame-start)//20,column]=1.
        knots=np.asarray(active_frames)
    dimensions=3*basis.shape[1]
    folder=WORK['stage']/f'path{path}';folder.mkdir()
    np.savez_compressed(folder/'temporal_basis.npz',basis=basis,start_tick=start,stop_tick=stop,knots=knots)
    history=[];generations=[]
    def evaluate(parameters,tag,identity=False):
        env=WORK['factory'](reference);env.restore(initial)
        if hasattr(policy,'bind'):policy.bind(env)
        chunks=[];actions=[];observations=[];innovations=[];labels=[];parent_latents=[];latent_actions=[]
        while not env.done:
            if time.perf_counter()>WORK['deadline']:raise TimeoutError('Joint teacher wall budget')
            x=env.observation();latent=policy.act(x,True)[0]
            _,features=innovation_features(env,latent)
            index=(env.ticks-start)//20
            weights=basis[index]@parameters.reshape(-1,3) if 0<=index<count else np.zeros(3)
            direction=SCALE if WORK['expert'] is None else WORK['expert'].act(x,True)[0]-latent
            offset=weights*direction
            action=latent if identity else (latent+offset).astype(latent.dtype)
            _,_,_,info=env.step(action)
            chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x)
            innovations.append(features);labels.append(action-latent);parent_latents.append(latent)
            latent_actions.append(action)
            costs['actor_calls']+=1;costs['physics_ticks']+=len(info['rows'])
        rows=np.vstack([nominal[:start],*chunks]);held=np.vstack([nominal_actions[:start//20],actions])
        r=measure(reference,rows,env)
        score=teacher_score(r,baseline,retained)
        file=folder/(tag+'.npz')
        np.savez_compressed(file,reference=reference,rows=rows,actions=held,
            observations=np.asarray(observations),innovations=np.asarray(innovations),
            residual_labels=np.asarray(labels),parent_latents=np.asarray(parent_latents),
            latent_actions=np.asarray(latent_actions),basis=basis,
            parameters=parameters,start_tick=start,stop_tick=stop)
        result=dict(tag=tag,path=path,parameters=parameters.tolist(),score=score,
                    trace=file.relative_to(WORK['stage']).as_posix(),record=r)
        history.append(result)
        if identity:
            assert np.array_equal(rows,nominal)
            assert np.array_equal(held,nominal_actions)
        return result
    if WORK['method'] in ('cobyla','block_cobyla'):
        from scipy.optimize import minimize
        guard_shape=dict(baseline['shape'])
        if WORK['reference_envelope']:
            for key in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                        'full_normalized_energy','full_spatial_peak_curvature_mm_inv'):
                guard_shape[key]=max(guard_shape[key],retained[key])
            guard_shape['apex_speed_mm_s']=min(guard_shape['apex_speed_mm_s'],retained['apex_speed_mm_s'])
        if WORK['protect_energy']:
            guard_shape['full_normalized_energy']=baseline['shape']['full_normalized_energy']
        if WORK['protect_exit']:
            guard_shape['out_edge_outward_mm']=baseline['shape']['out_edge_outward_mm']
        seed=job[3]
        if seed.get('evaluate_parameters'):
            seed=evaluate(np.asarray(seed['parameters']),'changed_support_seed')
            history.clear()
        zero=dict(tag='reused_identity',path=path,parameters=[0.]*dimensions,
                  score=teacher_score(baseline,baseline,retained),trace=baseline['trace'],record=baseline)
        history.extend([zero,seed])
        cache={tuple(np.zeros(dimensions)):zero,tuple(np.round(seed['parameters'],12)):seed}
        def inequalities(candidate):
            r=candidate['record'];s=r['shape'];b=guard_shape
            if not r['success']:return np.full(10+int('reverse_lateral_total_mm' in b),-100.)
            guards=[(b[k]-s[k])/scale for k,scale in (
                ('in_edge_outward_mm',.002),('out_edge_outward_mm',.005),
                ('full_physical_reverse_deg',.5),('full_normalized_energy',.02),
                ('full_spatial_peak_curvature_mm_inv',.2))]
            guards.extend([(s['apex_speed_mm_s']-b['apex_speed_mm_s'])/2.,
                (baseline['complete_time_s']+.02-r['complete_time_s'])/.02,
                (r['motion']['cruise_ge99_s']-baseline['motion']['cruise_ge99_s']+.02)/.05,
                1. if r['target_motion']['first_turn_inside'] else -1.,
                r['corridor']['margin_lower_mm']/.01])
            if 'reverse_lateral_total_mm' in b:
                guards.append((b['reverse_lateral_total_mm']-s['reverse_lateral_total_mm'])/.002)
            return np.asarray(guards)
        lower,upper=(-.6,.6) if WORK['expert'] is None else (-.25,1.25)
        def feasible(candidate):
            # COBYLA may evaluate outside its bounds before convergence. A history
            # candidate must satisfy BOTH the search box and the measured guards.
            return (parameters_in_bounds(candidate['parameters'],lower,upper)
                    and np.min(inequalities(candidate))>=-1e-8)
        def obtain(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                c=evaluate(u,f'constrained_{len(cache)-1}');cache[key]=c
                c['measured_inequalities']=inequalities(c).tolist()
                if len(cache)%24==0:
                    good=[r for r in history if feasible(r)]
                    best=min(good,key=lambda c:c['score'])
                    write_json(folder/'results.json',dict(candidates=history,best=best['tag'],
                        base_score=zero['score'],offline_teacher_only=True,deployable_policy=False))
                    print('CONSTRAINED_JOINT_TEACHER',path,len(cache)-2,best['score'],flush=True)
            return cache[key]
        if WORK['method']=='block_cobyla':
            block_audit=[];incumbent=min((r for r in history if feasible(r)),key=lambda c:c['score'])
            first_block=int(job[4]) if len(job)>4 else 0
            for block,label in enumerate(('exit_recovery','pre_apex','early_approach')):
                if block<first_block:
                    block_audit.append(dict(block=label,frames_ms=block_frames[block],
                        new_candidates=0,best=incumbent['tag'],score=incumbent['score'],reused=True))
                    continue
                anchor=np.asarray(incumbent['parameters']).copy()
                indices=np.arange(12*block,12*(block+1))
                def expand(v):
                    full=anchor.copy();full[indices]=v;return full
                count_before=len(cache)
                solution=minimize(lambda v:obtain(expand(v))['score'],anchor[indices],method='COBYLA',
                    bounds=[(-.25,1.25)]*12,
                    constraints=[dict(type='ineq',fun=lambda v:inequalities(obtain(expand(v))))],
                    options=dict(maxiter=80,rhobeg=.15,tol=1e-5,catol=1e-8))
                candidates=[r for r in history if feasible(r)]
                incumbent=min(candidates,key=lambda c:c['score'])
                block_audit.append(dict(block=label,frames_ms=block_frames[block],
                    new_candidates=len(cache)-count_before,best=incumbent['tag'],score=incumbent['score']))
                write_json(folder/'block_credit.json',dict(blocks=block_audit,offline_only=True))
                print('CAUSAL_BLOCK_JOINT_TEACHER',path,label,incumbent['score'],flush=True)
        else:
            solution=minimize(lambda u:obtain(u)['score'],np.asarray(seed['parameters']),method='COBYLA',
                bounds=([(-.6,.6)] if WORK['expert'] is None else [(-.25,1.25)])*dimensions,
                constraints=[dict(type='ineq',fun=lambda u:inequalities(obtain(u)))],
                options=dict(maxiter=240,rhobeg=.025 if WORK['expert'] is None else .15,tol=1e-5,catol=1e-8))
        good=[r for r in history if feasible(r)]
        best=min(good,key=lambda c:c['score'])
        write_json(folder/'results.json',dict(candidates=history,best=best['tag'],base_score=zero['score'],
            solver=dict(message=str(solution.message),evaluations=int(solution.nfev)),
            offline_teacher_only=True,deployable_policy=False))
        return dict(path=path,best=best,baseline=baseline,baseline_score=zero['score'],
            temporal_scope=WORK['temporal_scope'],
            shape_guard_bounds=guard_shape,
            reference_envelope=WORK['reference_envelope'],
            start_tick=start,stop_tick=stop,trials=len(history)-2,costs=costs,
            score_improved=best['score']<zero['score']-.001,offline_teacher_only=True,deployable_policy=False)
    if WORK['method']=='powell':
        from scipy.optimize import minimize
        zero=dict(tag='reused_identity',path=path,parameters=[0.]*12,
                  score=teacher_score(baseline,baseline,retained),trace=baseline['trace'],record=baseline)
        history.append(zero)
        cache={tuple(np.zeros(12)):zero}
        def objective(u):
            key=tuple(np.round(u,12))
            if key not in cache:
                cache[key]=evaluate(u,f'powell_{len(cache)}')
                if len(cache)%24==0:
                    best=min(history,key=lambda c:c['score'])
                    write_json(folder/'results.json',dict(candidates=history,best=best['tag'],
                        base_score=zero['score'],offline_teacher_only=True,deployable_policy=False))
                    print('POWELL_JOINT_TEACHER',path,len(cache),best['score'],flush=True)
            return cache[key]['score']
        solved=minimize(objective,np.zeros(12),method='Powell',bounds=[(-.3,.3)]*12,
            options=dict(maxfev=180,maxiter=5,xtol=.001,ftol=.0001,direc=.03*np.eye(12)))
        best=min(history,key=lambda c:c['score'])
        write_json(folder/'results.json',dict(candidates=history,best=best['tag'],base_score=zero['score'],
            solver=dict(message=str(solved.message),evaluations=int(solved.nfev)),
            offline_teacher_only=True,deployable_policy=False))
        return dict(path=path,best=best,baseline=baseline,baseline_score=zero['score'],
            start_tick=start,stop_tick=stop,trials=len(history)-1,costs=costs,
            score_improved=best['score']<zero['score']-.001,offline_teacher_only=True,deployable_policy=False)
    zero=evaluate(np.zeros(12),'identity',True)
    incumbent=zero
    mean=np.zeros(12);covariance=.3**2*np.eye(12)
    no_gain=0
    for generation in range(8):
        eig,vec=np.linalg.eigh(covariance);factor=vec@np.diag(np.sqrt(np.maximum(eig,1e-5)))
        noise=rng.normal(size=(12,12));batch=[]
        for i,z in enumerate(np.vstack([noise,-noise])):
            u=np.clip(mean+factor@z,-2.,2.)
            result=evaluate(u,f'g{generation}_{i}');batch.append(result)
        elite=sorted(batch+[incumbent],key=lambda c:c['score'])[:4]
        best=elite[0]
        improved=best['score']<incumbent['score']-.001
        if improved:incumbent=best;no_gain=0
        else:no_gain+=1
        u=np.array([c['parameters'] for c in elite]);target=u.mean(0);centered=u-target
        covariance=.65*covariance+.35*centered.T@centered/len(u)
        covariance*=1.1 if improved else .7
        mean=.25*mean+.75*target
        generations.append(dict(generation=generation,best=incumbent['tag'],score=incumbent['score'],
            complete=sum(c['record']['success'] for c in batch),no_gain=no_gain))
        write_json(folder/'results.json',dict(candidates=history,generations=generations,best=incumbent['tag'],
            base_score=zero['score'],offline_teacher_only=True,deployable_policy=False))
        print('JOINT_TEACHER',path,generation,incumbent['score'],incumbent['record']['shape']['full_physical_reverse_deg'],
              incumbent['record']['shape']['out_edge_outward_mm'],flush=True)
        if no_gain>=3:break
    return dict(path=path,best=incumbent,baseline=baseline,baseline_score=zero['score'],
        start_tick=start,stop_tick=stop,trials=len(history),generations=generations,costs=costs,
        score_improved=incumbent['score']<zero['score']-.001,offline_teacher_only=True,deployable_policy=False)


def main():
    torch.set_num_threads(1)
    data=read(BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json')
    baseline,retained=data['baseline'],data['retained']
    parent_file=BASE/'canonical_gate_band_iterative_motion_constraint_learning_20260919/research_candidate.pt'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[12,13,14,15]
    stage=register('canonical_gate_band_complete_joint_teacher_learning_20260919',
        [__file__,BASE/'decision_innovation_policy.py',BASE/'analyze_nonlinear_joint_response.py',
         BASE/'probe_coordinated_suffix.py',parent_file,
         BASE/'canonical_gate_band_current_bend_capacity_learning_20260919/baseline.json',
         BASE/'canonical_gate_band_innovation_motion_guard_learning_20260919/motion_credit_validation.json',
         BASE/'canonical_gate_band_innovation_motion_guard_learning_20260919/receipt.json'],
        dict(wall_s=900,workers=4,paths=paths,maximum_generations=8,population=24,
             coefficients_per_teacher=12,maximum_trials_per_path=193,stop_after_no_gain=3,
             action_decision_ms=20,physical_execution_ms=1,output_latent_scales=SCALE.tolist()),
        'New global/local response models failed a direct common-horizon check: predicted '
        'path31 distance0.81mm versus actual1.27mm, maximumXY error1.56mm. Do not optimize '
        'again on that invalid derivative model. Compare direct ACTUAL complete-return '
        'optimization of three actions together on four difficult training corners. '
        'Each OFFLINE teacher uses four smooth temporal basis coefficients per action, '
        'added to the feedback parent only within a recorded bend window, with zero '
        'endpoints. These temporal/path-specific teachers are explicitly NONDEPLOYABLE '
        'training oracles, not the learned state policy. CEM updates use actual full '
        'XY shape,completion,time,cruise and retained reference deficits, not predicted '
        'rollouts or hand-drawn target curves. Invalid paths get fixed failure scores. '
        'All20/1ms,V/A/J,1.25mm,ordered coverage,braking and rest remain unchanged. '
        'Record every measured candidate, selected joint residual labels and real states. '
        'Verify zero teacher bitwise identity. If useful teachers emerge, jointly distill '
        'steering/feed/preview into a shared observable-state network and test actual '
        'full64 closure before any research selection. Do not deploy an oracle schedule, '
        'claim PPO, add angle speed tables, or open sealed tests.')
    started=time.perf_counter();pool=None;outcomes=[]
    progress(stage,True,'直接用真实完整回报优化困难角三动作离线教师，随后联合蒸馏和闭环检查；时间序列教师绝不部署。')
    try:
        jobs=[]
        for path in paths:
            b=next(r for r in baseline['records'] if r['path']==path)
            o=next(r for r in retained if r['path']==path)['baseline_shape']
            jobs.append((path,b,o))
        pool=mp.get_context('spawn').Pool(4,initialize,(stage,parent,started+885))
        for result in pool.imap_unordered(optimize_path,jobs):
            outcomes.append(result)
            write_json(stage/'results.json',dict(teachers=outcomes,offline_teacher_only=True,
                deployable_policy=False,adopted=False,goal_complete=False))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_complete_joint_teacher_learning',paths=len(outcomes),
            improved_teachers=sum(c['score_improved'] for c in outcomes),trials=sum(c['trials'] for c in outcomes),
            wall_s=time.perf_counter()-started,offline_teacher_only=True,deployable_policy=False,
            adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_complete_joint_teacher_learning',error=str(exc),
            completed_paths=len(outcomes),wall_s=time.perf_counter()-started,
            offline_teacher_only=True,deployable_policy=False,adopted=False,goal_complete=False));raise
    finally:
        progress(stage,False,'依据离线联合教师的真实完整形状改进构造状态策略学习；教师序列不作为部署策略。')


if __name__=='__main__':mp.freeze_support();main()
