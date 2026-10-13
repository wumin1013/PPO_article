"""Learn shared state-dependent temporal actions from actual complete outcomes."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import copy
import argparse
import multiprocessing as mp
import time
import numpy as np
import torch
from scipy.optimize import minimize
from learned_action_pulse_policy import LearnedActionPulsePolicy,environment_factory
from learn_complete_joint_teachers import teacher_score,parameters_in_bounds
from analyze_nonlinear_joint_response import measure
from learn_full_pool_joint_credit import aggregate
from learn_retained_joint_growth import smooth_cruise
from study_io import BASE,register,receipt,write_json,sha
from search_complete_return import read


WORK={}


def initialize(stage,parent,baseline,retained,deadline,conditioned=False,cache_source=None,expert_state=None,envelope=False,
               feature_spec=None,envelope_paths=None):
    torch.set_num_threads(1)
    WORK.update(stage=stage,parent=parent,baseline=baseline,retained=retained,deadline=deadline,
                conditioned=conditioned,cache_source=cache_source,expert_state=expert_state,envelope=envelope,
                feature_spec=feature_spec,envelope_paths=envelope_paths or [104,204,206])


def inequalities(records):
    values=[]
    for r,b,o in zip(records,WORK['baseline'],WORK['retained']):
        if not r['success']:
            values.extend([-100.]*(11 if WORK['feature_spec'] else 10))
            continue
        s,t=r['shape'],b['shape']
        if WORK['envelope'] and r['path'] in WORK['envelope_paths']:
            t=dict(t)
            for key in ('in_edge_outward_mm','out_edge_outward_mm','full_physical_reverse_deg',
                        'full_normalized_energy','full_spatial_peak_curvature_mm_inv'):
                t[key]=max(t[key],o[key])
            t['apex_speed_mm_s']=min(t['apex_speed_mm_s'],o['apex_speed_mm_s'])
        if WORK['feature_spec']:
            t=dict(t)
            for k in ('out_edge_outward_mm','full_normalized_energy'):
                t[k]=b['shape'][k]
        values.extend((t[k]-s[k])/scale for k,scale in (
            ('in_edge_outward_mm',.002),('out_edge_outward_mm',.005),
            ('full_physical_reverse_deg',.5),('full_normalized_energy',.02),
            ('full_spatial_peak_curvature_mm_inv',.2)))
        values.extend([(s['apex_speed_mm_s']-t['apex_speed_mm_s'])/2,
            (b['complete_time_s']+.02-r['complete_time_s'])/.02,
            (r['motion']['cruise_ge99_s']-b['motion']['cruise_ge99_s']+.02)/.05,
            1. if r['target_motion']['first_turn_inside'] else -1.,r['corridor']['margin_lower_mm']/.01])
        if WORK['feature_spec']:
            values.append((b['shape']['reverse_lateral_total_mm']-s['reverse_lateral_total_mm'])/.002)
    return np.asarray(values)


def evaluate_complete_parameters(policy,coefficients,baseline,folder,tag,deadline,*,zero_support_paths=()):
    """One exact full-feedback vector evaluation, shared by bounded optimizers."""
    policy.coefficients=np.asarray(coefficients).reshape(-1,3)
    records=[]
    for b in baseline:
        if b['path'] in zero_support_paths:
            r=copy.deepcopy(b)
            r['reused_exact_zero_support']=True
            records.append(r)
            continue
        with np.load(b['trace']) as z:ref=z['reference'].copy()
        env=policy.bind(environment_factory(ref))
        chunks=[];actions=[];observations=[];executed=[]
        while not env.done:
            if time.perf_counter()>deadline:raise TimeoutError('Shared parameter learning budget')
            x=env.observation()
            _,_,_,info=env.step(policy.act(x,True)[0])
            chunks.append(info['rows']);actions.append(info['held_action']);observations.append(x)
            executed.append(info['executed_actions_1ms'])
        rows=np.vstack(chunks);r=measure(ref,rows,env)
        file=folder/f"{tag}_path{b['path']}.npz"
        np.savez_compressed(file,reference=ref,rows=rows,actions=np.asarray(actions),
            observations=np.asarray(observations),executed_actions_1ms=np.vstack(executed))
        r.update(path=b['path'],trace=str(file),curved_path=b['curved_path'])
        if r['success']:r['smooth_cruise99_s']=smooth_cruise(rows)
        records.append(r)
    return records


def optimize(mode):
    folder=WORK['stage']/mode
    folder.mkdir()
    dimensions=18 if WORK['conditioned'] else 9
    policy=LearnedActionPulsePolicy(WORK['parent'],np.zeros(dimensions),mode,WORK['expert_state'],WORK['feature_spec'])
    history=[]
    def score(records):
        return sum(teacher_score(r,b,o) for r,b,o in zip(records,WORK['baseline'],WORK['retained']))
    zero=dict(tag='reused_identity',coefficients=[0.]*dimensions,records=WORK['baseline'],
              score=score(WORK['baseline']),measured_inequalities=inequalities(WORK['baseline']).tolist())
    cache={tuple(np.zeros(dimensions)):zero}
    history.append(zero)
    if WORK['cache_source']:
        cached_mode=mode if WORK['envelope'] else 'constant' if mode=='target_coordinates' else 'expert_residual' if mode=='anticipatory_expert' else mode
        previous=read(WORK['cache_source']/cached_mode/'results.json')['candidates']
        for c in previous:
            if mode=='target_coordinates' and np.any(np.asarray(c['coefficients'])[2::3]):continue
            if mode=='anticipatory_expert' and not WORK['envelope'] and np.any(np.asarray(c['coefficients'])[9:]):continue
            u=np.r_[c['coefficients'],np.zeros(dimensions-len(c['coefficients']))]
            key=tuple(np.round(u,12))
            if key in cache:continue
            c=copy.deepcopy(c);c['coefficients']=u.tolist();c['tag']='reused_'+c['tag']
            c['measured_inequalities']=inequalities(c['records']).tolist()
            cache[key]=c;history.append(c)
    cached_count=len(cache)
    def selectable(c):
        return (parameters_in_bounds(c['coefficients'],-3.,3.)
                and min(c['measured_inequalities'])>=-1e-8)
    # Verify the inherited physical executor and zero residual against stored startup.
    for b in WORK['baseline']:
        with np.load(b['trace']) as z:
            ref,old,actions=[z[k].copy() for k in ('reference','rows','actions')]
        env=policy.bind(environment_factory(ref))
        latent=policy.act(env.observation(),True)[0]
        _,_,_,info=env.step(latent)
        assert np.array_equal(info['rows'],old[:20])
        assert np.array_equal(info['held_action'],actions[0])
    write_json(folder/'identity.json',dict(paths=[r['path'] for r in WORK['baseline']],
        zero_first20ms_bitwise=True,decision_ms=20,execution_ms=1,original_physics_inherited=True))
    def obtain(u):
        key=tuple(np.round(u,12))
        if key in cache:return cache[key]
        if time.perf_counter()>WORK['deadline']:raise TimeoutError('Shared temporal learning budget')
        tag=f'candidate_{len(cache)}'
        records=evaluate_complete_parameters(policy,u,WORK['baseline'],folder,tag,WORK['deadline'])
        c=dict(tag=tag,coefficients=np.asarray(u).tolist(),records=records,score=score(records),
               measured_inequalities=inequalities(records).tolist())
        cache[key]=c;history.append(c)
        if len(cache)%10==0:
            good=[c for c in history if selectable(c)]
            best=min(good,key=lambda c:c['score'])
            write_json(folder/'results.json',dict(candidates=history,best=best['tag']))
            print('SHARED_TEMPORAL_LEARNING',mode,len(cache)-1,best['score'],flush=True)
        return c
    result=minimize(lambda u:obtain(u)['score'],np.zeros(dimensions),method='COBYLA',bounds=[(-3.,3.)]*dimensions,
        constraints=[dict(type='ineq',fun=lambda u:obtain(u)['measured_inequalities'])],
        options=dict(maxiter=80,rhobeg=.25,tol=1e-4,catol=1e-8))
    good=[c for c in history if selectable(c)]
    best=min(good,key=lambda c:c['score'])
    write_json(folder/'results.json',dict(candidates=history,best=best['tag'],
        solver_message=str(result.message),evaluations=int(result.nfev)))
    md=dict(architecture='joint_expert_residual' if mode in ('expert_residual','anticipatory_expert','spatial_expert','innovation_coordinates') else 'learned_state_temporal_pulse',parent=WORK['parent'],
        coefficients=np.asarray(best['coefficients']),mode=mode,accepted=False,
        learning_method='actual_complete_return_joint_parameter_optimization',
        speed_conditioned=WORK['conditioned'] and mode not in ('anticipatory_expert','innovation_coordinates'),
        motion_conditioned=WORK['conditioned'] and mode=='innovation_coordinates',
        geometry_conditioned=mode=='anticipatory_expert',
        reference_envelope_paths=WORK['envelope_paths'] if WORK['envelope'] else [],
        action_interval_ms=20,physical_tick_ms=1)
    if mode in ('expert_residual','anticipatory_expert','spatial_expert','innovation_coordinates'):md['expert_state']=WORK['expert_state']
    if WORK['feature_spec']:md.update(feature_spec=WORK['feature_spec'],protect_current_lateral_energy_exit=True)
    torch.save(md,folder/'candidate.pt')
    return dict(mode=mode,best=best,baseline_score=zero['score'],new_candidates=len(cache)-cached_count,
                reused_candidates=cached_count,
                improved=best['score']<zero['score']-.001,full_pool_pass=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--speed-conditioned',action='store_true')
    parser.add_argument('--target-coordinate',action='store_true')
    parser.add_argument('--expert-residual',action='store_true')
    parser.add_argument('--anticipatory-expert',action='store_true')
    parser.add_argument('--reference-envelope',action='store_true')
    args=parser.parse_args()
    if args.reference_envelope:args.anticipatory_expert=True
    if args.anticipatory_expert:args.expert_residual=True
    if args.target_coordinate or args.expert_residual:args.speed_conditioned=True
    assert not (args.target_coordinate and args.expert_residual)
    torch.set_num_threads(1)
    source=BASE/'canonical_gate_band_target_jump_teacher_state_distillation_20260919'
    added=BASE/'canonical_gate_band_joint_training_interpolation_20260919'
    result=read(source/'results.json')
    baseline=copy.deepcopy(next(c for c in result['candidates'] if c['tag']==result['best']))
    for r in baseline['records']:r['trace']=str(source/r['trace'])
    retained=read(source/'baseline.json')['retained']
    new=read(added/'results.json')['records']
    for r in sorted((r for r in new if r['arm']=='joint_state'),key=lambda r:r['path']):
        r=copy.deepcopy(r);r['trace']=str(added/r['trace']);baseline['records'].append(r)
        old=next(t for t in new if t['path']==r['path'] and t['arm']=='retained')
        retained.append(dict(path=r['path'],baseline_shape=old['shape'],
                             complete_time_s=old['complete_time_s'],complete_time_change_s=0.))
    baseline['summary']=aggregate(baseline['records'])
    parent_file=source/'research_candidate.pt'
    assert sha(parent_file)=='20e60338eb4e6980eefc013766cf04a8d943d0e6f3d8bc97e658dfddf340cb00'
    parent=torch.load(parent_file,map_location='cpu',weights_only=False)
    paths=[14,104,204,206]
    training=[r for r in baseline['records'] if r['path'] in paths]
    old=[next(o['baseline_shape'] for o in retained if o['path']==r['path']) for r in training]
    cache_source=(BASE/'canonical_gate_band_anticipatory_joint_expert_learning_20260920' if args.reference_envelope else
                  BASE/'canonical_gate_band_joint_expert_residual_learning_20260920' if args.anticipatory_expert else
                  None if args.expert_residual else
                  BASE/'canonical_gate_band_speed_conditioned_temporal_joint_learning_20260919' if args.target_coordinate else
                  BASE/'canonical_gate_band_shared_temporal_joint_learning_20260919' if args.speed_conditioned else None)
    extra_sources=([cache_source/'receipt.json',cache_source/'anticipatory_expert/results.json',
                    BASE/'joint_anticipation_features.py'] if args.reference_envelope else
                   [cache_source/'receipt.json',cache_source/'expert_residual/results.json',
                    cache_source/'anticipation_feature_audit.json',BASE/'joint_anticipation_features.py'] if args.anticipatory_expert else
                    [cache_source/'receipt.json',cache_source/'constant/results.json']+
                    ([] if args.target_coordinate else [cache_source/'pulse/results.json']) if cache_source else [])
    expert_state=None
    if args.expert_residual:
        expert_file=BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918/candidate_0.5.pt'
        assert sha(expert_file)=='86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
        expert_state=torch.load(expert_file,map_location='cpu',weights_only=False)['policy']
        extra_sources.append(expert_file)
    arms=['anticipatory_expert'] if args.anticipatory_expert else ['expert_residual'] if args.expert_residual else ['target_coordinates'] if args.target_coordinate else ['pulse','constant']
    stage=register('canonical_gate_band_reference_envelope_joint_learning_20260920' if args.reference_envelope else
                   'canonical_gate_band_anticipatory_joint_expert_learning_20260920' if args.anticipatory_expert else
                   'canonical_gate_band_joint_expert_residual_learning_20260920' if args.expert_residual else
                   'canonical_gate_band_target_coordinate_joint_learning_20260919' if args.target_coordinate else
                   'canonical_gate_band_speed_conditioned_temporal_joint_learning_20260919' if args.speed_conditioned
                   else 'canonical_gate_band_shared_temporal_joint_learning_20260919',
        [__file__,BASE/'learned_action_pulse_policy.py',BASE/'joint_runtime.py',
         BASE/'gate_band_state.py',BASE/'learn_complete_joint_teachers.py',parent_file,
         source/'results.json',source/'receipt.json',source/'raw_result_audit.json',
         added/'results.json',added/'receipt.json']+extra_sources,
        dict(wall_s=600,workers=len(arms),training_paths=paths,full_training_pool=80,
             arms=arms,parameters_per_arm=18 if args.speed_conditioned else 9,
             maximum_optimizer_evaluations_per_arm=80,
             speed_conditioned=args.speed_conditioned and not args.anticipatory_expert,
             geometry_conditioned=args.anticipatory_expert,
             reference_envelope_paths=[104,204,206] if args.reference_envelope else [],
             initial_radius=.25,bounds=[-3.,3.],decision_ms=20,execution_ms=1),
        ('The measured anticipation candidate improved three difficult routes but traded '
         'some peak curvature/apex speed against severe outgoing excursion. Compare '
         'the OPTIMIZATION guard, not physical limits: keep current guards on original '
         'path14 and every other saved route; on unresolved added paths104,204,206 only, '
         'use max(current,retained86) upper shape bounds and min(current,retained86) '
         'apex-speed floors. Keep current time/cruise guards and unchanged1.25mm,V/A/J, '
         '20/1ms,ordered coverage,braking/rest. Reuse every completed anticipation '
         'candidate and recompute only its declared guard values, not its trajectory '
         'or score. Same18 parameters and80-call solver budget. Any candidate needs '
         'all80 paths with the same explicitly recorded research envelope; this is '
         'not final tangency/smoothness acceptance or replacement of retained/S2. '
         if args.reference_envelope else '')+
        ('Saved-state audit found reactive correction features around1e-6 during the '
         'first200ms of three-155 cases, while the same observable state already '
         'contained future-bend signals around0.24-0.35. Replace the nine speed-product '
         'coefficients with near/far continuous sensed geometry and geometry-gap '
         'interaction coefficients: still18 total parameters. This changes only '
         'the learned correction input representation, not the available sensor '
         'data,physical execution or guards. Reuse only old expert candidates whose '
         'replaced coefficients are exactly zero. No future path ID,angle bucket '
         'or manually timed switch is added. ' if args.anticipatory_expert and not args.reference_envelope else '')+
        ('Complete80 shared-kernel checks preserved completion but failed shape protection '
         'on midpoint training states. Test a richer learned function span without a '
         'manual switching schedule: each output learns a continuous observable-state '
         'weight on the difference between retained86cd and current20e actions. The '
         '18 coefficients use even target-jump,gap,lag features and the registered '
         'continuous conditioning inputs; both experts were learned policies. Zero coefficients exactly '
         'preserve20e; all three weights learn jointly from real complete trajectories. '
         'No angle/path/time IDs,speed rules,temporal pulses or hand-chosen switches '
         'are used in this arm. Any subset gain needs all80-path protection before saving. '
         if args.expert_residual else '')+
        ('Both temporal modes and explicit speed conditioning found no guard-preserving '
         'update. Test a learned joint TARGET-coordinate action: the steering residual '
         'changes desired combined heading, while a preview change is compensated in '
         'steering at the decision instant using the current guide geometry. The '
         'same18 state-conditioned parameters learn from real full trajectories. '
         'Compensation is an action-coordinate decoder, not a speed schedule or an '
         'assertion that future20ms targets match. Reuse only cached constant-arm '
         'candidates with identically zero preview coefficients; other candidates '
         'are physically different and must be measured. All guards remain unchanged. '
         if args.target_coordinate else '')+
        ('The completed nine-coefficient comparison found no feasible improvement: lower '
         'scores reduced bend speed and violated energy/curvature protection across different '
         'incoming states. Test the hypothesis that action response needs explicit current '
         'speed conditioning. Add continuous measured speed times each old feature, giving '
         '18 coefficients; speed is an input, not a requested speed or rule. Reuse ALL '
         'completed old candidates exactly through zero added coefficients; the zero-slope '
         'implementation retains identical arithmetic. Same optimizer evaluation cap and '
         'guards. This hypothesis is not asserted as an established universal cause. '
         if args.speed_conditioned and not args.target_coordinate and not args.expert_residual else '')+
        'Midpoint TRAINING augmentation exposed poor-155/+155 geometry despite exact '
        'protection at72 saved paths. Test a SHARED learned state-dependent joint temporal '
        'representation, not another path-local kernel. Matched coefficients and observed '
        'target-jump,target-gap,motion-lag features in the registered arms. Constant and '
        'target-coordinate/expert modes hold actions20ms; pulse mode decays a learned '
        'residual quadratically to zero within the original20ms decision. '
        'Amplitudes are learned by actual complete constrained returns, never manual angles, '
        'speed tables or trajectory templates. Original guide,vector projector,brake,1.25mm '
        'and1ms physics are inherited. Verify zero bitwise identity; log actual1ms actions. '
        'Optimize on four difficult training paths under every current per-path shape/speed '
        'guard, then test any feasible improved learned coefficients on all80 training paths. '
        'No adoption from subset or a probe, no PPO, no independent/generalization claim.')
    write_json(stage/'baseline.json',dict(baseline=baseline,retained=retained))
    started=time.perf_counter();pool=None;outcomes=[]
    try:
        pool=mp.get_context('spawn').Pool(len(arms),initialize,(stage,parent,training,old,started+580,
                                                     args.speed_conditioned,cache_source,expert_state,args.reference_envelope))
        for r in pool.imap_unordered(optimize,arms):
            outcomes.append(r)
            write_json(stage/'results.json',dict(arms=outcomes,training_only=True,adopted=False))
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_shared_temporal_joint_learning',arms=len(outcomes),
            new_candidates=sum(r['new_candidates'] for r in outcomes),
            improved_arms=[r['mode'] for r in outcomes if r['improved']],
            wall_s=time.perf_counter()-started,full_pool_pass=False,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shared_temporal_joint_learning',error=str(exc),
            completed_arms=len(outcomes),wall_s=time.perf_counter()-started,
            full_pool_pass=False,adopted=False,goal_complete=False));raise


if __name__=='__main__':mp.freeze_support();main()
