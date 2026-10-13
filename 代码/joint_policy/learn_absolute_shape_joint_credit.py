"""Matched full-policy learning of absolute shape goals versus baseline regret."""
import argparse
import itertools
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from compact_joint_growth_policy import CompactJointGrowthPolicy
from complete_local_credit import complete_metric_response,state_function_features
from learn_retained_joint_growth import growth_deficits,evaluate_growth
from learn_retained_joint_growth import smooth_cruise
from learn_trace_resolved_joint_credit import build_trace_response,trace_deficits
from learn_entry_preserving_trace_credit import entry_constraints
from learn_coupled_feasibility_credit import build_geometric_response
from learn_nonlinear_joint_feedback import promotion
from optimize_complete_joint_function import initialize_worker,residual
from study_io import BASE,register,receipt,write_json
from search_complete_return import read


def absolute_shape_response(value,jac,records,retained):
    """Keep the actual metric derivatives; change only declared shape targets."""
    value,jac=value.copy(),jac.copy()
    old={r['path']:r['baseline_shape'] for r in retained if 'baseline_shape' in r}
    offset=0
    for r in records:
        offset+=1
        if 'shape' in r:
            b=old[r['path']]
            for index,key,factor,scale in [
                (1,'out_edge_outward_mm',.5,.01),
                (2,'full_physical_reverse_deg',.5,1.),
                (3,'full_normalized_energy',.2,.1)]:
                i=offset+index
                value[i]=factor*value[i]+b[key]/scale
                jac[i]*=factor
            offset+=7
        offset+=1
    return value,jac


def absolute_actual(candidate,retained):
    value=growth_deficits(candidate,retained)
    value,_=absolute_shape_response(value,np.zeros((len(value),1)),candidate['records'],retained)
    value=np.maximum(value,0.)
    return float(value@value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--calibrated-geometry',action='store_true')
    parser.add_argument('--fast-source',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source=BASE/('canonical_gate_band_conditioned_joint_credit_20260919' if args.fast_source
                else 'canonical_gate_band_compact_joint_growth_completion_20260919')
    data=read(source/'results.json')
    assert read(source/'receipt.json')['status'].startswith('completed')
    current=data['candidates'][0] if args.fast_source else next(r for r in data['candidates'] if r['tag']=='reused_retained86')
    spec=torch.load(source/'policy_spec.pt',map_location='cpu',weights_only=False)
    n=spec['transform'].shape[1]
    indexed={r['tag']:r for r in data['candidates']}
    paired=[(indexed[f'iter0_axis{h}_-1'],indexed[f'iter0_axis{h}_+1']) for h in range(n)]
    step=.125 if args.fast_source else .25
    origin=np.asarray(current['coefficients'])
    calibrated=args.calibrated_geometry or args.fast_source
    initialization_control=args.calibrated_geometry and not args.fast_source
    retained=read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage=register('canonical_gate_band_fast_absolute_shape_joint_credit_20260919' if args.fast_source else
                  'canonical_gate_band_calibrated_geometry_joint_credit_20260919' if args.calibrated_geometry
                   else 'canonical_gate_band_absolute_shape_joint_credit_20260919',
        [__file__,BASE/'compact_joint_growth_policy.py',BASE/'learn_retained_joint_growth.py',
         BASE/'learn_trace_resolved_joint_credit.py',BASE/'complete_local_credit.py',
         BASE/'learn_entry_preserving_trace_credit.py',BASE/'learn_coupled_feasibility_credit.py',
         BASE/'optimize_complete_joint_function.py',source/'results.json',source/'receipt.json',source/'policy_spec.pt']+
         ([BASE/'calibrated_joint_geometry.py',BASE/'canonical_gate_band_joint_geometry_response_consistency_20260919/results.json',
           BASE/'canonical_gate_band_absolute_shape_joint_credit_20260919/results.json',
           BASE/'canonical_gate_band_absolute_shape_joint_credit_20260919/receipt.json'] if calibrated else [])+
         ([BASE/'conditioned_joint_policy.py',BASE/'canonical_gate_band_calibrated_geometry_joint_credit_20260919/receipt.json'] if args.fast_source else []),
        dict(reused_response_candidates=2*n,new_candidates=4,paths=20,workers=4,
             joint_normalized_radius=2.,wall_s=180,
             absolute_scales=dict(exit_mm=.01,reverse_deg=1.,normalized_energy=.1)),
        ('The retained-start absolute-goal and calibrated-geometry candidates improved '
         'some shape measures but did not reach true99mm/s curved cruise. Apply the '
         'SAME absolute shape objective to the already completed conditioned fast '
         'research center, which has about3.76s actual curved cruise and remains '
         'unadopted because shape fails. Reuse its92 complete-policy response '
         'candidates at the EXACT center; do not rerun them or treat this research '
         'model as the retained model. Preserve that branch original0.125 parameter '
         'response step and [.0025,.0125,.0125] latent function bounds. Calibrate '
         'the sampled-distance proposal model to reproduce known feasible axes, '
         'with the same normalized joint radius2. Compare baseline-regret joint, '
         'absolute-shape joint half/full, and absolute-shape feed-only full. '
         'Compute smooth cruise only from full saved actual trajectories; hard99 '
         'cruise and all original per-corner metrics remain the real acceptance. '
         'Cache repeated actual float32 coefficient vectors. Four maximum new '
         'full20-path checks, no labels, PPO claim, rule deployment or adoption. '
         'This directly tests whether correcting an objective that stops rewarding '
         'improvements below the old baseline helps shape at an already cruising '
         'operating point. Do not claim a matched causal comparison between '
         'different starting architectures. Preserve86cd and all final full64/'
         'fair/generalization requirements.' if args.fast_source else
        'The linear sampled-distance constraint falsely excluded10 already measured '
         'complete primary-preserving axes, including the best actually measured '
         'axis. Worst predicted violation was0.419mm while exact actual segment '
         'margin remained0.0115mm. Replace only this proposal geometry with a '
         'diagonal quadratic response that reproduces the observed +/- axis '
         'point distances exactly, including their asymmetry. Extend completed '
         'stopped paths only at their true endpoint; never extend failed prefixes. '
         'Require zero false exclusions among all known completed sampled axes '
         'before new simulation. Cross-axis effects remain unmodeled and every '
         'proposal still needs the original exact full closed-loop physical audit. '
         'Use the SAME absolute-shape objective, raw48-function actor space, '
         'function bounds, entry constraints and normalized joint radius2. '
         'Compare zero initialization and initialization at the best actually '
         'measured complete primary-preserving axis; evaluate half/full steps '
         'for each, four new candidates. Reuse prior linear-geometry outcomes '
         'as the control, without rerunning them. This is joint parameter '
         'optimization and local-response calibration, not PPO, proof of global '
         'feasibility, a new speed rule or relaxation of any actual acceptance. '
         'All original per-corner, hard99 cruise, full64 and later frozen fair/'
         'generalization requirements remain in force.' if args.calibrated_geometry else
        'The prior objective penalized only regressions from86cd. Reverse/exit '
        'and energy terms became inactive once below a still-imperfect baseline, '
        'which is insufficient for the original monotone/tangent/smooth-shape '
        'goal. Reuse the exact completed compact-policy response collection at '
        'retained86cd. Compare the original baseline-regret objective with a '
        'declared absolute-shape objective: exit excursion/0.01mm, physical '
        'reverse/1degree, normalized bending energy/0.1, all squared after '
        'the existing positive-part operation. Keep entry, peak-curvature/apex '
        'regression, time and smooth actual-cruise terms as before. This is '
        'reward/optimization-goal revision, never a trajectory template or '
        'action/speed label. Both joint arms use the same signed-trace response, '
        'physical geometry and entry constraints, function bounds and normalized '
        'joint radius2. Add a matched absolute-objective feed-only parameter '
        'mask to assess whether joint coordination helps. Evaluate original '
        'joint at full step, absolute joint at half/full, and absolute feed '
        'at full:4 new complete20-path candidates. Predicted scores do not '
        'count as results; incomplete paths get no time/cruise advantage. '
        'Save a research candidate only if complete primary conditions hold '
        'and actual absolute-goal loss improves. Original per-corner screens '
        'and genuine hard99 cruise still govern adoption; ideal shape and '
        'full64/fair/generalization checks remain unfulfilled unless verified.'))
    started=time.perf_counter()
    subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
        '--active',stage.name,'--kind','complete_closed_loop_parameter_optimization',
        '--next','对照绝对形状目标、原基线退化目标与仅进给更新；必须检查完整真实轨迹。'],
        cwd=BASE,check=True,stdout=subprocess.DEVNULL)
    pool=None
    try:
        if args.fast_source:
            smooth_summary=[]
            for candidate in data['candidates']:
                total=0.
                for r in candidate['records']:
                    if r['success']:
                        with np.load(source/r['trace']) as z:r['smooth_cruise99_s']=smooth_cruise(z['rows'])
                        if r['path'] not in (48,57):total+=r['smooth_cruise99_s']
                candidate['summary']['smooth_curved_cruise99_s']=total if all(r['success'] for r in candidate['records']) else None
                smooth_summary.append(dict(tag=candidate['tag'],smooth_curved_cruise99_s=candidate['summary']['smooth_curved_cruise99_s']))
            write_json(stage/'reused_smooth_motion.json',dict(candidates=smooth_summary,new_physics_ticks=0))
        base,matrix,bounds,modes,_=complete_metric_response(current,paired,retained)
        base[-1]=growth_deficits(current,retained)[-1]
        for h,(lo,hi) in enumerate(paired):
            low,high=lo['summary']['smooth_curved_cruise99_s'],hi['summary']['smooth_curved_cruise99_s']
            nominal=current['summary']['smooth_curved_cruise99_s']
            matrix[-1,h]=-5*((high-low)/2 if low is not None and high is not None else
                high-nominal if high is not None else nominal-low if low is not None else 0.)
        models=build_trace_response(source,current,paired)
        zero_value,zero_jac=trace_deficits(np.zeros(n),base,matrix,models,current,retained)
        absolute_zero,_=absolute_shape_response(zero_value,zero_jac,current['records'],retained)
        absolute_zero=np.maximum(absolute_zero,0.)
        identity_error=abs(float(absolute_zero@absolute_zero)-absolute_actual(current,retained))
        assert identity_error<1e-7,identity_error
        write_json(stage/'metric_identity.json',dict(absolute_loss_error=identity_error))
        if args.fast_source:
            from conditioned_joint_policy import ConditionedJointPolicy
            support=np.vstack([np.load(source/r['trace'])['observations'] for r in current['records']])
            p=ConditionedJointPolicy(spec['parent'],spec['transform'])
            limits=np.array([.0025,.0125,.0125])
        else:
            with np.load(BASE/'canonical_gate_band_retained86_parameter_geometry_20260918/geometry.npz') as z:support=z['observation'].copy()
            p=CompactJointGrowthPolicy(spec['retained'],spec['feed'],spec['basis'],spec['transform'])
            limits=np.array([.005,.6,.03])
        features=state_function_features(p,support)
        trust=(features*step/limits[None,:,None]).reshape(-1,n)
        margin,geometry,queries=build_geometric_response(source,current,paired)
        quadratic=np.zeros_like(geometry)
        if calibrated:
            from calibrated_joint_geometry import calibrated_geometry
            margin,geometry,quadratic,calibration=calibrated_geometry(source,current,paired,queries)
            write_json(stage/'geometry_calibration.json',calibration)
            assert calibration['false_exclusions']==0,calibration
            print('GEOMETRY_CALIBRATED',calibration['known_complete_axes'],calibration['false_exclusions'],flush=True)
        entry,entry_jac=entry_constraints(models)
        constraints=[dict(type='ineq',fun=lambda u:np.r_[1-trust@u,1+trust@u],jac=lambda u:np.vstack([-trust,trust])),
            dict(type='ineq',fun=lambda u:(margin-geometry@u-quadratic@(u*u))/.01,
                 jac=lambda u:(-geometry-2*quadratic*u[None,:])/.01),
            dict(type='ineq',fun=lambda u:4-u@u,jac=lambda u:-2*u)]
        if len(entry):constraints.append(dict(type='ineq',fun=lambda u:(entry+entry_jac@u)/.001,jac=lambda u:entry_jac/.001))
        head_size=spec['transform'].shape[0]//3
        feed_columns=np.any(abs(spec['transform'][head_size:2*head_size])>1e-12,axis=0)
        jobs,audits=[],[]
        eligible_axes=[c for pair in paired for c in pair if c.get('primary_preserved')]
        seed_axis=min(eligible_axes,key=lambda c:absolute_actual(c,retained)) if initialization_control else None
        for arm in (('calibrated_zero','calibrated_sample') if initialization_control else
                    ('regret_joint','absolute_joint','absolute_feed')):
            absolute=initialization_control or arm.startswith('absolute')
            def objective(u):
                v,j=trace_deficits(u,base,matrix,models,current,retained)
                if absolute:v,j=absolute_shape_response(v,j,current['records'],retained)
                v=np.maximum(v,0.)
                return float(v@v+.01*(u@u)),2*j.T@v+.02*u
            arm_bounds=[bound if arm!='absolute_feed' or feed_columns[h] else (0.,0.) for h,bound in enumerate(bounds)]
            initial=(np.asarray(seed_axis['coefficients'])-origin)/step if arm=='calibrated_sample' else np.zeros(n)
            sol=minimize(objective,initial,jac=True,method='SLSQP',bounds=arm_bounds,constraints=constraints,
                         options=dict(maxiter=300,ftol=1e-8))
            audits.append(dict(arm=arm,solver_success=bool(sol.success),message=sol.message,
                predicted_objective=objective(sol.x)[0]-.01*(sol.x@sol.x),normalized_l2=float(np.linalg.norm(sol.x)),
                normalized_step=sol.x.tolist(),baseline_objective=objective(np.zeros(n))[0],
                initial_sample=None if arm!='calibrated_sample' else seed_axis['tag']))
            if sol.success:
                for scale in ((.5,1.) if arm=='absolute_joint' or initialization_control else (1.,)):
                    jobs.append((f'{arm}_{scale}',origin+step*scale*sol.x))
        write_json(stage/'proposal.json',dict(arms=audits,complete_response_modes=modes))
        print('ABSOLUTE_SHAPE_PROPOSALS',[(r['arm'],r['predicted_objective'],r['normalized_l2']) for r in audits],flush=True)
        refs={}
        for r in current['records']:
            with np.load(source/r['trace']) as z:refs[r['path']]=z['reference'].copy()
        cache={np.asarray(r['coefficients'],dtype=np.float32).tobytes():r for r in data['candidates']}
        pending,reused,aliases=[],[],{}
        pending_keys={}
        for tag,coeff in jobs:
            key=np.asarray(coeff,dtype=np.float32).tobytes()
            if key in cache:
                old=cache[key]
                reused.append(dict(old,tag=tag,reused_source_candidate=old['tag'],records=[dict(r,
                    trace='../'+(source/r['trace']).resolve().relative_to(BASE).as_posix()) for r in old['records']]))
            elif key in pending_keys:
                aliases.setdefault(pending_keys[key],[]).append(tag)
            else:
                pending_keys[key]=tag;pending.append((tag,coeff))
        pool=mp.get_context('spawn').Pool(4,initialize_worker,(stage,spec,refs,'gate_band_canonical_kernel',started+175))
        results=[]
        baseline_absolute=absolute_actual(current,retained)
        for result in itertools.chain(reused,pool.imap_unordered(evaluate_growth,pending)):
            result['promotion']=promotion(result['records'],retained)
            result['primary_preserved']=all(result['promotion'][k] for k in ('all_complete','inward_first_turns','entry_nonworse'))
            result['hard_objective']=float(np.linalg.norm(residual(result,retained))**2)
            result['absolute_objective']=absolute_actual(result,retained) if result['summary']['successes']==20 else None
            results.append(result)
            for alias in aliases.get(result['tag'],[]):
                results.append(dict(result,tag=alias,reused_identical_candidate=result['tag']))
            write_json(stage/'results.json',dict(candidates=results,baseline_absolute_objective=baseline_absolute,
                adopted=False,goal_complete=False))
            print('ABSOLUTE_SHAPE_ACTUAL',result['tag'],result['absolute_objective'],result['summary'],result['promotion'],flush=True)
        pool.close();pool.join();pool=None
        usable=[r for r in results if r['primary_preserved'] and r['absolute_objective']<baseline_absolute-1e-5]
        best=min(usable,key=lambda r:r['absolute_objective']) if usable else None
        if best:
            torch.save(dict(architecture=spec['kind'],policy_spec=spec,coefficients=np.asarray(best['coefficients']),
                environment_architecture='gate_band_canonical_kernel',learning_method='absolute_shape_complete_policy_joint_optimization'),
                stage/'research_candidate.pt')
            write_json(stage/'results.json',dict(candidates=results,best=best['tag'],baseline_absolute_objective=baseline_absolute,
                adopted=False,goal_complete=False))
        receipt(stage,dict(status='completed_absolute_shape_joint_credit_comparison',new_candidates=len(pending),
            reported_candidates=len(results),reused_candidates=len(results)-len(pending),
            best=None if best is None else best['tag'],wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_absolute_shape_joint_credit_comparison',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False));raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
            '--kind','complete_closed_loop_parameter_optimization_finished',
            '--next','核对绝对形状目标的联合/仅进给对照；据真实收益继续三动作学习，不以候选失败或预算结束终止目标。'],
            cwd=BASE,check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':mp.freeze_support();main()
