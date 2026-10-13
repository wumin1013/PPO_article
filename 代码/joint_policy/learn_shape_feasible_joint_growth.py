"""Actual joint-growth update under explicit original per-corner shape constraints."""
import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from retained_joint_growth_policy import RetainedJointGrowthPolicy
from learn_retained_joint_growth import evaluate_growth, growth_deficits
from complete_local_credit import complete_metric_response, state_function_features
from learn_trace_resolved_joint_credit import build_trace_response, trace_deficits
from learn_entry_preserving_trace_credit import entry_constraints
from learn_coupled_feasibility_credit import build_geometric_response
from optimize_complete_joint_function import initialize_worker, residual
from learn_nonlinear_joint_feedback import promotion
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def shape_indices_and_tolerances(records):
    indices, tolerances = [], []
    offset = 0
    for r in records:
        offset += 1
        if 'shape' in r:
            indices.extend(range(offset, offset+6))
            tolerances.extend([1e-6/.002, 1e-6/.005, 1e-5/.5, 1e-6/.02, 1e-6/.2, 1e-4/2.])
            offset += 7
        indices.append(offset)
        tolerances.append(1e-9)
        offset += 1
    return np.asarray(indices), np.asarray(tolerances)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--time-attribution',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_retained_joint_growth_identity_corrected_20260919'
    data = read(source/'results.json')
    current = data['candidates'][0]
    paired = [(next(r for r in data['candidates'] if r['tag'] == f'iter0_axis{h}_-1'),
               next(r for r in data['candidates'] if r['tag'] == f'iter0_axis{h}_+1')) for h in range(18)]
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    spec = torch.load(source/'policy_spec.pt', map_location='cpu', weights_only=False)
    stage = register('canonical_gate_band_growth_time_constraint_attribution_20260919' if args.time_attribution
                     else 'canonical_gate_band_shape_feasible_joint_growth_20260919',
        [__file__, BASE/'learn_retained_joint_growth.py', BASE/'retained_joint_growth_policy.py',
         BASE/'learn_trace_resolved_joint_credit.py', BASE/'learn_entry_preserving_trace_credit.py',
         BASE/'complete_local_credit.py', source/'results.json', source/'receipt.json', source/'policy_spec.pt'],
        dict(reused_complete_policy_perturbations=36, maximum_new_candidates=3, paths=20,
             normalized_joint_radius=1., scales=[.25, .5, 1.], workers=3, wall_s=180),
        ('The first constrained local solve had essentially zero cruise gain. '
         'Before blaming policy capacity, isolate the added training-time completion '
         'constraints: compare per-path time inequalities, aggregate complete time, '
         'and no linearized time inequality, while every shape constraint stays '
         'identical. The third arm is offline attribution only. Completion times '
         'are quantized by the original decision/braking mechanism; their finite '
         'responses need not be smooth constraint derivatives. Only an aggregate '
         'time proposal may proceed to3 full-path checks, still with all unchanged '
         'shape and exact physical requirements. This changes an added optimizer '
         'constraint, never the20/1ms cadence, stop duration or failure accounting. '
         if args.time_attribution else '')+
        'The retained-start penalty optimizer reduced total reverse/exit but allowed '
        'individual shape regressions and produced no actual99mm/s curved cruise. '
        'At the exact retained starting point, reuse all36 response candidates. '
        'Replace soft shape penalties by explicit original per-corner entry/exit, '
        'physical reverse, normalized energy, peak curvature and apex-speed '
        'inequalities, plus each complete-path time condition. Keep exact existing '
        'screen tolerances; no physical bound changes. Maximize the measured smooth '
        'actual-speed cruise response, not a requested-speed label. Bound the joint '
        'normalized coefficient step by L2<=1 as well as saved-state action support '
        'and sampled corridor constraints: prior composed steps had L2 about5 despite '
        'single-axis calibration radius1. All inequalities are proposal models, never '
        'safety proofs. Run3 new full20-path candidates only if a nonzero predicted '
        'shape-feasible improving direction exists. Otherwise record a local '
        'representation/response limitation, not global impossibility, then change '
        'representation or credit. No acceptance unless fresh complete trajectories '
        'satisfy the original screening; actual hard99 cruise still determines cruise '
        'claims. This is parameter optimization, not PPO or deployed rules.')
    started = time.perf_counter()
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
        '--active', stage.name, '--kind', 'complete_closed_loop_parameter_optimization',
        '--next', '复用完整响应，求逐角形状约束下的联合提速方向，并检查真实闭环结果。'],
        cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    pool = None
    try:
        base, matrix, bounds, modes, _ = complete_metric_response(current, paired, retained)
        # Current source is exactly86cd. The only changed scalar training goal is
        # measured smooth cruise; its exact hard-threshold response is retained.
        hard_response = matrix[-1].copy()
        base[-1] = growth_deficits(current, retained)[-1]
        for h,(lo,hi) in enumerate(paired):
            low,high = lo['summary']['smooth_curved_cruise99_s'],hi['summary']['smooth_curved_cruise99_s']
            nominal = current['summary']['smooth_curved_cruise99_s']
            matrix[-1,h] = -5*((high-low)/2 if low is not None and high is not None else
                high-nominal if high is not None else nominal-low if low is not None else 0.)
        models = build_trace_response(source,current,paired)
        idx,tol = shape_indices_and_tolerances(current['records'])
        time_indices=[]
        offset=0
        for r in current['records']:
            offset+=8 if 'shape' in r else 1
            time_indices.append(offset)
            offset+=1
        time_indices=np.asarray(time_indices)
        with np.load(BASE/'canonical_gate_band_retained86_parameter_geometry_20260918/geometry.npz') as z:
            support = z['observation'].copy()
        p = RetainedJointGrowthPolicy(spec['retained'],spec['feed'],spec['basis'],spec['transform'])
        features = state_function_features(p,support)
        trust = (features*.25/np.array([.005,.25,.03])[None,:,None]).reshape(-1,18)
        margin,geometry,_ = build_geometric_response(source,current,paired)
        entry,entry_jac = entry_constraints(models)
        mode='per_path'
        def metric_constraint(u):
            value,jac = trace_deficits(u,base,matrix,models,current,retained)
            selected=idx if mode=='per_path' else idx[~np.isin(idx,time_indices)]
            tolerance=tol if mode=='per_path' else tol[~np.isin(idx,time_indices)]
            slack,gradient=tolerance-value[selected],-jac[selected]
            if mode=='aggregate':
                slack=np.r_[slack,1e-9-value[time_indices].sum()]
                gradient=np.vstack([gradient,-jac[time_indices].sum(0)])
            return slack,gradient
        constraints=[dict(type='ineq',fun=lambda u:np.r_[1-trust@u,1+trust@u],jac=lambda u:np.vstack([-trust,trust])),
            dict(type='ineq',fun=lambda u:(margin-geometry@u)/.01,jac=lambda u:-geometry/.01),
            dict(type='ineq',fun=lambda u:1-u@u,jac=lambda u:-2*u),
            dict(type='ineq',fun=lambda u:metric_constraint(u)[0],jac=lambda u:metric_constraint(u)[1])]
        if len(entry):
            constraints.append(dict(type='ineq',fun=lambda u:(entry+entry_jac@u)/.001,jac=lambda u:entry_jac/.001))
        gradient = matrix[-1]/5.
        comparisons=[]
        solutions={}
        for mode in (('per_path','aggregate','no_time_model') if args.time_attribution else ('per_path',)):
            sol = minimize(lambda u:(float(gradient@u+1e-5*(u@u)),gradient+2e-5*u),
                np.zeros(18),jac=True,method='SLSQP',bounds=bounds,constraints=constraints,
                options=dict(maxiter=400,ftol=1e-10))
            solutions[mode]=sol
            comparisons.append(dict(mode=mode,solver_success=bool(sol.success),
                predicted_smooth_cruise_gain_s=float(-gradient@sol.x),
                normalized_l2=float(np.linalg.norm(sol.x)),
                minimum_slack=float(metric_constraint(sol.x)[0].min())))
        if args.time_attribution:
            mode='aggregate'
            sol=solutions[mode]
            write_json(stage/'constraint_attribution.json',dict(comparisons=comparisons,
                only_aggregate_time_arm_may_be_executed=True))
            print('TIME_CONSTRAINT_ATTRIBUTION',comparisons,flush=True)
        u=sol.x
        audit=dict(solver_success=bool(sol.success),message=sol.message,
            predicted_smooth_cruise_gain_s=float(-gradient@u),joint_normalized_l2=float(np.linalg.norm(u)),
            minimum_shape_model_slack=float(metric_constraint(u)[0].min()),
            maximum_function_change=float(abs(trust@u).max()),normalized_step=u.tolist(),
            hard_cruise_response=hard_response.tolist(),smooth_cruise_response=(-gradient).tolist(),
            complete_response_modes=modes,shape_constraints=len(idx),incoming_constraints=len(entry))
        write_json(stage/'proposal.json',audit)
        print('SHAPE_FEASIBLE_GROWTH',audit,flush=True)
        if not sol.success or audit['minimum_shape_model_slack'] < -1e-7 or audit['predicted_smooth_cruise_gain_s'] < 1e-5:
            receipt(stage,dict(status='completed_local_shape_feasibility_limit',new_candidates=0,
                wall_s=time.perf_counter()-started,solver=audit,adopted=False,goal_complete=False))
            return
        refs={}
        for r in current['records']:
            with np.load(source/r['trace']) as z:refs[r['path']]=z['reference'].copy()
        pool=mp.get_context('spawn').Pool(3,initialize_worker,
            (stage,spec,refs,'gate_band_canonical_kernel',started+175))
        results=[]
        for result in pool.imap_unordered(evaluate_growth,
                [(f'feasible_{s}',.25*s*u) for s in (.25,.5,1.)]):
            result['promotion']=promotion(result['records'],retained)
            result['hard_objective']=float(np.linalg.norm(residual(result,retained))**2)
            results.append(result)
            write_json(stage/'results.json',dict(candidates=results,adopted=False,goal_complete=False))
            print('SHAPE_FEASIBLE_ACTUAL',result['tag'],result['summary'],result['promotion'],flush=True)
        pool.close();pool.join();pool=None
        receipt(stage,dict(status='completed_shape_feasible_joint_growth_learning',new_candidates=len(results),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        receipt(stage,dict(status='failed_shape_feasible_joint_growth_learning',error=str(exc),
            wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
            '--kind','complete_closed_loop_parameter_optimization_finished',
            '--next','读取逐角形状约束联合提速的真实结果或局部可行性限制；据证据调整表示和信用，保留86cd。'],
            cwd=BASE,check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':
    mp.freeze_support();main()
