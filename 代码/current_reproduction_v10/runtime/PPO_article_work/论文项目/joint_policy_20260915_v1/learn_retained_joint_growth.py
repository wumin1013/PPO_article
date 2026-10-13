"""Complete-policy joint learning from retained86cd, with smooth cruise credit."""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from retained_joint_growth_policy import RetainedJointGrowthPolicy, growth_basis_transform
from raw_state_joint_policy import construct_raw_basis
from complete_local_credit import complete_metric_response, state_function_features, signed_deficits
from learn_trace_resolved_joint_credit import build_trace_response, trace_deficits
from learn_entry_preserving_trace_credit import entry_constraints
from learn_coupled_feasibility_credit import build_geometric_response
from optimize_complete_joint_function import initialize_worker, evaluate_job, residual
from learn_nonlinear_joint_feedback import promotion, aggregate
from complete_corner_metrics import complete_corner_metrics
from complete_motion_metrics import complete_motion_metrics
from probe_linear_action_hold import target_motion
from reflection_policy import reflect_observation
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def smooth_cruise(rows):
    speed = np.linalg.norm(rows[:, 2:4], axis=1)
    return float((1/(1+np.exp(np.clip((99.-speed)/.5, -40, 40)))).sum()*.001)


def evaluate_growth(job):
    from optimize_complete_joint_function import WORKER
    result = evaluate_job(job)
    total = 0.
    for record in result['records']:
        if record['success']:
            with np.load(WORKER['stage']/record['trace']) as z:
                record['smooth_cruise99_s'] = smooth_cruise(z['rows'])
            if record['path'] not in (48, 57):
                total += record['smooth_cruise99_s']
    result['summary']['smooth_curved_cruise99_s'] = total if all(r['success'] for r in result['records']) else None
    return result


def growth_deficits(candidate, retained):
    values = signed_deficits(candidate, retained)
    values[-1] = (3.-candidate['summary']['smooth_curved_cruise99_s'])*5.
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    root = BASE/'canonical_gate_band_shape_cruise_completion_feedback_20260918'
    model_path = root/'candidate_0.5.pt'
    assert sha(model_path) == '86cdfe3e1bf550a34332bab2678fed9a12e3a33611173ba57a75a1818daf5a86'
    feed_path = BASE/'canonical_gate_band_nonlinear_cruise_capacity_20260918/diagnostic_policy.pt'
    geometry_path = BASE/'canonical_gate_band_retained86_parameter_geometry_20260918/geometry.npz'
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    source = read(root/'results.json')['full_candidate']
    wall_s,maximum_iterations=(420,1) if args.compact else (360,2)
    stage = register('canonical_gate_band_compact_joint_growth_20260919' if args.compact
                     else 'canonical_gate_band_retained_joint_growth_identity_corrected_20260919',
        [__file__, BASE/'retained_joint_growth_policy.py', BASE/'complete_local_credit.py',
         BASE/'learn_trace_resolved_joint_credit.py', BASE/'learn_entry_preserving_trace_credit.py',
         BASE/'learn_coupled_feasibility_credit.py', BASE/'optimize_complete_joint_function.py',
         model_path, root/'receipt.json', root/'results.json', feed_path,
         geometry_path, BASE/'canonical_gate_band_conditioned_joint_credit_20260919/receipt.json',
         BASE/'canonical_gate_band_retained_joint_growth_20260919/receipt.json']+
         ([BASE/'compact_joint_growth_policy.py',BASE/'canonical_gate_band_growth_time_constraint_attribution_20260919/receipt.json'] if args.compact else []),
        dict(workers=4, maximum_iterations=maximum_iterations, maximum_dimensions=48 if args.compact else 18, response_step=.25,
             proposal_scales=[.5, 1.], maximum_new_candidates=98 if args.compact else 76, paths=20, wall_s=wall_s,
             smooth_cruise_center_mm_s=99., smooth_cruise_width_mm_s=.5),
        ('The18-direction shape-constrained local model admitted only negligible '
         'cruise gain; three actual tiny updates still violated individual shape '
         'screens. Test a new local function representation:16 observable-state '
         'centers, reflection-equivariant Wendland C2 compact support, retaining '
         'all non-null steering/feed/preview functions instead of truncating each '
         'head to6 global principal directions. Radius derives only from retained '
         'state coverage; there is no angle, path identity, trajectory template '
         'or desired speed/action label. Increase feed feature scale to1.5 latent '
         'units at maximum support, with an explicit0.6 latent function trust '
         'bound; all physical limits and real acceptance stay unchanged. This is '
         'an exploratory representation-and-conditioning change, not an isolated '
         'causal proof for compact support. One complete response/update round; '
         'do not expand the same run if it has no useful actual primary-preserving '
         'progress. ' if args.compact else '')+
        'Repairing the failed fast-feed policy through small residuals preserved '
        'cruise but left persistent reversal and wide-corner defects. Change the '
        'learning starting point to the verified retained86cd actor, with exactly '
        'zero initial policy difference. Jointly learn steering/preview state '
        'functions and observable-state-dependent amplitude of the previously '
        'learned nonlinear feed feature. No new action/speed targets or regression. '
        'Use18 conditioned directions in the shared current-state radial feature '
        'span, six per output. Every coefficient response is an actual complete '
        'closed-loop rollout. Replace only the hard-threshold cruise objective '
        'term by the integral of sigmoid((actual speed-99)/0.5), over completed '
        'curved trajectories; raw99ms cruise remains the acceptance evidence. '
        'The smooth value is a training proxy, not claimed real cruise. All '
        'shape, per-path time, entry, exact geometry, V/A/J and stop checks remain. '
        'Reuse retained completed paths and saved observations without simulation. '
        'At most two response/update rounds; accept only primary-preserving '
        'complete loss reductions. Record both original hard and new smooth '
        'objectives and their matched local response. This is model-based '
        'parameter learning, not PPO, label fitting, a speed rule or adoption. '
        'No final model claim until original shape conditions plus full64 and '
        'later frozen fair/generalization validation all pass.')
    started = time.perf_counter()
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
        '--active', stage.name, '--kind', 'complete_closed_loop_parameter_optimization',
        '--next', '从86cd保留模型联合学习提速与形状协调；实际巡航和全部形状约束仍须独立通过。'],
        cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    with np.load(geometry_path) as z:
        support = z['observation'].copy()
    retained_md = torch.load(model_path, map_location='cpu', weights_only=False)
    feed_md = torch.load(feed_path, map_location='cpu', weights_only=False)
    if args.compact:
        from compact_joint_growth_policy import CompactJointGrowthPolicy,construct_compact_growth
        basis,transform,conditioning=construct_compact_growth(retained_md['policy'],feed_md['policy'],support)
        coverage=conditioning['coverage']
        policy_class=CompactJointGrowthPolicy
    else:
        basis, coverage = construct_raw_basis(support, count=16)
        transform, conditioning = growth_basis_transform(retained_md['policy'], feed_md['policy'], basis, support)
        policy_class=RetainedJointGrowthPolicy
    dimension = transform.shape[1]
    assert dimension<=48 if args.compact else dimension==18
    spec = dict(kind='retained_compact_growth' if args.compact else 'retained_joint_growth', retained=retained_md['policy'], feed=feed_md['policy'],
                basis=basis, transform=transform)
    policy = policy_class(spec['retained'], spec['feed'], basis, transform)
    torch.save(spec, stage/'policy_spec.pt')
    write_json(stage/'basis_audit.json', dict(coverage=coverage, conditioning=conditioning))
    sample = torch.as_tensor(support[::11], dtype=torch.float32)
    with torch.no_grad():
        f = policy.function_features(sample)
        parity_error = float(abs(policy.function_features(reflect_observation(sample))-
            f*torch.tensor([-1., 1., 1.])[None, :, None]).max())
    identity_error = max(float(abs(policy.act(x, True)[0]-policy.base.act(x, True)[0]).max()) for x in support[::11])
    assert parity_error < 1e-7 and identity_error == 0.
    write_json(stage/'initial_identity.json', dict(latent_error=identity_error, reflection_error=parity_error))
    refs, records = {}, []
    selected = {r['path'] for r in retained}
    for saved, trace in zip(source['results'], source['traces']):
        path = saved['original_path']
        if path not in selected:
            continue
        with np.load(BASE/trace) as z:
            ref, rows = z['reference'].copy(), z['rows'].copy()
        refs[path] = ref
        r = dict(path=path, trace='../'+trace.replace('\\', '/'), success=saved['success'],
            reason=saved['reason'], audit=saved['audit'], geometry=saved['geometry'],
            complete_time_s=saved['completion_time_s'], motion=complete_motion_metrics(ref, rows),
            smooth_cruise99_s=smooth_cruise(rows))
        if len(ref) == 3:
            r.update(shape=complete_corner_metrics(ref, rows), target_motion=target_motion(ref, rows))
        records.append(r)
    records.sort(key=lambda r:r['path'])
    refs = {r['path']:refs[r['path']] for r in records}
    summary = aggregate(records)
    summary['smooth_curved_cruise99_s'] = sum(r['smooth_cruise99_s'] for r in records if r['path'] not in (48,57))
    current = dict(tag='reused_retained86', coefficients=np.zeros(dimension).tolist(),
                   records=records, summary=summary, primary_preserved=True)
    current['objective'] = float(np.maximum(growth_deficits(current, retained), 0.)@np.maximum(growth_deficits(current, retained), 0.))
    current['hard_objective'] = float(np.linalg.norm(residual(current, retained))**2)
    current['promotion'] = promotion(records, retained)
    candidates, iterations = [current], []
    cache = {np.zeros(dimension, dtype=np.float32).tobytes(): current}
    pool = mp.get_context('spawn').Pool(4, initialize_worker,
        (stage, spec, refs, 'gate_band_canonical_kernel', started+wall_s-10))
    steps = np.full(dimension, .25)
    def save():
        write_json(stage/'results.json', dict(candidates=candidates, iterations=iterations,
            best=current['tag'], adopted=False, goal_complete=False))
    def evaluate(jobs):
        output, pending = {}, []
        for tag, coeff in jobs:
            key = np.asarray(coeff, dtype=np.float32).tobytes()
            if key in cache:
                output[tag] = cache[key]
            else:
                pending.append((tag, coeff))
        for result in pool.imap_unordered(evaluate_growth, pending):
            result['hard_objective'] = float(np.linalg.norm(residual(result, retained))**2)
            if all(r['success'] for r in result['records']):
                v = np.maximum(growth_deficits(result, retained), 0.)
                result['objective'] = float(v@v)
            else:
                result['objective'] = result['hard_objective']
            result['promotion'] = promotion(result['records'], retained)
            result['primary_preserved'] = all(result['promotion'][k] for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))
            candidates.append(result); output[result['tag']] = result
            cache[np.asarray(result['coefficients'], dtype=np.float32).tobytes()] = result
            save()
            if len(candidates)%12 == 0 or 'proposal' in result['tag']:
                print('RETAINED_GROWTH',result['tag'],round(result['objective'],5),result['summary'],result['primary_preserved'],flush=True)
        return output
    try:
        save()
        for it in range(maximum_iterations):
            if it and time.perf_counter() > started+170:
                break
            coeff = np.asarray(current['coefficients'])
            measured = evaluate([(f'iter{it}_axis{axis}_{sign:+d}', coeff+sign*steps[axis]*np.eye(dimension)[axis])
                for axis in range(dimension) for sign in (-1,1)])
            paired = [(measured[f'iter{it}_axis{axis}_-1'], measured[f'iter{it}_axis{axis}_+1']) for axis in range(dimension)]
            base, matrix, bounds, modes, _ = complete_metric_response(current, paired, retained)
            hard_cruise_response = matrix[-1].copy()
            base[-1] = growth_deficits(current, retained)[-1]
            for axis, (lo,hi) in enumerate(paired):
                low,high = lo['summary']['smooth_curved_cruise99_s'],hi['summary']['smooth_curved_cruise99_s']
                nominal = current['summary']['smooth_curved_cruise99_s']
                matrix[-1,axis] = -5*((high-low)/2 if low is not None and high is not None else
                    high-nominal if high is not None else nominal-low if low is not None else 0.)
            models = build_trace_response(stage,current,paired)
            current_support = support
            if current['tag'] != 'reused_retained86':
                current_support = np.vstack([np.load(stage/r['trace'])['observations'] for r in current['records']])
            features = state_function_features(policy,current_support)
            trust = (features*steps[None,None,:]/np.array([.005,.6 if args.compact else .25,.03])[None,:,None]).reshape(-1,dimension)
            margin,geometry,_ = build_geometric_response(stage,current,paired)
            entry,entry_jac = entry_constraints(models)
            def objective(u):
                signed,jac = trace_deficits(u,base,matrix,models,current,retained)
                value=np.maximum(signed,0.)
                return float(value@value+.01*(u@u)),2*jac.T@value+.02*u
            constraints=[dict(type='ineq',fun=lambda u:np.r_[1-trust@u,1+trust@u],jac=lambda u:np.vstack([-trust,trust])),
                         dict(type='ineq',fun=lambda u:(margin-geometry@u)/.01,jac=lambda u:-geometry/.01)]
            if len(entry):
                constraints.append(dict(type='ineq',fun=lambda u:(entry+entry_jac@u)/.001,jac=lambda u:entry_jac/.001))
            sol=minimize(objective,np.zeros(dimension),jac=True,method='SLSQP',bounds=bounds,
                constraints=constraints,options=dict(maxiter=250,ftol=1e-8))
            audit=dict(iteration=it,previous=current['tag'],previous_objective=current['objective'],
                solver_success=bool(sol.success),message=sol.message,predicted_objective=objective(sol.x)[0]-.01*(sol.x@sol.x),
                hard_cruise_response=hard_cruise_response.tolist(),smooth_cruise_response=matrix[-1].tolist(),
                response_modes=modes,normalized_step=sol.x.tolist())
            iterations.append(audit)
            write_json(stage/f'proposal_{it}.json',audit)
            print('RETAINED_GROWTH_SOLVE',it,audit['predicted_objective'],sol.message,flush=True)
            if not sol.success:
                save();break
            proposed=evaluate([(f'iter{it}_proposal_{scale}',coeff+scale*steps*sol.x) for scale in (.5,1.)])
            eligible=[r for r in list(measured.values())+list(proposed.values()) if r['primary_preserved'] and r['objective']<current['objective']-1e-5]
            if not eligible:
                audit['accepted']=False;save();break
            current=min(eligible,key=lambda r:r['objective'])
            audit.update(accepted=True,selected=current['tag'],selected_objective=current['objective'])
            save()
            print('RETAINED_GROWTH_ACCEPT',current['tag'],current['summary'],flush=True)
        pool.close();pool.join();pool=None
        torch.save(dict(architecture=spec['kind'],policy_spec=spec,
            coefficients=np.asarray(current['coefficients']),environment_architecture='gate_band_canonical_kernel',
            learning_method='complete_policy_joint_growth_with_smooth_motion_credit'),stage/'research_candidate.pt')
        save()
        receipt(stage,dict(status='completed_retained_joint_growth_learning',new_candidates=len(candidates)-1,
            best=current['tag'],iterations=len(iterations),wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
    except Exception as exc:
        if pool is not None:pool.terminate();pool.join()
        save();receipt(stage,dict(status='failed_retained_joint_growth_learning',error=str(exc),
            new_candidates=len(candidates)-1,wall_s=time.perf_counter()-started,adopted=False,goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable,'-B',str(BASE/'record_joint_live_status.py'),
            '--kind','complete_closed_loop_parameter_optimization_finished',
            '--next','核实保留模型起点的联合学习是否同时保住形状并产生真实巡航；未达标继续机制修正。'],
            cwd=BASE,check=True,stdout=subprocess.DEVNULL)


if __name__=='__main__':
    mp.freeze_support();main()
