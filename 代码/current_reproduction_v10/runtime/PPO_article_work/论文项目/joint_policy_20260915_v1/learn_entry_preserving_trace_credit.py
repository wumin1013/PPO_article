"""Recenter signed-trace parameter learning with explicit entry constraints."""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from complete_local_credit import complete_metric_response, state_function_features
from raw_state_joint_policy import RawStateJointPolicy
from learn_trace_resolved_joint_credit import build_trace_response, trace_deficits
from learn_coupled_feasibility_credit import build_geometric_response
from optimize_complete_joint_function import initialize_worker, evaluate_job, residual
from learn_nonlinear_joint_feedback import promotion
from study_io import BASE, register, receipt, write_json
from search_complete_return import read


def entry_constraints(models, tolerance=1e-8):
    # The constraints use signed positions, avoiding the zero derivative of an
    # inactive maximum excursion. They govern proposals only, never execution.
    baseline, jacobian = [], []
    for m in models.values():
        b, j = m['base']['incoming'], m['jac']['incoming']
        # Box bounds cannot change the sign of omitted rows, even at full step.
        active = (b-np.abs(j).sum(1) < 0.) & (np.abs(j).max(1) > 1e-14)
        baseline.append(b[active]+tolerance)
        jacobian.append(j[active])
    return np.concatenate(baseline), np.vstack(jacobian)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditioned', action='store_true')
    args = parser.parse_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    parent = BASE/('canonical_gate_band_entry_preserving_trace_credit_20260919' if args.conditioned
                  else 'canonical_gate_band_trace_resolved_joint_credit_20260919')
    data = read(parent/'results.json')
    prior = next(r for r in data['candidates'] if r['tag'] == data['best'])
    model = torch.load(parent/'research_candidate.pt', map_location='cpu', weights_only=False)
    spec = model['policy_spec']
    dimension, wall_s, maximum_iterations = 48, 360, 2
    basis_audit = None
    if args.conditioned:
        from conditioned_joint_policy import ConditionedJointPolicy, condition_basis
        support = []
        for r in prior['records']:
            with np.load(parent/r['trace']) as z:
                support.append(z['observations'].copy())
        transform, basis_audit = condition_basis(model, np.vstack(support))
        spec = dict(kind='conditioned_raw_state', parent=model, transform=transform)
        dimension, wall_s, maximum_iterations = transform.shape[1], 420, 1
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    stage = register('canonical_gate_band_conditioned_joint_credit_20260919' if args.conditioned
                     else 'canonical_gate_band_entry_preserving_trace_credit_20260919',
        [__file__, BASE/'learn_trace_resolved_joint_credit.py', BASE/'complete_local_credit.py',
         BASE/'raw_state_joint_policy.py', BASE/'learn_coupled_feasibility_credit.py',
         BASE/'optimize_complete_joint_function.py', parent/'results.json', parent/'receipt.json',
         parent/'research_candidate.pt']+([BASE/'conditioned_joint_policy.py'] if args.conditioned else []),
        dict(workers=4, maximum_iterations=maximum_iterations, dimensions=dimension, response_step=.125,
             proposal_scales=[.5, 1.], maximum_new_candidates=maximum_iterations*(2*dimension+2), paths=20, wall_s=wall_s),
        ('The observed raw48-parameter state-function matrix had condition number '
         'about3.5million, with two nearly null steering directions and weak steering '
         'support near some reverse-motion peaks. Test optimizer coordinate conditioning '
         'within the SAME learned raw-state function span, using separate-head SVD '
         'of current visited-state features, dropping only singular directions below '
         '0.001 of that head peak and normalizing each direction by its maximum '
         'observed action feature. This is a change of parameter credit/exploration '
         'coordinates, not a new observation, action label, trajectory or speed rule. '
         'Verify exact zero-increment policy identity and reflection parity. Fresh '
         'complete-policy differences are necessary because previously tiny raw '
         'directions can now have material closed-loop effects. One bounded round; '
         'stop if actual primary-preserving loss fails to improve. ' if args.conditioned else '')+
        'Signed physical trace credit produced a primary-preserving improvement, '
        'but the full step traded about5nm incoming excursion for lower aggregate '
        'loss. Do not loosen the existing1e-6mm screening tolerance. Recenter on '
        'the actually completed half-step candidate and add explicit signed '
        'incoming-half-plane constraints to parameter proposals with1e-8mm model '
        'tolerance. The sum-of-deficits objective and executor stay unchanged. '
        'At most two complete-policy response/update rounds, with all3 actions '
        'learning together. Use complete response sides only for efficiency and '
        'shape; sampled geometry of failed prefixes is diagnostic only. Cache '
        'actual float32 coefficient vectors. Every accepted research checkpoint '
        'must preserve20/20 completion,16 first-inside turns and incoming shape, '
        'and decrease the declared original objective. No pure offline fit, PPO '
        'claim, rollout interpolation deployment, template or speed rule. If '
        'no accepted improvement, stop this local update mechanism and reassess '
        'the representation. All retained shape conditions plus64-path check '
        'remain prerequisites for adoption; sealed tests remain closed.')
    started = time.perf_counter()
    subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
        '--active', stage.name, '--kind', 'complete_closed_loop_parameter_optimization',
        '--next', '在新候选上重估三动作闭环响应；入边保持作为提案约束，按登记预算执行。'],
        cwd=BASE, check=True, stdout=subprocess.DEVNULL)
    torch.save(spec, stage/'policy_spec.pt')
    if basis_audit is not None:
        write_json(stage/'basis_audit.json', basis_audit)
    # Reuse prior trajectories by explicit relative paths, with no rerun/copy.
    current = dict(prior, tag='reused_trace_half', reused_from=parent.name,
                   records=[dict(r, trace='../'+parent.name+'/'+r['trace']) for r in prior['records']])
    if args.conditioned:
        current.update(tag='reused_conditioned_parent', coefficients=np.zeros(dimension).tolist())
    refs = {}
    for r in current['records']:
        with np.load(stage/r['trace']) as z:
            refs[r['path']] = z['reference'].copy()
    candidates, iterations = [current], []
    cache = {np.asarray(current['coefficients'], dtype=np.float32).tobytes(): current}
    policy = (ConditionedJointPolicy(spec['parent'], spec['transform']) if args.conditioned else
              RawStateJointPolicy(spec['parent'], spec['basis']))
    if args.conditioned:
        from reflection_policy import reflect_observation
        samples = torch.as_tensor(np.vstack(support)[::7], dtype=torch.float32)
        error = max(float(abs(policy.act(x.numpy())[0]-policy.base.act(x.numpy())[0]).max()) for x in samples)
        f = policy.function_features(samples)
        mirrored = policy.function_features(reflect_observation(samples))
        reflection_error = float(abs(mirrored-f*torch.tensor([-1., 1., 1.])[None, :, None]).max())
        assert error == 0. and reflection_error < 1e-7
        write_json(stage/'initial_identity.json', dict(max_latent_error=error, reflection_error=reflection_error))
    pool = mp.get_context('spawn').Pool(4, initialize_worker,
        (stage, spec, refs, 'gate_band_canonical_kernel', started+wall_s-10))
    steps = np.full(dimension, .125)
    def save():
        write_json(stage/'results.json', dict(candidates=candidates, iterations=iterations,
            best=current['tag'], adopted=False, goal_complete=False))
    def evaluate(jobs):
        output, pending = {}, []
        for tag, coefficients in jobs:
            key = np.asarray(coefficients, dtype=np.float32).tobytes()
            if key in cache:
                output[tag] = dict(cache[key], reused_candidate=cache[key]['tag'])
            else:
                pending.append((tag, coefficients))
        for result in pool.imap_unordered(evaluate_job, pending):
            result['objective'] = float(np.linalg.norm(residual(result, retained))**2)
            result['promotion'] = promotion(result['records'], retained)
            result['primary_preserved'] = all(result['promotion'][k]
                for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))
            candidates.append(result)
            cache[np.asarray(result['coefficients'], dtype=np.float32).tobytes()] = result
            output[result['tag']] = result
            save()
            if 'proposal' in result['tag'] or len(candidates)%24 == 0:
                print('ENTRY_TRACE', result['tag'], round(result['objective'], 5),
                      result['summary'], result['primary_preserved'], flush=True)
        return output
    try:
        for it in range(maximum_iterations):
            if time.perf_counter() > started+(wall_s-10)/2 and it > 0:
                print('BUDGET_RESERVE_NEXT_ITERATION', flush=True)
                break
            coeff = np.asarray(current['coefficients'])
            jobs = [(f'iter{it}_axis{axis}_{sgn:+d}', coeff+sgn*steps[axis]*np.eye(dimension)[axis])
                    for axis in range(dimension) for sgn in (-1, 1)]
            results = evaluate(jobs)
            paired = [(results[f'iter{it}_axis{axis}_-1'], results[f'iter{it}_axis{axis}_+1']) for axis in range(dimension)]
            base, matrix, bounds, modes, _ = complete_metric_response(current, paired, retained)
            models = build_trace_response(stage, current, paired)
            support = []
            for r in current['records']:
                with np.load(stage/r['trace']) as z:
                    support.append(z['observations'].copy())
            features = state_function_features(policy, np.vstack(support))
            trust = (features*steps[None, None, :]/np.array([.0025, .0125, .0125])[None, :, None]).reshape(-1, dimension)
            margin, geometry, _ = build_geometric_response(stage, current, paired)
            entry, entry_jac = entry_constraints(models)
            value0, _ = trace_deficits(np.zeros(dimension), base, matrix, models, current, retained)
            assert abs(value0-base).max() < 1e-8
            def objective(u):
                signed, jac = trace_deficits(u, base, matrix, models, current, retained)
                v = np.maximum(signed, 0.)
                return float(v@v+.01*(u@u)), 2*jac.T@v+.02*u
            constraints = [dict(type='ineq', fun=lambda u: np.r_[1-trust@u, 1+trust@u],
                                jac=lambda u: np.vstack([-trust, trust])),
                           dict(type='ineq', fun=lambda u: (margin-geometry@u)/.01,
                                jac=lambda u: -geometry/.01)]
            if len(entry):
                constraints.append(dict(type='ineq', fun=lambda u:(entry+entry_jac@u)/.001,
                                        jac=lambda u:entry_jac/.001))
            sol = minimize(objective, np.zeros(dimension), jac=True, method='SLSQP', bounds=bounds,
                constraints=constraints, options=dict(maxiter=250, ftol=1e-8))
            delta = sol.x
            audit = dict(iteration=it, previous=current['tag'], previous_objective=current['objective'],
                solver_success=bool(sol.success), message=sol.message,
                predicted_objective=objective(delta)[0]-.01*(delta@delta),
                maximum_function_change=float(abs(trust@delta).max()),
                incoming_constraints=len(entry), minimum_predicted_incoming_mm=float((entry+entry_jac@delta).min()) if len(entry) else None,
                response_modes=modes, normalized_step=delta.tolist())
            iterations.append(audit)
            write_json(stage/f'proposal_{it}.json', audit)
            print('ENTRY_TRACE_SOLVE', it, audit['predicted_objective'], sol.message, len(entry), flush=True)
            if not sol.success:
                save(); break
            measured = evaluate([(f'iter{it}_proposal_{s}', coeff+s*steps*delta) for s in (.5, 1.)])
            eligible = [r for r in list(results.values())+list(measured.values())
                        if r.get('primary_preserved') and r['objective'] < current['objective']-1e-5]
            if not eligible:
                audit['accepted'] = False
                save(); break
            current = min(eligible, key=lambda r:r['objective'])
            audit.update(accepted=True, selected=current['tag'], selected_objective=current['objective'])
            save()
            print('ENTRY_TRACE_ACCEPT', current['tag'], current['objective'], current['summary'], flush=True)
        pool.close(); pool.join(); pool = None
        torch.save(dict(model, architecture='conditioned_joint_policy' if args.conditioned else model['architecture'],
            policy_spec=spec, coefficients=np.asarray(current['coefficients']),
            learning_method='entry_preserving_signed_trace_complete_parameter_optimization'),
            stage/'research_candidate.pt')
        save()
        receipt(stage, dict(status='completed_entry_preserving_trace_learning',
            new_candidates=len(candidates)-1, iterations=len(iterations), best=current['tag'],
            objective=current['objective'], wall_s=time.perf_counter()-started,
            adopted=False, goal_complete=False))
    except Exception as exc:
        if pool is not None:
            pool.terminate(); pool.join()
        save()
        receipt(stage, dict(status='failed_entry_preserving_trace_learning', error=str(exc),
            new_candidates=len(candidates)-1, wall_s=time.perf_counter()-started,
            adopted=False, goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
            '--kind', 'complete_closed_loop_parameter_optimization_finished',
            '--next', '比较带入边约束的逐时刻信用学习实测收益；未达标时继续机制修正，不替换保留模型。'],
            cwd=BASE, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    mp.freeze_support()
    main()
