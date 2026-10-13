"""Actual complete-policy finite differences and bounded joint parameter learning."""
import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import lsq_linear
from canonical_gate_state import checkpoint_environment_factory
from complete_joint_function_policy import CompleteJointFunctionPolicy
from learn_nonlinear_joint_feedback import rollout, aggregate, promotion
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha

WORKER = {}
STEP = np.array([.15, .3, .3, .4, .08])


def initialize_worker(stage, policy_spec, refs, architecture, deadline):
    torch.set_num_threads(1)
    if policy_spec['kind'] == 'retained_compact_growth':
        from compact_joint_growth_policy import CompactJointGrowthPolicy
        policy = CompactJointGrowthPolicy(policy_spec['retained'], policy_spec['feed'],
                                         policy_spec['basis'], policy_spec['transform'])
    elif policy_spec['kind'] == 'retained_joint_growth':
        from retained_joint_growth_policy import RetainedJointGrowthPolicy
        policy = RetainedJointGrowthPolicy(policy_spec['retained'], policy_spec['feed'],
                                          policy_spec['basis'], policy_spec['transform'])
    elif policy_spec['kind'] == 'conditioned_raw_state':
        from conditioned_joint_policy import ConditionedJointPolicy
        policy = ConditionedJointPolicy(policy_spec['parent'], policy_spec['transform'])
    elif policy_spec['kind'] == 'raw_state':
        from raw_state_joint_policy import RawStateJointPolicy
        policy = RawStateJointPolicy(policy_spec['parent'], policy_spec['basis'])
    elif policy_spec['kind'] == 'shared_feature':
        from shared_feature_joint_policy import SharedFeatureJointPolicy
        policy = SharedFeatureJointPolicy(policy_spec['parent'], policy_spec['basis'])
    else:
        policy = CompleteJointFunctionPolicy(policy_spec['joint'], policy_spec['feed'])
    WORKER.update(stage=stage, policy=policy,
                  refs=refs, factory=checkpoint_environment_factory(architecture), deadline=deadline)


def evaluate_job(job):
    tag, coeff = job
    stage, policy = WORKER['stage'], WORKER['policy']
    with torch.no_grad():
        policy.coefficients.copy_(torch.tensor(coeff, dtype=torch.float32))
    folder = stage/tag
    folder.mkdir()
    costs = dict(actor_calls=0, physics_ticks=0)
    records = []
    for path, ref in WORKER['refs'].items():
        result, rows, actions, observations = rollout(policy, WORKER['factory'], ref,
                                                      costs, WORKER['deadline'])
        np.savez_compressed(folder/f'path{path}.npz', reference=ref, rows=rows,
                            actions=actions, observations=observations)
        result.update(path=path, trace=f'{tag}/path{path}.npz')
        records.append(result)
    return dict(tag=tag, coefficients=np.asarray(coeff).tolist(), records=records,
                summary=aggregate(records), costs=costs)


def residual(candidate, retained):
    if all(r['success'] for r in candidate['records']):
        from complete_local_credit import signed_deficits
        return np.maximum(signed_deficits(candidate, retained), 0.)
    old = {r['path']: r for r in retained}
    values = []
    for r in candidate['records']:
        baseline = old[r['path']]
        values.append(20.*(not r['success']))
        # Only completed trajectories contribute time or cruise evidence.
        if 'shape' in r:
            s, b = r['shape'], baseline['baseline_shape']
            for key, scale in [('in_edge_outward_mm', .002), ('out_edge_outward_mm', .005),
                               ('full_physical_reverse_deg', .5), ('full_normalized_energy', .02),
                               ('full_spatial_peak_curvature_mm_inv', .2)]:
                values.append(max(s[key]-b[key], 0.)/scale)
            values.append(max(b['apex_speed_mm_s']-s['apex_speed_mm_s'], 0.)/2.)
            values.append(10.*(r['target_motion']['first_turn_inside'] is not True))
        if r['success']:
            base_time = baseline['complete_time_s']-baseline['complete_time_change_s']
            values.append(max(r['complete_time_s']-base_time, 0.)/.02)
        else:
            values.append(20.)
    summary = candidate['summary']
    values.append(max(3.-summary['curved_cruise99_s'], 0.)*5.
                  if summary['curved_cruise99_s'] is not None else 20.)
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-expansion', action='store_true')
    parser.add_argument('--local-calibration', action='store_true')
    parser.add_argument('--raw-local-state', action='store_true')
    args = parser.parse_args()
    uses_shared = args.feature_expansion or args.local_calibration or args.raw_local_state
    calibrated = args.local_calibration or args.raw_local_state
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    source = BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919'
    learned = BASE/'canonical_gate_band_nonlinear_joint_feedback_20260919'
    baseline = read(source/'results.json')['records']
    joint_md = torch.load(learned/'joint_1.0.pt', map_location='cpu', weights_only=False)
    feed_md = torch.load(learned/'feed_only_1.0.pt', map_location='cpu', weights_only=False)
    extra_sources = []
    prior = next(c for c in read(learned/'results.json')['candidates']
                 if c['arm'] == 'feed_only' and c['scale'] == 2.)
    coeff = np.array([0., 0., 0., 2., 1.])
    policy_spec = dict(kind='combined', joint=joint_md['policy'], feed=feed_md['policy'])
    steps, max_iterations, wall_s = STEP, 4, 600
    stage_name = 'canonical_gate_band_complete_joint_function_learning_20260919'
    prior_root = learned
    if args.feature_expansion:
        from shared_feature_joint_policy import construct_basis
        prior_root = BASE/stage_name
        parent = torch.load(prior_root/'research_candidate.pt', map_location='cpu', weights_only=False)
        prior_result = read(prior_root/'results.json')
        prior = next(r for r in prior_result['candidates'] if r['tag'] == prior_result['best'])
        support = []
        for r in prior['records']:
            with np.load(prior_root/r['trace']) as z:
                support.append(z['observations'].copy())
        basis, basis_audit = construct_basis(parent, np.vstack(support))
        policy_spec = dict(kind='shared_feature', parent=parent, basis=basis)
        coeff = np.zeros(14)
        steps, max_iterations, wall_s = np.full(14, .5), 3, 900
        stage_name = 'canonical_gate_band_shared_feature_joint_learning_20260919'
        extra_sources = [BASE/'shared_feature_joint_policy.py', prior_root/'research_candidate.pt',
                         prior_root/'results.json', prior_root/'receipt.json']
    if args.local_calibration:
        prior_root = BASE/'canonical_gate_band_shared_feature_joint_learning_20260919'
        previous = torch.load(prior_root/'research_candidate.pt', map_location='cpu', weights_only=False)
        policy_spec = previous['policy_spec']
        parent, basis = policy_spec['parent'], policy_spec['basis']
        coeff = np.asarray(previous['coefficients']).copy()
        prior_result = read(prior_root/'results.json')
        prior = next(r for r in prior_result['candidates'] if r['tag'] == prior_result['best'])
        basis_audit = read(prior_root/'basis_audit.json')
        steps, max_iterations, wall_s = np.full(14, .125), 4, 900
        stage_name = 'canonical_gate_band_complete_local_credit_learning_20260919'
        extra_sources = [BASE/'shared_feature_joint_policy.py', BASE/'complete_local_credit.py',
            prior_root/'research_candidate.pt', prior_root/'results.json', prior_root/'receipt.json',
            BASE/'canonical_gate_band_joint_target_coordinate_probe_20260919/results.json']
    if args.raw_local_state:
        from raw_state_joint_policy import construct_raw_basis
        prior_root = BASE/'canonical_gate_band_complete_local_credit_learning_20260919'
        prior_result = read(prior_root/'results.json')
        feasible = [r for r in prior_result['candidates'] if 'promotion' in r and
                    all(r['promotion'][k] for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))]
        prior = min(feasible, key=lambda r: r['objective'])
        parent = torch.load(prior_root/'research_candidate.pt', map_location='cpu', weights_only=False)
        parent['coefficients'] = np.asarray(prior['coefficients'])
        support = []
        for r in prior['records']:
            with np.load(prior_root/r['trace']) as z:
                support.append(z['observations'].copy())
        basis, basis_audit = construct_raw_basis(np.vstack(support), count=16)
        policy_spec = dict(kind='raw_state', parent=parent, basis=basis)
        coeff = np.zeros(48)
        steps, max_iterations, wall_s = np.full(48, .125), 2, 900
        stage_name = 'canonical_gate_band_raw_state_joint_learning_20260919'
        extra_sources = [BASE/'raw_state_joint_policy.py', BASE/'shared_feature_joint_policy.py',
            BASE/'complete_local_credit.py', prior_root/'research_candidate.pt',
            prior_root/'results.json', prior_root/'receipt.json']
    dimension = len(coeff)
    stage = register(stage_name,
        [__file__, BASE/'complete_joint_function_policy.py', BASE/'nonlinear_joint_adapter.py',
         BASE/'learn_nonlinear_joint_feedback.py', source/'results.json', source/'receipt.json',
         learned/'results.json', learned/'receipt.json', learned/'joint_1.0.pt', learned/'feed_only_1.0.pt']+extra_sources,
        dict(workers=4, coefficient_dimensions=dimension, finite_steps=steps.tolist(),
             maximum_iterations=max_iterations, maximum_candidates=max_iterations*(2*dimension+3),
             paths=[r['path'] for r in baseline], wall_s=wall_s, initial_coefficients=coeff.tolist()),
        ('The shared features were inherited from feed fitting, then adapted on only12 '
         'counterfactual states. Their14-parameter complete-policy updates improved exit '
         'but retained reversal/curvature deficits and sometimes sacrificed incoming '
         'shape. Test a NEW residual representation on local continuous physical and '
         'geometric inputs, with16 radial basis centers chosen by observable-state '
         'coverage modulo reflection. The added residual excludes global progress, '
         'time, gate ordinal and prior actions; the frozen parent input structure '
         'is unchanged. All48 three-head coefficients learn from actual complete-policy '
         'outcomes, without action/speed labels. This is an exploratory representation '
         'change, not a matched architecture causal claim. Start from the already '
         'completed minimum-loss candidate preserving all16 first-inside turns and '
         'retained incoming half-plane shape. Maintain those primary conditions on '
         'EVERY accepted new research checkpoint, alongside all20 complete successes. '
         'Individual exit/reversal/energy/apex tests still govern adoption. No rule, '
         'per-angle template or query-state switch is deployed. Use the complete-only '
         'signed local response and training-only function trust region. ' if args.raw_local_state else
         'Large whole-policy finite steps crossed completion boundaries and produced '
         'unreliable linear proposals. Keep the SAME14-parameter state function space '
         'but change credit estimation/optimization: quarter-sized finite response '
         'neighborhood, retain signed inactive metric margins, never differentiate '
         'failed-prefix shape/time as complete outcomes, use complete one-sided '
         'responses with excluded failed directions, and impose a parameter-step '
         'function trust region across every current saved observation. This trust '
         'region is ONLY a training proposal constraint, not deployed action clipping. '
         'No guide-compensation deployment: the matched physical probe had mixed '
         'shape effects despite cancelling initial target rotation. ' if args.local_calibration else
         'Four iterations in the five-function space reduced the measured loss from788 '
         'to135 but left substantial individual shape violations. Expand state-dependent '
         'three-head output weights using four odd and four even principal directions '
         'of the existing shared nonlinear hidden state, plus even biases. PCA uses '
         'only current visited observation covariance; no new speed/action labels. '
         'All14 coefficients are learned solely from complete-policy physical outcomes. '
         'Zero residual exactly preserves the previous candidate. No unchanged-volume '
         'extension of the restricted five-function model. ' if args.feature_expansion else '')+
        'Model-based parameter optimization from freshly measured COMPLETE nonlinear '
        'policy outcomes, not PPO or saved-action regression. Learned coefficients '
        'combine nonlinear steering/feed/preview feature functions; they are optimized '
        'jointly without path-specific values, thresholds, templates or speed schedules. '
        'Initialize exactly at the saved preceding candidate and reuse its20 '
        'finished paths. Central full-policy perturbations measure state feedback, '
        'all future actions, physical constraints, ordered completion and terminal rest. '
        'Bound the linear least-squares proposal and accept only a freshly evaluated '
        'complete-path reduction in a declared shape-regression/cruise deficit objective. '
        'Failure has a positive penalty and no time/cruise score. Three backtracking '
        'opportunities each iteration. Four independent local simulator worker processes '
        'are not AI agents. Stop the local model if no reduction; do not blindly increase '
        'training. All promotions still require the stricter individual retained86cd '
        'shape conditions. Preserve raw traces, parameters and all negative outcomes; '
        'no deployment, no sealed or independent-generalization tuning. Reuse cached '
        'coefficient vectors by their actual float32 representation, including revisits.')
    started = time.perf_counter()
    refs = {}
    for r in baseline:
        with np.load(source/r['trace']) as z:
            refs[r['path']] = z['reference'].copy()
    current = dict(tag='reused_parent', coefficients=coeff.tolist(), records=prior['records'],
                   summary=prior['summary'], reused_from=prior_root.name)
    history = [current]
    iterations = []
    jobs_done = 0
    costs = dict(actor_calls=0, physics_ticks=0)
    # Algebraic identity is checked against the saved source network, not rerunning paths.
    check = CompleteJointFunctionPolicy(joint_md['policy'], feed_md['policy'])
    from nonlinear_joint_adapter import NonlinearJointAdapter
    old = NonlinearJointAdapter()
    old.load_state_dict(torch.load(learned/'feed_only_2.0.pt', map_location='cpu', weights_only=False)['policy'])
    max_error = 0.
    if uses_shared:
        from shared_feature_joint_policy import SharedFeatureJointPolicy
        if args.raw_local_state:
            from raw_state_joint_policy import RawStateJointPolicy
            old_spec = parent['policy_spec']
            old = SharedFeatureJointPolicy(old_spec['parent'], old_spec['basis'])
            with torch.no_grad():
                old.coefficients.copy_(torch.as_tensor(parent['coefficients']))
            check = RawStateJointPolicy(parent, basis)
        elif args.local_calibration:
            old = SharedFeatureJointPolicy(parent, basis)
            check = SharedFeatureJointPolicy(parent, basis)
            with torch.no_grad():
                old.coefficients.copy_(torch.as_tensor(coeff))
                check.coefficients.copy_(torch.as_tensor(coeff))
        else:
            old = CompleteJointFunctionPolicy(parent['joint_state'], parent['feed_state'])
            with torch.no_grad():
                old.coefficients.copy_(torch.as_tensor(parent['coefficients']))
            check = SharedFeatureJointPolicy(parent, basis)
        torch.save(policy_spec, stage/'policy_spec.pt')
        write_json(stage/'basis_audit.json', basis_audit)
    with torch.no_grad():
        for p in (14, 31, 57):
            trace = next(r['trace'] for r in prior['records'] if r['path'] == p)
            with np.load(prior_root/trace) as z:
                obs = z['observations']
                saved_actions = z['actions'].copy()
            decode = checkpoint_environment_factory(joint_md['environment_architecture'])(refs[p]).decode_action
            for k, x in enumerate(obs):
                max_error = max(max_error, float(abs(check.act(x)[0]-old.act(x, True)[0]).max()))
                assert np.array_equal(decode(check.act(x)[0]), saved_actions[k])
    assert max_error == 0., 'Reused completed baseline requires exact actor identity'
    write_json(stage/'initial_identity.json', dict(max_latent_float32_error=max_error))
    pool = mp.get_context('spawn').Pool(4, initialize_worker,
        (stage, policy_spec, refs, joint_md['environment_architecture'], started+wall_s-10))
    cache = {np.asarray(coeff, dtype=np.float32).tobytes(): current}

    def evaluate(jobs):
        nonlocal jobs_done
        results, pending = [], []
        for tag, value in jobs:
            key = np.asarray(value, dtype=np.float32).tobytes()
            if key in cache:
                results.append(dict(cache[key], tag=tag, reused_candidate=cache[key]['tag']))
            else:
                pending.append((tag, value))
        for result in pool.imap_unordered(evaluate_job, pending):
            result['promotion'] = promotion(result['records'], baseline)
            result['objective'] = float(np.linalg.norm(residual(result, baseline))**2)
            results.append(result)
            cache[np.asarray(result['coefficients'], dtype=np.float32).tobytes()] = result
            history.append(result)
            jobs_done += 1
            for key in costs:
                costs[key] += result['costs'][key]
            write_json(stage/'results.json', dict(candidates=history, iterations=iterations,
                best=current['tag'], costs=costs, adopted=False, goal_complete=False))
            print('COMPLETE_PARAMETER', result['tag'],
                  'loss', round(result['objective'], 4), result['summary'], flush=True)
        return {r['tag']: r for r in results}

    try:
        current['objective'] = float(np.linalg.norm(residual(current, baseline))**2)
        for iteration in range(max_iterations):
            jobs = []
            for head in range(dimension):
                for direction in (-1, 1):
                    value = coeff.copy()
                    value[head] += steps[head]*direction
                    jobs.append((f'iter{iteration}_axis{head}_{direction:+d}', value))
            measured = evaluate(jobs)
            value = residual(current, baseline)
            response_audit = None
            if calibrated:
                from complete_local_credit import state_function_features, solve_local_response
                from shared_feature_joint_policy import SharedFeatureJointPolicy
                support = []
                for r in current['records']:
                    trace = stage/r['trace']
                    if not trace.exists():
                        trace = prior_root/r['trace']
                    with np.load(trace) as z:
                        support.append(z['observations'].copy())
                if args.raw_local_state:
                    from raw_state_joint_policy import RawStateJointPolicy
                    feature_policy = RawStateJointPolicy(parent, basis)
                else:
                    feature_policy = SharedFeatureJointPolicy(parent, basis)
                features = state_function_features(feature_policy, np.vstack(support))
                paired = [(measured[f'iter{iteration}_axis{h}_-1'],
                           measured[f'iter{iteration}_axis{h}_+1']) for h in range(dimension)]
                proposal, predicted_objective, response_audit = solve_local_response(
                    current, paired, baseline, features, steps)
            else:
                columns = []
                for head in range(dimension):
                    lo = measured[f'iter{iteration}_axis{head}_-1']
                    hi = measured[f'iter{iteration}_axis{head}_+1']
                    columns.append((residual(hi, baseline)-residual(lo, baseline))/2.)
                jacobian = np.column_stack(columns)
                solution = lsq_linear(np.vstack([jacobian, .1*np.eye(dimension)]),
                                      np.r_[-value, np.zeros(dimension)], bounds=(-1., 1.), tol=1e-9)
                proposal = solution.x*steps
                predicted_objective = float(np.linalg.norm(value+jacobian@solution.x)**2)
            trials = evaluate([(f'iter{iteration}_proposal_{scale}', coeff+scale*proposal)
                               for scale in (.25, .5, 1.)])
            choices = list(measured.values())+list(trials.values())
            completed = [r for r in choices if r['summary']['successes'] == len(baseline)]
            if args.raw_local_state:
                completed = [r for r in completed if all(r['promotion'][k]
                    for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))]
            best = min(completed, key=lambda r: r['objective']) if completed else current
            accepted = best['objective'] < current['objective']-1e-6
            iterations.append(dict(iteration=iteration, previous=current['tag'],
                previous_objective=current['objective'], proposed_coefficients=(coeff+proposal).tolist(),
                predicted_objective=predicted_objective, response_audit=response_audit,
                selected=best['tag'], selected_objective=best['objective'], accepted=accepted))
            if accepted:
                current = best
                coeff = np.asarray(best['coefficients'])
                checkpoint = dict(architecture='complete_joint_function_policy',
                    joint_state=joint_md['policy'], feed_state=feed_md['policy'],
                    coefficients=coeff, environment_architecture=joint_md['environment_architecture'],
                    learning_method='complete_closed_loop_parameter_optimization',
                    source_model_sha256=[sha(learned/'joint_1.0.pt'), sha(learned/'feed_only_1.0.pt')])
                if uses_shared:
                    checkpoint = dict(architecture='raw_state_joint_policy' if args.raw_local_state else 'shared_feature_joint_policy', policy_spec=policy_spec,
                        coefficients=coeff, environment_architecture=joint_md['environment_architecture'],
                        learning_method='complete_closed_loop_parameter_optimization')
                torch.save(checkpoint, stage/'research_candidate.pt')
            write_json(stage/'results.json', dict(candidates=history, iterations=iterations,
                best=current['tag'], costs=costs, adopted=False, goal_complete=False))
            print('COMPLETE_SELECTION', iterations[-1], flush=True)
            if not accepted:
                break
            if time.perf_counter()-started > wall_s*.75:
                break
        pool.close()
        pool.join()
        receipt(stage, dict(status='completed_joint_full_policy_parameter_learning',
            iterations=len(iterations), new_candidates=jobs_done, costs=costs,
            best=current['tag'], objective=current['objective'],
            eligible=all(promotion(current['records'], baseline).values()),
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
        summary = current['summary']
        (stage/'简短结论.md').write_text(
            f"当前为完整闭环参数优化，不是 PPO 交互训练。\n\n"
            f"新候选 {jobs_done} 个；选中 {current['tag']}；完整成功 {summary['successes']}/{summary['paths']}。\n"
            f"含弯99巡航 {summary['curved_cruise99_s']} s；反向转动总量 {summary['reverse_sum_deg']} deg；"
            f"出边越过总量 {summary['exit_sum_mm']} mm。\n"
            f"采用条件通过：{all(promotion(current['records'], baseline).values())}。保留86cd模型不变，目标未完成。\n"
            "执行链仍含引导器、物理约束投影和制动；未部署探针或速度表。\n",
            encoding='utf-8')
    except Exception as exc:
        pool.terminate()
        pool.join()
        receipt(stage, dict(status='failed_joint_full_policy_parameter_learning', error=str(exc),
            new_candidates=jobs_done, costs=costs, wall_s=time.perf_counter()-started,
            adopted=False, goal_complete=False))
        raise
    finally:
        if args.raw_local_state:
            subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
                '--kind', 'complete_closed_loop_parameter_optimization_finished',
                '--next', '读取最新原始轨迹和完成收据，核实采用条件及真实进程；根据局部状态联合学习结果继续研究，目标未完成，不把预算结束视为研究完成。'],
                cwd=BASE, check=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
