"""Bounded critic-free CEM in a learned actor-feature subspace.

This is policy-parameter learning, not action-sequence templates or online
search. The deployed candidate is the same 20 ms feedback network. The
optimizer and its trainable subspace differ from PPO; causal claims must
retain both changes. All complete/failure trajectories remain available.
"""
import argparse
import copy
import json
import math
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from joint_policy import JointPolicy
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from resume_codec_v5 import load_resume
from train_pilot import save_torch, evaluate
from validate_actions import difference_audit
from study_io import BASE, register, receipt, sha, write_json
from geometric_turn_metrics import geometric_turn_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def actor_parameter_key(coefficients, basis, weight, bias):
    change=np.asarray(coefficients)@basis
    return (np.asarray(weight+change[:,:weight.shape[1]],dtype=np.float32).tobytes(),
            np.asarray(bias+change[:,-1],dtype=np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objective', choices=('time', 'shape_constraint'), default='time')
    parser.add_argument('--reflection', action='store_true')
    parser.add_argument('--broader-curriculum', action='store_true')
    parser.add_argument('--full-covariance', action='store_true')
    parser.add_argument('--regularized-exit', action='store_true')
    parser.add_argument('--spatial-energy', action='store_true')
    parser.add_argument('--dynamic-residual', action='store_true')
    parser.add_argument('--preview-residual', action='store_true')
    parser.add_argument('--workers',type=int,default=1,choices=(1,4))
    args = parser.parse_args()
    constrained = args.objective == 'shape_constraint'
    if args.spatial_energy and not args.regularized_exit:
        raise ValueError('Spatial energy replaces the balanced exit regularizer only')
    if args.dynamic_residual and not args.spatial_energy:
        raise ValueError('Dynamic-state residual contrasts the spatial-objective search')
    if args.preview_residual and not args.dynamic_residual:
        raise ValueError('Preview residual starts from a learned dynamic residual')
    architecture='reflection_preview_residual' if args.preview_residual else 'reflection_dynamic_residual' if args.dynamic_residual else 'reflection_group_average' if args.reflection else 'original'
    if args.reflection and not constrained:
        raise ValueError('This architecture comparison keeps the shape-constrained objective')
    if args.broader_curriculum and not args.reflection:
        raise ValueError('Broader continuation retains the reflected actor')
    if args.full_covariance and not args.broader_curriculum:
        raise ValueError('Full covariance is registered against the broader diagonal control')
    if args.regularized_exit and not args.full_covariance:
        raise ValueError('Balanced objective retains the full-covariance method')
    if args.workers!=1 and not args.regularized_exit:
        raise ValueError('Parallel execution is registered only for the balanced objective')
    reuse_population = (constrained and not args.reflection) or args.full_covariance
    if args.dynamic_residual:reuse_population=False
    train_indices = [4,5,6,8,9,11,12,13,14,15,18,22,26,34,50,53] if args.broader_curriculum else [6,11,18,53]
    training_budget = 100000 if args.broader_curriculum else 32000
    physics_budget = 2300000 if args.broader_curriculum else 1500000
    source = BASE/'ar_conditional_joint_20260916'
    buffer = Path(read(source/'storage.json')['rolling_state_directory'])/'resume_1.pt'
    assert read(source/'receipt.json')['final_sha256'] == sha(source/'final.pt')
    names = ['joint_policy.py', 'joint_runtime.py', 'joint_runtime_v2.py', 'joint_runtime_v3.py',
             'joint_runtime_v5.py', 'terminal_candidate_runtime_v1.py', 'geometry_spec.py',
             'isolated_corner_shape.py', 'validate_actions.py', 'resume_codec_v5.py', 'train_pilot.py']
    original = BASE/'complete_return_search_20260916'
    reuse_origin = BASE/'dynamic_state_residual_learning_20260916' if args.preview_residual else BASE/'spatial_bending_learning_20260916' if args.dynamic_residual else BASE/'balanced_shape_time_learning_20260916' if args.spatial_energy else BASE/'correlated_parameter_search_20260916' if args.regularized_exit else BASE/'broader_shape_learning_20260916' if args.full_covariance else original
    initial_model = BASE/'shape_pareto_candidate_review_20260916/candidate_1.pt' if args.preview_residual else BASE/'reflection_shape_search_20260916/best_generation_5.pt' if args.broader_curriculum else source/'final.pt'
    reuse_sources = [original/n for n in ('receipt.json', 'training_results.json', 'generations.json', 'basis.npz')] if constrained else []
    if args.reflection:
        reuse_sources += [BASE/'reflection_policy.py', BASE/'reflection_policy_probe_20260916/receipt.json',
                          BASE/'reflection_policy_probe_20260916/eval_source/summary.json']
    if args.broader_curriculum:
        reuse_sources += [initial_model, BASE/'reflection_shape_search_20260916/receipt.json',
                          BASE/'reflection_shape_search_20260916/generations.json',
                          BASE/'reflection_shape_search_20260916/validation_5/summary.json']
        if args.preview_residual:
            selected=read(BASE/'shape_pareto_candidate_review_20260916/results.json')['results'][1]
            assert selected['successes']==12 and selected['model_sha256']==sha(initial_model)
        else:
            selected = next(r for r in read(BASE/'reflection_shape_search_20260916/generations.json')['generations'] if r['generation']==5)
            assert selected['validation_successes']==12 and selected['model_sha256']==sha(initial_model)
    if args.full_covariance:
        reuse_sources += [reuse_origin/n for n in ('receipt.json','training_results.json','generations.json','basis.npz')]
        reuse_sources += [BASE/'learned_shape_coordination_probe_20260916/receipt.json',
                          BASE/'learned_shape_coordination_probe_20260916/results.json']
    if args.workers!=1: reuse_sources += [BASE/'parameter_search_worker.py']
    if args.spatial_energy:
        reuse_sources += [BASE/'geometric_turn_metrics.py',BASE/'spatial_metric_resolution_audit_20260916/receipt.json']
    if args.dynamic_residual:
        reuse_sources += [BASE/'dynamic_residual_policy.py',BASE/'gate_observation_effect_audit_20260916/receipt.json',BASE/'gate_feed_jump_probe_20260916/receipt.json']
    if args.preview_residual:
        reuse_sources += [BASE/'wide_turn_reachability_probe_20260916/receipt.json',BASE/'wide_turn_reachability_probe_20260916/results.json',
            BASE/'shape_pareto_candidate_review_20260916/receipt.json',BASE/'shape_pareto_candidate_review_20260916/results.json']
    stage_name = 'preview_state_residual_learning_20260916' if args.preview_residual else 'dynamic_state_residual_learning_20260916' if args.dynamic_residual else 'spatial_bending_learning_20260916' if args.spatial_energy else 'balanced_shape_time_learning_20260916' if args.regularized_exit else 'correlated_parameter_search_20260916' if args.full_covariance else 'broader_shape_learning_20260916' if args.broader_curriculum else 'reflection_shape_search_20260916' if args.reflection else 'shape_constrained_search_20260916' if constrained else 'complete_return_search_20260916'
    stage = register(stage_name,
        [__file__, source/'final.pt', source/'receipt.json', source/'storage.json',
         BASE/'learning_preparation_v4/paths.json', BASE/'reverse_preswing_xy_audit_20260916/results.json']+
        [BASE/n for n in names]+reuse_sources,
        dict(training_decisions=training_budget, evaluation_decisions=12000, physics_ticks=physics_budget,
             wall_s=1200, generations=6, population=12, elites=3, seed=916271,
             train_indices=train_indices, validation_paths=12,
             feature_rank=8, starting_latent_rms=[.05, .12, .25], objective=args.objective,
             architecture=architecture,
             feature_space='PCA8 of existing observation[22:42] preview point vectors; zero-initialized affine residual' if args.preview_residual else 'existing normalized observation[0:8], state-linear learned residual; reflection leaves fewer effective coefficients than nominal27' if args.dynamic_residual else 'frozen hidden-feature PCA8',
             search_covariance='full' if args.full_covariance else 'diagonal',
             exit_violation_seconds_weight=1. if args.regularized_exit and not args.spatial_energy else None,
             spatial_energy=dict(spacing_mm=.1,sensitivity_spacing_mm=.2,seconds_weight=1.,normalization='1.25 mm * integral kappa^2 ds / angle_rad^2; sum isolated training paths',
                adoption='Development only: 12/12 complete; inward first/outward<=0.05 mm; lower energy >=20% at both 125/150deg, reverse turn<=5deg, total time<=1.02*balanced g5. Inspect XY. Cruise remains a separate unresolved requirement.') if args.spatial_energy else None,
             compute_workers=args.workers,
             shape_constraint=dict(outward_tolerance_mm=.05, reverse_turn_tolerance_deg=5.,
                requires_inward_first=True, success_has_priority=True) if constrained else None),
        ('Warm-start the preserved dynamic g3_c5 model and freeze it; learn a separate zero-initialized affine residual from the existing ten future XY points. All three outputs are optimized by complete episodes. Same objective, population and nominal27 coefficients. Warm-start plus feature-space extension is a mechanism refinement, not a single-factor superiority claim. No angle table, intervention timing, fixed-feed rule or per-corner trajectory template is deployed. ' if args.preview_residual else
         'Replace trainable frozen-hidden-feature subspace with a zero-initialized residual on existing error/velocity/acceleration/heading features. Freeze the same source network; use same episode objective, training paths, seed, nominal27 coefficients, optimizer and budget as spatial_bending_learning. Zero residual starts from exactly identical actions. Reflection annihilates some raw linear directions, so effective capacity and induced exploration differ; no pure dimension or state-sufficiency claim. No gate/feed-hold rule is deployed. ' if args.dynamic_residual else
         'Replace only the exit-reversal soft cost in balanced_shape_time_learning with finite-resolution spatial bending energy. Same source, seed, basis, architecture, population, covariance and train paths. Complete success and inward approach retain priority; maximize complete return minus sum of 1.25mm*E/angle^2 at 1 second per unit. Reuse any exact actor match. Stop after three generations without training improvement or registered budget, then analyze. ' if args.spatial_energy else
         'Matched objective contrast to full-covariance search: complete success and inward approach remain higher priorities; maximize complete return minus normalized excess exit-turn penalty at 1 second per unit. Same seed, initial model, paths, covariance and population. Parallel workers are checked for exact deterministic actions; original first population reused. ' if args.regularized_exit else
         'Matched optimizer-covariance-only contrast to broader_shape_learning: same source, seed, basis, objective, paths, elite count, mean/variance rates and candidate budget. Preserve cross-head/cross-feature elite covariance instead of its diagonal; exact first population reused. ' if args.full_covariance else
         'Continue the preserved 12/12 reflected generation-5 model with sixteen existing training paths including sharp turns; same optimization and shape constraints. This is curriculum refinement with a warm start and larger physical budget, not a single-factor superiority claim. ' if args.broader_curriculum else
         'Matched architecture-only contrast to shape-constrained search: same source weights, feature basis, seed, paths, objective and optimizer; exact reflection group averaging changes actor only. ' if args.reflection else
         'Matched objective-only contrast: retain complete-success priority, then minimize isolated-turn shape constraint violation, then full time; reuse exact baseline/first population. ' if constrained else '')+
        'Critic-free learning in training-feature PCA subspace; same observation/reward/controller. Development mechanism pilot, no independent-seed or final-comparison claim.')
    write_json(stage/'external_source.json', dict(path=str(buffer), sha256=sha(buffer)))
    started = time.perf_counter(); deadline = started+1200.
    costs = dict(training_actor_calls=0, evaluation_actor_calls=0,
                 training_physics_ticks=0, evaluation_physics_ticks=0, parameter_updates=0)
    executor = None
    try:
        torch.set_num_threads(1); torch.manual_seed(916271)
        rng = np.random.default_rng(916271)
        incoming = torch.load(initial_model, map_location='cpu', weights_only=False)
        if args.preview_residual:
            from dynamic_residual_policy import PreviewResidualPolicy
            policy=PreviewResidualPolicy(incoming['observation_dim'])
            missing=policy.load_state_dict(incoming['policy'],strict=False)
            assert set(missing.missing_keys)=={'preview_head.weight','preview_head.bias'} and not missing.unexpected_keys
        elif args.dynamic_residual:
            from dynamic_residual_policy import DynamicResidualPolicy
            policy=DynamicResidualPolicy(incoming['observation_dim'])
            missing=policy.load_state_dict(incoming['policy'],strict=False)
            assert set(missing.missing_keys)=={'dynamic_head.weight','dynamic_head.bias'} and not missing.unexpected_keys
        elif args.reflection:
            from reflection_policy import ReflectionPolicy
            policy = ReflectionPolicy(incoming['observation_dim'])
        else:
            policy = JointPolicy(incoming['observation_dim'])
        if not args.dynamic_residual:policy.load_state_dict(incoming['policy'])
        saved = load_resume(buffer)
        obs = torch.tensor(np.asarray(saved['batch']['observation']), dtype=torch.float32)
        assert len(obs) <= 2048 and len(obs) > 100
        with torch.no_grad():
            features = obs[:,22:42].numpy().astype(float) if args.preview_residual else obs[:,:8].numpy().astype(float) if args.dynamic_residual else policy.actor[:4](obs).numpy().astype(float)
        feature_count=features.shape[1]
        average = features.mean(0)
        _, singular, axes = np.linalg.svd(features-average, full_matrices=False)
        transform = axes[:8]/np.maximum(singular[:8]/math.sqrt(len(features)-1), .1)[:, None]
        basis = np.zeros((9, feature_count+1)); basis[:8, :feature_count] = transform
        basis[:8, feature_count] = -transform@average; basis[8, feature_count] = 1.
        np.savez_compressed(stage/'basis.npz', basis=basis, singular=singular, rows=len(features))
        if constrained and not args.dynamic_residual:
            assert np.array_equal(basis, np.load(original/'basis.npz')['basis'])
        previous_candidates = {r['tag']:r for r in read(reuse_origin/'training_results.json')['candidates']} if reuse_population else {}
        previous_generations = read(reuse_origin/'generations.json')['generations'] if reuse_population else []
        trainable_head=policy.preview_head if args.preview_residual else policy.dynamic_head if args.dynamic_residual else policy.actor[-1]
        base_weight=trainable_head.weight.detach().numpy().copy()
        base_bias=trainable_head.bias.detach().numpy().copy()

        def physical_key(coefficients):
            return actor_parameter_key(coefficients,basis,base_weight,base_bias)

        if reuse_population:
            previous_registration=read(reuse_origin/'registration.json')
            assert previous_registration['budget']['train_indices']==train_indices
            assert previous_registration['budget']['architecture']==('reflection_group_average' if args.reflection else 'original')
            current_sources=read(stage/'registration.json')['source_sha256']
            for file in [initial_model,BASE/'learning_preparation_v4/paths.json']+[BASE/n for n in names if 'runtime' in n or n=='geometry_spec.py']:
                key=file.relative_to(BASE.parent.parent).as_posix()
                assert previous_registration['source_sha256'][key]==current_sources[key], key

        physical_cache={physical_key(r['coefficients']):r for r in previous_candidates.values()
                        if r.get('candidate_complete',True)}

        source_catalog={}
        def attach_spatial_metrics(result,origin):
            if not args.spatial_energy:return result
            for record in result['results']:
                if record['shape'] is None:continue
                folder=origin; candidate=result
                while not (folder/'training_traces'/record['trace']).exists():
                    source_file=Path(candidate['reused_from'])
                    if str(source_file) not in source_catalog:
                        source_catalog[str(source_file)]={r['tag']:r for r in read(source_file)['candidates']}
                    candidate=source_catalog[str(source_file)][candidate.get('physical_source_tag',candidate['tag'])]
                    folder=source_file.parent
                with np.load(folder/'training_traces'/record['trace']) as data:
                    record['spatial']=geometric_turn_metrics(data['reference'],data['rows'])
                    record['spatial_coarse']=geometric_turn_metrics(data['reference'],data['rows'],.2)
            return result

        def cached_result(coefficients,tag):
            old=physical_cache.get(physical_key(coefficients))
            if old is None:return None
            result=copy.deepcopy(old)
            attach_spatial_metrics(result,reuse_origin)
            result.update(tag=tag,coefficients=np.asarray(coefficients).tolist(),
                reused_from=str(reuse_origin/'training_results.json'),physical_source_tag=old['tag'],
                exact_float32_actor_parameters_equal=True,new_actor_calls=0,new_physics_ticks=0)
            return result

        def install(coefficients):
            change = coefficients@basis
            with torch.no_grad():
                trainable_head.weight.copy_(torch.as_tensor(base_weight+change[:, :feature_count], dtype=torch.float32))
                trainable_head.bias.copy_(torch.as_tensor(base_bias+change[:, feature_count], dtype=torch.float32))

        install(np.zeros((3, 9)))
        for key, value in incoming['policy'].items():
            assert torch.equal(value, policy.state_dict()[key]), key
        if args.dynamic_residual:
            from dynamic_residual_policy import DynamicResidualPolicy
            from reflection_policy import ReflectionPolicy
            unchanged=DynamicResidualPolicy(incoming['observation_dim']) if args.preview_residual else ReflectionPolicy(incoming['observation_dim'])
            unchanged.load_state_dict(incoming['policy'])
            with torch.no_grad():
                assert torch.equal(unchanged.distribution(obs).mean,policy.distribution(obs).mean)
        # Independent matrix identity guards the learned-basis parameter mapping.
        probe = rng.normal(size=(3, 9))*.001
        install(probe)
        with torch.no_grad():
            actual = trainable_head(obs[:,22:42]).numpy() if args.preview_residual else trainable_head(obs[:,:8]).numpy() if args.dynamic_residual else policy.actor(obs).numpy()
        expected = features@base_weight.T+base_bias+np.c_[features, np.ones(len(features))]@(probe@basis).T
        assert np.max(np.abs(actual-expected)) < 2e-6
        install(np.zeros((3, 9)))
        paths = read(BASE/'learning_preparation_v4/paths.json')
        training = [np.asarray(paths['training'][i]) for i in train_indices]
        validation = [np.asarray(p) for p in paths['validation']]
        initial_directory = BASE/'shape_pareto_candidate_review_20260916/validation_1' if args.preview_residual else BASE/'reflection_shape_search_20260916/validation_5' if args.broader_curriculum else BASE/'reflection_policy_probe_20260916/eval_source' if args.reflection else source/'final_validation'
        initial_validation = read(initial_directory/'summary.json')
        initial_validation['successful_time_sum'] = sum(r['completion_time_s'] for r in initial_validation['results'][:12])
        initial_shapes = []
        for i in range(4):
            with np.load(initial_directory/f'{i}.npz') as d:
                initial_shapes.append(corner_shape(d['reference'], d['rows'], d['actions']))
        write_json(stage/'initial_reuse.json', dict(source=str(initial_directory),
            model_equal=True, new_validation_runs=0, summary_sha256=sha(initial_directory/'summary.json')))
        all_results, generations = [], []
        traces = stage/'training_traces'; traces.mkdir()

        def rollout(coeff, tag):
            result=cached_result(coeff,tag) if reuse_population else None
            if args.dynamic_residual and tag=='baseline':
                original_tag='g3_c5' if args.preview_residual else 'baseline'
                old=next(r for r in read(reuse_origin/'training_results.json')['candidates'] if r['tag']==original_tag)
                result=copy.deepcopy(old);attach_spatial_metrics(result,reuse_origin)
                result.update(tag=tag,coefficients=coeff.tolist(),reused_from=str(reuse_origin/'training_results.json'),physical_source_tag=original_tag,new_actor_calls=0,new_physics_ticks=0,zero_residual_exact_action_identity=True)
            if result is not None:
                all_results.append(result)
                write_json(stage/'training_results.json', dict(candidates=all_results, costs=costs))
                print('REUSED', tag, result['successes'], flush=True)
                return result
            install(coeff); records = []
            for index, ref in enumerate(training):
                env = environment_factory(ref); chunks, actions = [], []; reward_sum = 0.
                while not env.done:
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Registered 1200 second wall limit')
                    if costs['training_actor_calls'] >= training_budget:
                        raise RuntimeError('Registered training-decision budget reached')
                    if costs['training_physics_ticks']+costs['evaluation_physics_ticks'] >= physics_budget-2200:
                        raise RuntimeError('Registered physical tick budget reached')
                    latent, _, _ = policy.act(env.observation(), deterministic=True)
                    costs['training_actor_calls'] += 1
                    _, reward, _, info = env.step(latent)
                    costs['training_physics_ticks'] += len(info['rows'])
                    reward_sum += reward; chunks.append(info['rows']); actions.append(info['held_action'])
                rows = np.vstack(chunks); actions = np.asarray(actions)
                audit = difference_audit(ref[0], rows)
                success = bool(env.success and audit['kinematic_pass'] and audit['rest_pass'])
                trace = f'{tag}_{index}.npz'
                np.savez_compressed(traces/trace, reference=ref, rows=rows, actions=actions)
                records.append(dict(path=train_indices[index], success=success, reason=env.reason,
                    elapsed_s=env.ticks*.001, complete_time_s=env.ticks*.001 if success else None,
                    return_sum=reward_sum, audit=audit, geometry=env.tracker.snapshot(1.25), trace=trace,
                    shape=corner_shape(ref, rows, actions) if len(ref)==3 and len(rows)>21 else None))
            result = dict(tag=tag, coefficients=coeff.tolist(), results=records,
                          successes=sum(r['success'] for r in records),
                          return_sum=sum(r['return_sum'] for r in records))
            if result['successes'] == len(training):
                result['complete_time_sum_s'] = sum(r['complete_time_s'] for r in records)
                # Potential telescopes on success; hence reward ranks only full time.
                expected_return = -result['complete_time_sum_s']+sum(np.linalg.norm(np.diff(r,axis=0),axis=1).sum()/100. for r in training)
                assert abs(expected_return-result['return_sum']) < 1e-9
            attach_spatial_metrics(result,stage)
            all_results.append(result)
            write_json(stage/'training_results.json', dict(candidates=all_results, costs=costs))
            print(tag, result['successes'], round(result['return_sum'], 4), costs['training_actor_calls'], flush=True)
            return result

        def rank(row):
            if args.spatial_energy:
                incoming_violation,_=violation_components(row)
                return row['successes'],-incoming_violation,row['return_sum']-bending_cost(row)
            if args.regularized_exit:
                incoming_violation, exit_violation = violation_components(row)
                return row['successes'], -incoming_violation, row['return_sum']-exit_violation
            if constrained:
                return row['successes'], -violation(row), row['return_sum']
            return row['successes'], row['return_sum']

        def bending_cost(row):
            return sum(1.25*r['spatial']['normalized_bending_mm_inv'] for r in row['results'] if r['shape'] is not None)

        def violation(row):
            # This is an explicit training objective; it never overrides physics.
            # Components are dimensionless and reported individually in raw shape data.
            return sum(violation_components(row))

        def violation_components(row):
            shapes=[r['shape'] for r in row['results'] if r['shape'] is not None]
            incoming=sum(max(s['outward_excursion_mm']-.05,0)/1.25+
                .1*(s['first_physical_turn']!='inside') for s in shapes)
            exit_turn=sum(max(s['reverse_turning_total_deg']-5.,0)/abs(s['turn_deg']) for s in shapes)
            return incoming, exit_turn

        if args.workers!=1:
            from concurrent.futures import ProcessPoolExecutor
            import parameter_search_worker as worker
            config=dict(model=str(initial_model),basis=basis.tolist(),stage=str(stage),
                dynamic_residual=args.dynamic_residual,
                preview_residual=args.preview_residual,
                paths=[r.tolist() for r in training],indices=train_indices,
                deadline_unix=time.time()+max(0.,deadline-time.perf_counter()))
            executor=ProcessPoolExecutor(max_workers=args.workers,initializer=worker.initialize,initargs=(config,))
            check_obs=np.asarray(saved['batch']['observation'])[:16]
            install(probe)
            expected_actions=np.array([policy.act(o,deterministic=True)[0] for o in check_obs])
            checks=[executor.submit(worker.action_check,probe.tolist(),check_obs) for _ in range(4)]
            assert all(np.array_equal(f.result(),expected_actions) for f in checks)
            install(np.zeros((3,9)))
            write_json(stage/'parallel_action_check.json',dict(exact_equal=True,requests=4,observations_per_request=16,
                physics_ticks=0,training_updates=0,worker_source_sha256=sha(BASE/'parameter_search_worker.py')))

        mean = np.zeros((3, 9)); std = np.array([.05, .12, .25])[:, None]/3*np.ones((3, 9))
        covariance = np.diag(std.ravel()**2)
        baseline = rollout(mean, 'baseline'); incumbent = baseline
        no_gain = 0; stopped_reason = 'registered_generations_finished'
        for generation in range(1, 7):
            # Reserve enough physical work to finish a generation and its validation.
            reserve = 18000 if args.broader_curriculum else 7500
            if costs['training_actor_calls'] > training_budget-reserve or time.perf_counter() > deadline-(240 if args.broader_curriculum else 120):
                stopped_reason = 'remaining_registered_budget_requires_analysis'; break
            noises = rng.normal(size=(6, 3, 9))
            if args.full_covariance and generation>1:
                factor=np.linalg.cholesky(covariance)
                population=[mean+(factor@e.ravel()).reshape(3,9) for e in np.concatenate([noises,-noises])]
            else:
                population = [mean+std*e for e in np.concatenate([noises, -noises])]
            candidates = [incumbent]
            if executor is not None and generation>1:
                cached=[cached_result(coeff,f'g{generation}_c{number}') for number,coeff in enumerate(population)]
                fresh_count=sum(r is None for r in cached)
                actor_cap=2200
                if costs['training_actor_calls']+fresh_count*actor_cap>training_budget:
                    stopped_reason='remaining_budget_below_parallel_reservations'; break
                # Reserve the exact time-limit upper bounds of all remaining validations.
                validation_reserve=(7-generation)*sum(environment_factory(ref).max_ticks for ref in validation)
                available_ticks=physics_budget-costs['training_physics_ticks']-costs['evaluation_physics_ticks']-validation_reserve
                tick_cap=int(available_ticks/max(fresh_count,1))
                if fresh_count and tick_cap<50000:
                    stopped_reason='remaining_physics_budget_below_parallel_reservations'; break
                futures=[None if cached[number] is not None else executor.submit(worker.evaluate_candidate,coeff.tolist(),f'g{generation}_c{number}',actor_cap,tick_cap)
                    for number,coeff in enumerate(population)]
                for previous,future in zip(cached,futures):
                    result=previous if previous is not None else future.result()
                    if previous is None:
                        costs['training_actor_calls']+=result['actor_calls']
                        costs['training_physics_ticks']+=result['physics_ticks']
                    if result.get('candidate_complete',True):attach_spatial_metrics(result,stage if previous is None else reuse_origin)
                    all_results.append(result)
                    if result.get('candidate_complete',True): candidates.append(result)
                    write_json(stage/'training_results.json',dict(candidates=all_results,costs=costs))
                    print(result['tag'],result.get('successes'),'complete',result.get('candidate_complete',True),'reused',previous is not None,
                        round(result.get('return_sum',0.),4),costs['training_actor_calls'],flush=True)
                if len(candidates)<3:
                    stopped_reason='too_few_complete_candidates';break
            else:
                for number, coeff in enumerate(population):
                    candidates.append(rollout(coeff, f'g{generation}_c{number}'))
            ordered = sorted(candidates, key=rank, reverse=True)
            previous = incumbent; incumbent = ordered[0]
            elites = np.array([row['coefficients'] for row in ordered[:3]])
            mean = .5*mean+.5*elites.mean(0)
            if args.full_covariance:
                centered=elites.reshape(3,27)-elites.reshape(3,27).mean(0)
                covariance=.8*covariance+.2*(centered.T@centered/3.)
                std=np.sqrt(np.diag(covariance)).reshape(3,9)
            else:
                std = np.sqrt(.8*std**2+.2*np.var(elites, axis=0))
            costs['parameter_updates'] += 1
            changed = incumbent['tag'] != previous['tag']
            no_gain = 0 if changed else no_gain+1
            record = dict(generation=generation, best_tag=incumbent['tag'], changed=changed,
                          training_successes=incumbent['successes'],
                          training_time_s=incumbent.get('complete_time_sum_s'), mean=mean.tolist(), std=std.tolist(),
                          training_shape_violation=violation(incumbent))
            if args.spatial_energy:
                record['training_spatial_cost_seconds']=bending_cost(incumbent)
            if args.full_covariance:
                record['covariance']=covariance.tolist()
                record['minimum_covariance_eigenvalue']=float(np.linalg.eigvalsh(covariance).min())
            if changed:
                install(np.asarray(incumbent['coefficients']))
                model_path = stage/f'best_generation_{generation}.pt'
                save_torch(model_path, dict(policy=policy.state_dict(), observation_dim=incoming['observation_dim'],
                           mode='joint', optimizer_method='complete_return_CEM', coefficients=incumbent['coefficients'],
                           architecture=architecture))
                reuse = next((g for g in previous_generations if g['best_tag']==incumbent['tag'] and
                    g.get('changed') and np.array_equal(incumbent['coefficients'],previous_candidates[incumbent['tag']]['coefficients'])), None)
                if reuse:
                    validation_directory = reuse_origin/f"validation_{reuse['generation']}"
                    result = read(validation_directory/'summary.json')['results']
                    write_json(stage/f'validation_reuse_{generation}.json',dict(source=str(validation_directory),
                        summary_sha256=sha(validation_directory/'summary.json'), coefficients_equal=True))
                else:
                    result = evaluate(policy, validation, stage, f'validation_{generation}', None, costs, deadline, environment_factory)
                    validation_directory = stage/f'validation_{generation}'
                shapes = []
                for i in range(4):
                    with np.load(validation_directory/f'{i}.npz') as d:
                        shapes.append(corner_shape(d['reference'], d['rows'], d['actions']))
                        if args.spatial_energy:
                            shapes[-1]['spatial']=geometric_turn_metrics(d['reference'],d['rows'])
                            shapes[-1]['spatial_coarse']=geometric_turn_metrics(d['reference'],d['rows'],.2)
                complete = all(r['success'] for r in result)
                time_sum = sum(r['completion_time_s'] for r in result) if complete else None
                record.update(validation_successes=sum(r['success'] for r in result),
                    validation_time_s=time_sum, shapes=shapes, model_sha256=sha(model_path),
                    validation_directory=str(validation_directory),
                    initial_shape_screen_pass=bool(complete and time_sum<=initial_validation['successful_time_sum'] and
                        all(s['shape_screen_pass'] for s in shapes)),
                    final_shape_acceptance=False)
                print('GENERATION', generation, record['validation_successes'], time_sum,
                      [round(s['outward_excursion_mm'],4) for s in shapes], record['initial_shape_screen_pass'], flush=True)
            generations.append(record)
            write_json(stage/'generations.json', dict(generations=generations, costs=costs, stopped_reason=None))
            if no_gain >= 3:
                stopped_reason = 'three_generations_without_training_gain'; break
            if costs['evaluation_actor_calls'] >= 10000:
                stopped_reason = 'reserve_evaluation_budget'; break
        write_json(stage/'results.json', dict(generations=generations, costs=costs, stopped_reason=stopped_reason,
            best_training=incumbent, initial_time_s=initial_validation['successful_time_sum'],
            training_paths=train_indices,
            boundary='Critic-free parameter search in restricted learned feature subspace. No final algorithm comparison, no isolated causal attribution to critic, no independent generalization claim.',
            goal_complete=False))
        receipt(stage, dict(status='bounded_complete_return_search_completed', costs=costs,
            wall_s=time.perf_counter()-started, stopped_reason=stopped_reason, goal_complete=False))
    except Exception:
        write_json(stage/'failure.json', dict(traceback=traceback.format_exc(), costs=costs,
            wall_s=time.perf_counter()-started, preserved_partial_data=True, automatic_retry=False))
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
