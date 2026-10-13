"""Replay saved network samples/optimizer updates; do not repeat physics training."""
import time
import traceback
import numpy as np
import torch
from contextual_mixture_policy import ContextualMixturePolicy, DeployedContextualView
from train_bounded_shape_ppo import update
from train_pilot import save_torch, evaluate
from conditional_feedback_policy import RHO
from joint_runtime_v5 import environment_factory
from isolated_corner_shape import corner_shape
from geometric_turn_metrics import geometric_turn_metrics
from search_complete_return import read
from study_io import BASE, ROOT, register, receipt, write_json, sha


def main():
    source = BASE/'contextual_learned_function_ppo_20260916'
    original = read(source/'results.json')
    episodes = read(source/'episode_results.json')['episodes']
    registered = read(source/'registration.json')['source_sha256']
    for name, digest in registered.items():
        if name.endswith('.py'):
            assert sha(ROOT/name) == digest, name
    a_path = BASE/'constrained_parameter_step_probe_20260916/candidate_1.pt'
    b_path = BASE/'heading_frame_feedback_learning_20260916/epoch_2000.pt'
    stage = register('contextual_checkpoint_recovery_20260916',
        [__file__, source/'receipt.json', source/'results.json', source/'episode_results.json',
         source/'paired_groups.json', source/'registration.json', a_path, b_path,
         BASE/'contextual_mixture_policy.py', BASE/'conditional_feedback_policy.py',
         BASE/'train_bounded_shape_ppo.py', BASE/'train_pilot.py', BASE/'joint_policy.py'],
        dict(new_training_decisions=0, new_training_physics_ticks=0,
             replayed_network_samples=22509, replayed_optimizer_updates=19,
             supplemental_validation_checkpoints=3, evaluation_decisions=10000,
             evaluation_physics_ticks=210000, wall_s=600),
        'Recover omitted checkpoints from complete saved observations and original RNG stream. '
        'Every sample/log/value and both existing model tensors must match exactly. '
        'Only after exact recovery, supplement missing development evaluations. No new training physics, '
        'no sealed tests, no changed objective or extended training. Stop on any replay mismatch.')
    costs = dict(training_actor_calls=0, training_physics_ticks=0, replay_actor_calls=0,
                 replay_optimizer_updates=0, evaluation_actor_calls=0, evaluation_physics_ticks=0)
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(9161009)
        a = torch.load(a_path, map_location='cpu', weights_only=False)
        b = torch.load(b_path, map_location='cpu', weights_only=False)
        policy = ContextualMixturePolicy(72, a['policy'], b['policy'])
        torch.manual_seed(9161009)
        policy.reset_context()
        optimizer = {'actor': torch.optim.Adam(policy.actor_parameters(), lr=.0003),
                     'critic': torch.optim.Adam(policy.critic.parameters(), lr=.001)}
        batch, group, verified, recovered, manifest = [], [], [], [], []
        next_check, update_index, decisions = 5000, 0, 0
        for item in episodes:
            trace = source/'episodes'/item['trace']
            manifest.append(dict(trace=item['trace'], sha256=sha(trace)))
            with np.load(trace) as saved:
                data = {key: saved[key].copy() for key in ('observation', 'latent',
                    'previous_observation', 'previous_latent', 'context_live',
                    'old_log', 'reward', 'value', 'terminal')}
            policy.reset_context()
            for i, obs in enumerate(data['observation']):
                latent, logp, value = policy.act(obs)
                assert np.array_equal(latent, data['latent'][i]), ('sample', item['episode'], i)
                assert logp == data['old_log'][i], ('logp', item['episode'], i)
                assert value == data['value'][i], ('value', item['episode'], i)
                for key, actual in policy.last_context.items():
                    assert np.array_equal(actual, data[key][i]), (key, item['episode'], i)
                costs['replay_actor_calls'] += 1
            batch.append(data)
            group.append(data)
            decisions += len(data['reward'])
            if len(group) == 4:
                total = np.array([g['reward'].sum() for g in group])
                baseline = (total.sum()-total)/3
                for g, advantage in zip(group, total-baseline):
                    g['paired_advantage'] = np.full(len(g['reward']), advantage)
                group = []
            expected = original['updates'][update_index]
            assert decisions <= expected['decisions']
            if decisions != expected['decisions']:
                continue
            assert not group
            stats = update(policy, optimizer, batch, paired_advantage=True)
            for key, value in stats.items():
                assert value == expected[key], ('update_stat', decisions, key, value, expected[key])
            costs['replay_optimizer_updates'] += 1
            update_index += 1
            batch = []
            print('REPLAY_UPDATE', decisions, 'exact', flush=True)
            final = update_index == len(original['updates'])
            if decisions >= next_check or final:
                old_path = source/f'policy_{decisions}.pt'
                if old_path.exists():
                    old = torch.load(old_path, map_location='cpu', weights_only=False)
                    assert old['policy'].keys() == policy.state_dict().keys()
                    assert all(torch.equal(old['policy'][k], v) for k, v in policy.state_dict().items())
                    verified.append(dict(decisions=decisions, all_tensors_bitwise_equal=True,
                                         source_sha256=sha(old_path)))
                else:
                    model = stage/f'policy_{decisions}.pt'
                    save_torch(model, dict(policy=policy.state_dict(), observation_dim=72,
                        mode='joint', architecture='contextual_learned_function_mixture',
                        learning_method='PPO_on_sampled_mixture_coefficients',
                        training_decisions=decisions, parents=[sha(a_path), sha(b_path)],
                        coefficient_clip=[0., 1.], exploration=dict(rho=RHO, std=policy.exploration_std),
                        recovery='Exact saved-observation optimizer replay; no physics training'))
                    recovered.append(dict(decisions=decisions, model=model.name, sha256=sha(model)))
                if decisions >= next_check:
                    next_check += 5000
        assert decisions == 22509 and update_index == 19 and not batch and not group
        assert [v['decisions'] for v in verified] == [5206, 10407]
        assert len(recovered) == 3
        save_torch(stage/'final_optimizer_state.pt', dict(
            optimizer={k: v.state_dict() for k, v in optimizer.items()},
            torch_rng=torch.get_rng_state(), decisions=decisions))
        write_json(stage/'recovery_contract.json', dict(verified=verified, recovered=recovered,
            sample_log_value_context_exact=True, update_statistics_exact=True,
            episode_manifest=manifest, costs=costs.copy()))
        paths = [np.asarray(p) for p in read(BASE/'learning_preparation_v4/paths.json')['validation']]
        checkpoints = []
        for item in recovered:
            policy.load_state_dict(torch.load(stage/item['model'], map_location='cpu', weights_only=False)['policy'])
            tag = f"validation_{item['decisions']}"
            result = evaluate(DeployedContextualView(policy), paths, stage, tag, None,
                              costs, started+600, environment_factory)
            shapes, cruise = [], []
            for i in range(len(paths)):
                with np.load(stage/tag/f'{i}.npz') as z:
                    speed = np.linalg.norm(z['rows'][:, 2:4], axis=1)
                    cruise.append(dict(path=i, peak=float(speed.max()), ge95_seconds=float((speed >= 95).sum()*.001)))
                    if i < 4:
                        shape = corner_shape(z['reference'], z['rows'], z['actions'])
                        shape['spatial'] = geometric_turn_metrics(z['reference'], z['rows'])
                        shapes.append(shape)
            row = dict(**item, successes=sum(r['success'] for r in result),
                       full_time_s=sum(r['completion_time_s'] for r in result) if all(r['success'] for r in result) else None,
                       shapes=shapes, cruise=cruise, final_acceptance=False)
            checkpoints.append(row)
            write_json(stage/'results.json', dict(checkpoints=checkpoints, costs=costs, goal_complete=False))
            print('RECOVERED_CHECKPOINT', item['decisions'], row['successes'], row['full_time_s'], flush=True)
        assert costs['evaluation_actor_calls'] <= 10000 and costs['evaluation_physics_ticks'] <= 210000
        receipt(stage, dict(status='exact_recovery_and_missing_evaluations_completed', costs=costs,
                            recovered=recovered, wall_s=time.perf_counter()-started))
    except Exception:
        write_json(stage/'failure.json', dict(traceback=traceback.format_exc(), costs=costs,
                                              wall_s=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()
