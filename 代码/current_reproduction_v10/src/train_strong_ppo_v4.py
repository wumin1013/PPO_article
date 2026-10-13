"""One own-seed fitted-actor PPO smoke; every failed/truncated episode retained."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, sha, utc, write_json
from strong_policy_v4 import tensor_hash, load_strong_policy, STATIC_KEYS
from strong_ppo_v4 import from_fit, interleaved_gae, load_strong_ppo
from train_strong_fit_v4 import equivalent_loss, DATA, DATA_HASH
from terminal_completion_v4d import install, VERSION

SEED, ARM, STEPS = 101, 'adaptive', 2048
FIT = ROOT/'results/strong_fit_v4/pilot_seed101/adaptive/final.pt'
FIT_HASH = '188b02080069d7381a8ce9573ee936c89886e5b600e35f8e394e5f38f8256f94'
OUT = ROOT/'results/strong_ppo_v4/smoke_seed101/adaptive'
SOURCES = ['train_strong_ppo_v4.py', 'strong_ppo_v4.py', 'strong_policy_v4.py',
           'train_strong_fit_v4.py', 'paired_learning.py', 'terminal_completion_v4d.py',
           'runtime_no_target_override_v4b.py', 'runtime_configuration.py',
           'execution_settings.py', 'corridor_v3.py', 'runtime_bootstrap.py']


def training_paths():
    manifest = json.loads((ROOT/'data/manifest.json').read_text('utf-8'))
    rows = [dict(id=r['id'], reference='data/'+r['id']+'.csv', tolerance_mm=1.25,
                 arc_chord_error_mm=0.) for r in manifest['paths'] if r['split']=='train']
    assert len(rows) == 4
    directory = ROOT/'baseline_records/supplementary_reproduction/cases/dense'
    geometry = json.loads((directory/'geometry.json').read_text('utf-8'))
    chord = max([p.get('chord_error_bound_mm', 0.) for p in geometry['primitives']]+[0.])
    rows.append(dict(id='dense', reference=(directory/'reference.csv').relative_to(ROOT).as_posix(),
                     tolerance_mm=1.25-chord, arc_chord_error_mm=chord))
    for row in rows:
        row['reference_sha256'] = sha(ROOT/row['reference'])
    return rows


def audit_interface(policy, x):
    fitted, _ = load_strong_policy(FIT)
    before = tensor_hash(dict(policy.actor.named_parameters()))
    with torch.no_grad():
        error = float((fitted.mean(x[:512])-policy.mean(x[:512])).abs().max())
        action, logp, _ = policy.act(x[0])
        replay = float(policy.dist(x[0]).log_prob(torch.tensor(action)).sum())
    policy.action_generator.manual_seed(SEED+200000)
    # Two interleaved programs with distinct terminals and a bootstrapped tail.
    batch = [dict(path='a', episode=0, value=0., reward=1., next_value=2., done=False),
             dict(path='b', episode=0, value=0., reward=10., next_value=100., done=True),
             dict(path='a', episode=0, value=2., reward=3., next_value=0., done=True),
             dict(path='b', episode=1, value=4., reward=5., next_value=6., done=False)]
    adv, _ = interleaved_gae(batch)
    expected = np.array([1.+.99*2.+.99*.95*(3.-2.), 10., 1., 5.+.99*6.-4.])
    gae_error = float(abs(adv-expected).max())
    assert error == 0. and replay == logp and gae_error < 1e-6
    assert before == tensor_hash(dict(fitted.actor.named_parameters()))
    return dict(fit_deterministic_mean_max_error=error, log_probability_replay_error=abs(replay-logp),
        interleaved_gae_max_error=gae_error, actor_tensor_sha256=before,
        actor_parameters_all_trainable=all(p.requires_grad for p in policy.actor.parameters()),
        critic_rng_seed=SEED+100000, action_rng_seed=SEED+200000, action_rng_reset_after_audit=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    assert sha(FIT) == FIT_HASH and sha(DATA) == DATA_HASH
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'episodes').mkdir()
    (OUT/'batches').mkdir()
    started, cpu_started = time.perf_counter(), time.process_time()
    policy, fit_metadata = from_fit(FIT)
    with np.load(DATA) as d:
        x = torch.tensor(d['observations'], dtype=torch.float32)
        y = torch.tensor(d['targets_three' if ARM=='adaptive' else 'targets_two'], dtype=torch.float32)
        anchor_groups = [np.flatnonzero(d['case']==case) for case in ('original', 'dense')]
    with torch.no_grad():
        features = [policy.actor.features(part) for part in x.split(512)]
        preview, joint = [torch.cat([a[i] for a in features]) for i in (0, 1)]
    audit = audit_interface(policy, x)
    write_json(OUT/'interface_audit.json', audit)
    rows = training_paths()
    registration = dict(created_utc=utc(), version='strong_ppo_v4_smoke_1', scope='DEVELOPMENT ONLY',
        seed=SEED, arm=ARM, actor_parent=str(FIT.relative_to(ROOT)), actor_parent_sha256=FIT_HASH,
        initialization='Own seed101 fresh-fit actor is the previous stage of this training pipeline; critic newly seeded and log_std fixed anew. Not a separately independent random actor.',
        historical_88_ppo_reproduced=False, inherited_v4_fit_iterations=12000,
        runtime_version=VERSION, runtime_manifest_sha256=sha(ROOT/'runtime/source_manifest.json'),
        source_sha256={name:sha(ROOT/'src'/name) for name in SOURCES}, paths=rows,
        sampling='Deterministic round-robin decision interleaving across five persistent environments; reset only an ended episode. GAE grouped by program and episode; every batch-end tail bootstraps its observed next value.',
        allowed_training='Four split=train short programs and historical dense only; no validation, observed-v3-final or new final program.',
        decisions=STEPS, rollout=256, ppo_updates=8, epochs=4, minibatch=64, gamma=.99, gae_lambda=.95,
        clip=.2, actor_lr=1e-5, critic_lr=1e-4, critic_loss_coefficient=.5, global_gradient_norm=.5,
        std_initial=[.002, .01, .005], log_std_clamp=[-8., -2.], actor_all_eight_tensors_trainable=True,
        static_feature_maps_fixed=True, actor_optimizer='Adam actor plus log_std', critic_optimizer='Separate Adam critic',
        anchor=dict(dataset_sha256=DATA_HASH, coefficient=.1, samples_each_original_dense_per_minibatch=32,
                    loss='equivalent_loss with clipped-boundary margin .02', scales=[.015,.15,.3],
                    source='Same historical recorded actions; not c16d45 function relabels; fixed later uses targets_two with all20 clipped labels retained'),
        random_streams=dict(critic=SEED+100000, action=SEED+200000, minibatches=SEED+300000, anchors=SEED+400000,
                            paths='Deterministic round-robin, no random draws'),
        checkpoints=[0,1024,2048], selection='Fixed2048 final; diagnostics never select a different checkpoint',
        no_validation_during_optimization=True, no_five_seed_launch=True, wall_limit_s=1680,
        exception_rule='Retain failed partial trace and reason; terminal reward -10 for unexpected step exceptions, matching v3 handling. Initialization or optimization errors end run with a checkpoint.',
        reused_algorithm='paired_learning.py/update embedded PPO clipped-surrogate formula; separate critic lr and anchor are explicitly new smoke settings',
        kl_scope='Logged full-batch Gaussian KL, no undisclosed rollback or early stopping')
    write_json(OUT/'registration.json', registration)
    reg_hash = sha(OUT/'registration.json')
    actor_optimizer = torch.optim.Adam(list(policy.actor.parameters())+[policy.log_std], lr=1e-5)
    critic_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-4)
    minibatch_rng = np.random.default_rng(SEED+300000)
    anchor_rng = np.random.default_rng(SEED+400000)
    cost = dict(training_interactions=0, physics_ticks=0, ppo_updates=0, actor_optimizer_steps=0,
                critic_optimizer_steps=0, ppo_sample_presentations=0, anchor_sample_presentations=0,
                actor_wall_s=0., executor_wall_s=0., optimization_wall_s=0.)
    episodes, curves, transition_records = [], [], []
    current, buffer = {}, []
    next_episode = 0
    error = None

    def save_checkpoint(step, name=None):
        path = OUT/(name or f'checkpoint_{step:05d}.pt')
        torch.save(dict(format='strong_ppo_v4', policy=policy.state_dict(), seed=SEED, arm=ARM,
            fixed_preview_mm=3., steps_completed=step, inherited_fit_sha256=FIT_HASH,
            registration_sha256=reg_hash, runtime_version=VERSION, cost=cost.copy(),
            trainable_tensor_sha256=tensor_hash(dict(policy.named_parameters())),
            actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(), action_rng_state=policy.action_generator.get_state(),
            minibatch_rng_state=minibatch_rng.bit_generator.state, anchor_rng_state=anchor_rng.bit_generator.state), path)
        return path

    def new_episode(row):
        nonlocal next_episode
        install(dt=.001, tolerance=row['tolerance_mm'])
        from compact_farvector_runtime import FarVectorRuntime
        from direct_two_action_control import DirectTwoRuntime
        reference = np.loadtxt(ROOT/row['reference'], delimiter=',', skiprows=1)
        env = (FarVectorRuntime(reference, potential_speed=100.) if ARM=='adaptive' else
               DirectTwoRuntime(reference, preview_rule_mm=3., potential_speed=100.))
        env.env.max_ticks=int(round((5.+np.linalg.norm(np.diff(reference,axis=0),axis=1).sum()/5.)/.001))
        item = dict(env=env, reference=reference, path=row['id'], episode=next_episode,
                    positions=[reference[0].copy()], observations=[], actions=[], rewards=[], rows=[])
        next_episode += 1
        original_update = env.tracker.update
        def observe(position):
            item['positions'].append(np.asarray(position).copy())
            return original_update(position)
        env.tracker.update = observe
        return item

    def retain(item, complete, info):
        file = OUT/'episodes'/f'episode_{item["episode"]:04d}_{item["path"]}.npz'
        if file.exists():
            raise FileExistsError(file)
        np.savez_compressed(file, reference=item['reference'], positions=np.asarray(item['positions']),
            rows=np.vstack(item['rows']) if item['rows'] else np.empty((0,13)),
            observations=np.asarray(item['observations']), actions=np.asarray(item['actions']), rewards=np.asarray(item['rewards']))
        episodes.append(dict(path=item['path'], episode=item['episode'], complete=complete,
            executor_success=bool(info.get('success')) if complete else None, reason=info.get('reason'),
            decisions=len(item['actions']), ticks=item['env'].ticks, trace_sha256=sha(file),
            trace_file=str(file.relative_to(ROOT)), action_source='fresh_PPO_Gaussian_sample',
            partial_decision_positions_retained=len(item['positions'])-1-sum(len(r) for r in item['rows'])))
        write_json(OUT/'episodes.json', episodes)

    save_checkpoint(0)
    try:
        for row in rows:
            current[row['id']] = new_episode(row)
        for step in range(1, STEPS+1):
            if time.perf_counter()-started > 1680.:
                raise TimeoutError('Registered smoke wall limit')
            row = rows[(step-1)%len(rows)]
            item = current[row['id']]
            env = item['env']
            obs = env.observation()
            before = time.perf_counter()
            action, logp, value = policy.act(obs)
            cost['actor_wall_s'] += time.perf_counter()-before
            item['observations'].append(obs)
            item['actions'].append(action)
            tick_before, before = env.ticks, time.perf_counter()
            try:
                next_obs, reward, done, info = env.step(action)
                item['rows'].append(info['rows'])
                with torch.no_grad():
                    next_value = 0. if done else float(policy.value(torch.tensor(next_obs)))
            except Exception as exc:
                done, reward, next_value = True, -10., 0.
                info = dict(success=False, reason=type(exc).__name__+': '+str(exc))
                frame = exc.__traceback__
                while frame is not None:
                    if frame.tb_frame.f_code.co_name=='step' and frame.tb_frame.f_code.co_filename.endswith('joint_runtime.py'):
                        partial = frame.tb_frame.f_locals.get('rows', [])
                        if len(partial):
                            item['rows'].append(np.asarray(partial))
                    frame = frame.tb_next
            cost['executor_wall_s'] += time.perf_counter()-before
            cost['physics_ticks'] += env.ticks-tick_before
            cost['training_interactions'] += 1
            item['rewards'].append(reward)
            record = dict(path=item['path'], episode=item['episode'], value=value, reward=reward,
                          next_value=next_value, done=done, observation=obs, action=action, logp=logp)
            buffer.append(record)
            transition_records.append(dict(step=step, path=item['path'], episode=item['episode'], tick_before=tick_before,
                tick_after=env.ticks, reward=reward, done=done, reason=info.get('reason'),
                executor_success=bool(info.get('success')), action_source='Gaussian_actor',
                old_log_probability=logp, old_value=value, next_value=next_value))
            if done:
                retain(item, True, info)
                current[row['id']] = new_episode(row)
            if len(buffer)==256:
                optimize_started = time.perf_counter()
                ob = torch.tensor(np.asarray([r['observation'] for r in buffer]), dtype=torch.float32)
                ac = torch.tensor(np.asarray([r['action'] for r in buffer]), dtype=torch.float32)
                old = torch.tensor([r['logp'] for r in buffer], dtype=torch.float32)
                advantage, targets = interleaved_gae(buffer)
                advantage = torch.tensor(advantage)
                advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
                targets = torch.tensor(targets)
                with torch.no_grad():
                    old_dist = policy.dist(ob)
                    old_mean, old_std = old_dist.mean.clone(), old_dist.scale.clone()
                    start_log_error = float(abs(old_dist.log_prob(ac).sum(-1)-old).max())
                if start_log_error > .02:
                    raise AssertionError('Unexpected pre-update likelihood discrepancy: '+str(start_log_error))
                np.savez_compressed(OUT/'batches'/f'batch_{step:05d}.npz', observations=ob.numpy(), actions=ac.numpy(),
                    old_log_probability=old.numpy(), advantage=advantage.numpy(), targets=targets.numpy(),
                    path=np.asarray([r['path'] for r in buffer]), episode=np.asarray([r['episode'] for r in buffer]),
                    rewards=np.asarray([r['reward'] for r in buffer]), done=np.asarray([r['done'] for r in buffer]),
                    old_value=np.asarray([r['value'] for r in buffer]), next_value=np.asarray([r['next_value'] for r in buffer]))
                stats, anchor_ids_log = [], []
                for epoch in range(4):
                    order = minibatch_rng.permutation(256)
                    for start in range(0, 256, 64):
                        ids = order[start:start+64]
                        anchors = np.concatenate([g[anchor_rng.integers(len(g), size=32)] for g in anchor_groups])
                        anchor_ids_log.append(anchors)
                        dist = policy.dist(ob[ids])
                        logp_new = dist.log_prob(ac[ids]).sum(-1)
                        ratio = (logp_new-old[ids]).exp()
                        surrogate = -torch.minimum(ratio*advantage[ids], ratio.clamp(.8,1.2)*advantage[ids]).mean()
                        anchor_prediction = policy.actor.from_features(x[anchors], preview[anchors], joint[anchors])[:, :policy.action_dim]
                        anchor_loss = equivalent_loss(anchor_prediction, y[anchors], torch.tensor([.015,.15,.3][:policy.action_dim]))
                        critic_loss = .5*(policy.value(ob[ids])-targets[ids]).square().mean()
                        loss = surrogate+.1*anchor_loss+critic_loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite PPO loss')
                        actor_optimizer.zero_grad(set_to_none=True)
                        critic_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
                        if not torch.isfinite(gradient):
                            raise FloatingPointError('Nonfinite PPO gradient')
                        actor_optimizer.step()
                        critic_optimizer.step()
                        cost['actor_optimizer_steps'] += 1
                        cost['critic_optimizer_steps'] += 1
                        cost['ppo_sample_presentations'] += 64
                        cost['anchor_sample_presentations'] += 64
                        stats.append([float(surrogate.detach()), float(anchor_loss.detach()), float(critic_loss.detach()), float(gradient)])
                np.savez_compressed(OUT/'batches'/f'anchors_{step:05d}.npz', indices=np.asarray(anchor_ids_log))
                with torch.no_grad():
                    new = policy.dist(ob)
                    kl = (torch.log(new.scale/old_std)+(old_std.square()+(old_mean-new.mean).square())/(2*new.scale.square())-.5).sum(-1).mean()
                cost['ppo_updates'] += 1
                cost['optimization_wall_s'] += time.perf_counter()-optimize_started
                record = dict(step=step, update=cost['ppo_updates'], surrogate=float(np.mean(stats,axis=0)[0]),
                    anchor_loss=float(np.mean(stats,axis=0)[1]), critic_loss=float(np.mean(stats,axis=0)[2]),
                    max_preclip_gradient=float(np.max(stats,axis=0)[3]), full_batch_gaussian_kl=float(kl),
                    pre_update_log_probability_max_error=start_log_error,
                    completed_episodes=len(episodes), executor_successes=sum(bool(e['executor_success']) for e in episodes))
                curves.append(record)
                write_json(OUT/'training_curve.json', curves)
                write_json(OUT/'transitions.json', transition_records)
                buffer = []
                print('STRONG_PPO_V4', step, record, flush=True)
                if step in (1024,2048):
                    save_checkpoint(step)
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
    finally:
        for item in current.values():
            if item['actions']:
                retain(item, False, dict(reason='budget_truncated' if error is None else error))
        cost['total_wall_s'] = time.perf_counter()-started
        cost['total_cpu_s'] = time.process_time()-cpu_started
        final = save_checkpoint(cost['training_interactions'], 'final.pt')
        loaded, _ = load_strong_ppo(final)
        with torch.no_grad():
            reload_error = float((loaded.mean(x[:512])-policy.mean(x[:512])).abs().max())
        unchanged = all(sha(ROOT/'src'/n)==h for n,h in registration['source_sha256'].items())
        old = torch.load(FIT,map_location='cpu',weights_only=False)['policy']
        static_unchanged = all(torch.equal(old['actor.'+key], policy.state_dict()['actor.'+key]) for key in STATIC_KEYS)
        write_json(OUT/'transitions.json', transition_records)
        receipt = dict(status='completed' if error is None and cost['training_interactions']==STEPS else 'failed',
            seed=SEED, arm=ARM, error=error, cost=cost, model_sha256=sha(final), registration_sha256=reg_hash,
            fit_parent_unchanged=sha(FIT)==FIT_HASH, dataset_unchanged=sha(DATA)==DATA_HASH,
            registered_sources_unchanged=unchanged, static_feature_buffers_unchanged=static_unchanged,
            reload_max_error=reload_error, inherited_fit_is_same_seed_training_stage=True,
            decisions_per_path={r['id']:sum(t['path']==r['id'] for t in transition_records) for r in rows},
            episodes_completed=sum(e['complete'] for e in episodes), executor_successful_episodes=sum(bool(e['executor_success']) for e in episodes),
            budget_truncated_episodes=sum(not e['complete'] for e in episodes), no_formal_test_evaluated=True,
            physical_completion_or_efficiency_of_final_policy_not_yet_evaluated=True)
        write_json(OUT/'receipt.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
