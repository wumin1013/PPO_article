"""Bounded information/credit audit using preserved batches and synthetic pairs."""
import copy
import json
import math
from pathlib import Path
import numpy as np
from study_io import BASE, register, receipt, sha, write_json
from resume_codec_v5 import load_resume
from joint_runtime_v5 import environment_factory
from joint_policy import advantages
from compact_observation_candidate import compact_observation, feature_names

GROUPS = ['joint_independent_refinement_v7', 'joint_gsde_refinement_v7',
          'fixed3_independent_refinement_v7', 'fixed3_gsde_refinement_v7']


def main():
    sources = [Path(__file__), BASE/'compact_observation_candidate.py', BASE/'joint_runtime.py',
        BASE/'joint_runtime_v2.py', BASE/'joint_runtime_v3.py', BASE/'joint_runtime_v5.py',
        BASE/'joint_policy.py', BASE/'learning_update_v4.py', BASE/'learning_preparation_v4/paths.json']
    sources += [BASE/(name+'_reboot_recovery_v3')/'restoration.json' for name in GROUPS]
    stage = register('minimal_information_credit_audit_v1', sources,
        dict(saved_batch_loads=4, policy_forwards=0, optimizer_updates=0,
             synthetic_environment_constructions=12, maximum_macro_steps=12,
             maximum_physical_ticks=25440, full_evaluation_runs=0),
        'Source-derived state redundancy and task alias checks; exact reward/critic transformation on frozen buffers; no trained compact-policy claim.')
    paths = json.loads((BASE/'learning_preparation_v4/paths.json').read_text(encoding='utf-8'))
    lengths = [float(np.linalg.norm(np.diff(np.asarray(p), axis=0), axis=1).sum()) for p in paths['training']]
    batches = []
    for name in GROUPS:
        restoration = json.loads((BASE/(name+'_reboot_recovery_v3')/'restoration.json').read_text())
        snapshot = Path(restoration['snapshot'])
        assert sha(snapshot) == restoration['snapshot_sha256']
        state = load_resume(snapshot)
        batch = state['batch']
        obs = np.asarray(batch['observation'], dtype=float)
        values = np.asarray(batch['value'], dtype=float)
        next_values = np.asarray(batch['next_value'], dtype=float)
        rewards = np.asarray(batch['reward'], dtype=float)
        live = 1.-np.asarray(batch['terminal'], dtype=float)
        # Phi = -remaining/100; observation slot 15 is log1p(remaining/30)/8.
        phi = -.3*np.expm1(8.*obs[:, 15])
        fixed_d = 3. if state['mode'] == 'fixed' else None
        env = environment_factory(np.asarray(paths['training'][state['path_index']]), fixed_d=fixed_d)
        env.restore(state['environment'])
        next_phi = np.r_[phi[1:], env.potential()]*live
        base_rewards = rewards-next_phi+phi
        shifted_values = values+phi
        shifted_next_values = next_values+next_phi
        original_delta = rewards+live*next_values-values
        shifted_delta = base_rewards+live*shifted_next_values-shifted_values
        old_adv, _ = advantages(rewards, values, next_values, batch['terminal'])
        new_adv, _ = advantages(base_rewards, shifted_values, shifted_next_values, batch['terminal'])
        # Compare successful return identities across saved completed episodes.
        errors = []
        success_returns, failure_returns = [], []
        margins = []
        for episode in state['episodes']:
            length = lengths[episode['path']]
            raw_return = episode['return_sum']-length/100.
            if episode['success']:
                errors.append(abs(raw_return+episode['elapsed_s']))
                success_returns.append(raw_return)
            else:
                failure_returns.append(raw_return)
                horizon = round(max(5., length/20.+1.)/.001)*.001
                margins.append(-horizon-raw_return)
        batches.append(dict(stage=name, decision=state['decision'], rows=len(obs),
            snapshot_sha256=sha(snapshot),
            constant_features=[i for i in range(72) if np.ptp(obs[:, i]) < 1e-9],
            slot_ranges=[[float(x), float(y)] for x, y in zip(obs.min(0), obs.max(0))],
            exact_projection_duplicate_max_error=float(np.max(abs(obs[:, 22:24]+obs[:, :2]/24.))),
            pbrs_td_max_difference=float(np.max(abs(original_delta-shifted_delta))),
            pbrs_gae_max_difference=float(np.max(abs(old_adv-new_adv))),
            reward_quantiles=np.quantile(rewards,[0,.01,.5,.99,1]).tolist(),
            unshaped_reward_quantiles=np.quantile(base_rewards,[0,.01,.5,.99,1]).tolist(),
            gae_std=float(np.std(old_adv)),
            successes=len(success_returns), failures=len(failure_returns),
            successful_return_identity_max_error=max(errors, default=None),
            failure_below_slowest_success_margin_min_s=min(margins, default=None)))
    # Frozen physical transition contrasts; commands are identical within a pair.
    reference = np.asarray([[-20.,0.],[0.,0.],[0.,25.]])
    pairs = []
    physical_ticks = 0
    for changed in ('held', 'acceleration', 'heading', 'velocity'):
        left, right = environment_factory(reference), environment_factory(reference)
        left.position[:] = [-8., .15]
        left.tracker.position = left.position.copy()
        left.velocity[:] = [40.,0.]
        right.restore(left.state())
        if changed == 'held': right.held[:] = [.8, 90., 11.]
        if changed == 'acceleration': right.acceleration[:] = [-600., 200.]
        if changed == 'heading': right.heading += .4
        if changed == 'velocity': right.velocity[:] = [20.,5.]
        c0, c1 = compact_observation(left), compact_observation(right)
        action = np.array([.03, .35, -.7])
        _, r0, d0, i0 = left.step(action)
        _, r1, d1, i1 = right.step(action)
        physical_ticks += len(i0['rows'])+len(i1['rows'])
        pairs.append(dict(changed=changed, compact_observation_difference=float(np.max(abs(c0-c1))),
            physical_rows_difference=float(np.max(abs(i0['rows']-i1['rows']))),
            reward_difference=abs(r0-r1), terminal_equal=d0==d1,
            interpretation='exactly removable previous held action' if changed=='held' else 'physical state must remain distinguishable'))
    assert pairs[0]['physical_rows_difference'] == pairs[0]['reward_difference'] == 0
    assert all(p['compact_observation_difference'] > 0 and p['physical_rows_difference'] > 0 for p in pairs[1:])
    for row in batches:
        assert row['pbrs_td_max_difference'] < 1e-10 and row['pbrs_gae_max_difference'] < 1e-6
        assert row['successful_return_identity_max_error'] < 1e-8
        assert row['failure_below_slowest_success_margin_min_s'] > 1.
    result = dict(batches=batches, physical_pairs=pairs, candidate_dimension=len(feature_names()),
        candidate_feature_names=feature_names(), wider_candidate_dimension=len(feature_names((3.,6.,12.,24.))),
        maximum_training_path_length_mm=max(lengths),
        gae_lam=.98, gae_half_life_decisions=math.log(.5)/math.log(.98),
        gae_weight_at_50_decisions=.98**50, gae_weight_at_100_decisions=.98**100,
        compact_state_sufficiency_proved=False, compact_policy_trained=False, goal_complete=False,
        caveat='A finite local geometry encoding remains partial; pairwise checks detect specific omissions and do not prove sufficiency on arbitrary paths.')
    write_json(stage/'results.json', result)
    receipt(stage, dict(status='completed_bounded_information_credit_audit', saved_batch_loads=4,
        synthetic_environment_constructions=12, macro_steps=8, physics_ticks=physical_ticks,
        model_forwards=0, optimizer_updates=0, training_decisions=0, goal_complete=False))
    print(json.dumps(dict(stage=str(stage), candidate_dimension=result['candidate_dimension'],
        physical_pairs=pairs, batch_findings=[{k:r[k] for k in ('stage','rows','pbrs_gae_max_difference',
        'successful_return_identity_max_error','failure_below_slowest_success_margin_min_s')} for r in batches]),ensure_ascii=False))


if __name__ == '__main__':
    main()
