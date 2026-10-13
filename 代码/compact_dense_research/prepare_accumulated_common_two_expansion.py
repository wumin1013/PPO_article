"""Prepare paired action availability from a successful own-seed two-output model."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT, write, sha
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS
from localized_two_action_policy import policy_from
from localized_joint_correction import LocalizedJointPolicy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    parent_stage = SUPPLEMENT/'two_accumulated_feedback'/f'seed_{args.seed}'
    assert read(parent_stage/'evaluation/replay_receipt.json')['all_four_complete_success'], 'Common two must first complete all four checks.'
    parent = parent_stage/'fit_30000.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    two = policy_from(saved, args.seed).eval()
    three = LocalizedJointPolicy(args.seed, count=192, base_count=96).eval()
    feature_source = ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt'
    historical = policy_from(torch.load(feature_source, map_location='cpu', weights_only=False), args.seed)
    latent = float(np.float32((3.-.8)/5.6-1.))
    distance = .8+5.6*(1.+latent)
    with torch.no_grad():
        for index in [0, 2]:
            three.actor.backbone.backbone[index].load_state_dict(two.actor.backbone[index].state_dict())
        three.actor.backbone.backbone[4].weight.zero_()
        three.actor.backbone.backbone[4].bias.fill_(latent)
        three.actor.backbone.backbone[4].weight[:2].copy_(two.actor.backbone[4].weight)
        three.actor.backbone.backbone[4].bias[:2].copy_(two.actor.backbone[4].bias)
        for name in ['centers', 'feature_scale', 'radius']:
            getattr(three.actor, name).copy_(getattr(two.actor, name))
            getattr(three.actor.backbone, name).copy_(getattr(historical.actor.backbone, name))
        three.actor.correction.weight.zero_()
        three.actor.correction.weight[:2].copy_(two.actor.correction.weight)
        three.actor.backbone.correction.weight.zero_()
        three.critic.load_state_dict(two.critic.state_dict())
        three.log_std[:2].copy_(two.log_std)
        three.log_std[2] = np.log(.02)
    samples, bindings = [], {str(parent): sha(parent), str(feature_source): sha(feature_source), str(Path(__file__)): sha(__file__)}
    for path in sorted((parent_stage/'evaluation').glob('eval_30000_*.npz')):
        result = read(path.with_suffix('.json'))
        assert result['success'] and result['checkpoint_sha256'] == sha(parent)
        assert result['trace_sha256'] == sha(path)
        with np.load(path) as data:
            samples.append(data['observations'].copy())
        bindings[str(path)] = sha(path)
    assert len(samples) == 4
    observations = torch.tensor(np.vstack(samples), dtype=torch.float32)
    with torch.no_grad():
        batched_error = max(float((three.actor(x)[:, :2]-two.actor(x)).abs().max()) for x in observations.split(512))
        pointwise_error = max(float((three.actor(x)[:2]-two.actor(x)).abs().max()) for x in observations[::97])
        preview_error = max(float((three.actor(x)[:, 2]-latent).abs().max()) for x in observations.split(512))
    assert batched_error < 2e-6 and pointwise_error < 2e-6 and preview_error == 0.
    stage = SUPPLEMENT/'common_two_action_expansion'/f'seed_{args.seed}'
    stage.mkdir(parents=True, exist_ok=False)
    records = []
    for arm, policy in [('two', two), ('three', three)]:
        folder = stage/arm
        folder.mkdir()
        payload = copy.copy(saved)
        payload.update(policy=policy.state_dict(), interface=[14, 2 if arm=='two' else 3, args.seed],
            actor_outputs=2 if arm=='two' else 3, updates=0, new_ppo_updates=0,
            preview_rule_mm=distance, parent_checkpoint=str(parent), parent_checkpoint_sha256=sha(parent),
            policy_class='LocalizedTwoPolicy' if arm=='two' else 'LocalizedJointPolicy',
            joint_basis_count=192, base_local_basis_count=96, local_basis_count=192,
            training_kind='Paired shared two-action initialization; exact weight expansion, zero new training',
            offline_iterations_this_operation=0, paired_arm=arm)
        checkpoint = folder/'initial.pt'
        torch.save(payload, checkpoint)
        records.append(dict(arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    write(stage/'initialization_audit.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed, source_sha256=bindings, records=records,
        witnessed_observations=len(observations), max_batched_first_two_error=batched_error,
        max_sampled_pointwise_first_two_error=pointwise_error, preview_latent=latent,
        common_preview_mm=distance, preview_only_weights_zero=True,
        common_steering_feed_weights_identical=True, full_execution_equivalence_pending=True,
        rule='Both arms must pass strict initial evaluation; inspect actual trajectory discrepancy before paired PPO.',
        independence='Own-seed optimized two-action coefficients; shared static feature maps and teacher corpus; no independent teacher rediscovery.'))
    print('EXPANSION_PREPARED', args.seed, records, 'action_errors', batched_error, pointwise_error)


if __name__ == '__main__':
    main()
