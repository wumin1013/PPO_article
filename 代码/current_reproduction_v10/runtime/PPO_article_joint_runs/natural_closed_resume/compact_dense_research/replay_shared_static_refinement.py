"""Refit static output coefficients on fixed historical data, per own-seed base.

This is conditional replication with shared fitted feature maps and teacher
corpora. It does not independently repeat discovery of the historical maps.
No historical trained output coefficient is copied into an independent actor.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from research_policy_loader import from_saved
from localized_preview_head import LocalizedPreviewPolicy
from localized_joint_correction import LocalizedJointPolicy
from replay_lowdim_training_stage import SUPPLEMENT, SEEDS


def call(module, x):
    with torch.no_grad():
        return np.concatenate([module(torch.from_numpy(a.astype(np.float32))).numpy()
                               for a in np.array_split(x, max(1, (len(x)+2047)//2048))])


def arrays(path):
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def ridge(features, target, weight, penalty=1e-4):
    f = features.astype(np.float64)
    return np.linalg.solve(f.T @ (f*weight[:, None])+penalty*np.eye(f.shape[1]),
                           f.T @ (target*weight[:, None]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--historical-verification', action='store_true')
    parser.add_argument('--verification-label', default='preserved_anchors')
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_bound_inputs()
    started = time.perf_counter()
    seedroot = SUPPLEMENT/'independent_training'/f'seed_{args.seed}'
    initial = seedroot/'local_feedback_fit/fit_08000.pt'
    if args.historical_verification:
        initial = ROOT/'direct14_farvector_local_feedback_fit_round21/fit_08000.pt'
    assert Path(args.verification_label).name == args.verification_label
    stage = (SUPPLEMENT/('historical_static_reconstruction_'+args.verification_label) if args.historical_verification
             else seedroot/'local_features')
    stage.mkdir(exist_ok=False, parents=True)
    (stage/'source').mkdir()
    sources = [Path(__file__)] + [ROOT/n for n in (
        'localized_preview_head.py', 'localized_joint_correction.py', 'research_policy_loader.py')]
    for path in sources:
        shutil.copy2(path, stage/'source'/path.name)
    files = {
        'preview_data': ROOT/'direct14_localized_preview_fit_round49/fitting_data.npz',
        'preview_teacher': ROOT/'direct14_head_bias_search_round29/candidate_08/model.pt',
        'joint_data': ROOT/'direct14_actual_arrival_fit_round55_initialization/fit_data.npz',
        'joint_teacher': ROOT/'direct14_localized_preview_fit_round49/checkpoint_0002.pt',
        'retained_data': ROOT/'direct14_retained_basis_repair_round68_initialization/fit_data.npz',
        'dictionary_data': ROOT/'direct14_dictionary02_repair_round71_initialization/fit_data.npz',
    }
    bindings = {str(p): sha(p) for p in [initial, *sources, *files.values()]}
    write(stage/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=args.seed, initial_checkpoint=str(initial), initial_checkpoint_sha256=sha(initial),
        source_sha256=bindings, wall_budget_s=1800, new_ppo_updates=0,
        evidence_gap='Independent random initializations have not completed final static refinement and strict paired evaluation.',
        procedure='Own network plus the fixed feed-bias increment; fit 96 preview and 192 joint coefficients, then two fixed64-center refinements. Historical teacher labels and feature maps are shared; anchor actions are recomputed from the own-seed incumbent. No trained historical coefficients are installed.',
        fixed_settings=dict(feed_bias=.04, preview_radius=1.2, joint_radius=1.2,
                            retained_anchor_weight=4., dictionary_anchor_weight=4., ridge=1e-4),
        independence_scope='Independent base-network initialization and optimization, conditional on shared historical labels and fitted feature maps. Not independent feature-map discovery or a pure PPO repeat.',
        adoption='One fixed output per seed, no seed or checkpoint replacement; complete both cases at both physical steps before any outcome claim.',
        stop='Four fixed linear fits, nonfinite values, provenance error, or 1800s limit.',
        historical_verification=args.historical_verification, full_pipeline_complete=False))
    saved = torch.load(initial, map_location='cpu', weights_only=False)
    base = from_saved(saved, args.seed)
    with torch.no_grad():
        base.actor[4].bias[1] += .04
    expected_base = copy.deepcopy(base.actor.state_dict())
    records = []

    def export(policy, index, name, **metadata):
        assert time.perf_counter()-started < 1800
        payload = dict(saved)
        payload.update(policy=policy.state_dict(), interface=[14, 3, args.seed],
            updates=index, new_ppo_updates=0, ppo_updates_this_round=0,
            actor_outputs=3, observation_dimension=14, policy_memory_scalars=0,
            head_bias_delta=[0., .04, 0.], source_checkpoint=str(initial),
            source_checkpoint_sha256=sha(initial), training_kind=name,
            shared_static_feature_maps=True, **metadata)
        checkpoint = stage/f'checkpoint_{index:04d}.pt'
        torch.save(payload, checkpoint)
        record = dict(stage=name, checkpoint=str(checkpoint), sha256=sha(checkpoint))
        records.append(record)
        return record

    d = arrays(files['preview_data'])
    x, weight = d['observations'], d['weights']
    teacher = from_saved(torch.load(files['preview_teacher'], map_location='cpu', weights_only=False))
    absolute = call(teacher.actor, x)[:, 2] + d['target_delta']
    baseline = call(base.actor, x)
    anchor = weight == 4.
    absolute[anchor] = baseline[anchor, 2]
    target = absolute - baseline[:, 2]
    preview = LocalizedPreviewPolicy(args.seed)
    preview.initialize(base)
    with torch.no_grad():
        preview.actor.centers.copy_(torch.tensor(d['centers']))
        preview.actor.feature_scale.copy_(torch.tensor(d['scale']))
        preview.actor.radius.fill_(1.2)
    f = call(preview.actor.features, x)
    coefficient = ridge(f, np.arctanh(np.clip(target/.4, -.999, .999))[:, None], weight)
    with torch.no_grad():
        preview.actor.correction.weight.copy_(torch.tensor(coefficient.T, dtype=torch.float32))
    assert np.array_equal(call(preview.actor, x)[:, :2], baseline[:, :2])
    export(preview, 1, 'Shared-data preview coefficient refit',
           policy_class='LocalizedPreviewPolicy', local_basis_count=96)

    d = arrays(files['joint_data'])
    x, weight = d['observations'], d['weights']
    teacher = from_saved(torch.load(files['joint_teacher'], map_location='cpu', weights_only=False))
    absolute = call(teacher.actor, x) + d['target_delta']
    baseline = call(preview.actor, x)
    anchor = np.isin(d['source_kind'], ['unchanged_full_execution_anchor', 'unchanged_dense_anchor'])
    # Some historical anchors preserve an earlier fitted correction, not the
    # uncorrected preview backbone. Transfer that recorded correction too.
    absolute[anchor] = baseline[anchor] + d['target_delta'][anchor]
    target = absolute-baseline
    joint = LocalizedJointPolicy(args.seed, count=192)
    joint.initialize(preview)
    with torch.no_grad():
        joint.actor.centers.copy_(torch.tensor(d['centers']))
        joint.actor.feature_scale.copy_(torch.tensor(d['scale']))
        joint.actor.radius.fill_(1.2)
    f = call(joint.actor.features, x)
    scale = joint.actor.output_scale.numpy()
    coefficient = ridge(f, np.arctanh(np.clip(target/scale, -.999, .999)), weight)
    with torch.no_grad():
        joint.actor.correction.weight.copy_(torch.tensor(coefficient.T, dtype=torch.float32))
    export(joint, 2, 'Shared-data joint coefficient refit',
           policy_class='LocalizedJointPolicy', joint_basis_count=192, base_local_basis_count=96)

    for index, name in [(3, 'retained_data'), (4, 'dictionary_data')]:
        d = arrays(files[name])
        x, y, count, reuse = d['observations'], d['targets'], int(d['positive_count']), d['reused_indices']
        incumbent = copy.deepcopy(joint)
        y[count:] = call(incumbent.actor, x[count:])
        with torch.no_grad():
            joint.actor.centers[reuse] = torch.tensor(d['centers'])
            joint.actor.correction.weight[:, reuse] = 0.
        feature = call(joint.actor.features, x).astype(np.float64)
        held = joint.actor.correction.weight.detach().numpy().copy()
        target = np.arctanh(np.clip((y-call(joint.actor.backbone, x))/scale, -.999999, .999999))-feature@held.T
        weight = np.r_[np.full(count, 2.), np.full(len(x)-count, 4.)]
        selected = reuse if index == 3 else np.arange(192)
        increment = ridge(feature[:, selected], target, weight)
        held[:, selected] += increment.T
        with torch.no_grad():
            joint.actor.correction.weight.copy_(torch.tensor(held))
        error = call(joint.actor, x)-y
        record = export(joint, index, 'Shared-data '+name+' coefficient refit',
            policy_class='LocalizedJointPolicy', joint_basis_count=192, base_local_basis_count=96)
        record.update(teacher_max_error=abs(error[:count]).max(0).tolist(),
                      anchor_max_error=abs(error[count:]).max(0).tolist())
    assert all(torch.equal(v, joint.actor.backbone.backbone.state_dict()[k]) for k, v in expected_base.items())
    assert all(torch.isfinite(v).all() for v in joint.state_dict().values())
    if args.historical_verification:
        historical = [
            ROOT/'direct14_localized_preview_fit_round49/checkpoint_0002.pt',
            ROOT/'direct14_actual_arrival_fit_round55/checkpoint_0001.pt',
            ROOT/'direct14_retained_basis_repair_round68/checkpoint_0001.pt',
            ROOT/'direct14_dictionary02_resume_round73/checkpoint_0000.pt']
        for item, path in zip(records, historical):
            a = from_saved(torch.load(item['checkpoint'], map_location='cpu', weights_only=False))
            b = from_saved(torch.load(path, map_location='cpu', weights_only=False))
            item['historical_checkpoint'] = str(path)
            item['maximum_output_reconstruction_error'] = float(abs(call(a.actor, x)-call(b.actor, x)).max())
    write(stage/'fit_audit.json', records)
    if args.historical_verification:
        assert max(r['maximum_output_reconstruction_error'] for r in records) < 3e-5, records
    write(stage/'replay_receipt.json', dict(status='finished', exit_code=0,
        wall_s=time.perf_counter()-started, source_unchanged=all(sha(p)==h for p,h in bindings.items()),
        base_network_preserved=True, full_pipeline_complete=False,
        final_checkpoint=str(stage/'checkpoint_0004.pt'), new_ppo_updates=0))
    print('STATIC_REFINEMENT_COMPLETE', args.seed, records, flush=True)


if __name__ == '__main__':
    main()
