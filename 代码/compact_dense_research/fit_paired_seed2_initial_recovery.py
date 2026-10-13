"""One registered shared-start repair, using only effective feed requests."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from evaluate_historical_dense import ROOT, write, sha, verify_bound_inputs
from paired_variable_radius_policy import policy_from
from fit_matched_local_recovery import call


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    verify_bound_inputs()
    seed = 923202
    sup = ROOT/'experimental_supplement_20260923'
    parent_stage = sup/'common_two_action_expansion'/f'seed_{seed}'
    probe = sup/'paired_seed2_original_probe/half_ms'
    teacher = probe/'trial_10.npz'
    parents = {arm: parent_stage/arm/'initial.pt' for arm in ['two', 'three']}
    stage = sup/'paired_seed2_initial_recovery'/f'seed_{seed}'/'attempt_2'
    stage.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bindings = {str(p): sha(p) for p in [*parents.values(), teacher, probe/'trials.json',
        probe/'registration.json', Path(__file__), ROOT/'paired_variable_radius_policy.py',
        ROOT/'variable_radius_recovery_policy.py', ROOT/'fit_matched_local_recovery.py']}
    write(stage/'registration.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, source_sha256=bindings, wall_budget_s=120, new_ppo_updates=0,
        training_kind='One offline local static-feature coefficient fit, shared by both arms',
        evidence_gap='Both expanded starts fail the same original half-ms arrival. A zero-preview-offset local probe recovers it.',
        procedure='Use trial10 feed-minus0.02 targets only where exact single-input clipped requests change (indices4/8). Reuse two inactive features with per-center radii at most0.8times nearest protected distance. Copy the same centers/radii/first-two coefficients to both arms; leave preview constant.',
        protected='Three successful full traces plus failed half-ms prefix excluding its final12 decisions.',
        architecture='192 static features and14 observations unchanged; per-center support radii provided identically to both arms. No policy memory.',
        adoption='Exact protected outputs, equal first-two outputs, constant preview, then local closed-loop and unconditional four-case evaluation of both arms. This is not full success evidence.',
        stop='One candidate,120seconds, changed protected output or unequal paired requests. Retain every failure.'))
    trial = read(probe/'trials.json')[10]
    assert trial['robust_local_candidate'] and trial['delta'] == [0., -.02, 0.]
    assert trial['trace_sha256'] == sha(teacher)
    bound = read(probe/'registration.json')['source_sha256']
    assert any((ROOT/Path(p)).resolve() == parents['three'].resolve() and h == sha(parents['three']) for p, h in bound.items())
    saved = {a: torch.load(p, map_location='cpu', weights_only=False) for a, p in parents.items()}
    incumbent = policy_from(saved['three'], seed).eval()
    three = policy_from(saved['three'], seed).eval()
    two = policy_from(saved['two'], seed).eval()
    protected = []
    for case in ['original', 'dense']:
        for key in ['1ms', 'half_ms']:
            path = parent_stage/'three/evaluation'/f'eval_0000_{case}_{key}.npz'
            result = read(path.with_suffix('.json'))
            assert result['trace_sha256'] == sha(path) and result['checkpoint_sha256'] == sha(parents['three'])
            with np.load(path) as data:
                x = data['observations'].copy().astype(np.float32)
            if case == 'original' and key == 'half_ms':
                assert not result['success']
                x = x[:-12]
            else:
                assert result['success']
            protected.append(x)
            for p in [path, path.with_suffix('.json')]:
                bindings[str(p)] = sha(p)
    protected = np.vstack(protected)
    with np.load(teacher) as data:
        all_x = data['observations'].copy().astype(np.float32)
        all_y = data['latents'].copy().astype(np.float32)
    initial = np.array([incumbent.act(x, True)[0] for x in all_x])
    effective = np.flatnonzero(np.max(abs(np.clip(all_y, -1, 1)-np.clip(initial, -1, 1)), axis=1) > 1e-6)
    assert effective.tolist() == [4, 8], effective
    px, py = all_x[effective], all_y[effective]
    unused = np.flatnonzero(call(incumbent.actor.features, protected).max(0) == 0.)
    assert len(unused) >= len(px)
    reuse = unused[:len(px)]
    nearest = np.full(len(px), np.inf)
    scale = incumbent.actor.feature_scale.numpy()
    for part in np.array_split(protected, 100):
        nearest = np.minimum(nearest, np.sqrt((((px[:, None]-part[None])/scale)**2).sum(2)).min(1))
    assert nearest.min() > 1e-5
    radii = np.full(192, float(incumbent.actor.radius), dtype=np.float32)
    radii[reuse] = np.minimum(radii[reuse], .8*nearest)
    with torch.no_grad():
        three.actor.radius = torch.tensor(radii)
        three.actor.centers[reuse] = torch.tensor(px)
        three.actor.correction.weight[:, reuse] = 0.
    assert call(three.actor.features, protected)[:, reuse].max() == 0.
    f = call(three.actor.features, px)
    target = np.arctanh(np.clip((py-call(three.actor.backbone, px))/three.actor.output_scale.numpy(), -.999999, .999999))
    target -= f@three.actor.correction.weight.detach().numpy().T
    matrix = f[:, reuse].astype(float)
    coefficients = np.linalg.solve(matrix.T@matrix+1e-10*np.eye(len(px)), matrix.T@target)
    with torch.no_grad():
        # Only the feed row changes; steering and preview remain bitwise unchanged.
        three.actor.correction.weight[1, reuse] = torch.tensor(coefficients[:, 1], dtype=torch.float32)
        two.actor.radius = three.actor.radius.clone()
        two.actor.centers.copy_(three.actor.centers)
        two.actor.correction.weight.copy_(three.actor.correction.weight[:2])
    protected_error = float(abs(call(three.actor, protected)-call(incumbent.actor, protected)).max())
    assert protected_error == 0., protected_error
    fit_error = abs(call(three.actor, px)-py).max(0)
    assert np.all(fit_error < 1e-5), fit_error
    all_inputs = np.vstack([protected, all_x])
    pair_error = float(abs(call(three.actor, all_inputs)[:, :2]-call(two.actor, all_inputs)).max())
    preview_error = float(abs(call(three.actor, all_inputs)[:, 2]-saved['three']['policy']['actor.backbone.backbone.4.bias'][2].item()).max())
    assert pair_error < 2e-6 and preview_error == 0.
    point_error = max(float(abs(three.act(x, True)[0][:2]-two.act(x, True)[0]).max()) for x in all_inputs[::97])
    assert point_error < 2e-6
    records = []
    for arm, model in [('two', two), ('three', three)]:
        folder = stage/arm
        folder.mkdir()
        payload = dict(saved[arm])
        payload.update(policy=model.state_dict(),
            policy_class='LocalizedTwoVariableRadiusPolicy' if arm == 'two' else 'LocalizedJointVariableRadiusPolicy',
            updates=0, new_ppo_updates=0, parent_checkpoint=str(parents[arm]),
            parent_checkpoint_sha256=sha(parents[arm]), joint_support_radius_kind='per_center',
            new_static_radius_parameters=191,
            training_kind='Shared paired initial offline feed recovery; zero PPO', checkpoint_index_kind='fixed_candidate')
        checkpoint = folder/'initial.pt'
        torch.save(payload, checkpoint)
        records.append(dict(arm=arm, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint)))
    write(stage/'fit_audit.json', dict(source_sha256=bindings, protected_input_count=len(protected),
        effective_target_indices=effective.tolist(), reused_indices=reuse.tolist(), new_radii=radii[reuse].tolist(),
        nearest_protected_distances=nearest.tolist(), protected_max_output_error=protected_error,
        teacher_max_output_error=fit_error.tolist(), paired_max_output_error=pair_error,
        paired_sampled_pointwise_error=point_error, preview_constant_error=preview_error, records=records,
        common_preview_mm=saved['two']['preview_rule_mm'], no_complete_success_claim=True))
    assert all(sha(p) == h for p, h in bindings.items())
    assert time.perf_counter()-started < 120.
    write(stage/'replay_receipt.json', dict(status='finished', wall_s=time.perf_counter()-started,
        source_unchanged=True, new_ppo_updates=0, complete_evaluation_pending=True))
    print('PAIRED_INITIAL_FIT', protected_error, pair_error, fit_error.tolist(), flush=True)


if __name__ == '__main__':
    main()
