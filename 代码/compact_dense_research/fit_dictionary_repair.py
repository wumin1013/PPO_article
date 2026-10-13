"""Redistribute a fixed static dictionary and fit verified local action targets."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from evaluate_historical_dense import ROOT, BASE, write, sha, verify_bound_inputs
from research_policy_loader import from_saved
from fit_retained_basis_repair import array_call, read
from fit_actual_arrival_corrections import observed_progress
from observable_feedback_gate import modify, strength


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--reuse-count', type=int, default=64)
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    assert 1 <= args.reuse_count < 192
    verify_bound_inputs()
    torch.set_num_threads(1)
    started = time.perf_counter()
    stage = ROOT / (args.name + '_initialization')
    stage.mkdir(exist_ok=False)
    (stage / 'source').mkdir()
    source = args.source.resolve()
    teacher = args.teacher.resolve()
    parent = source / f'checkpoint_{args.index:04d}.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=False)
    incumbent = from_saved(saved)
    assert len(incumbent.actor.centers) == 192
    registration = read(teacher / 'registration.json')
    assert registration['checkpoint_sha256'] == sha(parent)
    selected = read(teacher / 'result.json')['selected']
    assert all(v['selected'] and len(v['selected']['traces']) == 10 for v in selected)
    sources = [Path(__file__)] + [ROOT / n for n in [
        'research_policy_loader.py', 'localized_joint_correction.py',
        'localized_preview_head.py', 'screen_saved_mixed_checkpoints.py',
        'fit_retained_basis_repair.py', 'fit_actual_arrival_corrections.py',
        'observable_feedback_gate.py']]
    hashes = {str(p): sha(p) for p in sources}
    for path in sources:
        shutil.copy2(path, stage / 'source' / path.name)
    write(stage / 'registration.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Offline fit with fixed-capacity static dictionary redistribution; not PPO',
        parent=str(parent), parent_sha256=sha(parent),
        teacher=str(teacher), teacher_result_sha256=sha(teacher / 'result.json'),
        source_sha256=hashes, inputs=14, outputs=3, policy_memory_scalars=0,
        basis_count=192, redistributed_centers=args.reuse_count,
        anchor_weights=[4., 16., 64.], ridge=1e-4, new_ppo_updates=0,
        training_anchor_exclusion_half_width_mm=40., wall_budget_s=1800,
        evidence_gap='Complete candidate68_1 passes other gates but nominal corner02 loses historical main; round70 provides a phase-checked feasible teacher. Thirty unused centers alone leave strong targets uncovered.',
        procedure='Rank existing centers by their maximum output contribution outside the repair training window across four complete source executions. Redistribute64 low-contribution centers using farthest-point residual coverage of teacher inputs after retaining other128 centers. Fit coefficient increments on all192 features against teacher actions and unchanged complete-execution anchors; freeze the MLP, scales, radius, and remaining128 centers.',
        adoption='Same checkpoint must pass unchanged full paired numerical, physical, all34, and dense5percent gates. Local fit error is not success.',
        stop='Three prespecified anchor weights or combined30minute wall limit; uncovered material targets stop before simulation.',
        goal_complete=False, method_frozen=False))
    bindings = {str(parent): sha(parent)}
    for name in ['registration.json', 'result.json']:
        bindings[str(teacher / name)] = sha(teacher / name)
    geometry_path = BASE / 'paper_evidence_review/natural_closed_case/geometry.json'
    geometry = read(geometry_path)
    bindings[str(geometry_path)] = sha(geometry_path)
    repair_ids = {v['id'] for v in selected}
    repair_s = np.array([c['reference_s_mm'] for c in geometry['sharp_corners'] if c['id'] in repair_ids])
    all_x, masks, selection_audit = [], [], []
    for case in ['original', 'dense']:
        for key, dt in [('1ms', .001), ('half_ms', .0005)]:
            path = source / f'eval_{args.index:04d}_{case}_{key}.npz'
            result_path = path.with_suffix('.json')
            result = read(result_path)
            assert result['success'] and result['checkpoint_sha256'] == sha(parent)
            assert result['trace_sha256'] == sha(path)
            with np.load(path) as data:
                x = data['observations'].copy()
            keep = np.ones(len(x), bool)
            if case == 'original':
                gates = path.with_name(path.stem + '_gates.json')
                xx, progress = observed_progress(path, gates, dt)
                assert np.array_equal(x, xx)
                keep = (abs(progress[:, None] - repair_s[None, :]) > 40.).all(1)
                bindings[str(gates)] = sha(gates)
            all_x.append(x)
            masks.append(keep)
            selection_audit.append(dict(path=str(path), excluded_training_anchors=int((~keep).sum())))
            for bound in [path, result_path]:
                bindings[str(bound)] = sha(bound)
    all_x = np.vstack(all_x)
    keep = np.concatenate(masks)
    anchor_x = all_x[keep]
    anchor_y = array_call(incumbent.actor, anchor_x)
    positive_x, positive_y = [], []
    for item in selected:
        chosen = item['selected']
        steer, feed, gain = chosen['response']
        for name in chosen['traces']:
            path = Path(name)
            with np.load(path) as data:
                positive_x.append(data['observations'].copy())
                positive_y.append(data['latents'].copy())
            bindings[str(path)] = sha(path)
        for key in ['1ms', 'half_ms']:
            path = teacher / key / f'{item["id"]}_r04_p00.npz'
            with np.load(path) as data:
                x = data['observations'].copy()
            y = []
            for row in x:
                z = modify(incumbent.act(row, True)[0], row, feed, gain)
                z[0] += strength(row) * steer
                y.append(z)
            positive_x.append(x)
            positive_y.append(np.array(y))
            bindings[str(path)] = sha(path)
    px = np.vstack(positive_x).astype(np.float32)
    py = np.vstack(positive_y).astype(np.float32)
    output_scale = incumbent.actor.output_scale.numpy()
    strong = np.linalg.norm((py - array_call(incumbent.actor, px)) / output_scale, axis=1) > .005
    active = px[strong]
    feature_scale = incumbent.actor.feature_scale.numpy()
    old_weights = incumbent.actor.correction.weight.detach().numpy()
    anchor_features = array_call(incumbent.actor.features, anchor_x)
    importance = anchor_features.max(0) * np.linalg.norm(old_weights * output_scale[:, None], axis=0)
    reuse = np.argsort(importance, kind='stable')[:args.reuse_count]
    retained = np.setdiff1d(np.arange(192), reuse)
    scaled = active / feature_scale
    fixed_centers = incumbent.actor.centers.numpy()[retained] / feature_scale
    distance = ((scaled[:, None] - fixed_centers[None, :]) ** 2).sum(2).min(1)
    chosen = []
    for _ in reuse:
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, ((scaled - scaled[index]) ** 2).sum(1))
    centers = active[chosen]
    assert len(np.unique(centers, axis=0)) == args.reuse_count
    coverage = dict(strong_samples=int(strong.sum()), uncovered=int((distance >= float(incumbent.actor.radius) ** 2).sum()),
                    maximum_distance=float(np.sqrt(distance.max())), reused_indices=reuse.tolist(),
                    importance=importance[reuse].tolist(), selection=selection_audit)
    write(stage / 'coverage_audit.json', coverage)
    assert coverage['uncovered'] == 0, 'Redistributed dictionary still misses material teacher targets'
    initial = from_saved(saved)
    with torch.no_grad():
        initial.actor.centers[reuse] = torch.tensor(centers)
        initial.actor.correction.weight[:, reuse] = 0.
    lookup = {row.tobytes(): target for row, target in zip(anchor_x, anchor_y)}
    conflicts = [float(abs(target - lookup[row.tobytes()]).max()) for row, target in zip(px, py) if row.tobytes() in lookup]
    assert not any(v > 1e-5 for v in conflicts), 'Conflicting exact teacher and anchor labels'
    x = np.vstack([px, anchor_x])
    y = np.vstack([py, anchor_y])
    backbone = array_call(initial.actor.backbone, x)
    features = array_call(initial.actor.features, x).astype(float)
    initial_weights = initial.actor.correction.weight.detach().numpy().copy()
    target = np.arctanh(np.clip((y - backbone) / output_scale, -.999999, .999999)) - features @ initial_weights.T
    baseline_delta = array_call(initial.actor, anchor_x) - anchor_y
    write(stage / 'initialization_audit.json', dict(
        unchanged_initialization_claim=False,
        anchor_max_delta=abs(baseline_delta).max(0).tolist(),
        anchor_rms_delta=np.sqrt((baseline_delta ** 2).mean(0)).tolist(),
        exact_label_conflicts_above1e_5=0,
        note='Some redistributed centers were active; initial differences are measured and retained, not claimed zero.'))
    np.savez_compressed(stage / 'fit_data.npz', observations=x, targets=y,
                        positive_count=len(px), reused_indices=reuse, centers=centers)
    write(stage / 'data_binding.json', dict(source_sha256=bindings, fit_data_sha256=sha(stage / 'fit_data.npz'), counterfactual_labels_explicit=True))
    gram_positive = features[:len(px)].T @ features[:len(px)]
    gram_anchor = features[len(px):].T @ features[len(px):]
    rhs_positive = features[:len(px)].T @ target[:len(px)]
    rhs_anchor = features[len(px):].T @ target[len(px):]
    records = []
    for index, weight in enumerate([4., 16., 64.]):
        delta_w = np.linalg.solve(2. * gram_positive + weight * gram_anchor + 1e-4 * np.eye(192),
                                  2. * rhs_positive + weight * rhs_anchor)
        policy = from_saved(saved)
        with torch.no_grad():
            policy.actor.centers[reuse] = torch.tensor(centers)
            policy.actor.correction.weight.copy_(torch.tensor(initial_weights + delta_w.T, dtype=torch.float32))
        for key, value in policy.state_dict().items():
            if key == 'actor.centers':
                assert torch.equal(value[retained], saved['policy'][key][retained])
            elif key != 'actor.correction.weight':
                assert torch.equal(value, saved['policy'][key]), key
        errors = array_call(policy.actor, x) - y
        records.append(dict(index=index, anchor_weight=weight,
                            teacher_max_error=abs(errors[:len(px)]).max(0).tolist(),
                            anchor_max_error=abs(errors[len(px):]).max(0).tolist(),
                            anchor_rms_error=np.sqrt((errors[len(px):] ** 2).mean(0)).tolist()))
        payload = dict(saved)
        payload.update(policy=policy.state_dict(), updates=index, new_ppo_updates=0,
                       parent_checkpoint=str(parent), parent_checkpoint_sha256=sha(parent),
                       training_kind='Offline fixed-capacity dictionary redistribution', checkpoint_index_kind='anchor_weight_index')
        torch.save(payload, stage / f'checkpoint_{index:04d}.pt')
    write(stage / 'fit_audit.json', records)
    print('DICTIONARY_REPAIR_FIT', json.dumps(records), flush=True)
    assert time.perf_counter() - started < 60., 'Insufficient remaining budget for standard screening'
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'screen_saved_mixed_checkpoints.py'),
                             '--name', args.name, '--source', str(stage), '--counts', '0', '1', '2',
                             '--localized-joint-head'])
    write(stage / 'receipt.json', dict(status='finished', exit_code=result.returncode,
        combined_wall_s=time.perf_counter() - started, sources_unchanged=all(sha(p) == h for p, h in hashes.items()),
        new_ppo_updates=0, goal_complete=False, method_frozen=False))


if __name__ == '__main__':
    main()
