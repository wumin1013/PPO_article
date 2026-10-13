"""Complete one missing signed response within a declared current metric radius."""
import time
import numpy as np
import torch
from canonical_gate_state import CanonicalGatePolicy, checkpoint_environment_factory
from cruise_update_geometry import CruiseUpdateGeometry
from learn_single_decision_credit import review
from probe_direct_constraint_composition import metric_rows
from train_pilot import evaluate, save_torch
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_segment_cruise_reference_20260918'
    geometry = BASE/'canonical_gate_band_matched_pure_straights_geometry_20260918'
    data = read(source/'results.json')
    model = BASE/data['parent_model']
    metric = CruiseUpdateGeometry(geometry, 'physical', data['parent_sha256']).metric
    with np.load(source/'basis.npz') as z:
        arrays = {k: z[k].copy() for k in z.files}
    directions = arrays['basis']*arrays['amplitudes'][:, None]
    norms = np.sqrt(np.einsum('ij,jk,ik->i', directions, metric, directions))
    local_ids = np.flatnonzero(norms <= .007).tolist()
    assert local_ids == [0, 1]
    stage = register('canonical_gate_band_segment_cruise_small_reference_20260918',
        [__file__, model, source/'results.json', source/'receipt.json', source/'basis.npz',
         geometry/'geometry.npz', geometry/'results.json', geometry/'receipt.json',
         BASE/'canonical_gate_band_segment_cruise_coverage_20260918/results.json',
         BASE/'canonical_gate_band_segment_cruise_distance_error_20260918/results.json',
         BASE/'train_pilot.py', BASE/'canonical_gate_state.py'],
        dict(new_full_paths=29, reused_full_paths=58, actor_decisions=3500,
             physics_ticks=75000, wall_s=120, historical_metric_radius=.007,
             proposed_learning_radius=.0069, new_direction=0, new_sign=-1,
             policy_training_updates=0),
        'Preserve all four-direction broad coverage failures. The two existing '
        'responses within physical radius0.007 passed; the larger stress responses '
        'do not. Before optimization, declare a smaller0.0069 local proposal radius '
        'and complete only the missing negative of historical direction0. Reuse '
        'all58 positive local paths, keeping their exact model/trace bindings. '
        'Check both signs, full XY and terminal modes before matched cruise-target '
        'learning. The two local directions are nearly collinear and do not certify '
        'a parameter ball. Each new learned direction needs its own both-sign '
        'full-path check. Failure stops this local proposal; no rate inflation or '
        'larger step is authorized by this diagnostic.')
    start = time.perf_counter()
    costs = dict(evaluation_actor_calls=0, evaluation_physics_ticks=0)
    metadata = torch.load(model, map_location='cpu', weights_only=False)
    policy = CanonicalGatePolicy(256)
    policy.load_state_dict(metadata['policy'])
    with torch.no_grad():
        policy.residual_coefficients.add_(torch.tensor(-directions[0].reshape(3, 256), dtype=torch.float32))
    name = 'opposite_history.pt'
    save_torch(stage/name, dict(metadata, policy=policy.state_dict(), diagnostic_only=True,
        parent_sha256=data['parent_sha256'], learning_method='signed_locality_diagnostic'))
    ids = [r['original_path'] for r in data['baseline']['results']]
    paths = read(BASE/'learning_preparation_v4/paths.json')['training']
    results = evaluate(policy, [np.asarray(paths[p]) for p in ids], stage, 'negative',
        None, costs, start+120, checkpoint_environment_factory(metadata['architecture']))
    results = [dict(r, original_path=p) for r, p in zip(results, ids)]
    traces = [stage/'negative'/f'{j}.npz' for j in range(len(ids))]
    summary = review(results, traces)
    candidates = [c for c in data['candidates'] if c['direction'] in local_ids]
    candidates.append(dict(direction=0, sign=-1, amplitude=float(arrays['amplitudes'][0]),
        model=str((stage/name).relative_to(BASE)), model_sha256=sha(stage/name),
        results=results, summary=summary, traces=[str(t.relative_to(BASE)) for t in traces],
        metrics=metric_rows(results, traces, summary, data['metric_names'])))
    np.savez_compressed(stage/'basis.npz', **arrays)
    write_json(stage/'results.json', dict(data, candidates=candidates,
        full_stress_reference=source.name, local_direction_ids=local_ids,
        physical_metric_norms=norms.tolist(), proposed_learning_radius=.0069,
        diagnostic_only=True, costs=costs, goal_complete=False))
    assert costs['evaluation_actor_calls'] <= 3500 and costs['evaluation_physics_ticks'] <= 75000
    receipt(stage, dict(status='completed_small_complete_response_reference', costs=costs,
        new_paths=29, reused_paths=58, successes=summary['successes'],
        wall_s=time.perf_counter()-start, diagnostic_only=True, goal_complete=False))
    print('SIGNED_LOCALITY', summary['successes'], norms.tolist(), costs, flush=True)


if __name__ == '__main__':
    main()
