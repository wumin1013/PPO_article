"""Bind the same feasible initializer and constraints to two cruise objectives."""
import shutil
import numpy as np
from cruise_update_geometry import CruiseUpdateGeometry
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    init = BASE/'canonical_gate_band_segment_cruise_pure_init_20260918'
    result = read(init/'optimizer.json')
    assert result['eligible_for_small_probe']
    pure = BASE/'canonical_gate_band_matched_pure_straights_geometry_20260918'
    interior = BASE/'canonical_gate_band_matched_segment_interiors_geometry_20260918'
    errors = [BASE/f'canonical_gate_band_segment_cruise_{arm}_wave_error_20260918'
              for arm in ('pure', 'interiors')]
    varying = {'cruise_features', 'cruise_latent', 'cruise_weight', 'cruise_state_indices'}
    with np.load(pure/'geometry.npz') as a, np.load(interior/'geometry.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files if k not in varying)
    a, b = [read(s/'error_model.json') for s in errors]
    assert {k:v for k,v in a.items() if k!='parameter_geometry'} == {k:v for k,v in b.items() if k!='parameter_geometry'}
    for item in a['waveform_bounds']:
        assert sha(errors[0]/item['file']) == sha(errors[1]/item['file'])
    assert result['preserve_cruise_states'] and not result['preserve_cruise_deficit']
    assert result['requested_common_reduction'] == .01 and result['metric_radius'] == .0069
    stage = register('canonical_gate_band_segment_cruise_matched_round_20260918',
        [__file__, init/'optimizer.json', init/'direction.npz', init/'receipt.json',
         pure/'geometry.npz', interior/'geometry.npz', errors[0]/'error_model.json',
         errors[1]/'error_model.json', errors[0]/'verification.json',
         BASE/'solve_transition_shape_step.py', BASE/'convex_cut_solver.py',
         BASE/'probe_continuous_sensor_composition.py', BASE/'learn_canonical_credit_step.py'],
        dict(arms=['pure_straights', 'segment_interiors'], shared_initializer=True,
             optimizer_wall_s_per_arm=180, maximum_interpolation_wall_s_per_arm=60,
             both_sign_screen_wall_s_per_arm=120, screen_actor_decisions_per_arm=6000,
             learned_candidate_wall_s_per_arm=300, learned_candidate_decisions_per_arm=16000,
             candidate_scales=[.5, 1.], metric_radius=.0069),
        'Primary question: does adding conservative curved-route straight-segment '
        'interiors to the cruise objective improve actual cruise without degrading '
        'turn shape? Both arms share parentff3, all768 trainable residual coefficients, '
        'the identical physical metric/action bounds, signed waveform error model, '
        'terminal event restrictions, half corridor reserves, shape constraints and '
        '599 pure-straight feed-preservation states. Only99.5mm/s target membership '
        'and its equal-path weighting differ. Reuse one verified minimum-norm '
        'initializer because its objective and feasible set are exactly identical. '
        'Run one180s cruise optimization per arm; one saved-iterate convex-segment '
        'feasibility recovery of at most60s is permitted. Fresh+/-0.125 full29 '
        'screen checks precede learned0.5/1 candidate checks. Retention requires '
        'every measured corner metric nonworse, each positive shape total at least1% '
        'better, full64 success/constraints/stop, full time gain>=0.01s and full '
        'actual>=99mm/s cruise gain>=0.05s. In addition, the NEW mechanism is not '
        'successful unless curved-route>=99mm/s cruise itself gains>=0.05s and '
        'eligible-interior actual speed improves. Do not compare the two objective '
        'loss values as effects. On any physical/shape failure, reject that arm '
        'and analyze the cause; no repeated error inflation, larger radius or '
        'unbounded optimization continuation. This is development only; no sealed '
        'tests, runtime edits or claim of final research completion.')
    rebound = register('canonical_gate_band_segment_cruise_interiors_init_20260918',
        [__file__, stage/'registration.json', init/'optimizer.json', init/'direction.npz',
         init/'receipt.json', pure/'geometry.npz', interior/'geometry.npz',
         errors[0]/'error_model.json', errors[1]/'error_model.json'],
        dict(new_optimizer_iterations=0, new_actor_decisions=0, new_physics_ticks=0),
        'Exact feasible-set equivalence checked array-by-array; reuse the same '
        'minimum-norm direction for the second objective arm. Full constraints '
        'are re-evaluated before the cruise optimizer can start.')
    g = CruiseUpdateGeometry(interior, 'physical', result['parent_sha256'])
    with np.load(init/'direction.npz') as z: x = z['direction'].ravel()
    copied = dict(result, parameter_geometry=g.identity,
        empirical_error_model=str((errors[1]/'error_model.json').relative_to(BASE)).replace('\\','/'),
        baseline_cruise_request_deficit=g.objective(np.zeros(768))[0],
        selected_cruise_request_deficit=g.objective(x)[0],
        identical_feasible_initializer_source=init.name)
    shutil.copy2(init/'direction.npz', rebound/'direction.npz')
    write_json(rebound/'optimizer.json', copied)
    receipt(rebound, dict(status='reused_exact_matched_feasible_initializer',
        eligible_for_small_probe=True, new_actor_decisions=0, new_physics_ticks=0))
    receipt(stage, dict(status='registered_matched_cruise_objective_experiment',
        identical_feasible_set_verified=True, new_actor_decisions=0,
        new_physics_ticks=0, goal_complete=False))
    print('MATCHED_CRUISE_ROUND_READY', result['parent_sha256'], flush=True)


if __name__ == '__main__':
    main()
