"""Minimum-norm saved-state feed fits, separating range from interference."""
import sys
import time
import argparse
import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from search_complete_return import read
from study_io import BASE, register, receipt, write_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extra-feed-head', action='store_true')
    parser.add_argument('--stage', default='canonical_gate_band_selective_cruise_capacity_20260918')
    parser.add_argument('--geometry', default='canonical_gate_band_current_hold_geometry_20260918')
    args = parser.parse_args()
    sys.path.insert(0, str(BASE/'.runtime_dependencies'))
    import clarabel
    source = BASE/args.geometry
    bound = BASE/'canonical_gate_band_current_cruise_search_capacity_20260918'
    metadata = read(source/'results.json')
    prepared = BASE/'canonical_gate_band_current_actor_head_support_20260918'
    extra_sources = ([prepared/'support.npz', prepared/'results.json', prepared/'receipt.json',
        BASE/'canonical_gate_state.py', BASE/'kernel_local_feedback_policy.py',
        BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918/candidate_1.pt']
        if args.extra_feed_head else [])
    stage = register(args.stage,
        [__file__, source/'geometry.npz', source/'results.json', source/'receipt.json',
         bound/'results.json', bound/'receipt.json']+extra_sources,
        dict(new_physics_ticks=0, new_actor_decisions=0, wall_s=180,
             conic_solves=3, target_mm_s=99., non_target_bound_factors=[1., 2., None],
             extra_feed_output_parameters=97 if args.extra_feed_head else 0),
        ('Matched representation contrast: append the existing96 feed actor output '
         'weights and its bias to the256 kernel coefficients. Reuse saved observations '
         'and check affine effects against actual float32 policy output; hidden weights '
         'and the other action heads stay fixed. All three optimization arms, targets '
         'and physical metric construction match the preceding kernel-only capacity '
         'audit. This is an output-parameter-scope change, not extra state information. '
         if args.extra_feed_head else '')+
        'The existing radius and saved-action bounds exclude99mm/s at all576 '
        'curved-route target states. Test the next necessary condition without '
        'a runtime intervention: can the SAME shared256 feed coefficients request '
        'at least99 on all1175 saved cruise states? Remove only the algorithmic '
        'radius and expand target-state feed allowance to99.9. Minimize the '
        'original physical metric norm. Compare non-target feed preservation '
        'at its original bounds, twice those PHYSICAL change bounds, and no '
        'non-target bounds. The99.9 ceiling is imposed only on saved targets '
        'and never in the runtime. Steering and preview parameter changes are '
        'zero. This is an optimistic capacity diagnostic: it omits closed-loop '
        'state shifts, shape, corridor and braking constraints, proves no actual '
        'cruise or ideal trajectory, and cannot promote a policy. Independently '
        'check all decoded saved-state requests and report unresolved numerical '
        'status without claiming infeasibility. Stop after three solves; a '
        'large required norm or preservation conflict calls for changing the '
        'representation/learning mechanism, not blindly widening physical limits.')
    start = time.perf_counter()
    with np.load(source/'geometry.npz') as z:
        d = {k:z[k].copy() for k in z.files}
    features = d['support'][1::3, 256:512]
    ids = d['cruise_state_indices']
    np.testing.assert_allclose(features[ids], d['cruise_features'], atol=1e-14)
    latent = d['latent'][:, 1]
    nominal = d['decoded_actions'][:, 1]
    np.testing.assert_allclose(nominal, 50*(1+np.tanh(latent)), atol=1e-12)
    metric = d['metric_physical'][256:512, 256:512]
    affine_error = None
    if args.extra_feed_head:
        import torch
        from canonical_gate_state import CanonicalGatePolicy
        torch.set_num_threads(1)
        model = BASE/'canonical_gate_band_per_corner_shape_priority_learning_20260918/candidate_1.pt'
        assert sha(model) == metadata['parent_sha256'] == read(prepared/'results.json')['parent_sha256']
        md = torch.load(model, map_location='cpu', weights_only=False)
        policy = CanonicalGatePolicy(256); policy.load_state_dict(md['policy'])
        with np.load(prepared/'support.npz') as z:
            observations = z['observation'].copy()
            assert np.array_equal(z['state_path'], d['state_path'])
        extra = []
        scale = float(policy.residual_scale[1])
        with torch.no_grad():
            for obs in torch.tensor(observations).split(128):
                a, b = policy.hidden_pair(obs)
                extra.append((.5*(a+b)*scale).numpy().astype(float))
        extra = np.c_[np.concatenate(extra), np.full(len(features), scale)]
        features = np.c_[features, extra]
        derivative = 50*(1-np.tanh(latent)**2)/5.
        physical = features*derivative[:, None]
        extended = physical.T@(d['state_weight'][:, None]*physical)/3.+.001*np.eye(features.shape[1])
        assert np.max(abs(metric-extended[:256, :256])) < 1e-10
        metric = extended
        probe = torch.tensor(observations[::31])
        delta = np.random.default_rng(9182055).normal(size=97)*.001
        with torch.no_grad():
            before = policy.distribution(probe).mean.numpy().astype(float)
            policy.actor[-1].weight[1].add_(torch.tensor(scale*delta[:96], dtype=torch.float32))
            policy.actor[-1].bias[1].add_(float(scale*delta[-1]))
            after = policy.distribution(probe).mean.numpy().astype(float)
        affine_error = float(np.max(abs(after[:, 1]-before[:, 1]-extra[::31]@delta)))
        assert affine_error < 1e-6 and np.array_equal(before[:, [0, 2]], after[:, [0, 2]])
        np.savez_compressed(stage/'extended_support.npz', features=features, metric=metric,
            output_scale=scale, observations=observations, latent=latent)
    dimension = features.shape[1]
    chol = np.linalg.cholesky(metric)
    f = solve_triangular(chol, features.T, lower=True).T
    target = np.zeros(len(features), dtype=bool)
    target[ids] = True
    other = ~target
    records = []
    for factor in (1., 2., None):
        assert time.perf_counter()-start < 170
        lower = np.full(len(features), -np.inf)
        upper = np.full(len(features), np.inf)
        lower[target] = np.arctanh(.98)-latent[target]
        upper[target] = np.arctanh(.998)-latent[target]
        if factor is not None:
            # Original bounds are exactly +/-2.5 mm/s, mapped through tanh.
            old_lower = d['lower_physical'].reshape(-1, 3)[:, 1]
            old_upper = d['upper_physical'].reshape(-1, 3)[:, 1]
            old_low_feed = 50*(1+np.tanh(latent+old_lower))
            old_high_feed = 50*(1+np.tanh(latent+old_upper))
            low = np.maximum(0., nominal+factor*(old_low_feed-nominal))
            high = np.minimum(100., nominal+factor*(old_high_feed-nominal))
            use = other & (low > 0.)
            lower[use] = np.arctanh(low[use]/50.-1.)-latent[use]
            use = other & (high < 100.)
            upper[use] = np.arctanh(high[use]/50.-1.)-latent[use]
        lo, hi = np.isfinite(lower), np.isfinite(upper)
        a = np.vstack([-f[lo], f[hi]])
        b = np.r_[-lower[lo], upper[hi]]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        settings.max_iter = 150
        settings.time_limit = min(50., 175.-(time.perf_counter()-start))
        settings.tol_gap_abs = 1e-9
        settings.tol_gap_rel = 1e-9
        settings.tol_feas = 1e-9
        fit = clarabel.DefaultSolver(sparse.eye(dimension, format='csc'), np.zeros(dimension),
            sparse.csc_matrix(a), b, [clarabel.NonnegativeConeT(len(b))], settings).solve()
        y = np.asarray(fit.x)
        coeff = solve_triangular(chol.T, y, lower=False)
        delta = features@coeff
        feed = 50*(1+np.tanh(latent+delta))
        violation = float(np.max(a@y-b))
        valid = str(fit.status) in ('Solved', 'AlmostSolved') and violation <= 1e-7
        name = 'unprotected' if factor is None else f'preserve_{factor:g}'
        direction = np.zeros((3, 256)); direction[1] = coeff[:256]
        np.savez_compressed(stage/f'{name}.npz', direction=direction, feed=feed,
            nominal=nominal, target_indices=ids, state_path=d['state_path'],
            lower=lower, upper=upper, actor_output_delta=coeff[256:], dual=np.asarray(fit.z))
        record = dict(arm=name, non_target_physical_bound_factor=factor,
            conic_status=str(fit.status), saved_constraints_verified=valid,
            maximum_latent_violation=violation,
            original_metric_norm=float(np.linalg.norm(y)),
            original_radius_ratio=float(np.linalg.norm(y)/.1),
            target_feed_min=float(feed[target].min()),
            target_below99_with_1e8_slack=int((feed[target]<99.-1e-8).sum()),
            non_target_max_absolute_feed_change=float(np.max(abs(feed[other]-nominal[other]))),
            non_target_rms_feed_change=float(np.sqrt(np.mean((feed[other]-nominal[other])**2))),
            solver_iterations=fit.iterations, direction_file=f'{name}.npz')
        record['extra_feed_parameters'] = dimension-256
        if not valid:
            record['returned_primal_is_not_a_valid_policy_direction'] = True
        if str(fit.status) == 'PrimalInfeasible':
            dual = np.asarray(fit.z)
            record['numerical_dual_certificate'] = dict(min_multiplier=float(dual.min()),
                stationarity_residual=float(np.max(abs(a.T@dual))),
                bound_pairing=float(b@dual), formal_exact_certificate=False)
        records.append(record)
        write_json(stage/'results.json', dict(parent_sha256=metadata['parent_sha256'],
            records=records, target_states=int(target.sum()), non_target_states=int(other.sum()),
            affine_runtime_latent_error=affine_error,
            learned_candidate_adopted=False, physical_feasibility_proven=False, goal_complete=False))
        print('SELECTIVE_CRUISE_CAPACITY', record, flush=True)
    receipt(stage, dict(status='completed_selective_saved_cruise_capacity',
        new_physics_ticks=0, new_actor_decisions=0, wall_s=time.perf_counter()-start,
        learned_candidate_adopted=False, goal_complete=False))


if __name__ == '__main__':
    main()
