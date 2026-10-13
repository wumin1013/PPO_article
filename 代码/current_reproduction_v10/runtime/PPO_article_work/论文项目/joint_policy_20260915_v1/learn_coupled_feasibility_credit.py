"""Matched parameter learning using coupled geometric constraints instead of axis bans."""
import multiprocessing as mp
import subprocess
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize
from raw_state_joint_policy import RawStateJointPolicy
from complete_local_credit import complete_metric_response, state_function_features
from optimize_complete_joint_function import initialize_worker, evaluate_job, residual
from learn_nonlinear_joint_feedback import promotion
from audit_function_constraint_interference import distances
from study_io import BASE, register, receipt, write_json, sha
from search_complete_return import read


def build_geometric_response(source, current, paired):
    offsets, matrices, records = [], [], []
    for index, nominal in enumerate(current['records']):
        with np.load(source/nominal['trace']) as z:
            ref, rows = z['reference'].copy(), z['rows'].copy()
        distance = distances(ref, rows[:, :2])
        queries = set(np.flatnonzero((distance > 1.1) & (rows[:, 12] == 0))[::5].tolist())
        queries.add(int(distance.argmax()))
        branch_rows = []
        for lo, hi in paired:
            branches = []
            for candidate in (lo, hi):
                r = candidate['records'][index]
                assert r['path'] == nominal['path']
                with np.load(source/r['trace']) as z:
                    xy = z['rows'][:, :2].copy()
                branches.append(xy)
                if not r['success']:
                    queries.update(q for q in (len(xy)-21, len(xy)-1) if 0 <= q < len(rows))
            branch_rows.append(branches)
        q = np.asarray(sorted(queries), dtype=int)
        gradient = np.zeros((len(q), len(paired)))
        known = np.ones(len(q), dtype=bool)
        for axis, (low, high) in enumerate(branch_rows):
            lo_available, hi_available = q < len(low), q < len(high)
            lo = np.zeros(len(q)); hi = np.zeros(len(q))
            lo[lo_available] = distances(ref, low[q[lo_available]])
            hi[hi_available] = distances(ref, high[q[hi_available]])
            both = lo_available & hi_available
            gradient[both, axis] = (hi[both]-lo[both])/2.
            only_low = lo_available & ~hi_available
            only_high = hi_available & ~lo_available
            gradient[only_low, axis] = distance[q[only_low]]-lo[only_low]
            gradient[only_high, axis] = hi[only_high]-distance[q[only_high]]
            # Both-failed axes are fixed by the complete-metric estimator.
            active = any(all(r['success'] for r in c['records']) for c in paired[axis])
            if active:
                known &= lo_available | hi_available
        q, gradient = q[known], gradient[known]
        margin = np.maximum(1.25-distance[q], 0.)
        offsets.append(margin)
        matrices.append(gradient)
        records.append(dict(path=nominal['path'], sampled_ticks=(q+1).tolist(),
                            minimum_baseline_margin_mm=float(margin.min()), samples=len(q)))
    return np.concatenate(offsets), np.vstack(matrices), records


def main():
    torch.set_num_threads(1)
    source = BASE/'canonical_gate_band_raw_state_joint_learning_20260919'
    data = read(source/'results.json')
    iteration = data['iterations'][-1]
    current = next(r for r in data['candidates'] if r['tag'] == iteration['previous'])
    point = int(iteration['iteration'])
    paired = [(next(r for r in data['candidates'] if r['tag'] == f'iter{point}_axis{h}_-1'),
               next(r for r in data['candidates'] if r['tag'] == f'iter{point}_axis{h}_+1')) for h in range(48)]
    retained = read(BASE/'canonical_gate_band_nonlinear_feed_closed_loop_20260919/results.json')['records']
    spec = torch.load(source/'policy_spec.pt', map_location='cpu', weights_only=False)
    steps = np.full(48, .125)
    stage = register('canonical_gate_band_coupled_feasibility_credit_20260919',
        [__file__, BASE/'complete_local_credit.py', BASE/'raw_state_joint_policy.py',
         BASE/'optimize_complete_joint_function.py', source/'results.json', source/'receipt.json',
         source/'policy_spec.pt', BASE/'audit_function_constraint_interference.py'],
        dict(reused_response_candidates=96, new_complete_candidates=3, paths=20,
             parameter_dimensions=48, scales=[.25, .5, 1.], workers=3, wall_s=180),
        'At the exact final response-collection center of the new raw-state learner, '
        'reuse96 complete-policy perturbation traces and the already finished axis-exclusion '
        'control proposals. Change ONLY proposal constraints: replace independently banned '
        'coefficient signs by joint linearized sampled XY corridor inequalities. Completion '
        'shape/time/cruise derivatives use ONLY full-success sides; failed prefixes supply '
        'physical point positions and constraint-violation times only, never efficiency. '
        'Both-failed axes remain fixed because complete-metric response is unavailable. '
        'Use the same signed shape deficits, finite amplitude and saved-state function '
        'trust region. Sampled linear geometry is an approximate proposal model, NOT '
        'a feasibility certificate; every new candidate must finish all original exact '
        'segment/gate/kinematic/braking/rest checks. Three fresh full20-path trials only. '
        'No runtime control switch, speed rule or intervention is deployed. Accept a '
        'research update only with first-inside/incoming preservation plus full success; '
        'all retained shape conditions still required for adoption. This tests whether '
        'single-direction failures unnecessarily remove learnable coordinated repairs.')
    started = time.perf_counter()
    refs, support = {}, []
    for r in current['records']:
        with np.load(source/r['trace']) as z:
            refs[r['path']] = z['reference'].copy()
            support.append(z['observations'].copy())
    p = RawStateJointPolicy(spec['parent'], spec['basis'])
    f = state_function_features(p, np.vstack(support))
    base, matrix, bounds, modes, ratios = complete_metric_response(current, paired, retained, False)
    trust = (f*steps[None, None, :]/np.array([.0025, .0125, .0125])[None, :, None]).reshape(-1, 48)
    margin, geometry, samples = build_geometric_response(source, current, paired)
    def objective(u):
        v = np.maximum(base+matrix@u, 0.)
        return float(v@v+.01*(u@u)), 2*matrix.T@v+.02*u
    solution = minimize(objective, np.zeros(48), jac=True, method='SLSQP', bounds=bounds,
        constraints=[dict(type='ineq', fun=lambda u: np.r_[1-trust@u, 1+trust@u],
                          jac=lambda u: np.vstack([-trust, trust])),
                     dict(type='ineq', fun=lambda u: (margin-geometry@u)/.01,
                          jac=lambda u: -geometry/.01)],
        options=dict(maxiter=300, ftol=1e-8))
    u = solution.x
    audit = dict(solver_success=bool(solution.success), message=solution.message,
        predicted_objective=float(np.maximum(base+matrix@u, 0.)@np.maximum(base+matrix@u, 0.)),
        current_objective=float(np.maximum(base, 0.)@np.maximum(base, 0.)),
        maximum_function_change=float(abs(trust@u).max()),
        minimum_linear_geometry_margin_mm=float((margin-geometry@u).min()),
        complete_metric_modes=modes, sampled_geometry=samples, normalized_step=u.tolist(),
        matched_axis_exclusion_iteration=iteration)
    write_json(stage/'proposal.json', audit)
    np.savez_compressed(stage/'response_model.npz', signed_deficit=base, metric_jacobian=matrix,
                        geometry_margin=margin, geometry_jacobian=geometry, function_trust=trust)
    if not solution.success or abs(trust@u).max() > 1.+1e-6 or (margin-geometry@u).min() < -1e-7:
        receipt(stage, dict(status='failed_coupled_feasibility_solve', solver=audit,
                           wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
        return
    pool = mp.get_context('spawn').Pool(3, initialize_worker,
        (stage, spec, refs, 'gate_band_canonical_kernel', started+175))
    current_coeff = np.asarray(current['coefficients'])
    output = []
    try:
        for result in pool.imap_unordered(evaluate_job,
                [(f'coupled_{scale}', current_coeff+scale*steps*u) for scale in (.25, .5, 1.)]):
            result['objective'] = float(np.linalg.norm(residual(result, retained))**2)
            result['promotion'] = promotion(result['records'], retained)
            result['primary_preserved'] = all(result['promotion'][k]
                for k in ('all_complete', 'inward_first_turns', 'entry_nonworse'))
            output.append(result)
            write_json(stage/'results.json', dict(candidates=output, current=current['tag'],
                baseline=current['summary'], predicted_objective=audit['predicted_objective'],
                matched_control=iteration, adopted=False, goal_complete=False))
            print('COUPLED_GEOMETRY', result['tag'], result['objective'], result['summary'],
                  result['primary_preserved'], flush=True)
        pool.close(); pool.join()
        usable = [r for r in output if r['primary_preserved']]
        best = min(usable, key=lambda r: r['objective']) if usable else None
        if best:
            torch.save(dict(architecture='raw_state_joint_policy', policy_spec=spec,
                coefficients=np.asarray(best['coefficients']), environment_architecture='gate_band_canonical_kernel',
                learning_method='coupled_complete_policy_parameter_optimization'), stage/'research_candidate.pt')
        receipt(stage, dict(status='completed_coupled_feasibility_parameter_probe', candidates=len(output),
            best=None if best is None else best['tag'], best_objective=None if best is None else best['objective'],
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
    except Exception as exc:
        pool.terminate(); pool.join()
        receipt(stage, dict(status='failed_coupled_feasibility_parameter_probe', error=str(exc),
            wall_s=time.perf_counter()-started, adopted=False, goal_complete=False))
        raise
    finally:
        subprocess.run([sys.executable, '-B', str(BASE/'record_joint_live_status.py'),
            '--kind', 'complete_closed_loop_parameter_optimization_finished',
            '--next', '读取联合可行性参数对照的完整轨迹与收据，判断是否解除单方向失败造成的保守限制；继续推进形状和巡航，目标未完成。'],
            cwd=BASE, check=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
